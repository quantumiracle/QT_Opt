'''
QT-Opt: Q-value assisted CEM policy learning,
for reinforcement learning on robotics.

QT-Opt: https://arxiv.org/pdf/1806.10293.pdf
CEM: https://www.youtube.com/watch?v=tNAIHEse7Ms

Pytorch implementation
CEM for fitting the weights: w * s + b = a
'''


import math
import random
import argparse

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher


# use_cuda = torch.cuda.is_available()
# device   = torch.device("cuda" if use_cuda else "cpu")
# print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class ContinuousActionLinearPolicy(object):
    def __init__(self, theta, state_dim, action_dim):
        assert len(theta) == (state_dim + 1) * action_dim
        self.W = theta[0 : state_dim * action_dim].reshape(state_dim, action_dim)
        self.b = theta[state_dim * action_dim : None].reshape(1, action_dim)
    def act(self, state):
        # a = state.dot(self.W) + self.b
        a = np.dot(state, self.W) + self.b
        return a
    def update(self, theta):
        self.W = theta[0 : state_dim * action_dim].reshape(state_dim, action_dim)
        self.b = theta[state_dim * action_dim : None].reshape(1, action_dim)


class CEM():
    ''' cross-entropy method, as optimization of the action policy 
    the policy weights theta are stored in this CEM class instead of in the policy
    '''
    def __init__(self, theta_dim, ini_mean_scale=0.0, ini_std_scale=1.0):
        self.theta_dim = theta_dim
        self.initialize(ini_mean_scale=ini_mean_scale, ini_std_scale=ini_std_scale)

        
    def sample(self):
        theta = self.mean + np.random.randn(self.theta_dim) * self.std
        return theta

    def initialize(self, ini_mean_scale=0.0, ini_std_scale=1.0):
        self.mean = ini_mean_scale*np.ones(self.theta_dim)
        self.std = ini_std_scale*np.ones(self.theta_dim)

    def sample_multi(self, n):
        theta_list=[]
        for i in range(n):
            theta_list.append(self.sample())
        return np.array(theta_list)


    def update(self, selected_samples):
        self.mean = np.mean(selected_samples, axis = 0)
        # print('mean: ', self.mean)
        self.std = np.std(selected_samples, axis = 0) # plus the entropy offset, or else easily get 0 std
        # print('mean std: ', np.mean(self.std))

        return self.mean, self.std




class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QT_Opt():
    def __init__(self, replay_buffer, hidden_dim, q_lr=3e-4, cem_update_itr=4, select_num=6, num_samples=64):
        self.num_samples = num_samples
        self.select_num = select_num
        self.cem_update_itr = cem_update_itr
        self.replay_buffer = replay_buffer
        self.qnet = QNetwork(state_dim+action_dim, hidden_dim).to(device) # gpu
        self.target_qnet1 = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.target_qnet2 = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.cem = CEM(theta_dim = (state_dim + 1) * action_dim)  # cross-entropy method for updating
        theta = self.cem.sample()
        self.policy = ContinuousActionLinearPolicy(theta, state_dim, action_dim)

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.step_cnt = 0

    def update(self, batch_size, gamma=0.9, soft_tau=1e-2, update_delay=100):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        self.step_cnt+=1

        
        state_      = torch.FloatTensor(state).to(device)
        next_state_ = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predict_q = self.qnet(state_, action) # predicted Q(s,a) value

        # get argmax_a' from the CEM for the target Q(s', a'), together with updating the CEM stored weights
        new_next_action = []
        for i in range(batch_size):      # batch of states, use them one by one
            new_next_action.append(self.cem_optimal_action(next_state[i]))

        new_next_action=torch.FloatTensor(new_next_action).to(device)

        target_q_min = torch.min(self.target_qnet1(next_state_, new_next_action), self.target_qnet2(next_state_, new_next_action))
        target_q = reward + (1-done)*gamma*target_q_min

        q_loss = ((predict_q - target_q.detach())**2).mean()  # MSE loss, note that original paper uses cross-entropy loss
        print('Q Loss: ',q_loss)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # update the target nets, according to original paper:
        # one with Polyak averaging, another with lagged/delayed update
        self.target_qnet1=self.target_soft_update(self.qnet, self.target_qnet1, soft_tau)
        self.target_qnet2=self.target_delayed_update(self.qnet, self.target_qnet2, update_delay)
    

    def cem_optimal_action(self, state):
        ''' evaluate action wrt Q(s,a) to select the optimal using CEM
        return the only one largest, very gready
        '''
        cuda_state = torch.FloatTensor(state).to(device)

        ''' the following line is critical:
        every time use a new/initialized cem, and cem is only for deriving the argmax_a', 
        but not for storing the weights of the policy.
        Without this line, the Q-network cannot converge, the loss will goes to infinity through time.
        I think the reason is that if you try to use the cem (gaussian distribution of policy weights) fitted 
        to the last state for the next state, it will generate samples mismatched to the global optimum for the 
        current state, which will lead to a local optimum for current state after cem iterations. And there may be
        several different local optimum for a similar state using cem from different last state, which will cause
        the optimal Q-value cannot be learned and even have a divergent loss for Q learning.
        '''
        self.cem.initialize()  # the critical line
        for itr in range(self.cem_update_itr):
            q_values=[]
            theta_list = self.cem.sample_multi(self.num_samples)
            # print(theta_list)
            for j in range(self.num_samples):
                self.policy.update(theta_list[j])
                one_action = torch.FloatTensor(self.policy.act(state)).to(device)
                # print(one_action)
                q_values.append( self.target_qnet1(cuda_state.unsqueeze(0), one_action).detach().cpu().numpy()[0][0]) # 2 dim to scalar
            idx=np.array(q_values).argsort()[-int(self.select_num):]  # select maximum q
            max_idx=np.array(q_values).argsort()[-1]  # select maximal one q
            selected_theta = theta_list[idx]
            mean, _= self.cem.update(selected_theta)  # mean as the theta for argmax_a'
            self.policy.update(mean)
        max_theta=theta_list[max_idx]
        self.policy.update(max_theta)
        action = self.policy.act(state)[0] # [0]: 2 dim -> 1 dim
        return action

    def target_soft_update(self, net, target_net, soft_tau):
        ''' Soft update the target net '''
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def target_delayed_update(self, net, target_net, update_delay):
        ''' delayed update the target net '''
        if self.step_cnt%update_delay == 0:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    param.data 
                )

        return target_net

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path)
        torch.save(self.target_qnet1.state_dict(), path)
        torch.save(self.target_qnet2.state_dict(), path)

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet1.load_state_dict(torch.load(path))
        self.target_qnet2.load_state_dict(torch.load(path))
        self.qnet.eval()
        self.target_qnet1.eval()
        self.target_qnet2.eval()




def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.plot(rewards)
    plt.savefig('qt_opt_v2.png')
    # plt.show()
if __name__ == '__main__':

    # choose env
    ENV = ['Pendulum', 'Reacher'][0]
    if ENV == 'Reacher':
        NUM_JOINTS=2
        LINK_LENGTH=[200, 140]
        INI_JOING_ANGLES=[0.1, 0.1]
        # NUM_JOINTS=4
        # LINK_LENGTH=[200, 140, 80, 50]
        # INI_JOING_ANGLES=[0.1, 0.1, 0.1, 0.1]
        SCREEN_SIZE=1000
        SPARSE_REWARD=False
        SCREEN_SHOT=False
        action_range = 10.0

        env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
        ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True, change_goal=False)
        action_dim = env.num_actions
        state_dim  = env.num_observations
    elif ENV == 'Pendulum':
        env = gym.make("Pendulum-v0").unwrapped
        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]
        action_range=1.


    hidden_dim = 512
    batch_size=100
    model_path = './qt_opt_model/model3'

    replay_buffer_size = 5e5
    replay_buffer = ReplayBuffer(replay_buffer_size)

    qt_opt = QT_Opt(replay_buffer, hidden_dim)
    
    if args.train:
        # hyper-parameters
        max_episodes  = 2000
        max_steps   = 20 if ENV ==  'Reacher' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
        frame_idx   = 0
        episode_rewards = []

        for i_episode in range (max_episodes):
            
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            elif ENV == 'Pendulum':
                state =  env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # action = qt_opt.policy.act(state)  
                action = qt_opt.cem_optimal_action(state)
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                elif ENV ==  'Pendulum':
                    next_state, reward, done, _ = env.step(action) 
                    env.render()
                episode_reward += reward
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

            if len(replay_buffer) > batch_size:
                qt_opt.update(batch_size)
                qt_opt.save_model(model_path)

            episode_rewards.append(episode_reward)
            
            if i_episode% 10==0:
                plot(episode_rewards)
                
            print('Episode: {}  | Reward:  {}'.format(i_episode, episode_reward))
    
    if args.test:
        qt_opt.load_model(model_path)
        # hyper-parameters
        max_episodes  = 10
        max_steps   = 20 if ENV ==  'Reacher' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
        frame_idx   = 0
        episode_rewards = []

        for i_episode in range (max_episodes):
            
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            elif ENV == 'Pendulum':
                state =  env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # action = qt_opt.policy.act(state)  
                action = qt_opt.cem_optimal_action(state)
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                elif ENV ==  'Pendulum':
                    next_state, reward, done, _ = env.step(action)  
                    env.render()               
                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)
            # plot(episode_rewards)
            print('Episode: {}  | Reward:  {}'.format(i_episode, episode_reward))
    
        
        
