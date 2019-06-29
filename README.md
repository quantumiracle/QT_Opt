# QT-Opt

Pytorch implementation of QT-Opt algorithm for vision-based robotic manipulation using deep reinforcement learning and non-gradient-based cross-entropy (CE) method.

## Description:
Instead of using an actor network, using the cross-entropy (CE) method for optimizing the Q network in a non-gradient-based manner, which is able to handle the non-convex cases and search the global structure more efficiently in practice, especially when leveraging the expert demonstrations. 

[Here]() is a notebook about CE method.

Paper: [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/abs/1806.10293)

Note: this implementation doesn't use the expert demonstrations, but purely learning from explorations. The demonstrations are environment-wise, and it's straightforward to apply the demonmstrations: just add another demonstrations buffer and sample from it whenever sampling from current exploration buffer for udpating the policy

## Two Versions of Implementation:
As CE method is very flexible in practice as shown in above [notebook](), there are at least two feasible versions of implementation for QT-Opt algorithms, and original paper didn't tell clearly about which one to apply. 
* the first version I tried didn't work, so not shown here;
* version 2: 

`qt_opt_v2.py` : the Gaussian distribution in CE method is fitted on the policy weights W, and action A = W * S + B, where S is state and B is bias. And the criteria for CE method is Q(S,A). This version is more usual to see in general CE method implementation, but could only be applied for low-dimensional cases as the dimension of weights are basically dimension of S multiplied by dimension of S.

Another key point in this implementation is that although we have action A = W * S + B  as a policy (can even use exactly a neural network here), it is initialized to be a normal distribtution with mean=0 and std=1 for each evaluation using the CE method. Therefore the W and B (parameters of the policy) actually does not stored any information for the policy during the changing of input state, but ideally we should let it contain some information for the policy to give an action dependent on the input state even for right after the initialization of CE distribution. I tried to not initialize the distribution for each evaluation, and it makes the Q-network not able to converge. I still haven't find out the way to solve this yet.

If the distribution dose not need to be initialized for each evaluation, the role of CE method is: the action policy + evaluating the argmax_a'(Q(s',a')).

* version 3: 

`qt_opt_v3.py` : the Gaussian distribution in CE method is fitted on the action A directly. And the criteria for CE method is Q(S,A). This version has advantages in the dimension of the fitted Gaussian distribution, which is the dimension of the action instead of the weights. And the dimension of the weights could be large for like visual-based input images as states S. Therefore, this version is more likely to be the one applied in original QT-Opt paper.

the role of CE method is just: evaluating the argmax_a'(Q(s',a')).


## To run:
For example to use `qt_opt_v3.py`:

`python qt_opt_v3.py --train` for training;

`python qt_opt_v3.py --test` for testing;

To use `qt_opt_v2.py` just change the name correspondingly.


