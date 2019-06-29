# QT-Opt

Pytorch implementation of QT-Opt algorithm for vision-based robotic manipulation using deep reinforcement learning and non-gradient-based cross-entropy (CE) method.

## Description:
Instead of using an actor network, using the cross-entropy (CE) method for optimizing the Q network in a non-gradient-based manner, which is able to handle the non-convex cases and search the global structure more efficiently in practice, especially when leveraging the expert demonstrations. 

[Here]() is a notebook about CE method.

Paper: [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/abs/1806.10293)

Note: this implementation doesn't use the expert demonstrations, but purely learning from explorations.

## Two Versions of Implementation:
As CE method is very flexible in practice as shown in above [notebook](), there are at least two feasible versions of implementation for QT-Opt algorithms, and original paper didn't tell clearly about which one to apply. 
* the first version I tried didn't work, so not shown here;
* version 2: 

`qt_opt_v2.py` : the Gaussian distribution in CE method is fitted on the policy weights W, and action A = W * S + B, where S is state and B is bias. And the criteria for CE method is Q(S,A). This version is more usual to see in general CE method implementation, but could only be applied for low-dimensional cases as the dimension of weights are basically dimension of S multiplied by dimension of S.
* version 3: 

`qt_opt_v3.py` : the Gaussian distribution in CE method is fitted on the action A directly. And the criteria for CE method is Q(S,A). This version has advantages in the dimension of the fitted Gaussian distribution, which is the dimension of the action instead of the weights. And the dimension of the weights could be large for like visual-based input images as states S. Therefore, this version is more likely to be the one applied in original QT-Opt paper.


## To run:
For example to use `qt_opt_v3.py`:

`python qt_opt_v3.py --train` for training;

`python qt_opt_v3.py --test` for testing;

To use `qt_opt_v2.py` just change the name correspondingly.


