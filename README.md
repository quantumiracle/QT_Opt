# QT-Opt

Pytorch implementation of QT-Opt algorithm for vision-based robotic manipulation using deep reinforcement learning and non-gradient-based cross-entropy (CE) method.

## Description:
Instead of using an actor network, using the cross-entropy (CE) method for optimizing the Q network in a non-gradient-based manner, which is able to handle the non-convex cases and search the global structure more efficiently in practice, especially when leveraging the expert demonstrations. 

[Here] is a notebook about CE method.

Paper: [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/abs/1806.10293)

Note: this implementation doesn't use the expert demonstrations, but purely learning from explorations.

## To run:
`python qt_opt_v3.py --train` for training;

`python qt_opt_v3.py --test` for testing;


