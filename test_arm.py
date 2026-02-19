#!/usr/bin/env python3
import numpy as np
# Workaround for NumPy 2.0 compatibility with old gym library
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import assistive_gym

env = gym.make('ScratchItchStretch-v1')
env.reset()

action = np.array([0, 0, 0.1, 0, 0])

a1 = env.robot.get_joint_angles([3, 5, 9])
print('Initial arm angles:', a1)

env.step(action * 100)
a2 = env.robot.get_joint_angles([3, 5, 9])
print('After 1 step:', a2)
print('Delta 1:', a2 - a1)

env.step(action * 100)
a3 = env.robot.get_joint_angles([3, 5, 9])
print('After 2 steps:', a3)
print('Delta 2:', a3 - a2)
