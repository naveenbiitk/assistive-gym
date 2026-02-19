#!/usr/bin/env python3
"""Direct test of Stretch arm control"""
import gym
import assistive_gym
import numpy as np
import pybullet as p

env = gym.make('ScratchItchStretch-v1')
env.reset()

print(f"Robot controllable_joint_indices: {env.robot.controllable_joint_indices}")
print(f"Robot all_controllable_joints: {env.robot.all_controllable_joints}")
print(f"Robot gains: {env.robot.gains}")
print(f"Robot forces: {env.robot.forces}")
print(f"Robot action_duplication: {env.robot.action_duplication}")
print(f"Action robot len: {env.action_robot_len}")

# Get initial angles
initial_angles = env.robot.get_joint_angles(env.robot.controllable_joint_indices)
print(f"\nInitial angles: {initial_angles}")

# Send a lift action (index 2)
print("\nSending lift action (0.01 * 100 = 1 radian)...")
action = np.array([0, 0, 1.0, 0, 0])  # Large lift action
for i in range(10):
    env.step(action)
    angles = env.robot.get_joint_angles(env.robot.controllable_joint_indices)
    print(f"Step {i}: angles = {angles}, delta = {angles - initial_angles}")

print("\nDone!")
