#!/usr/bin/env python3
import gym, assistive_gym
import numpy as np

env = gym.make('ScratchItchStretch-v1')
env.reset()

print(f"Robot controllable_joint_indices: {env.robot.controllable_joint_indices}")
print(f"Action robot len: {env.action_robot_len}")
print(f"Action space shape: {env.action_space.shape}")

# Try moving each action independently
for i in range(env.action_robot_len):
    print(f"\nTesting action index {i}")
    action = np.zeros(env.action_robot_len)
    action[i] = 0.5  # 50% of max action
    print(f"Action: {action}")
    
    # Get joint angles before
    before = env.robot.get_joint_angles(env.robot.controllable_joint_indices)
    print(f"Before: {before}")
    
    # Step simulation with this action
    for _ in range(10):
        env.step(action * 100)
    
    # Get joint angles after
    after = env.robot.get_joint_angles(env.robot.controllable_joint_indices)
    print(f"After: {after}")
    print(f"Delta: {after - before}")
