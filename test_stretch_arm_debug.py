"""
Debug script to understand Stretch arm control flow.
"""
import gym
import assistive_gym
import pybullet as p
import numpy as np

# NumPy 2.0 compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Create environment
env = gym.make('ScratchItchStretch-v1')
env.reset()
env.render()
env.reset()

print("\n" + "="*60)
print("STRETCH ARM DEBUG")
print("="*60)

robot = env.robot

print(f"\n1. Robot Type: {type(robot).__name__}")
print(f"2. mobile: {robot.mobile}")
print(f"3. controllable_joint_indices: {robot.controllable_joint_indices}")
print(f"4. all_controllable_joints: {robot.all_controllable_joints}")
print(f"5. right_arm_joint_indices: {robot.right_arm_joint_indices}")
print(f"6. wheel_joint_indices: {robot.wheel_joint_indices}")
print(f"7. action_duplication: {robot.action_duplication}")
print(f"8. action_multiplier: {robot.action_multiplier}")
print(f"9. gains: {robot.gains}")
print(f"10. forces: {robot.forces}")

print(f"\n11. action_space: {env.action_space.shape}")
print(f"12. controllable_joint_lower_limits: {robot.controllable_joint_lower_limits}")
print(f"13. controllable_joint_upper_limits: {robot.controllable_joint_upper_limits}")

# Get current joint angles
current_angles = robot.get_joint_angles(robot.controllable_joint_indices)
print(f"\n14. Current joint angles (controllable_joint_indices): {current_angles}")

# Get URDF joint info
print(f"\n15. Joint Info from URDF:")
for idx in robot.all_controllable_joints:
    info = p.getJointInfo(robot.body, idx, physicsClientId=env.id)
    name = info[1].decode('utf-8')
    joint_type = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED'][info[2]]
    lower = info[8]
    upper = info[9]
    max_force = info[10]
    max_velocity = info[11]
    state = p.getJointState(robot.body, idx, physicsClientId=env.id)
    print(f"    Joint {idx} ({name}): type={joint_type}, limits=[{lower:.4f}, {upper:.4f}], "
          f"current={state[0]:.4f}, force={max_force}, vel={max_velocity}")

# Test: apply action and see what happens
print("\n" + "="*60)
print("TESTING ARM CONTROL")
print("="*60)

# Test action: move lift (index 2) and arm extend (index 3)
test_action = np.array([0, 0, 0.5, 0.5, 0])  # wheels=0, lift=0.5, extend=0.5, wrist=0
print(f"\nTest action: {test_action}")

# Simulate what take_step does
action = test_action.copy()
action *= robot.action_multiplier
print(f"After action_multiplier: {action}")

agent_joint_angles = robot.get_joint_angles(robot.controllable_joint_indices).copy()
print(f"Current joint angles: {agent_joint_angles}")

# Check limits (same as take_step)
for _ in range(env.frame_skip):
    below_lower_limits = agent_joint_angles + action < robot.controllable_joint_lower_limits
    above_upper_limits = agent_joint_angles + action > robot.controllable_joint_upper_limits
    print(f"below_lower_limits: {below_lower_limits}")
    print(f"above_upper_limits: {above_upper_limits}")
    action[below_lower_limits] = 0
    action[above_upper_limits] = 0
    print(f"Action after limit check: {action}")
    agent_joint_angles += action
    print(f"New joint angles: {agent_joint_angles}")

# Apply action duplication
if robot.action_duplication is not None:
    duplicated = np.concatenate([[a]*d for a, d in zip(agent_joint_angles, robot.action_duplication)])
    print(f"\nAfter duplication (maps to all_controllable_joints):")
    print(f"  duplicated angles: {duplicated}")
    print(f"  all_controllable_joints: {robot.all_controllable_joints}")
    for joint_idx, angle in zip(robot.all_controllable_joints, duplicated):
        info = p.getJointInfo(robot.body, joint_idx, physicsClientId=env.id)
        name = info[1].decode('utf-8')
        print(f"    Joint {joint_idx} ({name}): target={angle:.4f}")

# Now actually step the env
print("\n" + "="*60)
print("ACTUAL ENV.STEP")
print("="*60)

initial_angles = {idx: p.getJointState(robot.body, idx, physicsClientId=env.id)[0] 
                  for idx in robot.all_controllable_joints}
print(f"Before step joint angles: {initial_angles}")

# Step with a lift+extend action
action = np.array([0, 0, 0.1, 0.1, 0])
for i in range(20):
    env.step(action)

final_angles = {idx: p.getJointState(robot.body, idx, physicsClientId=env.id)[0] 
                for idx in robot.all_controllable_joints}
print(f"After 20 steps joint angles: {final_angles}")

print("\nAngle changes:")
for idx in robot.all_controllable_joints:
    info = p.getJointInfo(robot.body, idx, physicsClientId=env.id)
    name = info[1].decode('utf-8')
    delta = final_angles[idx] - initial_angles[idx]
    print(f"  Joint {idx} ({name}): {initial_angles[idx]:.4f} -> {final_angles[idx]:.4f} (delta={delta:.4f})")

env.close()
print("\nDone.")
