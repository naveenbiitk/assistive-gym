"""
Teleop control for Stretch robot in Assistive Gym (position control).
Uses ScratchItchStretch-v1 environment with static sitting human.

Controls:
    Arrow keys : Move robot base (forward/back/rotate)
    S/X        : Lift arm up/down
    C/Z        : Arm extend/retract
    A/D        : Wrist rotate
    Q          : Quit

IMPORTANT: Click on the PyBullet window to give it focus for keyboard input.
"""
import gym
import assistive_gym
import pybullet as p
import numpy as np
import argparse

# NumPy 2.0 compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

parser = argparse.ArgumentParser(description='Stretch Robot Teleop Control')
parser.add_argument('--env', default='BedBathingStretch-v1',
                    help='Environment (default: BedBathingStretch-v1)')
args = parser.parse_args()

# Create environment
env = gym.make(args.env)
env.reset()
env.render()
env.reset()  # Reset again after render for GUI

print(f"DEBUG: action_space shape: {env.action_space.shape}")
print(f"DEBUG: action_robot_len: {env.action_robot_len}")
print(f"DEBUG: action_human_len: {env.action_human_len}")
print(f"DEBUG: controllable_joint_indices: {env.robot.controllable_joint_indices}")

# Key mappings for position control via env.step()
# Action indices: [0]=wheel_L, [1]=wheel_R, [2]=lift, [3]=arm_extend, [4]=wrist
# Match stretch_baseline.py key mapping, but keep arm deltas small
KEY_ACTIONS = {
    # Wheels
    p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]),
    p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]),
    p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]),
    p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]),
    # Lift (0.0..1.1)
    ord('s'): np.array([0, 0, 0.01, 0, 0]),
    ord('x'): np.array([0, 0, -0.01, 0, 0]),
    # Arm extend (each prismatic joint 0.0..0.13)
    ord('z'): np.array([0, 0, 0, -0.002, 0]),
    ord('c'): np.array([0, 0, 0, 0.002, 0]),
    # Wrist (yaw)
    ord('a'): np.array([0, 0, 0, 0, 0.02]),
    ord('d'): np.array([0, 0, 0, 0, -0.02]),
}

print("\n=== Stretch Teleop Controls ===")
print("Arrow keys : Move base")
print("S/X        : Lift up/down")
print("C/Z        : Arm extend/retract")
print("A/D        : Wrist rotate")
print("Q          : Quit")
print("=" * 30 + "\n")

# Main loop
frame = 0
while True:
    env.render()
    
    keys = p.getKeyboardEvents()
    
    # Quit check
    if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
        break
    
    # Build action from pressed keys
    action = np.zeros(env.action_space.shape[0])
    for key, delta in KEY_ACTIONS.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            # Pad to full action size (matches stretch_baseline.py)
            padded = np.zeros(env.action_space.shape[0])
            padded[:len(delta)] = delta
            action += padded

    # Scale wheels only (arm joints have small limits)
    if np.any(action != 0):
        action[:2] = action[:2] * 100

    # Clamp arm actions to stay within joint limits (prevents zeroing by limit checks)
    if len(action) >= 5:
        current = env.robot.get_joint_angles(env.robot.controllable_joint_indices)
        lower = env.robot.controllable_joint_lower_limits
        upper = env.robot.controllable_joint_upper_limits
        for i in range(2, 5):
            if i < len(current):
                target = np.clip(current[i] + action[i], lower[i], upper[i])
                action[i] = target - current[i]

    if frame % 60 == 0:
        print(f"DEBUG: action={action}, joint_angles={env.robot.get_joint_angles(env.robot.controllable_joint_indices)}")

    env.step(action)

    frame += 1

env.close()
print("Done.")
