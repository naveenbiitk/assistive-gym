"""
Teleop control for Stretch robot in Assistive Gym.
Uses ScratchItchStretch-v1 environment with static sitting human.

Controls:
    Arrow keys : Move robot base (forward/back/rotate)
    S/X        : Lift arm up/down
    C/Z        : Arm extend/retract
    A/D        : Wrist rotate
    Q          : Quit
"""
import gym
import assistive_gym
import pybullet as p
import numpy as np

# NumPy 2.0 compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Use Stretch robot - same as teleop_stretch_example.py
env = gym.make('ScratchItchStretch-v1')
env.reset()
env.render()
env.reset()  # Reset again after render for GUI

# Key mappings - same as teleop_stretch_example.py
# Action indices: [0]=wheel_L, [1]=wheel_R, [2]=lift, [3]=arm_extend, [4]=wrist
KEY_ACTIONS = {
    p.B3G_UP_ARROW:    np.array([0.01, 0.01, 0, 0, 0]),     # forward
    p.B3G_DOWN_ARROW:  np.array([-0.01, -0.01, 0, 0, 0]),   # backward
    p.B3G_LEFT_ARROW:  np.array([0.01, -0.01, 0, 0, 0]),    # rotate left
    p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]),    # rotate right
    ord('s'): np.array([0, 0, 0.01, 0, 0]),     # lift up
    ord('x'): np.array([0, 0, -0.01, 0, 0]),    # lift down
    ord('c'): np.array([0, 0, 0, 0.01, 0]),     # arm extend
    ord('z'): np.array([0, 0, 0, -0.01, 0]),    # arm retract
    ord('a'): np.array([0, 0, 0, 0, 0.01]),     # wrist rotate +
    ord('d'): np.array([0, 0, 0, 0, -0.01]),    # wrist rotate -
}

print("\n=== Stretch Teleop Controls ===")
print("Arrow keys : Move base")
print("S/X        : Lift up/down")
print("C/Z        : Arm extend/retract")
print("A/D        : Wrist rotate")
print("Q          : Quit")
print("=" * 30 + "\n")

# Main loop
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
            action[:len(delta)] += delta
    
    # Scale when keys pressed
    if np.any(action != 0):
        action *= 100
    
    env.step(action)

env.close()
print("Done.")

# Key mappings for end-effector control
POS_KEYS = {
    ord('j'): np.array([-0.01, 0, 0]),
    ord('l'): np.array([0.01, 0, 0]),
    ord('u'): np.array([0, -0.01, 0]),
    ord('o'): np.array([0, 0.01, 0]),
    ord('k'): np.array([0, 0, -0.01]),
    ord('i'): np.array([0, 0, 0.01]),
}
RPY_KEYS = {
    ord('k'): np.array([-0.05, 0, 0]),
    ord('i'): np.array([0.05, 0, 0]),
    ord('u'): np.array([0, -0.05, 0]),
    ord('o'): np.array([0, 0.05, 0]),
    ord('j'): np.array([0, 0, -0.05]),
    ord('l'): np.array([0, 0, 0.05]),
}

# Initialize target from current pose - ScratchItch uses LEFT arm
start_pos, orient = env.robot.get_pos_orient(env.robot.left_end_effector)
start_rpy = env.get_euler(orient)
target_pos_offset = np.zeros(3)
target_rpy_offset = np.zeros(3)
target_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

print("\n=== Teleop Controls ===")
print("U/O : Move Y axis")
print("I/K : Move Z axis")
print("J/L : Move X axis")
print("Shift + keys: Rotate")
print("Q   : Quit")
print("=" * 24 + "\n")

while True:
    env.render()
    keys = p.getKeyboardEvents()
    
    # Quit check
    if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
        break
    
    key_pressed = False
    
    # Position control (no shift)
    for key, action in POS_KEYS.items():
        if p.B3G_SHIFT not in keys and key in keys and keys[key] & p.KEY_IS_DOWN:
            target_pos_offset += action
            key_pressed = True
    
    # Rotation control (with shift)
    for key, action in RPY_KEYS.items():
        if p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] & p.KEY_IS_DOWN:
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                target_rpy_offset += action
                key_pressed = True

    # Compute target pose
    target_pos = start_pos + target_pos_offset
    target_rpy = start_rpy + target_rpy_offset

    # Recompute IK only when key pressed
    if key_pressed:
        target_joint_angles = env.robot.ik(
            env.robot.left_end_effector,
            target_pos,
            env.get_quaternion(target_rpy),
            env.robot.left_arm_ik_indices,
            max_iterations=200,
            use_current_as_rest=True
        )
    
    # Compute action from joint angle difference
    current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
    arm_action = (target_joint_angles - current_joint_angles) * 10
    
    # Pad to full action space
    action = np.zeros(env.action_space.shape[0])
    action[:len(arm_action)] = arm_action
    
    env.step(action)

env.close()
print("Done.")

