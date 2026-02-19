import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

# Workaround for NumPy 2.0 compatibility with old gym library
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

parser = argparse.ArgumentParser(description='Test Arm Manipulation Environment')
parser.add_argument('--env', default='ArmManipulationStretch-v1',
                    help='Environment to test')
args = parser.parse_args()

print(f"Creating environment: {args.env}")
env = gym.make(args.env)

print("Resetting environment...")
observation = env.reset()

print("Rendering...")
env.render()

print("Resetting again after render...")
observation = env.reset()

print(f"action_space shape: {env.action_space.shape}")
print(f"Human position: {env.human.get_pos_orient(env.human.base)}")

# Just render and wait - don't step, just visualize
print("\n=== Press Q to quit ===")
print("Human should be lying flat on the bed.")
import time
while True:
    env.render()
    keys = p.getKeyboardEvents()
    if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
        break
    time.sleep(0.02)

env.close()
