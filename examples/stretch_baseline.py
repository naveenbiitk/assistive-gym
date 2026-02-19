import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import sys
import math

# Workaround for NumPy 2.0 compatibility with old gym library
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


def evaluate_baseline():
    seed=5

    n_episodes = 5
    env = gym.make(args.env)
    env.seed(seed)
    env.reset()
    env.render()
    
    print(f"DEBUG: action_space shape: {env.action_space.shape}")
    print(f"DEBUG: action_robot_len: {env.action_robot_len}")
    print(f"DEBUG: action_human_len: {env.action_human_len}")
    
    rewards_ls = []
    forces_ls = []
    task_successes_ls = []
    for episode in range(n_episodes):
        obs = env.reset()
        terminated = False
        reward_total = 0.0
        force_list = []
        task_success = 0.0
        pos, orient = env.robot.get_pos_orient(env.robot.left_end_effector)
        human_target = np.array(env.target_pos)
        # human_pos_real, _ = env.robot.convert_to_realworld(env.target_pos)
        #print('difference', human_pos_real-human_target)
        target_pose_angles = env.robot.ik(env.robot.left_end_effector, human_target, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
        count_= 0
        tool_pos, tool_orient = env.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = env.robot.convert_to_realworld(tool_pos, tool_orient)

        while not terminated:
            env.render()

            base_pose,base_orient = env.robot.get_pos_orient(env.robot.base)
            
            # Create full-sized action array for action space
            action = np.zeros(env.action_space.shape[0])
            current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
            target_pose_angles = env.robot.ik(env.robot.left_end_effector, human_target, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
            # Only assign IK angles to the corresponding arm joints (first 3 after wheel joints)
            action[len(env.robot.wheel_joint_indices):len(env.robot.wheel_joint_indices)+len(target_pose_angles)] = target_pose_angles-current_joint_angles
              
            y_offset=-0.1

            if count_<500:
                action[0] = human_target[1]-base_pose[1]+y_offset
                action[1] = action[0]
                #action[2] = -0.5*(target_pose_angles[0]-current_joint_angles[0])
                action[2]=0
            else:
                action[2] = (target_pose_angles[0]-current_joint_angles[0])+0.02

            action[0] = human_target[1]-base_pose[1]+y_offset
            action[1] = action[0]
            action[3] = (target_pose_angles[1]-current_joint_angles[1])
            action[4] = (target_pose_angles[2]-current_joint_angles[2])-0.15
            
            pos, orient_1 = env.robot.get_pos_orient(env.robot.left_end_effector)

            #print("Actions ",np.linalg.norm(action), "Poses",np.linalg.norm(pos-human_target))
            count_ = count_ + 1

            # if np.linalg.norm(action)<0.04 and count_>90:
            #     action[2]=0
            #     action[3]=0
            #     action[4]=0

            print(f"DEBUG: action shape: {action.shape}, len: {len(action)}, first 7: {action[:7]}")
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            task_success = info['task_success']
            force_list.append(info['total_force_on_human'])

        print('Episode:',episode,'   reward: ',reward_total,'  task_success:',task_success)
        rewards_ls.append(reward_total)
        forces_ls.append(np.mean(force_list))
        task_successes_ls.append(task_success)
        print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))


    env.disconnect()

    print('\n', '-'*50, '\n')
    print('Number of episodes', n_episodes)
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards_ls))
    print('Reward Std:', np.std(rewards_ls))

    # print('Forces:', forces)
    print('Force Mean:', np.mean(forces_ls))
    print('Force Std:', np.std(forces_ls))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes_ls))
    print('Task Success Std:', np.std(task_successes_ls))
    print('Task_successes total', np.sum(task_successes_ls))
    print('Task_successes total', task_successes_ls)
    sys.stdout.flush()



def run_baseline():
    print('--------------------------------------')
    env = gym.make(args.env)
    seed=2
    env.seed(seed)
    env.render()
    observation = env.reset()
    
    print(f"DEBUG: action_space shape: {env.action_space.shape}")
    print(f"DEBUG: action_robot_len: {env.action_robot_len}")
    print(f"DEBUG: action_human_len: {env.action_human_len}")
    #env.robot.print_joint_info()

    print('--------------------------------------')
    print('human limmb pose',env.human.get_pos_orient(env.limb))

    pos, orient = env.robot.get_pos_orient(env.robot.left_end_effector)

    #print(pos)
    #print(orient)
    human_target = np.array(env.target_pos)
    print('target pose',human_target)
    #-0.36859864 -0.06440637  0.80693936  1.
    #-0.36859864 -0.06440637  0.80693936  2.

    #human_target[0]=-0.3859
    #human_target[1]=0.46769
    #human_target[2]=0.849474
    print('--------------------------------------')
    # Arrow keys for moving the base, 
    # s/x for the lift, third array
    # z/c for the prismatic joint, fourth array (compression/retraction)
    # a/d for the wrist joint, fifth array  (yaw movements)

    keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

    target_pose = env.target_pos

    target_pose_angles = env.robot.ik(env.robot.left_end_effector, human_target, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)

    print('robot target angle',target_pose_angles)

    current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

    print('current_joint_angles of robot',current_joint_angles)

    right_side=False
    count_= 0

    reward_total = 0
    while count_<250:
        env.render()
        # Create full-sized action array for action space
        action = np.zeros(env.action_space.shape[0])
        # if count_<50:
        #     base_position, _, _ = env.robot.position_robot_toc(env.task, 'left', pos, target_pose, env.human, step_sim=False, check_env_collisions=True, max_ik_iterations=100, max_ik_random_restarts=1, randomize_limits=False, right_side=right_side, base_euler_orient=[0, 0, 0 if right_side else np.pi], attempts=50)
        #print('inside loop')
        current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
        #current_joint_angles = env.robot.get_joint_angles(env.robot.left_gripper_indices)
        action[len(env.robot.wheel_joint_indices):] = target_pose_angles-current_joint_angles
        
        #print('gripper joints',env.robot.get_joint_angles(env.robot.left_gripper_indices))

        if count_<40:
            action[2] = -0.5*(target_pose_angles[0]-current_joint_angles[0])
        else:
            action[2] = (target_pose_angles[0]-current_joint_angles[0])+0.025
        action[3] = (target_pose_angles[1]-current_joint_angles[1])
        action[4] = (target_pose_angles[2]-current_joint_angles[2])-0.15
        
        keys = p.getKeyboardEvents()
        count_ = count_ + 1

        if np.linalg.norm(action)<0.04 and count_>90:
            action[2]=0
            action[3]=0
            action[4]=0
        
        print("actions", np.linalg.norm(action),"count",count_)
        #print(env.robot.get_pos_orient(env.robot.left_end_effector))
        #-0.41552773,  0.16920179,  0.81055504  without any motion (not correct spot)  1.
        #-0.41352561, -0.05890547,  0.85007751 correct position 2. orientation:  7.14263355e-04,  6.32512238e-05, -2.54243258e-02,  9.99676526e-01 correct one
        #-0.44282907, -0.07709748,  0.81385118 correct position 2. orientation:  4.70801868e-04, -3.55046359e-04,  6.66704595e-01,  7.45321810e-01
        # on correct position action stiil is actions [ 0.          0.         -0.04154994  0.05238283  1.78015788]


        # target pose -0.36859864 -0.06440637  0.80693936 for 1,2

        #print(env.robot.get_joint_angles(env.robot.left_arm_joint_indices))
        for key, a in keys_actions.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                # Pad the action to full size
                padded_a = np.zeros(env.action_space.shape[0])
                padded_a[:len(a)] = a
                action += padded_a
                action = action*100

        print(f"DEBUG: action shape: {action.shape}, len: {len(action)}")
        observation, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        task_success = info['task_success']


    print('reward')
    print(reward_total)
    print('task_success')
    print(task_success)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='BedBathingStretch-v1',
                    help='Environment to test (default: JointReachingStretchHuman-v1)')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--seed', type=int, default=5,
                        help='Random seed (default: 5)')
    args = parser.parse_args()
    evaluate_baseline()