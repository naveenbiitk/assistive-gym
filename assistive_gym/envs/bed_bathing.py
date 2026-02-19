import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture

class BedBathingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(BedBathingEnv, self).__init__(robot=robot, human=human, task='bed_bathing', obs_robot_len=(17 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(18 + len(human.controllable_joint_indices)))

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        # Keep human frozen after physics step
        if not self.human.controllable:
            self._freeze_human()

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_on_human)

        closest_points = self.tool.get_closest_points(self.human, distance=5.0)[-1]
        # Safeguard: if no closest points are returned, skip distance penalty
        if len(closest_points) == 0:
            reward_distance = 0.0
        else:
            reward_distance = -min(closest_points)
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_new_contact_points = self.new_contact_points # Reward new contact points on a person

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('wiping_reward_weight')*reward_new_contact_points + preferences_score

        if self.gui and self.tool_force_on_human > 0:
            print('Task success:', self.task_success, 'Force at tool on human:', self.tool_force_on_human, reward_new_contact_points)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= (self.total_target_count*self.config('task_success_threshold'))), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, False, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': False, 'human': False}, {'robot': info, 'human': info}

    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_on_human = 0
        new_contact_points = 0
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            if linkA in [1]:
                tool_force_on_human += force
                # Only consider contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > len(self.human.all_joint_indices):
                    continue

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_upperarm_world, self.targets_upperarm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in indices_to_delete]
                self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in indices_to_delete]
                self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in indices_to_delete]

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_forearm_world, self.targets_forearm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in indices_to_delete]
                self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in indices_to_delete]
                self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in indices_to_delete]

        return tool_force, tool_force_on_human, total_force_on_human, new_contact_points

    def _get_obs(self, agent=None):
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        self.tool_force, self.tool_force_on_human, self.total_force_on_human, self.new_contact_points = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.total_force_on_human, self.tool_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(BedBathingEnv, self).reset()
        # Fix human base to prevent instability/jumping
        self.build_assistive_env('bed', fixed_human_base=True)

        self.furniture.set_friction(self.furniture.base, friction=5)

        # Set joint angles for human joints (in degrees) - lying down pose
        # All joints near 0 for a flat lying position, slight arm angle for reachability
        joints_positions = [
            (self.human.j_right_shoulder_x, 30),   # Right arm slightly raised
            (self.human.j_right_elbow, 0),         # Elbow straight
            (self.human.j_left_elbow, 0),          # Left elbow straight
            (self.human.j_right_hip_x, 0),         # Hips straight (lying flat)
            (self.human.j_right_knee, 0),          # Knees straight
            (self.human.j_left_hip_x, 0),          # Left hip straight
            (self.human.j_left_knee, 0),           # Left knee straight
        ]

        # Re-freeze human after pose initialization
        if not self.human.controllable:
            self._freeze_human()
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)

        # Position human lying on bed - rotation [-pi/2, 0, 0] makes them lie on back
        self.human.set_base_pos_orient([-0.15, 0.2, 0.85], [-np.pi/2.0, 0, 0])

        # Make human completely static (same approach as scratch_itch.py)
        # 1. Make human base kinematic (mass=0 means no dynamics)
        p.changeDynamics(self.human.body, -1, mass=0, physicsClientId=self.id)
        
        # 2. Set lying pose joint angles directly using resetJointState
        # All joints at 0 for lying flat, except slight arm angle
        lying_joints = {
            self.human.j_right_hip_x: 0,       # Hips straight
            self.human.j_left_hip_x: 0,        # Left hip straight
            self.human.j_right_knee: 0,        # Knees straight
            self.human.j_left_knee: 0,         # Left knee straight
            self.human.j_right_shoulder_x: 0.5, # Arm slightly raised for robot access
            self.human.j_right_elbow: 0,       # Elbow straight
            self.human.j_left_elbow: 0,        # Left elbow straight
        }
        
        # 3. Reset all joint states and make all links kinematic
        for joint_idx in range(p.getNumJoints(self.human.body, physicsClientId=self.id)):
            # Set target position (use lying pose if defined, else 0)
            target_pos = lying_joints.get(joint_idx, 0)
            # Directly set joint state (bypasses physics)
            p.resetJointState(self.human.body, joint_idx, target_pos, 0, physicsClientId=self.id)
            # Make link kinematic (mass=0)
            p.changeDynamics(self.human.body, joint_idx, mass=0, physicsClientId=self.id)

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[1]*3)

        # Reset selected target for new episode
        self._selected_target_pos = None

        # Position robot closer to the human arm (use elbow position as reference)
        arm_center = (shoulder_pos + elbow_pos) / 2
        target_ee_pos = np.array([arm_center[0] - 0.3, arm_center[1], arm_center[2] + 0.1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        base_position = self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], arm='left', tools=[self.tool], collision_objects=[self.human, self.furniture], wheelchair_enabled=False)

        if self.robot.wheelchair_mounted:
            # Load a nightstand in the environment for mounted arms
            self.nightstand = Furniture()
            self.nightstand.init('nightstand', self.directory, self.id, self.np_random)
            self.nightstand.set_base_pos_orient(np.array([-0.9, 0.7, 0]) + base_position, [0, 0, 0, 1])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        self.generate_targets()

        # For non-mobile robots, disable gravity on arm links
        if not self.robot.mobile:
            for joint_idx in self.robot.controllable_joint_indices:
                p.changeDynamics(self.robot.body, joint_idx, mass=0, physicsClientId=self.id)
        
        # Disable gravity on tool links
        for link_idx in range(-1, p.getNumJoints(self.tool.body, physicsClientId=self.id)):
            p.changeDynamics(self.tool.body, link_idx, mass=0, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def _freeze_human(self):
        """Re-apply kinematic state to keep human completely frozen."""
        for joint_idx in range(p.getNumJoints(self.human.body, physicsClientId=self.id)):
            joint_state = p.getJointState(self.human.body, joint_idx, physicsClientId=self.id)
            p.resetJointState(self.human.body, joint_idx, joint_state[0], 0, physicsClientId=self.id)
        p.resetBaseVelocity(self.human.body, [0, 0, 0], [0, 0, 0], physicsClientId=self.id)

    @property
    def target_pos(self):
        """Return a random target position on the arm for baseline compatibility."""
        # Return the selected random target, or pick one if not set
        if hasattr(self, '_selected_target_pos') and self._selected_target_pos is not None:
            return self._selected_target_pos
        
        # Combine all targets and pick a random one
        all_targets = self.targets_pos_upperarm_world + self.targets_pos_forearm_world
        if all_targets:
            # Pick a random target from the arm
            idx = self.np_random.integers(0, len(all_targets))
            self._selected_target_pos = all_targets[idx]
            return self._selected_target_pos
        else:
            return np.array([0, 0, 0])

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if self.human.gender == 'male':
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.279, 0.043
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.257, 0.033
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.264, 0.0355
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.234, 0.027

        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03)

        self.targets_upperarm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.targets_forearm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_forearm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.total_target_count = len(self.targets_pos_on_upperarm) + len(self.targets_pos_on_forearm)
        self.update_targets()

    def update_targets(self):
        upperarm_pos, upperarm_orient = self.human.get_pos_orient(self.upperarm)
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        forearm_pos, forearm_orient = self.human.get_pos_orient(self.forearm)
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

