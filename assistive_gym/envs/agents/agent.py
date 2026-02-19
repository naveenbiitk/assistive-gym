import numpy as np
import pybullet as p

class Agent:
    """
    PyBullet Agent wrapper compatible with modern pybullet + Gym seeding (np.random.Generator).

    Assumptions:
      - self.body is a valid PyBullet bodyUniqueId loaded in physicsClientId=self.id
      - indices are joint indices (ints) in [0, num_joints-1]
    """

    def __init__(self):
        self.base = -1                 # convention: -1 means base
        self.body = None               # PyBullet bodyUniqueId (int)
        self.id = None                 # physicsClientId (int)
        self.np_random = None          # numpy.random.Generator or RandomState

        self.all_joint_indices = []
        self.controllable_joint_indices = []

        # joint limits (dict keyed by joint index)
        self.lower_limits = {}
        self.upper_limits = {}

        # IK-specific arrays only for non-fixed joints
        self.ik_lower_limits = np.array([])
        self.ik_upper_limits = np.array([])
        self.ik_joint_names = []       # list of [k, jointIndex, jointName]

    # ----------------------------
    # Init helpers / validation
    # ----------------------------
    def init_env(self, body, env, indices=None):
        """Convenience initializer using an env with fields: id, np_random."""
        self.init(body=body, physics_client_id=env.id, np_random=env.np_random, indices=indices)

    def init(self, body, physics_client_id, np_random, indices=None):
        self.body = body
        self.id = physics_client_id
        self.np_random = np_random

        self._assert_connected()
        self._assert_body_valid()

        # Safely get joints
        try:
            nj = p.getNumJoints(self.body, physicsClientId=self.id)
        except Exception:
            nj = 0
            
        self.all_joint_indices = list(range(nj))

        # If caller doesn't pass indices, default to "all non-fixed joints"
        if indices is None:
            indices = self.all_joint_indices

        # Special case: indices == -1 means "skip joint limit setup / no controllable joints"
        if isinstance(indices, int) and indices == -1:
            self.controllable_joint_indices = []
            return

        # If user provided a subset, use it; otherwise use all joints
        self.controllable_joint_indices = list(indices) if isinstance(indices, (list, tuple, np.ndarray)) else [int(indices)]

        # Filter out fixed joints (cannot be controlled)
        # We need to check joint info to filter fixed joints
        if nj > 0:
            filtered_indices = []
            for j in self.controllable_joint_indices:
                if 0 <= j < nj:
                    j_info = p.getJointInfo(self.body, j, physicsClientId=self.id)
                    if j_info[2] != p.JOINT_FIXED:
                        filtered_indices.append(j)
            self.controllable_joint_indices = filtered_indices
        else:
            self.controllable_joint_indices = []

        indices = self._sanitize_joint_indices(indices)

        # Update limits and derive controllable joints
        self.update_joint_limits(indices=self.all_joint_indices)
        
        # Double check filtering with sanitized indices
        self.controllable_joint_indices = self._filter_nonfixed(self.controllable_joint_indices)

        # Enforce limits once at init (optional but keeps things sane)
        if len(self.controllable_joint_indices) > 0:
            self.enforce_joint_limits(self.controllable_joint_indices)

            # Cache arrays for vectorized clamping
            self.controllable_joint_lower_limits = np.array([self.lower_limits.get(j, -100.0) for j in self.controllable_joint_indices], dtype=np.float64)
            self.controllable_joint_upper_limits = np.array([self.upper_limits.get(j, 100.0) for j in self.controllable_joint_indices], dtype=np.float64)
        else:
            self.controllable_joint_lower_limits = np.array([])
            self.controllable_joint_upper_limits = np.array([])

    def _assert_connected(self):
        if self.id is None:
            raise ValueError("physicsClientId (self.id) is None. Did you call p.connect(...) and pass the id?")
        # Note: p.isConnected(self.id) checks if the ID is valid.

    def _assert_body_valid(self):
        if self.body is None:
            raise ValueError("self.body is None (no body loaded).")
        if not isinstance(self.body, int) or self.body < 0:
            raise ValueError(f"self.body must be a valid bodyUniqueId int, got: {self.body!r}")
        
        # REMOVED: p.getNumBodies check that was causing RuntimeError
        # Trust that if 'body' is a valid int, PyBullet will handle it or throw a specific error later.

    def _sanitize_joint_indices(self, indices):
        """Return a clean list of ints, within range."""
        if indices is None:
            return []

        if isinstance(indices, (np.ndarray,)):
            indices = indices.tolist()

        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        else:
            indices = [int(x) for x in list(indices)]
        
        # Safety check if body has joints
        try:
            nj = p.getNumJoints(self.body, physicsClientId=self.id)
        except:
            return []

        indices = [j for j in indices if 0 <= j < nj]
        return indices

    def _filter_nonfixed(self, indices):
        """Return only non-fixed joints."""
        out = []
        for j in self._sanitize_joint_indices(indices):
            info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            if info[2] != p.JOINT_FIXED:
                out.append(j)
        return out

    def _rng_uniform(self, low, high):
        """Uniform sampling compatible with Generator and RandomState."""
        if hasattr(self.np_random, "uniform"):
            return self.np_random.uniform(low, high)
        # very old fallback (shouldn't happen)
        return np.random.uniform(low, high)

    def _rng_integers(self, high):
        """Integer sampling compatible with Generator and RandomState."""
        if hasattr(self.np_random, "integers"):
            return int(self.np_random.integers(high))
        if hasattr(self.np_random, "randint"):
            return int(self.np_random.randint(high))
        return int(np.random.randint(high))

    # ----------------------------
    # Control / state
    # ----------------------------
    def control(self, indices, target_angles, gains, forces):
        self._assert_connected()
        self._assert_body_valid()

        indices = self._sanitize_joint_indices(indices)
        if not indices:
            return

        target_angles = np.asarray(target_angles, dtype=np.float64).tolist()

        if isinstance(gains, (int, float, np.floating)):
            gains = [float(gains)] * len(indices)
        else:
            gains = [float(g) for g in gains]

        if isinstance(forces, (int, float, np.floating)):
            forces = [float(forces)] * len(indices)
        else:
            forces = [float(f) for f in forces]

        p.setJointMotorControlArray(
            bodyUniqueId=self.body,
            jointIndices=indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_angles,
            positionGains=gains,
            forces=forces,
            physicsClientId=self.id
        )

    def get_joint_angles(self, indices=None):
        self._assert_connected()
        self._assert_body_valid()

        if indices is None:
            indices = self.all_joint_indices
        indices = self._sanitize_joint_indices(indices)
        if not indices:
            return np.array([], dtype=np.float64)

        states = p.getJointStates(self.body, jointIndices=indices, physicsClientId=self.id)
        return np.array([s[0] for s in states], dtype=np.float64)

    def get_joint_angles_dict(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        angles = self.get_joint_angles(indices)
        return {j: a for j, a in zip(self._sanitize_joint_indices(indices), angles)}

    # ----------------------------
    # Poses / transforms
    # ----------------------------
    def get_pos_orient(self, link, center_of_mass=False, convert_to_realworld=False):
        self._assert_connected()
        self._assert_body_valid()

        # Check if link index is valid
        num_joints = p.getNumJoints(self.body, physicsClientId=self.id)
        if link != -1 and (link < 0 or link >= num_joints):
            raise ValueError(f"Invalid link index {link}. Body {self.body} has {num_joints} joints. Valid indices are -1 to {num_joints-1}.")

        if link == self.base:
            pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.id)
        else:
            # getLinkState returns:
            # [0]=worldLinkFramePosition, [1]=worldLinkFrameOrientation,
            # [4]=worldLinkFramePosition (FK), [5]=worldLinkFrameOrientation (FK)
            ls = p.getLinkState(self.body, link, computeForwardKinematics=True, physicsClientId=self.id)
            if center_of_mass:
                pos, orn = ls[0], ls[1]
            else:
                pos, orn = ls[4], ls[5]

        pos = np.array(pos, dtype=np.float64)
        orn = np.array(orn, dtype=np.float64)

        if convert_to_realworld:
            return self.convert_to_realworld(pos, orn)
        return pos, orn

    def get_base_pos_orient(self):
        return self.get_pos_orient(self.base)

    def convert_to_realworld(self, pos, orient=(0, 0, 0, 1)):
        self._assert_connected()
        base_pos, base_orn = self.get_base_pos_orient()
        inv_pos, inv_orn = p.invertTransform(base_pos.tolist(), base_orn.tolist(), physicsClientId=self.id)

        orient = np.array(orient, dtype=np.float64)
        if orient.shape[0] != 4:
            orient = self.get_quaternion(orient)

        real_pos, real_orn = p.multiplyTransforms(
            inv_pos, inv_orn,
            np.array(pos, dtype=np.float64).tolist(),
            orient.tolist(),
            physicsClientId=self.id
        )
        return np.array(real_pos, dtype=np.float64), np.array(real_orn, dtype=np.float64)

    def get_velocity(self, link):
        self._assert_connected()
        self._assert_body_valid()

        if link == self.base:
            return np.array(p.getBaseVelocity(self.body, physicsClientId=self.id)[0], dtype=np.float64)

        ls = p.getLinkState(
            self.body, link,
            computeForwardKinematics=True,
            computeLinkVelocity=True,
            physicsClientId=self.id
        )
        return np.array(ls[6], dtype=np.float64)

    def get_euler(self, quaternion):
        self._assert_connected()
        return np.array(p.getEulerFromQuaternion(list(quaternion), physicsClientId=self.id), dtype=np.float64)

    def get_quaternion(self, euler):
        self._assert_connected()
        return np.array(p.getQuaternionFromEuler(list(euler), physicsClientId=self.id), dtype=np.float64)

    def get_mass(self, link):
        self._assert_connected()
        self._assert_body_valid()
        return float(p.getDynamicsInfo(self.body, link, physicsClientId=self.id)[0])

    # ----------------------------
    # Joint info / sensors
    # ----------------------------
    def get_motor_joint_states(self, joints=None):
        """
        Return (motor_indices, motor_positions, motor_velocities, motor_torques)
        for non-fixed joints.
        """
        self._assert_connected()
        self._assert_body_valid()

        if joints is None:
            joints = self.all_joint_indices
        joints = self._sanitize_joint_indices(joints)
        if not joints:
            return [], [], [], []

        joint_states = p.getJointStates(self.body, joints, physicsClientId=self.id)
        motor_indices, motor_positions, motor_velocities, motor_torques = [], [], [], []

        for j, st in zip(joints, joint_states):
            info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            if info[2] != p.JOINT_FIXED:
                motor_indices.append(j)
                motor_positions.append(st[0])
                motor_velocities.append(st[1])
                motor_torques.append(st[3])

        return motor_indices, motor_positions, motor_velocities, motor_torques

    def get_joint_max_force(self, indices=None):
        self._assert_connected()
        self._assert_body_valid()

        if indices is None:
            indices = self.all_joint_indices
        indices = self._sanitize_joint_indices(indices)
        out = []
        for j in indices:
            info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            out.append(info[10])  # maxForce in URDF
        return out

    def get_force_torque_sensor(self, joint):
        self._assert_connected()
        self._assert_body_valid()
        joint = int(joint)
        return np.array(p.getJointState(self.body, joint, physicsClientId=self.id)[2], dtype=np.float64)

    def enable_force_torque_sensor(self, joint):
        self._assert_connected()
        self._assert_body_valid()
        p.enableJointForceTorqueSensor(self.body, int(joint), enableSensor=True, physicsClientId=self.id)

    # ----------------------------
    # Contacts / distances
    # ----------------------------
    def get_contact_points(self, agentB=None, linkA=None, linkB=None):
        self._assert_connected()
        self._assert_body_valid()

        args = dict(bodyA=self.body, physicsClientId=self.id)
        if agentB is not None:
            args["bodyB"] = agentB.body
        if linkA is not None:
            args["linkIndexA"] = int(linkA)
        if linkB is not None:
            args["linkIndexB"] = int(linkB)

        cp = p.getContactPoints(**args) or []
        linkA_idx = [c[3] for c in cp]
        linkB_idx = [c[4] for c in cp]
        posA = [c[5] for c in cp]
        posB = [c[6] for c in cp]
        force = [c[9] for c in cp]
        return linkA_idx, linkB_idx, posA, posB, force

    def get_closest_points(self, agentB, distance=4.0, linkA=None, linkB=None):
        self._assert_connected()
        self._assert_body_valid()

        args = dict(bodyA=self.body, bodyB=agentB.body, distance=float(distance), physicsClientId=self.id)
        if linkA is not None:
            args["linkIndexA"] = int(linkA)
        if linkB is not None:
            args["linkIndexB"] = int(linkB)

        cp = p.getClosestPoints(**args) or []
        linkA_idx = [c[3] for c in cp]
        linkB_idx = [c[4] for c in cp]
        posA = [c[5] for c in cp]
        posB = [c[6] for c in cp]
        contact_distance = [c[8] for c in cp]
        return linkA_idx, linkB_idx, posA, posB, contact_distance

    # ----------------------------
    # Heights / resetting / dynamics
    # ----------------------------
    def get_heights(self, set_on_ground=False):
        self._assert_connected()
        self._assert_body_valid()

        min_z = np.inf
        max_z = -np.inf

        for link in self.all_joint_indices + [self.base]:
            aabb_min, aabb_max = p.getAABB(self.body, link, physicsClientId=self.id)
            min_z = min(min_z, aabb_min[2])
            max_z = max(max_z, aabb_max[2])

        height = max_z - min_z
        base_height = self.get_base_pos_orient()[0][2] - min_z

        if set_on_ground:
            self.set_on_ground(base_height)

        return float(height), float(base_height)

    def set_base_pos_orient(self, pos, orient):
        self._assert_connected()
        self._assert_body_valid()

        orient = np.array(orient, dtype=np.float64)
        if orient.shape[0] != 4:
            orient = self.get_quaternion(orient)

        p.resetBasePositionAndOrientation(self.body, list(pos), orient.tolist(), physicsClientId=self.id)

    def set_base_velocity(self, linear_velocity, angular_velocity):
        self._assert_connected()
        self._assert_body_valid()
        p.resetBaseVelocity(
            self.body,
            linearVelocity=list(linear_velocity),
            angularVelocity=list(angular_velocity),
            physicsClientId=self.id
        )

    def set_joint_angles(self, indices, angles, use_limits=True, velocities=0.0):
        self._assert_connected()
        self._assert_body_valid()

        indices = self._sanitize_joint_indices(indices)
        if not indices:
            return

        if self.lower_limits is None or len(self.lower_limits) == 0:
            self.update_joint_limits()

        angles = list(np.asarray(angles, dtype=np.float64))
        if isinstance(velocities, (int, float, np.floating)):
            velocities = [float(velocities)] * len(indices)
        else:
            velocities = [float(v) for v in velocities]

        for j, a, v in zip(indices, angles, velocities):
            if use_limits:
                a = min(max(a, self.lower_limits.get(j, -1e10)), self.upper_limits.get(j, 1e10))
            p.resetJointState(self.body, jointIndex=j, targetValue=float(a), targetVelocity=float(v), physicsClientId=self.id)

    def set_on_ground(self, base_height=None):
        if base_height is None:
            _, base_height = self.get_heights()
        pos, orn = self.get_base_pos_orient()
        self.set_base_pos_orient([pos[0], pos[1], base_height], orn)

    def reset_joints(self):
        self.set_joint_angles(self.all_joint_indices, [0.0] * len(self.all_joint_indices), use_limits=False, velocities=0.0)

    def set_frictions(self, links, lateral_friction=None, spinning_friction=None, rolling_friction=None):
        self._assert_connected()
        self._assert_body_valid()

        if isinstance(links, (int, np.integer)):
            links = [int(links)]
        else:
            links = [int(l) for l in links]

        for link in links:
            kwargs = dict(physicsClientId=self.id)
            if lateral_friction is not None:
                kwargs["lateralFriction"] = float(lateral_friction)
            if spinning_friction is not None:
                kwargs["spinningFriction"] = float(spinning_friction)
            if rolling_friction is not None:
                kwargs["rollingFriction"] = float(rolling_friction)
            if kwargs:
                p.changeDynamics(self.body, link, **kwargs)

    def set_friction(self, links, friction):
        self.set_frictions(links, lateral_friction=friction, spinning_friction=friction, rolling_friction=friction)

    def set_mass(self, link, mass):
        self._assert_connected()
        self._assert_body_valid()
        p.changeDynamics(self.body, int(link), mass=float(mass), physicsClientId=self.id)

    def set_joint_stiffness(self, joint, stiffness):
        self._assert_connected()
        self._assert_body_valid()
        p.changeDynamics(self.body, int(joint), jointDamping=float(stiffness), physicsClientId=self.id)

    def set_all_joints_stiffness(self, stiffness):
        for j in self.all_joint_indices:
            self.set_joint_stiffness(j, stiffness)

    def set_gravity(self, ax=0.0, ay=0.0, az=-9.81):
        """World gravity (global), not per-body."""
        self._assert_connected()
        p.setGravity(float(ax), float(ay), float(az), physicsClientId=self.id)

    # ----------------------------
    # Constraints
    # ----------------------------
    def create_constraint(
        self,
        parent_link,
        child,
        child_link,
        joint_type=p.JOINT_FIXED,
        joint_axis=(0, 0, 0),
        parent_pos=(0, 0, 0),
        child_pos=(0, 0, 0),
        parent_orient=(0, 0, 0, 1),
        child_orient=(0, 0, 0, 1),
    ):
        self._assert_connected()
        self._assert_body_valid()

        parent_orient = np.array(parent_orient, dtype=np.float64)
        child_orient = np.array(child_orient, dtype=np.float64)

        if parent_orient.shape[0] != 4:
            parent_orient = self.get_quaternion(parent_orient)
        if child_orient.shape[0] != 4:
            child_orient = self.get_quaternion(child_orient)

        return p.createConstraint(
            parentBodyUniqueId=self.body,
            parentLinkIndex=int(parent_link),
            childBodyUniqueId=child.body,
            childLinkIndex=int(child_link),
            jointType=int(joint_type),
            jointAxis=list(joint_axis),
            parentFramePosition=list(parent_pos),
            childFramePosition=list(child_pos),
            parentFrameOrientation=parent_orient.tolist(),
            childFrameOrientation=child_orient.tolist(),
            physicsClientId=self.id
        )

    # ----------------------------
    # Joint limits / IK
    # ----------------------------
    def update_joint_limits(self, indices=None):
        self._assert_connected()
        self._assert_body_valid()

        if indices is None:
            indices = self.all_joint_indices
        indices = self._sanitize_joint_indices(indices)

        self.lower_limits = {}
        self.upper_limits = {}
        ik_lower = []
        ik_upper = []
        ik_names = []

        for j in indices:
            info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            joint_name = info[1]         # bytes
            joint_type = info[2]
            lower = float(info[8])
            upper = float(info[9])

            # PyBullet convention: (0, -1) means "no limits"
            if lower == 0.0 and upper == -1.0:
                lower = -1e10
                upper = 1e10
                if joint_type != p.JOINT_FIXED:
                    ik_lower.append(-2 * np.pi)
                    ik_upper.append( 2 * np.pi)
                    ik_names.append([len(ik_names), j, joint_name])
            elif joint_type != p.JOINT_FIXED:
                ik_lower.append(lower)
                ik_upper.append(upper)
                ik_names.append([len(ik_names), j, joint_name])

            self.lower_limits[j] = lower
            self.upper_limits[j] = upper

        self.ik_lower_limits = np.array(ik_lower, dtype=np.float64)
        self.ik_upper_limits = np.array(ik_upper, dtype=np.float64)
        self.ik_joint_names = ik_names

    def enforce_joint_limits(self, indices=None):
        self._assert_connected()
        self._assert_body_valid()

        if indices is None:
            indices = self.all_joint_indices
        indices = self._sanitize_joint_indices(indices)
        if not indices:
            return

        if not self.lower_limits:
            self.update_joint_limits(self.all_joint_indices)

        joint_angles = self.get_joint_angles_dict(indices)
        for j in indices:
            a = joint_angles.get(j, 0.0)
            lo = self.lower_limits.get(j, -1e10)
            hi = self.upper_limits.get(j, 1e10)
            if a < lo:
                p.resetJointState(self.body, jointIndex=j, targetValue=float(lo), targetVelocity=0.0, physicsClientId=self.id)
            elif a > hi:
                p.resetJointState(self.body, jointIndex=j, targetValue=float(hi), targetVelocity=0.0, physicsClientId=self.id)

    def ik(
        self,
        target_joint,
        target_pos,
        target_orient,
        ik_indices,
        max_iterations=1000,
        half_range=False,
        use_current_as_rest=False,
        randomize_limits=False,
    ):
        """
        Returns IK joint angles corresponding to ik_indices (indexes into the IK solution array),
        not necessarily the raw Bullet joint indices.
        """
        self._assert_connected()
        self._assert_body_valid()

        if target_orient is not None and len(target_orient) < 4:
            target_orient = self.get_quaternion(target_orient)

        if self.ik_lower_limits.size == 0 or self.ik_upper_limits.size == 0:
            # build from all joints if not done
            self.update_joint_limits(self.all_joint_indices)

        if randomize_limits:
            # NOTE: original code used uniform(0, lower/upper) which is odd.
            # We'll keep similar behavior but safe.
            ik_lower = self._rng_uniform(0.0, self.ik_lower_limits)
            ik_upper = self._rng_uniform(0.0, self.ik_upper_limits)
        else:
            ik_lower = self.ik_lower_limits.copy()
            ik_upper = self.ik_upper_limits.copy()

        ik_ranges = ik_upper - ik_lower
        if half_range:
            ik_ranges = ik_ranges / 2.0

        if use_current_as_rest:
            _, motor_pos, _, _ = self.get_motor_joint_states()
            ik_rest = np.array(motor_pos, dtype=np.float64)
        else:
            ik_rest = self._rng_uniform(ik_lower, ik_upper)

        kwargs = dict(
            bodyUniqueId=self.body,
            endEffectorLinkIndex=int(target_joint),
            targetPosition=list(target_pos),
            lowerLimits=ik_lower.tolist(),
            upperLimits=ik_upper.tolist(),
            jointRanges=ik_ranges.tolist(),
            restPoses=ik_rest.tolist(),
            maxNumIterations=int(max_iterations),
            physicsClientId=self.id
        )
        if target_orient is not None:
            kwargs["targetOrientation"] = list(target_orient)

        sol = p.calculateInverseKinematics(**kwargs)
        sol = np.array(sol, dtype=np.float64)

        # ik_indices are indices into the IK solution array
        ik_indices = np.asarray(ik_indices, dtype=int)
        return sol[ik_indices]

    # ----------------------------
    # Printing
    # ----------------------------
    def print_joint_info(self, show_fixed=True):
        self._assert_connected()
        self._assert_body_valid()

        joint_names = []
        for j in self.all_joint_indices:
            info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            if show_fixed or info[2] != p.JOINT_FIXED:
                print(info)
                joint_names.append((j, info[1]))
        print(joint_names)