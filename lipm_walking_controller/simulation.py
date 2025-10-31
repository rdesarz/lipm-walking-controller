import typing
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pinocchio as pin
import pybullet as pb
import pybullet_data


def _yaw_of(R):
    """
    Return the yaw angle (rotation about +z) of a 3x3 rotation matrix.
    Args:
        R (np.ndarray): 3x3 rotation matrix.

    Returns:
        float: yaw in radians

    Notes:
        Assumes ZYX convention where yaw is extracted from the top-left 2x2 block.
    """
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _rotz(yaw):
    """
    Rotation matrix for a yaw about +z.

    Args:
        yaw (float): angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix Rz(yaw).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _snap_feet_to_plane(
    oMf_lf: pin.SE3, oMf_rf: pin.SE3, z_offset: float = 0.0, keep_yaw: bool = False
) -> Tuple[pin.SE3, pin.SE3]:
    """
    Project both feet to a horizontal plane shifted by z_offset. Optionally keep yaw orientation of the foot.

    Args:
        oMf_lf (pin.SE3): world pose of left foot frame.
        oMf_rf (pin.SE3): world pose of right foot frame.
        z_offset (float): target plane height in world z. Default 0.0.
        keep_yaw (bool): if True keep each foot's yaw; otherwise set identity R.

    Returns:
        Tuple[pin.SE3, pin.SE3]: new world poses (left, right) snapped to the plane.

    Notes:
        Only translation z is changed. Roll/pitch are set to zero if keep_yaw=False.
    """
    yl, yr = _yaw_of(oMf_lf.rotation), _yaw_of(oMf_rf.rotation)

    Rl = _rotz(yl) if keep_yaw else np.eye(3)
    Rr = _rotz(yr) if keep_yaw else np.eye(3)

    Pl = np.array([oMf_lf.translation[0], oMf_lf.translation[1], z_offset])
    Pr = np.array([oMf_rf.translation[0], oMf_rf.translation[1], z_offset])

    Ml = pin.SE3(Rl, Pl)
    Mr = pin.SE3(Rr, Pr)

    return Ml, Mr


def _compute_base_from_foot_target(
    model: pin.Model, data: pin.Data, q: np.ndarray, foot_frame_id: int, oMf_target: pin.SE3
):
    """
    Compute world base pose oMb that makes a given foot frame match a target pose.

    Args:
        model (pin.Model): Pinocchio model with free-flyer first.
        data (pin.Data): Pinocchio data.
        q (np.ndarray): configuration. q[:7] = [p_b(3), quat_b(x,y,z,w)].
        foot_frame_id (int): Pinocchio frame id of the foot.
        oMf_target (pin.SE3): desired world pose of the foot.

    Returns:
        pin.SE3: new world base pose oMb_new.

    Method:
        Keep joint part of q fixed. Let bMf(q_joints) be current foot pose in base.
        Solve oMf_target = oMb_new * bMf  =>  oMb_new = oMf_target * bMf^{-1}.
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Extract current base-in-world to convert bMf to base frame:
    p_b = q[:3]
    q_b = q[3:7]
    oMb_cur = pin.SE3(pin.Quaternion(q_b).toRotationMatrix(), p_b)

    bMf = data.oMf[foot_frame_id]  # currently expressed in world
    bMf_in_base = oMb_cur.inverse() * bMf
    oMb_new = oMf_target * bMf_in_base.inverse()

    return oMb_new


def _reset_pybullet_from_q(robot_id: int, q: np.ndarray, map_joint_idx_to_q_idx: Dict[int, int]):
    """
    Set a PyBullet robot state from a Pinocchio configuration q.

    Args:
        robot_id (int): PyBullet body unique id.
        q (np.ndarray): configuration. q[:3]=base pos, q[3:7]=base quat (x,y,z,w), rest joints.
        map_joint_idx_to_q_idx (Dict[int,int]): Bullet joint id -> q index mapping
            for actuated joints; fixed joints should be absent or mapped to <0.

    Side effects:
        Calls pb.resetBasePositionAndOrientation and pb.resetJointState for each joint.

    Notes:
        PyBullet base pose is the CoM/inertial frame. We transform Pinocchio base_link
        pose to CoM using getDynamicsInfo(local inertial) before resetting.
    """
    # base
    p_base = list(q[:3])
    q_base = list(q[3:7])

    # Inertial (CoM) pose relative to base_link
    _, _, _, p_li, q_li, *_ = pb.getDynamicsInfo(robot_id, -1)

    # World pose of CoM/inertial frame
    p_com, q_com = pb.multiplyTransforms(p_base, q_base, p_li, q_li)

    pb.resetBasePositionAndOrientation(robot_id, p_com, q_com)

    # joints
    for j_id, q_id in map_joint_idx_to_q_idx.items():
        if q_id >= 0:
            val = float(q[q_id])
            pb.resetJointState(robot_id, j_id, val)


def _apply_position_to_pybullet(robot_id: int, q_des: np.ndarray, j_to_q_idx: Dict[int, int]):
    """
    Apply position control to Bullet joints using desired joint positions.

    Args:
        robot_id (int): PyBullet body id.
        q_des (np.ndarray): desired joint configuration vector aligned with Pinocchio q.
        j_to_q_idx (Dict[int,int]): Bullet joint id -> q index mapping.

    Side effects:
        Calls pb.setJointMotorControl2 in POSITION_CONTROL with per-joint force limits.
    """
    for j_id, q_id in j_to_q_idx.items():
        joint_max = pb.getJointInfo(robot_id, j_id)[10]
        pb.setJointMotorControl2(
            robot_id,
            j_id,
            pb.POSITION_CONTROL,
            targetPosition=q_des[q_id],
            positionGain=0.4,
            force=joint_max * 0.8,
        )


def _build_pb_to_pin_joint_vel_vmap(robot_id, model):
    """
    Build a map from Bullet joint id -> Pinocchio velocity index start (idx_vs).

    Args:
        robot_id (int): PyBullet body id.
        model (pin.Model): Pinocchio model.

    Returns:
        Dict[int,int]: mapping Bullet movable joint id -> model.idx_vs[joint].

    Notes:
        Skips fixed joints in Bullet and Pinocchio. Names must match between URDFs.
        Useful to index v (size nv) blocks per joint in Pinocchio.
    """
    name_to_bullet = {}
    for jid in range(pb.getNumJoints(robot_id)):
        jn, jtype = pb.getJointInfo(robot_id, jid)[1].decode(), pb.getJointInfo(robot_id, jid)[2]
        if jtype != pb.JOINT_FIXED:
            name_to_bullet[jn] = jid
    j_to_v_idx = {}
    for j in range(1, model.njoints):  # 0 = universe
        if model.joints[j].nv == 0:
            continue
        jname = model.names[j]
        if jname in name_to_bullet:
            j_to_v_idx[name_to_bullet[jname]] = model.idx_vs[j]  # velocity start index
    return j_to_v_idx


def _apply_velocity_to_pybullet(robot_id, v_des, pb_to_pin_joint_vel):
    """
    Velocity control for Bullet joints using desired joint velocities.

    Args:
        robot_id (int): PyBullet body id.
        v_des (np.ndarray): desired joint velocity vector aligned with Pinocchio v.
        pb_to_pin_joint_vel (Dict[int,int]): Bullet joint id -> q index mapping.

    Side effects:
        Calls pb.setJointMotorControl2 in VELOCITY_CONTROL with force limits.
    """
    for j_id, q_id in pb_to_pin_joint_vel.items():
        joint_max = pb.getJointInfo(robot_id, j_id)[10]
        pb.setJointMotorControl2(
            robot_id,
            j_id,
            pb.VELOCITY_CONTROL,
            targetVelocity=v_des[q_id],
            velocityGain=1.0,
            force=0.8 * joint_max,
        )


def _reset_pybullet_position(robot_id, q_des, j_to_q_idx):
    """
    Hard reset of Bullet joint positions.

    Args:
        robot_id (int): PyBullet body id.
        q_des (np.ndarray): desired joint configuration vector aligned with Pinocchio q.
        j_to_q_idx (Dict[int,int]): Bullet joint id -> q index mapping.

    Side effects:
        Calls pb.resetJointState on each mapped joint. No motor control used.
    """
    for j_id, q_id in j_to_q_idx.items():
        val = q_des[q_id]
        pb.resetJointState(robot_id, j_id, val, 0.0)


def _get_q_from_pybullet(robot_id, nq, map_joint_idx_to_q_idx):
    """
    Read a full Pinocchio-style configuration q from PyBullet.

    Args:
        robot_id (int): PyBullet body id.
        nq (int): size of configuration vector.
        map_joint_idx_to_q_idx (Dict[int,int]): Bullet joint id -> q index mapping.
            Fixed joints should be mapped to <0 or omitted.

    Returns:
        np.ndarray: q with base pos, base quaternion (x,y,z,w), then joint positions.

    Notes:
        PyBullet returns CoM pose. Convert to base_link pose using local inertial
        transform from getDynamicsInfo and its inverse.
    """
    q = np.zeros(nq)
    com_pos, com_quat = pb.getBasePositionAndOrientation(robot_id)

    # Inertial (CoM) pose relative to base_link
    _, _, _, p_li, q_li, *_ = pb.getDynamicsInfo(robot_id, -1)
    p_ib, q_ib = pb.invertTransform(p_li, q_li)  # inertial_T_base

    # World pose of CoM/inertial frame
    p_base, q_base = pb.multiplyTransforms(com_pos, com_quat, p_ib, q_ib)

    q[:7] = np.concatenate([p_base, q_base])  # base position + orientation

    for j_id, q_id in map_joint_idx_to_q_idx.items():
        if q_id < 0:  # skip fixed joints
            continue
        q[q_id] = pb.getJointState(robot_id, j_id)[0]
    return q


def _build_pb_to_pin_joints_map(robot_id, model):
    """
    Build Bullet joint id -> Pinocchio q index map using joint names.

    Args:
        robot_id (int): PyBullet body id.
        model (pin.Model): Pinocchio model exposing get_joint_id(name)->int.

    Returns:
        Dict[int,int]: map_joint_idx_to_q_idx for movable joints.

    Notes:
        Name matching must be exact. Fixed joints should be excluded or mapped to -1.
    """
    pb_to_pin_joints_map = {}
    for j in range(pb.getNumJoints(robot_id)):
        joint_name = pb.getJointInfo(robot_id, j)[1]

        jid = model.get_joint_id(joint_name)
        if jid is not None and jid >= 0:
            pb_to_pin_joints_map[j] = jid

    return pb_to_pin_joints_map


def _link_index(body_id, link_name):
    """
    Return the Bullet link index for a given link (joint) name.

    Args:
        body_id (int): PyBullet body id.
        link_name (str): link name as stored in Bullet.

    Returns:
        Optional[int]: link index if found, else None.

    Notes:
        Uses pb.getJointInfo(...)[12] which stores the link name string.
    """
    n = pb.getNumJoints(body_id)
    for i in range(n):
        name = pb.getJointInfo(body_id, i)[12].decode()
        if name == link_name:  # joint/link name
            return i

    return None


class Simulator:
    """
    Thin PyBullet wrapper for loading a robot URDF and driving it from
    joint positions/velocities. Also exposes utilities for camera control,
    contact-force/ZMP computation, and setter/getter for joints configuration.
    The robot is loaded at the origin and a flat horizontal plane is added at z=0.
    """

    def __init__(self, dt, path_to_robot_urdf: Path, model, launch_gui=True):
        """
        Initialize PyBullet, load ground and TALOS, and set physics.

        Parameters
        ----------
        dt : float
            Fixed simulation time step in seconds.
        path_to_robot_urdf : pathlib.Path
            Base path containing the URDF data.
        model : Any
            Pinocchio model container used to build pb↔pin joint maps.
            Must expose ``model`` for velocity map construction.
        launch_gui : bool, default True
            If True, starts PyBullet with GUI. Otherwise uses DIRECT mode.

        Side Effects
        ------------
        - Connects to PyBullet and configures gravity, time step, and engine params.
        - Loads a plane and the TALOS robot in the world.
        - Builds maps from PyBullet joint indices to Pinocchio q/v indices.
        """
        # Initialize the simulator
        self.cid = pb.connect(
            pb.GUI if launch_gui else pb.DIRECT,
            options="--window_title=PyBullet --width=1920 --height=1080",
        )
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(dt)
        pb.setRealTimeSimulation(0)
        pb.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=100,
            numSubSteps=1,
            useSplitImpulse=1,
            splitImpulsePenetrationThreshold=0.01,
            contactSlop=0.001,
            erp=0.2,
            contactERP=0.2,
            frictionERP=0.05,
        )

        # Load elements in the simulation
        self.plane_id: int = pb.loadURDF("plane.urdf")

        self.robot_id: int = pb.loadURDF(
            str(path_to_robot_urdf),
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=False,
            flags=pb.URDF_MERGE_FIXED_LINKS,
        )

        # Build mapping between joints for both velocities and position
        self.pb_to_pin_joint_vel = _build_pb_to_pin_joint_vel_vmap(self.robot_id, model.model)
        self.pb_to_pin_joints_pos = _build_pb_to_pin_joints_map(self.robot_id, model)

        self._displayed_lines = None
        self._displayed_points = None

    def step(self):
        """
        Advance the physics simulation by one fixed time step.

        Notes
        -----
        Uses the engine time step configured in `__init__`.
        """
        pb.stepSimulation()

    def reset_robot_configuration(self, q: np.ndarray):
        """
        Hard-reset robot state from a configuration vector.

        Parameters
        ----------
        q : np.ndarray, shape (nq,)
            Full Pinocchio configuration, including free-flyer (7).
            Applied to PyBullet base + joints via helper mapping.
        """
        _reset_pybullet_from_q(self.robot_id, q, self.pb_to_pin_joints_pos)

    def apply_joints_pos_to_robot(self, q: np.ndarray):
        """
        Send position targets to PyBullet joint position controllers.

        Parameters
        ----------
        q : np.ndarray, shape (nq,)
            Desired configuration in Pinocchio ordering.
        """
        _apply_position_to_pybullet(
            robot_id=self.robot_id, q_des=q, j_to_q_idx=self.pb_to_pin_joints_pos
        )

    def apply_joints_vel_to_robot(self, v: np.ndarray):
        """
        Send joint-space velocity targets to the simulator.

        Parameters
        ----------
        v : np.ndarray, shape (nv,)
            Desired generalized velocities in Pinocchio ordering.
        """
        _apply_velocity_to_pybullet(
            robot_id=self.robot_id, v_des=v, pb_to_pin_joint_vel=self.pb_to_pin_joint_vel
        )

    def get_q(self, nq: int) -> np.ndarray:
        """
        Read the current configuration vector from PyBullet.

        Parameters
        ----------
        nq : int
            Total configuration size expected by Pinocchio.

        Returns
        -------
        np.ndarray, shape (nq,)
            Configuration with base pose first (x,y,z, qw,qx,qy,qz), then joints.
        """
        return _get_q_from_pybullet(self.robot_id, nq, self.pb_to_pin_joints_pos)

    def update_camera_to_follow_pos(self, x: float, y: float, z: float):
        """
        Aim the debug camera at a world position.

        Parameters
        ----------
        x, y, z : float
            Target position in world frame for the camera to look at.

        Notes
        -----
        Keeps distance, yaw, and pitch fixed.
        """
        pb.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=50,
            cameraPitch=-40,
            cameraTargetPosition=[x, y, z],
        )

    def draw_contact_forces(self, color=(0, 1, 0), scale=0.002):
        """
        Render an averaged ground-reaction normal as a debug line.

        Parameters
        ----------
        color : tuple[float, float, float], default (0, 1, 0)
            RGB color in [0,1].
        scale : float, default 0.002
            Visual scale factor converting Newtons to line length.

        Notes
        -----
        - Aggregates all robot–plane contacts, averages contact locations,
          sums normal forces, and draws a single arrow along the last
          observed contact normal.
        - Updates a persistent debug line if it already exists.
        """
        # iterate all foot links if you want per-foot colors
        cps_all = []

        cps_all.extend(pb.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id))

        mean_x = 0.0
        mean_y = 0.0
        mean_z = 0.0
        total_force = 0.0
        n_B = [0.0, 0.0, 0.0]
        for cp in cps_all:
            # tuple fields (relevant):
            # cp[4]=posOnA (world xyz), cp[5]=posOnB, cp[6]=normalOnB (world xyz),
            # cp[7]=distance, cp[8]=normalForce
            posB = cp[6]
            n_B = cp[7]
            fN = cp[9]  # Newtons
            mean_x += posB[0]
            mean_y += posB[1]
            mean_z += posB[2]
            total_force += fN

        if len(cps_all) is not 0:
            start = [mean_x / len(cps_all), mean_y / len(cps_all), mean_z / len(cps_all)]
            end = (
                start[0] + n_B[0] * total_force * scale,
                start[1] + n_B[1] * total_force * scale,
                start[2] + n_B[2] * total_force * scale,
            )

            # arrow for normal force
            if self._displayed_lines is None:
                self._displayed_lines = pb.addUserDebugLine(
                    start, end, lineColorRGB=color, lineWidth=3
                )
            else:
                pb.addUserDebugLine(
                    start,
                    end,
                    lineColorRGB=color,
                    lineWidth=3,
                    replaceItemUniqueId=self._displayed_lines,
                )

    def draw_points(
        self,
        points: typing.List[typing.Tuple],
        colors: typing.Optional[typing.List[typing.Tuple[float, float, float]]] = None,
        point_size: int = 5,
    ):
        """
        Draw or update a set of debug points.

        Parameters
        ----------
        points : array-like, shape (N, 3)
            World coordinates of points.
        colors : array-like, shape (N, 3)
            RGB colors in [0,1] per point.
        point_size: size of the displayed points

        Notes
        -----
        Uses a persistent debug item so updates replace the existing set (for performance reasons)
        """
        colors = [(1, 1, 1) for _ in points] if None else colors
        if self._displayed_points is None:
            self._displayed_points = pb.addUserDebugPoints(points, colors, pointSize=point_size)
        else:
            pb.addUserDebugPoints(
                points, colors, pointSize=point_size, replaceItemUniqueId=self._displayed_points
            )

    def get_robot_com_position(self) -> np.ndarray:
        """
        Compute the whole-body center of mass in world coordinates.

        Returns
        -------
        list[float]
            [x, y, z] position of the CoM in the world frame.

        Method
        ------
        - Retrieves base inertial COM, then iterates articulated links.
        - Mass-weights each link COM and normalizes by total mass.
        """
        # base (link index = -1)
        base_pos, base_orn = pb.getBasePositionAndOrientation(self.robot_id)
        dyn_info = pb.getDynamicsInfo(self.robot_id, -1)
        m_base = dyn_info[0]

        # base COM in world = base frame + inertial offset rotated
        # get base inertial offset from getDynamicsInfo (items 3–6)
        base_inertial_pos = dyn_info[3]
        base_inertial_orn = dyn_info[4]

        base_inertial_world = pb.multiplyTransforms(
            base_pos, base_orn, base_inertial_pos, base_inertial_orn
        )[0]

        m_total = m_base
        com_sum = m_base * np.array(base_inertial_world)

        # each articulated link
        for link_idx in range(pb.getNumJoints(self.robot_id)):
            # get COM position of the link directly in world
            state = pb.getLinkState(self.robot_id, link_idx, computeForwardKinematics=True)
            link_com_world = np.array(state[0])  # COM position in world
            m_link = pb.getDynamicsInfo(self.robot_id, link_idx)[0]
            com_sum += m_link * link_com_world
            m_total += m_link

        return (com_sum / m_total).tolist()

    def get_robot_pos(self):
        """
        Get the floating base world pose.

        Returns
        -------
        tuple[list[float], list[float]]
            (position [x,y,z], quaternion [x,y,z,w]) in world frame.
        """
        return pb.getBasePositionAndOrientation(self.robot_id)

    def get_robot_frame_pos(self, frame_name: str):
        """
        Get world pose of a named link frame.

        Parameters
        ----------
        frame_name : str
            Name of the link in the URDF.

        Returns
        -------
        tuple[tuple[float, float, float], tuple[float, float, float, float]]
            (position, quaternion) of the link frame in world.
            Uses PyBullet forward kinematics.

        Raises
        ------
        KeyError
            If `frame_name` is not found.
        """
        # Link/world poses
        i = _link_index(self.robot_id, frame_name)
        state = pb.getLinkState(self.robot_id, i, computeForwardKinematics=1)

        return state[4], state[5]

    def get_zmp_pose(self):
        """
        Estimate the Zero-Moment Point using contact wrenches against the plane.

        Returns
        -------
        np.ndarray | None
            [px, py, 0.0] in world if vertical force Fz != 0, else None.

        Method
        ------
        - Iterates robot–plane contact points.
        - Accumulates total contact force F and moment M about world origin
          using r × f at the contact position.
        - ZMP on the ground plane is computed as:
          px = -M_y / F_z,  py = M_x / F_z.

        Notes
        -----
        This assumes a flat ground at z=0 and uses only normal forces
        reported by PyBullet at each contact.
        """
        contact_pts = pb.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id)

        F = np.zeros(3)
        M = np.zeros(3)

        for contact_pt in contact_pts:
            a = contact_pt[1]
            b = contact_pt[2]

            pos_on_a = np.array(contact_pt[5])
            pos_on_b = np.array(contact_pt[6])

            n_b2a = np.array(contact_pt[7])
            fn = contact_pt[9]

            # Force vector defined along B's normal and friction directions
            f_on_b = fn * n_b2a

            # Action–reaction: force on A is opposite
            if a == self.robot_id and b != self.robot_id:
                f = -f_on_b
                r = pos_on_a
            elif b == self.robot_id and a != self.robot_id:
                f = f_on_b
                r = pos_on_b
            else:
                continue

            F += f
            M += np.cross(r, f)

        if abs(F[2]) < 1e-6:
            return None  # undefined ZMP

        px = -M[1] / F[2]
        py = M[0] / F[2]
        return np.array([px, py, 0.0])
