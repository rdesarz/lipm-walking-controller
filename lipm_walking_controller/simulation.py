from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pinocchio as pin
import pybullet as pb
import pybullet_data


def yaw_of(R):
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


def rotz(yaw):
    """
    Rotation matrix for a yaw about +z.

    Args:
        yaw (float): angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix Rz(yaw).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def snap_feet_to_plane(
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
    yl, yr = yaw_of(oMf_lf.rotation), yaw_of(oMf_rf.rotation)

    Rl = rotz(yl) if keep_yaw else np.eye(3)
    Rr = rotz(yr) if keep_yaw else np.eye(3)

    Pl = np.array([oMf_lf.translation[0], oMf_lf.translation[1], z_offset])
    Pr = np.array([oMf_rf.translation[0], oMf_rf.translation[1], z_offset])

    Ml = pin.SE3(Rl, Pl)
    Mr = pin.SE3(Rr, Pr)

    return Ml, Mr


def compute_base_from_foot_target(
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


def reset_pybullet_from_q(robot_id: int, q: np.ndarray, map_joint_idx_to_q_idx: Dict[int, int]):
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


def apply_position_to_pybullet(robot_id: int, q_des: np.ndarray, j_to_q_idx: Dict[int, int]):
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


def build_bullet_to_pin_vmap(robot_id, model):
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


def apply_velocity_to_pybullet(robot_id, v_des, j_to_q_idx):
    """
    Velocity control for Bullet joints using desired joint velocities.

    Args:
        robot_id (int): PyBullet body id.
        v_des (np.ndarray): desired joint velocity vector aligned with Pinocchio v.
        j_to_q_idx (Dict[int,int]): Bullet joint id -> q index mapping.

    Side effects:
        Calls pb.setJointMotorControl2 in VELOCITY_CONTROL with force limits.
    """
    for j_id, q_id in j_to_q_idx.items():
        joint_max = pb.getJointInfo(robot_id, j_id)[10]
        pb.setJointMotorControl2(
            robot_id,
            j_id,
            pb.VELOCITY_CONTROL,
            targetVelocity=v_des[q_id],
            velocityGain=1.0,
            force=0.8 * joint_max,
        )


def reset_position(robot_id, q_des, j_to_q_idx):
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


def get_q_from_pybullet(robot_id, nq, map_joint_idx_to_q_idx):
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


def build_map_joints(robot_id, model):
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
    map_joint_idx_to_q_idx = {}
    for j in range(pb.getNumJoints(robot_id)):
        joint_name = pb.getJointInfo(robot_id, j)[1]

        jid = model.get_joint_id(joint_name)
        if jid is not None and jid >= 0:
            map_joint_idx_to_q_idx[j] = jid

    return map_joint_idx_to_q_idx


def link_index(body_id, link_name):
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
    def __init__(self, dt, path_to_model: Path, model, launch_gui=True):
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
            frictionERP=0.2,
        )

        self.plane = pb.loadURDF("plane.urdf")
        path_to_urdf = path_to_model / "talos_data" / "urdf" / "talos_full.urdf"
        self.robot = pb.loadURDF(
            str(path_to_urdf),
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=False,
            flags=pb.URDF_MERGE_FIXED_LINKS,
        )

        self.vel_map = build_bullet_to_pin_vmap(self.robot, model.model)

        self.map_joints = build_map_joints(self.robot, model)

        self.line = None
        self.text = None
        self.point = None

    def step(self):
        pb.stepSimulation()

    def reset_robot(self, q: np.ndarray):
        reset_pybullet_from_q(self.robot, q, self.map_joints)

    def apply_position_to_robot(self, q: np.ndarray):
        apply_position_to_pybullet(robot_id=self.robot, q_des=q, j_to_q_idx=self.map_joints)

    def apply_velocity_to_robot(self, v: np.ndarray):
        apply_velocity_to_pybullet(robot_id=self.robot, v_des=v, j_to_q_idx=self.vel_map)

    def get_q(self, nq: int):
        return get_q_from_pybullet(self.robot, nq, self.map_joints)

    def update_camera_to_follow_pos(self, x: float, y: float, z: float):
        pb.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=50,
            cameraPitch=-40,
            cameraTargetPosition=[x, y, z],
        )

    def draw_contact_forces(self, color=(0, 1, 0), scale=0.002):
        # iterate all foot links if you want per-foot colors
        cps_all = []

        cps_all.extend(pb.getContactPoints(bodyA=self.robot, bodyB=self.plane))

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
            if self.line is None:
                self.line = pb.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=3)
                # self.text = pb.addUserDebugText(f"{total_force} N", end)
            else:
                pb.addUserDebugLine(
                    start,
                    end,
                    lineColorRGB=color,
                    lineWidth=3,
                    replaceItemUniqueId=self.line,
                )
                # pb.addUserDebugText(f"{total_force} N", end, replaceItemUniqueId=self.text)

    def draw_point(self, points, colors):
        if self.point is None:
            self.point = pb.addUserDebugPoints(points, colors, pointSize=5.0)
        else:
            pb.addUserDebugPoints(points, colors, pointSize=5.0, replaceItemUniqueId=self.point)

    def get_robot_com_position(self):
        # base (link index = -1)
        base_pos, base_orn = pb.getBasePositionAndOrientation(self.robot)
        dyn_info = pb.getDynamicsInfo(self.robot, -1)
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
        for link_idx in range(pb.getNumJoints(self.robot)):
            # get COM position of the link directly in world
            state = pb.getLinkState(self.robot, link_idx, computeForwardKinematics=True)
            link_com_world = np.array(state[0])  # COM position in world
            m_link = pb.getDynamicsInfo(self.robot, link_idx)[0]
            com_sum += m_link * link_com_world
            m_total += m_link

        return (com_sum / m_total).tolist()

    def get_robot_pos(self):
        return pb.getBasePositionAndOrientation(self.robot)

    def get_robot_frame_pos(self, frame_name: str):
        # Link/world poses
        i = link_index(self.robot, frame_name)
        state = pb.getLinkState(self.robot, i, computeForwardKinematics=1)

        return state[4], state[5]

    def get_zmp_pose(self):
        cps = pb.getContactPoints(bodyA=self.robot, bodyB=self.plane)

        F = np.zeros(3)
        M = np.zeros(3)

        for cp in cps:
            a = cp[1]  # bodyUniqueIdA
            b = cp[2]  # bodyUniqueIdB
            pos_on_a = np.array(cp[5])  # positionOnA in world
            pos_on_b = np.array(cp[6])  # positionOnB in world
            n_b2a = np.array(cp[7])  # contactNormalOnB (points B -> A)
            fn = cp[9]  # normalForce
            ft1 = cp[10] if len(cp) > 10 else 0.0
            dir1 = np.array(cp[11]) if len(cp) > 11 else np.zeros(3)
            ft2 = cp[12] if len(cp) > 12 else 0.0
            dir2 = np.array(cp[13]) if len(cp) > 13 else np.zeros(3)

            # Force vector defined along B's normal and friction directions
            f_on_b = fn * n_b2a + ft1 * dir1 + ft2 * dir2
            # Action–reaction: force on A is opposite
            if a == self.robot and b != self.robot:
                f = -f_on_b
                r = pos_on_a
            elif b == self.robot and a != self.robot:
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
