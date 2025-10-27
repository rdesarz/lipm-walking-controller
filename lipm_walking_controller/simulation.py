from pathlib import Path

import numpy as np
import pinocchio as pin
import pybullet as pb
import pybullet_data


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def snap_feet_to_plane(oMf_lf, oMf_rf, z_offset=0.0, keep_yaw=True):
    yl, yr = yaw_of(oMf_lf.rotation), yaw_of(oMf_rf.rotation)

    Rl = rotz(yl) if keep_yaw else np.eye(3)
    Rr = rotz(yr) if keep_yaw else np.eye(3)

    Pl = np.array([oMf_lf.translation[0], oMf_lf.translation[1], z_offset])
    Pr = np.array([oMf_rf.translation[0], oMf_rf.translation[1], z_offset])

    Ml = pin.SE3(Rl, Pl)
    Mr = pin.SE3(Rr, Pr)

    return Ml, Mr


def compute_base_from_foot_target(model, data, q, foot_frame_id, oMf_target):
    """
    With a free-flyer q[:7] = [p_b, q_b], compute base pose oMb so that
    the chosen foot frame matches oMf_target, keeping joint part of q.
    Uses: oMf_target = oMb * bMf(q_joints)
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    bMf = data.oMf[foot_frame_id]  # currently expressed in world
    # Extract current base-in-world to convert bMf to base frame:
    p_b = q[:3]
    q_b = q[3:7]
    oMb_cur = pin.SE3(pin.Quaternion(q_b).toRotationMatrix(), p_b)
    bMf_in_base = oMb_cur.inverse() * bMf  # ^b M_f at current joints
    oMb_new = oMf_target * bMf_in_base.inverse()
    return oMb_new


def reset_pybullet_from_q(robot, q, map_joint_idx_to_q_idx):
    # base
    p_base = list(q[:3])
    q_base = list(q[3:7])

    # Inertial (CoM) pose relative to base_link
    _, _, _, p_li, q_li, *_ = pb.getDynamicsInfo(robot, -1)

    # World pose of CoM/inertial frame
    p_com, q_com = pb.multiplyTransforms(p_base, q_base, p_li, q_li)

    pb.resetBasePositionAndOrientation(robot, p_com, q_com)

    # joints
    for j_id, q_id in map_joint_idx_to_q_idx.items():
        if q_id >= 0:
            val = float(q[q_id])
            pb.resetJointState(robot, j_id, val)


def apply_position(robot, q_des, j_to_q_idx):
    for j_id, q_id in j_to_q_idx.items():
        pb.setJointMotorControl2(
            robot,
            j_id,
            pb.POSITION_CONTROL,
            targetPosition=q_des[q_id],
            positionGain=0.4,
            force=1000,
        )


def build_bullet_to_pin_vmap(robot, model):
    name_to_bullet = {}
    for jid in range(pb.getNumJoints(robot)):
        jn, jtype = pb.getJointInfo(robot, jid)[1].decode(), pb.getJointInfo(robot, jid)[2]
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


def apply_velocity(robot, v_des, j_to_q_idx):
    for j_id, q_id in j_to_q_idx.items():
        pb.setJointMotorControl2(
            robot,
            j_id,
            pb.VELOCITY_CONTROL,
            targetVelocity=v_des[q_id],
            velocityGain=1.0,
            force=200,
        )


def reset_position(robot, q_des, j_to_q_idx):
    for j_id, q_id in j_to_q_idx.items():
        val = q_des[q_id]
        pb.resetJointState(robot, j_id, val, 0.0)


def get_q_from_pybullet(robot, nq, map_joint_idx_to_q_idx):
    q = np.zeros(nq)
    com_pos, com_quat = pb.getBasePositionAndOrientation(robot)

    # Inertial (CoM) pose relative to base_link
    _, _, _, p_li, q_li, *_ = pb.getDynamicsInfo(robot, -1)
    p_ib, q_ib = pb.invertTransform(p_li, q_li)  # inertial_T_base

    # World pose of CoM/inertial frame
    p_base, q_base = pb.multiplyTransforms(com_pos, com_quat, p_ib, q_ib)

    q[:7] = np.concatenate([p_base, q_base])  # base position + orientation

    for j_id, q_id in map_joint_idx_to_q_idx.items():
        if q_id < 0:  # skip fixed joints
            continue
        q[q_id] = pb.getJointState(robot, j_id)[0]
    return q


def build_map_joints(robot, model):
    map_joint_idx_to_q_idx = {}
    for j in range(pb.getNumJoints(robot)):
        joint_name = pb.getJointInfo(robot, j)[1]

        jid = model.get_joint_id(joint_name)
        if jid is not None and jid >= 0:
            map_joint_idx_to_q_idx[j] = jid

    return map_joint_idx_to_q_idx


def link_index(body_id, link_name):
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

    def reset_robot(self, q):
        reset_pybullet_from_q(self.robot, q, self.map_joints)

    def apply_position_to_robot(self, q):
        apply_position(robot=self.robot, q_des=q, j_to_q_idx=self.map_joints)

    def apply_velocity_to_robot(self, v):
        apply_velocity(robot=self.robot, v_des=v, j_to_q_idx=self.vel_map)

    def get_q(self, nq):
        return get_q_from_pybullet(self.robot, nq, self.map_joints)

    def update_camera_to_follow_pos(self, x, y, z):
        pb.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=50,
            cameraPitch=-40,
            cameraTargetPosition=[x, y, z],
        )

    def draw_contact_forces(self, color=(0, 1, 0), scale=0.002, life=0.2):
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
        i = link_index(self.robot, frame_name)  # example
        state = pb.getLinkState(self.robot, i, computeForwardKinematics=1)

        return state[4], state[5]
