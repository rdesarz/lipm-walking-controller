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


def q_from_base_and_joints(q, oMb):
    R = oMb.rotation
    p = oMb.translation
    quat = pin.Quaternion(R)  # xyzw
    q_out = q.copy()
    q_out[:3] = p
    q_out[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])
    return q_out


def reset_pybullet_from_q(robot, q, map_joint_idx_to_q_idx):
    # base
    base_pos = q[:3]
    base_quat = q[3:7]  # xyzw expected by PyBullet
    pb.resetBasePositionAndOrientation(robot, base_pos, base_quat)
    # joints
    for j_id, q_id in map_joint_idx_to_q_idx.items():
        name = pb.getJointInfo(robot, j_id)[1]
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
            targetVelocity=0.0,
            positionGain=0.4,
            velocityGain=1.0,
            force=200,
        )


def reset_position(robot, q_des, j_to_q_idx):
    for j_id, q_id in j_to_q_idx.items():
        val = q_des[q_id]
        pb.resetJointState(robot, j_id, val, 0.0)


def get_q_from_pybullet(robot, nq, map_joint_idx_to_q_idx):
    q = np.zeros(nq)
    base_pos, base_quat = pb.getBasePositionAndOrientation(robot)
    q[:7] = np.concatenate([base_pos, base_quat])  # base position + orientation

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


class Simulator:
    def __init__(self, dt, path_to_model: Path, model):
        self.cid = pb.connect(pb.GUI, options="--window_title=PyBullet --width=1920 --height=1080")
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
        plane = pb.loadURDF("plane.urdf")
        path_to_urdf = path_to_model / "talos_data" / "urdf" / "talos_full.urdf"
        self.robot = pb.loadURDF(
            str(path_to_urdf),
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=False,
            flags=pb.URDF_MERGE_FIXED_LINKS,
        )

        self.map_joints = build_map_joints(self.robot, model)

    def step(self):
        pb.stepSimulation()

    def reset_robot(self, q):
        reset_pybullet_from_q(self.robot, q, self.map_joints)

    def apply_position_to_robot(self, q):
        apply_position(robot=self.robot, q_des=q, j_to_q_idx=self.map_joints)

    def get_q(self, nq):
        return get_q_from_pybullet(self.robot, nq, self.map_joints)
