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
        self.plane = pb.loadURDF("plane.urdf")
        path_to_urdf = path_to_model / "talos_data" / "urdf" / "talos_full.urdf"
        self.robot = pb.loadURDF(
            str(path_to_urdf),
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=False,
            flags=pb.URDF_MERGE_FIXED_LINKS,
        )

        self.map_joints = build_map_joints(self.robot, model)

        self.line = None

    def step(self):
        pb.stepSimulation()

    def reset_robot(self, q):
        reset_pybullet_from_q(self.robot, q, self.map_joints)

    def apply_position_to_robot(self, q):
        apply_position(robot=self.robot, q_des=q, j_to_q_idx=self.map_joints)

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

        start = [mean_x / len(cps_all), mean_y / len(cps_all), mean_z / len(cps_all)]
        end = (
            start[0] + n_B[0] * total_force * scale,
            start[1] + n_B[1] * total_force * scale,
            start[2] + n_B[2] * total_force * scale,
        )

        # arrow for normal force
        if self.line is None:
            self.line = pb.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=3)
        else:
            pb.addUserDebugLine(
                start,
                end,
                lineColorRGB=color,
                lineWidth=3,
                replaceItemUniqueId=self.line,
            )
