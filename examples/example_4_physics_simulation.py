import numpy as np
import pybullet as pb
import pybullet_data
import os
import pinocchio as pin

from lipm_walking_controller.inverse_kinematic import InvKinSolverParams, solve_inverse_kinematics

from lipm_walking_controller.controller import (
    PreviewControllerParams,
    compute_preview_control_matrices,
    update_control,
)

from lipm_walking_controller.model import Talos


def se3(R, p):
    M = pin.SE3()
    M.rotation = R
    M.translation = np.asarray(p, dtype=float)
    return M


def yaw_of(R):
    # ZYX yaw from rotation
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def snap_feet_to_plane(oMf_lf, oMf_rf, z_plane=0.0, keep_yaw=True):
    # keep current yaw, zero roll/pitch, put soles on z=z_plane
    yl, yr = yaw_of(oMf_lf.rotation), yaw_of(oMf_rf.rotation)
    Rl = rotz(yl) if keep_yaw else np.eye(3)
    Rr = rotz(yr) if keep_yaw else np.eye(3)
    Pl = np.array([oMf_lf.translation[0], oMf_lf.translation[1], z_plane])
    Pr = np.array([oMf_rf.translation[0], oMf_rf.translation[1], z_plane])
    Ml = se3(Rl, Pl)
    Mr = se3(Rr, Pr)
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
        if q_id >= 0:
            pb.resetJointState(robot, j_id, float(q[q_id]))


def apply_position(q_des, j_to_q_idx):
    for j_id, q_id in j_to_q_idx.items():
        pb.setJointMotorControl2(
            robot,
            j_id,
            pb.POSITION_CONTROL,
            targetPosition=q_des[q_id],
            targetVelocity=0.0,
            positionGain=0.6,
            velocityGain=1.0,
            force=200,
        )


def reset_position(q_des, j_to_q_idx):
    for j_id, q_id in j_to_q_idx.items():
        pb.resetJointState(robot, j_id, q_des[q_id])


def get_q_from_pybullet(robot, model, map_joint_idx_to_q_idx):
    q = np.zeros(model.nq)
    base_pos, base_quat = pb.getBasePositionAndOrientation(robot)
    q[:7] = np.concatenate([base_pos, base_quat])  # base position + orientation

    for j_id, q_id in map_joint_idx_to_q_idx.items():
        if q_id < 0:  # skip fixed joints
            continue
        q[q_id] = pb.getJointState(robot, j_id)[0]
    return q


if __name__ == "__main__":
    dt = 1.0 / 240.0
    cid = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=50, erp=0.2, contactERP=0.2)

    # Preview controller parameters
    t_preview = 1.6  # Time horizon used for the preview controller

    # ZMP reference parameters
    t_ss = 0.6  # Single support phase time window
    t_ds = 0.1  # Double support phase time window
    n_steps = 25
    l_stride = 0.3
    max_height_foot = 0.05

    ctrler_params = PreviewControllerParams(
        zc=0.80,
        g=9.81,
        Qe=1.0,
        Qx=np.zeros((3, 3)),
        R=1e-6,
        n_preview_steps=int(round(t_preview / dt)),
    )
    ctrler_mat = compute_preview_control_matrices(ctrler_params, dt)

    plane = pb.loadURDF("plane.urdf")
    PKG_PARENT = os.path.expanduser(os.environ.get("PKG_PARENT", "~/projects"))
    URDF = os.path.join(PKG_PARENT, "talos_data/urdf/talos_full.urdf")
    robot = pb.loadURDF(
        URDF, [0, 0, 0], [0, 0, 0, 1], useFixedBase=False, flags=pb.URDF_MERGE_FIXED_LINKS
    )

    talos = Talos(path_to_model="~/projects", reduced=False)

    map_joint_idx_to_q_idx = {}
    for j in range(pb.getNumJoints(robot)):
        joint_name = pb.getJointInfo(robot, j)[1]
        map_joint_idx_to_q_idx[j] = talos.get_joint_id(joint_name)

    q = talos.set_and_get_default_pose()
    pin.forwardKinematics(talos.model, talos.data, q)
    pin.updateFramePlacements(talos.model, talos.data)

    com_initial_pose = pin.centerOfMass(talos.model, talos.data, q)

    k = 0
    x_k = np.array([0.0, com_initial_pose[0], 0.0, 0.0], dtype=float)
    y_k = np.array([0.0, com_initial_pose[1], 0.0, 0.0], dtype=float)

    ik_sol_params = InvKinSolverParams(
        fixed_foot_frame=talos.left_foot_id,
        moving_foot_frame=talos.right_foot_id,
        torso_frame=talos.torso_id,
        model=talos.model,
        data=talos.data,
        w_torso=10.0,
        w_com=10.0,
        w_mf=10.0,
        w_ff=1000.0,
        mu=1e-5,
        dt=dt,
    )

    oMf_rf0 = talos.data.oMf[talos.right_foot_id].copy()
    oMf_lf0 = talos.data.oMf[talos.left_foot_id].copy()
    oMf_lf_tgt, oMf_rf_tgt = snap_feet_to_plane(oMf_lf0, oMf_rf0, z_plane=-0.08, keep_yaw=True)

    oMb_init = compute_base_from_foot_target(
        talos.model, talos.data, q, talos.left_foot_id, oMf_lf_tgt
    )
    q = q_from_base_and_joints(q, oMb_init)

    # set initial CoM target centered between feet at height zc
    feet_mid = 0.5 * (oMf_lf_tgt.translation + oMf_rf_tgt.translation)
    com_target = np.array([feet_mid[0], feet_mid[1], ctrler_params.zc])

    # one IK solve to match: left foot fixed = oMf_lf_tgt, right foot = oMf_rf_tgt, CoM = com_target
    ik_sol_params.fixed_foot_frame = talos.left_foot_id
    ik_sol_params.moving_foot_frame = talos.right_foot_id
    q_des, dq = solve_inverse_kinematics(q, com_target, oMf_lf_tgt, oMf_rf_tgt, ik_sol_params)
    q = q_des

    pin.forwardKinematics(talos.model, talos.data, q)
    pin.updateFramePlacements(talos.model, talos.data)

    # hard reset PyBullet once, then enable controllers
    reset_pybullet_from_q(robot, q, map_joint_idx_to_q_idx)
    pb.setGravity(0, 0, 0.0)
    for _ in range(5):
        pb.setGravity(0, 0, 0.0)
        pb.stepSimulation()  # settle contacts before enabling motors

    pb.setGravity(0, 0, -9.81)
    while True:
        apply_position(q, map_joint_idx_to_q_idx)

        # q_pybullet = get_q_from_pybullet(robot, talos.model, map_joint_idx_to_q_idx)
        #
        # q[7:] = q_pybullet[7:]
        #
        # # Get zmp ref horizon
        # zmp_ref_horizon = np.zeros((ctrler_params.n_preview_steps - 1, 2))
        #
        # _, x_k, y_k = update_control(
        #     ctrler_mat, np.array([0.0, 0.0]), zmp_ref_horizon, x_k.copy(), y_k.copy()
        # )
        #
        # # The CoM target is meant to follow the computed x and y and stay at constant height zc from the feet
        # com_target = np.array([x_k[1], y_k[1], oMf_rf0.translation[2] + ctrler_params.zc])
        #
        # # Stabilize at the position
        # ik_sol_params.fixed_foot_frame = talos.left_foot_id
        # ik_sol_params.moving_foot_frame = talos.right_foot_id
        #
        # q_des, dq = solve_inverse_kinematics(q, com_target, oMf_lf_tgt, oMf_rf_tgt, ik_sol_params)
        # q = q_des
        #
        # apply_position(q_des=q, j_to_q_idx=map_joint_idx_to_q_idx)
        pb.stepSimulation()
        k += 1
