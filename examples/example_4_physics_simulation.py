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


def apply_position(q_des, j_to_q_idx):
    for j_id, q_id in j_to_q_idx.items():
        pb.setJointMotorControl2(
            robot,
            j_id,
            pb.POSITION_CONTROL,
            targetPosition=q_des[q_id],
            positionGain=0.4,
            velocityGain=1.0,
            force=200,
        )


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
        zc=0.89,
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
        URDF, [0, 0, 1.1], [0, 0, 0, 1], useFixedBase=False, flags=pb.URDF_MERGE_FIXED_LINKS
    )

    talos = Talos(path_to_model="~/projects", reduced=False)

    map_joint_idx_to_q_idx = {}
    for j in range(pb.getNumJoints(robot)):
        joint_name = pb.getJointInfo(robot, j)[1]
        map_joint_idx_to_q_idx[j] = talos.get_joint_id(joint_name)

    q = talos.set_and_get_default_pose()

    k = 0
    x_k = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    y_k = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    ik_sol_params = InvKinSolverParams(
        fixed_foot_frame=talos.right_foot_id,
        moving_foot_frame=talos.left_foot_id,
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

    while True:
        # Get zmp ref horizon
        zmp_ref_horizon = np.zeros((ctrler_params.n_preview_steps - 1, 2))

        _, x_k, y_k = update_control(
            ctrler_mat, np.array([0.0, 0.0]), zmp_ref_horizon, x_k.copy(), y_k.copy()
        )

        # The CoM target is meant to follow the computed x and y and stay at constant height zc from the feet
        com_target = np.array([x_k[1], y_k[1], ctrler_params.zc])

        # Stabilize at the position
        ik_sol_params.fixed_foot_frame = talos.left_foot_id
        ik_sol_params.moving_foot_frame = talos.right_foot_id

        q_new, dq = solve_inverse_kinematics(q, com_target, oMf_rf0, ik_sol_params)

        apply_position(q_des=q, j_to_q_idx=map_joint_idx_to_q_idx)
        pb.stepSimulation()
        k += 1
