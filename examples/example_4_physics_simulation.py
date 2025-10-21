import math
import argparse
from pathlib import Path

import numpy as np
import pybullet as pb
import pinocchio as pin

from lipm_walking_controller.foot import compute_feet_path_and_poses

from lipm_walking_controller.inverse_kinematic import InvKinSolverParams, solve_inverse_kinematics

from lipm_walking_controller.preview_control import (
    PreviewControllerParams,
    compute_preview_control_matrices,
    update_control,
    compute_zmp_ref,
)
from lipm_walking_controller.model import Talos, q_from_base_and_joints

from lipm_walking_controller.simulation import (
    snap_feet_to_plane,
    compute_base_from_foot_target,
    Simulator,
)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path-talos-data", type=Path, help="Path to talos_data root")
    args = p.parse_args()

    dt = 1.0 / 250.0

    # Preview controller parameters
    t_preview = 1.6  # Time horizon used for the preview controller

    # ZMP reference parameters
    t_ss = 0.4  # Single support phase time window
    t_ds = 0.4  # Double support phase time window
    t_init = 2.0  # Initialization phase (transition from still position to first step)
    t_end = 1.0
    n_steps = 15  # Number of steps executed by the robot
    l_stride = 0.25  # Length of the stride
    max_height_foot = 0.02  # Maximal height of the swing foot

    ctrler_params = PreviewControllerParams(
        zc=0.8,
        g=9.81,
        Qe=1.0,
        Qx=np.zeros((3, 3)),
        R=1e-6,
        n_preview_steps=int(round(t_preview / dt)),
    )
    ctrler_mat = compute_preview_control_matrices(ctrler_params, dt)

    # Load pinocchio model of Talos
    talos = Talos(path_to_model=args.path_talos_data.expanduser(), reduced=False)
    q = talos.set_and_get_default_pose()

    # Initialize simulation
    simulator = Simulator(dt, args.path_talos_data.expanduser(), talos)

    # Compute the right and left foot position as well as the base initial position
    oMf_rf0 = talos.data.oMf[talos.right_foot_id].copy()
    oMf_lf0 = talos.data.oMf[talos.left_foot_id].copy()
    oMf_lf_tgt, oMf_rf_tgt = snap_feet_to_plane(oMf_lf0, oMf_rf0, z_offset=-0.075, keep_yaw=True)

    oMf_torso = talos.data.oMf[talos.torso_id].copy()
    oMb_init = compute_base_from_foot_target(
        talos.model, talos.data, q, talos.left_foot_id, oMf_lf_tgt
    )
    q = q_from_base_and_joints(q, oMb_init)

    # set initial CoM target centered between feet at height zc
    feet_mid = 0.5 * (oMf_lf_tgt.translation + oMf_rf_tgt.translation)
    com_initial_target = np.array([feet_mid[0], feet_mid[1], ctrler_params.zc])

    # We run a single inverse kinematic iteration to get the desired initial position of the robot
    ik_sol_params = InvKinSolverParams(
        fixed_foot_frame=talos.left_foot_id,
        moving_foot_frame=talos.right_foot_id,
        torso_frame=talos.torso_id,
        model=talos.model,
        data=talos.data,
        w_torso=10.0,
        w_com=10.0,
        w_mf=100.0,
        mu=1e-4,
        dt=dt,
        locked_joints=talos.get_locked_joints_idx(),
    )
    q_des, dq = solve_inverse_kinematics(
        q, com_initial_target, oMf_lf_tgt, oMf_rf_tgt, oMf_torso, ik_sol_params
    )
    q = q_des

    pin.forwardKinematics(talos.model, talos.data, q)
    pin.updateFramePlacements(talos.model, talos.data)

    simulator.reset_robot(q)

    # First we hard reset the position of the robot and let the simulation run for a few iterations with 0 gravity to
    # stabilize the robot
    pb.setGravity(0, 0, 0)
    for _ in range(5):
        simulator.step()  # settle contacts before enabling motors

    # Then we start a stabilization phase where the robot has to maintain its CoM at a fixed position
    pb.setGravity(0, 0, -9.81)

    x_k = np.array([0.0, com_initial_target[0], 0.0, 0.0], dtype=float)
    y_k = np.array([0.0, com_initial_target[1], 0.0, 0.0], dtype=float)

    for _ in range(math.ceil(2.0 / dt)):
        q = simulator.get_q(talos.model.nq)

        # # Get zmp ref horizon
        zmp_ref_horizon = np.ones((ctrler_params.n_preview_steps - 1, 2)) * feet_mid[0:2]

        _, x_k, y_k = update_control(
            ctrler_mat, feet_mid[0:2], zmp_ref_horizon, x_k.copy(), y_k.copy()
        )

        # The CoM target is meant to follow the computed x and y and stay at constant height zc from the feet
        com_target = np.array([x_k[1], y_k[1], ctrler_params.zc])

        # Stabilize at the position
        q_des, dq = solve_inverse_kinematics(
            q, com_target, oMf_lf_tgt, oMf_rf_tgt, oMf_torso, ik_sol_params
        )

        # Uncomment to follow the center of mass of the robot
        simulator.update_camera_to_follow_pos(x_k[1], 0.0, 0.0)

        simulator.apply_position_to_robot(q_des)
        simulator.step()
        simulator.draw_contact_forces(color=(0, 1, 0))

    lf_initial_pose = oMf_lf_tgt.translation
    rf_initial_pose = oMf_rf_tgt.translation

    t, lf_path, rf_path, steps_pose, phases = compute_feet_path_and_poses(
        rf_initial_pose,
        lf_initial_pose,
        n_steps,
        t_ss,
        t_ds,
        t_init,
        t_end,
        l_stride,
        dt,
        max_height_foot,
    )

    zmp_ref = compute_zmp_ref(t, com_initial_target[0:2], steps_pose, t_ss, t_ds, t_init, t_end)

    zmp_padded = np.vstack(
        [zmp_ref, np.repeat(zmp_ref[-1][None, :], ctrler_params.n_preview_steps, axis=0)]
    )

    # We start the walking phase
    for k, _ in enumerate(phases):
        q = simulator.get_q(talos.model.nq)

        zmp_ref_horizon = zmp_padded[k + 1 : k + ctrler_params.n_preview_steps]

        _, x_k, y_k = update_control(
            ctrler_mat, zmp_padded[k], zmp_ref_horizon, x_k.copy(), y_k.copy()
        )

        # The CoM target is meant to follow the computed x and y and stay at constant height zc from the feet
        com_target = np.array([x_k[1], y_k[1], ctrler_params.zc])

        # Alternate between feet
        if phases[k] < 0.0:
            ik_sol_params.fixed_foot_frame = talos.right_foot_id
            ik_sol_params.moving_foot_frame = talos.left_foot_id

            oMf_lf = pin.SE3(oMf_lf_tgt.rotation, lf_path[k])
            oMf_lf_tgt = oMf_lf
            q_new, dq = solve_inverse_kinematics(
                q,
                com_target,
                oMf_fixed_foot=oMf_rf_tgt,
                oMf_moving_foot=oMf_lf,
                oMf_torso=oMf_torso,
                params=ik_sol_params,
            )
            q = q_new
        else:
            ik_sol_params.fixed_foot_frame = talos.left_foot_id
            ik_sol_params.moving_foot_frame = talos.right_foot_id

            oMf_rf = pin.SE3(oMf_rf_tgt.rotation, rf_path[k])
            oMf_rf_tgt = oMf_rf
            q_new, dq = solve_inverse_kinematics(
                q,
                com_target,
                oMf_fixed_foot=oMf_lf_tgt,
                oMf_moving_foot=oMf_rf,
                oMf_torso=oMf_torso,
                params=ik_sol_params,
            )
            q = q_new

        simulator.apply_position_to_robot(q)

        # Uncomment to follow the center of mass of the robot
        simulator.update_camera_to_follow_pos(x_k[1], 0.0, 0.0)

        simulator.step()

        # Uncomment to draw contact forces
        simulator.draw_contact_forces(color=(0, 1, 0))

    # Infinite loop to display the ending position
    while True:
        simulator.step()
