import argparse
from pathlib import Path
from time import sleep, clock_gettime
import numpy as np
import pinocchio as pin

from biped_walking_controller.preview_control import (
    compute_preview_control_matrices,
    compute_zmp_ref,
    update_control,
    PreviewControllerParams,
    cubic_spline_interpolation,
)

from biped_walking_controller.foot import (
    compute_feet_trajectories,
    BezierCurveFootPathGenerator,
    compute_steps_sequence,
)

from biped_walking_controller.inverse_kinematic import solve_inverse_kinematics, InvKinSolverParams
from biped_walking_controller.model import Talos
from biped_walking_controller.visualizer import Visualizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path-talos-data", type=Path, help="Path to talos_data root")
    p.add_argument(
        "-v",
        "--open-visualizer",
        action="store_true",
        help="If enabled, will open the web visualizer when the program is launched",
    )
    args = p.parse_args()

    # General parameters
    dt = 0.02  # Delta of time of the model simulation

    # Preview controller parameters
    t_preview = 1.6  # Time horizon used for the preview controller

    # ZMP reference parameters
    t_ss = 2.0  # Single support phase time window
    t_ds = 1.0  # Double support phase time window
    t_init = 2.0  # Initialization phase (transition from still position to first step)
    t_end = 2.0
    n_steps = 25
    l_stride = 0.2
    max_height_foot = 0.02

    ctrler_params = PreviewControllerParams(
        zc=0.89,
        g=9.81,
        Qe=1.0,
        Qx=np.zeros((3, 3)),
        R=1e-6,
        n_preview_steps=int(round(t_preview / dt)),
    )
    ctrler_mat = compute_preview_control_matrices(ctrler_params, dt)

    # Initialize the model position
    talos = Talos(path_to_model=args.path_talos_data.expanduser(), reduced=True)
    q = talos.set_and_get_default_pose()
    oMf_rf_fixed = talos.data.oMf[talos.right_foot_id].copy()
    oMf_lf_fixed = talos.data.oMf[talos.left_foot_id].copy()

    # Initialize visualizer
    viz = Visualizer(robot_model=talos, open_viewer=args.open_visualizer)

    lf_initial_pose = oMf_lf_fixed.translation
    rf_initial_pose = oMf_rf_fixed.translation
    com_initial_pose = pin.centerOfMass(talos.model, talos.data, q)
    oMf_torso = talos.data.oMf[talos.torso_id].copy()

    # Build ZMP reference to track
    steps_pose, steps_ids = compute_steps_sequence(
        rf_initial_pose=rf_initial_pose,
        lf_initial_pose=lf_initial_pose,
        n_steps=n_steps,
        l_stride=l_stride,
    )

    t, lf_path, rf_path, phases = compute_feet_trajectories(
        rf_initial_pose=rf_initial_pose,
        lf_initial_pose=lf_initial_pose,
        n_steps=n_steps,
        steps_pose=steps_pose,
        t_ss=t_ss,
        t_ds=t_ds,
        t_init=t_init,
        t_final=t_end,
        dt=dt,
        traj_generator=BezierCurveFootPathGenerator(max_height_foot),
    )

    zmp_ref = compute_zmp_ref(
        t=t,
        com_initial_pose=com_initial_pose[0:2],
        steps=steps_pose[:, 0:2],
        ss_t=t_ss,
        ds_t=t_ds,
        t_init=t_init,
        t_final=t_end,
        interp_fn=cubic_spline_interpolation,
    )

    zmp_padded = np.vstack(
        [zmp_ref, np.repeat(zmp_ref[-1][None, :], ctrler_params.n_preview_steps, axis=0)]
    )

    x_k = np.array([0.0, com_initial_pose[0], 0.0, 0.0], dtype=float)
    y_k = np.array([0.0, com_initial_pose[1], 0.0, 0.0], dtype=float)

    # Simulate
    ik_sol_params = InvKinSolverParams(
        fixed_foot_frame=talos.right_foot_id,
        moving_foot_frame=talos.left_foot_id,
        torso_frame=talos.torso_id,
        model=talos.model,
        data=talos.data,
        w_torso=10.0,
        w_com=10.0,
        w_mf=100.0,
        mu=1e-5,
        dt=dt,
    )

    for k in range(len(t)):
        start = clock_gettime(0)

        # Get zmp ref horizon
        zmp_ref_horizon = zmp_padded[k + 1 : k + ctrler_params.n_preview_steps]

        _, x_k, y_k = update_control(
            ctrler_mat, zmp_ref[k], zmp_ref_horizon, x_k.copy(), y_k.copy()
        )

        # The CoM target is meant to follow the computed x and y and stay at constant height zc from the feet
        com_target = np.array([x_k[1], y_k[1], lf_initial_pose[2] + ctrler_params.zc])

        # Alternate between feet
        if phases[k] < 0.0:
            ik_sol_params.fixed_foot_frame = talos.right_foot_id
            ik_sol_params.moving_foot_frame = talos.left_foot_id

            oMf_lf = pin.SE3(oMf_lf_fixed.rotation, lf_path[k])
            oMf_lf_fixed = oMf_lf
            q_new, dq = solve_inverse_kinematics(
                q,
                com_target,
                oMf_fixed_foot=oMf_rf_fixed,
                oMf_moving_foot=oMf_lf,
                oMf_torso=oMf_torso,
                params=ik_sol_params,
            )
            q = q_new
        else:
            ik_sol_params.fixed_foot_frame = talos.left_foot_id
            ik_sol_params.moving_foot_frame = talos.right_foot_id

            oMf_rf = pin.SE3(oMf_rf_fixed.rotation, rf_path[k])
            oMf_rf_fixed = oMf_rf
            q_new, dq = solve_inverse_kinematics(
                q,
                com_target,
                oMf_torso=oMf_torso,
                oMf_fixed_foot=oMf_lf_fixed,
                oMf_moving_foot=oMf_rf,
                params=ik_sol_params,
            )

            q = q_new

        pin.forwardKinematics(talos.model, talos.data, q)
        pin.updateFramePlacements(talos.model, talos.data)

        # Uncomment to display the path of the CoM in the viewer
        # viz.display_point(pin.centerOfMass(talos.model, talos.data, q), k)

        # Uncomment to have the camera follow the robot. Requires master branch of meshcat.
        # viz.point_camera_at_robot(robot_model=talos, camera_offset=np.array([2.0, 1.0, 1.0]))

        # Update the model visualization
        viz.update_display(q)

        # Compute the remaining time to render in real time the visualization
        stop = clock_gettime(0)
        elapsed_dt = stop - start
        remaining_dt = dt - elapsed_dt
        sleep(max(0.0, remaining_dt))


if __name__ == "__main__":
    main()
