import math
import argparse
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as pb
import pinocchio as pin
from matplotlib import pyplot as plt

from biped_walking_controller.foot import (
    compute_feet_trajectories,
    BezierCurveFootPathGenerator,
    compute_steps_sequence,
)

from biped_walking_controller.inverse_kinematic import InvKinSolverParams, solve_inverse_kinematics
from biped_walking_controller.plot import plot_feet_and_com, plot_contact_forces

from biped_walking_controller.preview_control import (
    PreviewControllerParams,
    compute_preview_control_matrices,
    update_control,
    compute_zmp_ref,
    cubic_spline_interpolation,
)

from biped_walking_controller.model import Talos, q_from_base_and_joints

from biped_walking_controller.simulation import (
    _snap_feet_to_plane,
    _compute_base_from_foot_target,
    Simulator,
)


@dataclass
class GeneralParams:
    dt = 1.0 / 240.0
    t_ss = 0.8  # Single support phase time window
    t_ds = 0.3  # Double support phase time window
    t_init = 2.0  # Initialization phase (transition from still position to first step)
    t_end = 0.4
    n_steps = 15  # Number of steps executed by the robot
    l_stride = 0.1  # Length of the stride
    max_height_foot = 0.01  # Maximal height of the swing foot
    t_preview = 1.6
    n_solver_iter = 200


def get_standard_params() -> typing.Tuple[GeneralParams, PreviewControllerParams]:
    general_params = GeneralParams()

    ctrler_params = PreviewControllerParams(
        zc=0.89,
        g=9.81,
        Qe=1.0,
        Qx=np.zeros((3, 3)),
        R=1e-6,
        n_preview_steps=int(round(general_params.t_preview / general_params.dt)),
    )

    return general_params, ctrler_params


def get_accurate_sim_params() -> typing.Tuple[GeneralParams, PreviewControllerParams]:
    general_params, ctrler_params = get_standard_params()

    # Specific params
    general_params.dt = 1.0 / 1000.0
    general_params.t_ss = 0.6
    general_params.t_ds = 0.1
    general_params.n_steps = 15
    general_params.l_stride = 0.15
    general_params.max_height_foot = 0.02
    general_params.n_solver_iter = 1500

    return general_params, ctrler_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path-talos-data", type=Path, help="Path to talos_data root")
    p.add_argument("--plot-results", action="store_true")
    p.add_argument("--launch-gui", action="store_true")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--accurate-sim", action="store_true")
    args = p.parse_args()

    np.set_printoptions(suppress=True, precision=3)

    if args.accurate_sim:
        scen_params, ctrler_params = get_accurate_sim_params()
    else:
        scen_params, ctrler_params = get_standard_params()

    # Initialize Talos pinocchio model
    talos = Talos(path_to_model=args.path_talos_data.expanduser(), reduced=False)
    q_init = talos.set_and_get_default_pose()

    # Initialize simulator
    simulator = Simulator(
        dt=scen_params.dt,
        path_to_robot_urdf=args.path_talos_data.expanduser()
        / "talos_data"
        / "urdf"
        / "talos_full.urdf",
        model=talos,
        launch_gui=args.launch_gui,
        n_solver_iter=scen_params.n_solver_iter,
    )

    # Compute the right and left foot position as well as the base initial position
    oMf_rf0 = talos.data.oMf[talos.right_foot_id].copy()
    oMf_lf0 = talos.data.oMf[talos.left_foot_id].copy()
    oMf_lf_tgt, oMf_rf_tgt = _snap_feet_to_plane(oMf_lf0, oMf_rf0)

    oMf_torso = talos.data.oMf[talos.torso_id].copy()
    oMb_init = _compute_base_from_foot_target(
        talos.model, talos.data, q_init, talos.left_foot_id, oMf_lf_tgt
    )

    # Compute the required initial configuration and apply them to the kinematic model
    q_init = q_from_base_and_joints(q_init, oMb_init)
    pin.forwardKinematics(talos.model, talos.data, q_init)
    pin.updateFramePlacements(talos.model, talos.data)

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
        dt=scen_params.dt,
        locked_joints=list(talos.get_locked_joints_idx()),
    )
    q_des, dq = solve_inverse_kinematics(
        q_init, com_initial_target, oMf_lf_tgt, oMf_rf_tgt, oMf_torso, ik_sol_params
    )
    q_init = q_des

    # Update kinematic model and simulator
    pin.forwardKinematics(talos.model, talos.data, q_init)
    pin.updateFramePlacements(talos.model, talos.data)
    simulator.reset_robot_configuration(q_init)

    # First we hard reset the position of the robot and let the simulation run for a few iterations with 0 gravity to
    # stabilize the robot
    pb.setGravity(0, 0, 0)
    for _ in range(5):
        simulator.step()  # settle contacts before enabling motors

    # Then we start a stabilization phase where the robot has to maintain its CoM at a fixed position
    pb.setGravity(0, 0, -9.81)

    # Setup the controller matrices
    ctrler_mat = compute_preview_control_matrices(ctrler_params, scen_params.dt)

    x_k = np.array([0.0, com_initial_target[0], 0.0, 0.0], dtype=float)
    y_k = np.array([0.0, com_initial_target[1], 0.0, 0.0], dtype=float)

    for _ in range(math.ceil(1.0 / scen_params.dt)):
        # Get the current configuration of the robot from the simulator
        q_init = simulator.get_q(talos.model.nq)

        # Apply the configuration to the kinematic model
        pin.forwardKinematics(talos.model, talos.data, q_init)
        pin.updateFramePlacements(talos.model, talos.data)

        # # Get zmp ref horizon
        zmp_ref_horizon = np.ones((ctrler_params.n_preview_steps - 1, 2)) * feet_mid[0:2]

        _, x_k, y_k = update_control(
            ctrler_mat, feet_mid[0:2], zmp_ref_horizon, x_k.copy(), y_k.copy()
        )

        # The CoM target is meant to follow the computed x and y and stay at constant height zc from the feet
        com_target = np.array([x_k[1], y_k[1], ctrler_params.zc])

        # Stabilize at the position
        q_des, dq = solve_inverse_kinematics(
            q_init, com_target, oMf_lf_tgt, oMf_rf_tgt, oMf_torso, ik_sol_params
        )

        # Uncomment to follow the center of mass of the robot
        simulator.update_camera_to_follow_pos(x_k[1], 0.0, 0.0)

        # This step is only here to start with the right initial position. Therefore we perform a reset
        simulator.reset_robot_configuration(q_des)
        simulator.step()

    lf_initial_pose = oMf_lf_tgt.translation
    rf_initial_pose = oMf_rf_tgt.translation

    # Build ZMP reference to track
    steps_pose, steps_ids = compute_steps_sequence(
        rf_initial_pose=rf_initial_pose,
        lf_initial_pose=lf_initial_pose,
        n_steps=scen_params.n_steps,
        l_stride=scen_params.l_stride,
    )

    t, lf_path, rf_path, phases = compute_feet_trajectories(
        rf_initial_pose=rf_initial_pose,
        lf_initial_pose=lf_initial_pose,
        n_steps=scen_params.n_steps,
        steps_pose=steps_pose,
        t_ss=scen_params.t_ss,
        t_ds=scen_params.t_ds,
        t_init=scen_params.t_init,
        t_final=scen_params.t_end,
        dt=scen_params.dt,
        traj_generator=BezierCurveFootPathGenerator(scen_params.max_height_foot),
    )

    zmp_ref = compute_zmp_ref(
        t=t,
        com_initial_pose=com_initial_target[0:2],
        steps=steps_pose[:, 0:2],
        ss_t=scen_params.t_ss,
        ds_t=scen_params.t_ds,
        t_init=scen_params.t_init,
        t_final=scen_params.t_end,
        interp_fn=cubic_spline_interpolation,
    )

    zmp_padded = np.vstack(
        [zmp_ref, np.repeat(zmp_ref[-1][None, :], ctrler_params.n_preview_steps, axis=0)]
    )

    com_pin_pos = np.zeros((len(phases), 3))
    com_ref_pos = np.zeros((len(phases), 3))
    com_pb_pos = np.zeros((len(phases), 3))

    lf_ref_pos = np.zeros((len(phases), 3))
    lf_pin_pos = np.zeros((len(phases), 3))
    lf_pb_pos = np.zeros((len(phases), 3))

    rf_ref_pos = np.zeros((len(phases), 3))
    rf_pin_pos = np.zeros((len(phases), 3))
    rf_pb_pos = np.zeros((len(phases), 3))

    zmp_pos = np.zeros((len(phases), 3))

    rf_forces = np.zeros((len(phases), 1))
    lf_forces = np.zeros((len(phases), 1))

    if args.record_video:
        simulator.start_video_record()

    # We start the walking phase
    for k, _ in enumerate(phases[:-2]):
        # Get the current configuration of the robot from the simulator
        q_init = simulator.get_q(talos.model.nq)

        # Apply the configuration to the kinematic model
        pin.forwardKinematics(talos.model, talos.data, q_init)
        pin.updateFramePlacements(talos.model, talos.data)

        zmp_ref_horizon = zmp_padded[k + 1 : k + ctrler_params.n_preview_steps]

        _, x_k, y_k = update_control(
            ctrler_mat, zmp_padded[k], zmp_ref_horizon, x_k.copy(), y_k.copy()
        )

        zmp_pos[k] = simulator.get_zmp_pose()

        # The CoM target is meant to follow the computed x and y and stay at constant height zc from the feet
        com_target = np.array([x_k[1], y_k[1], ctrler_params.zc])

        # Alternate between feet
        if phases[k] < 0.0:
            ik_sol_params.fixed_foot_frame = talos.right_foot_id
            ik_sol_params.moving_foot_frame = talos.left_foot_id

            oMf_lf = pin.SE3(oMf_lf_tgt.rotation, lf_path[k])
            q_des, dq = solve_inverse_kinematics(
                q_init,
                com_target,
                oMf_fixed_foot=oMf_rf_tgt,
                oMf_moving_foot=oMf_lf,
                oMf_torso=oMf_torso,
                params=ik_sol_params,
            )

            oMf_lf_tgt = pin.SE3(oMf_lf_tgt.rotation, lf_path[k + 1])

        else:
            ik_sol_params.fixed_foot_frame = talos.left_foot_id
            ik_sol_params.moving_foot_frame = talos.right_foot_id

            oMf_rf = pin.SE3(oMf_rf_tgt.rotation, rf_path[k])
            q_des, dq = solve_inverse_kinematics(
                q_init,
                com_target,
                oMf_fixed_foot=oMf_lf_tgt,
                oMf_moving_foot=oMf_rf,
                oMf_torso=oMf_torso,
                params=ik_sol_params,
            )

            oMf_rf_tgt = pin.SE3(oMf_rf_tgt.rotation, rf_path[k + 1])

        simulator.apply_joints_pos_to_robot(q_des)

        # Uncomment to follow the center of mass of the robot
        simulator.update_camera_to_follow_pos(x_k[1], 0.0, 0.0)

        simulator.step()

        if args.plot_results:
            pin.computeCentroidalMap(talos.model, talos.data, q_init)
            com_pin = pin.centerOfMass(talos.model, talos.data, q_init)

            # Store position of CoM, left and right feet
            com_ref_pos[k] = com_target
            com_pin_pos[k] = com_pin
            com_pb_pos[k] = simulator.get_robot_com_position()

            lf_ref_pos[k] = lf_path[k]
            lf_pin_pos[k] = talos.data.oMf[talos.left_foot_id].translation
            lf_pb_pos[k], _ = simulator.get_robot_frame_pos("leg_left_6_link")

            rf_ref_pos[k] = rf_path[k]
            rf_pin_pos[k] = talos.data.oMf[talos.right_foot_id].translation
            rf_pb_pos[k], _ = simulator.get_robot_frame_pos("leg_right_6_link")

            rf_forces[k], lf_forces[k] = simulator.get_contact_forces()

    if args.record_video:
        simulator.stop_video_record(duration=t[-1])

    if args.plot_results:

        zmp_ref_plot = np.zeros((zmp_ref.shape[0], 3))
        zmp_ref_plot[:, :2] = zmp_ref

        plot_feet_and_com(
            title_prefix="Walking controller",
            t=t,
            lf_pin_pos=lf_pin_pos,
            rf_pin_pos=rf_pin_pos,
            lf_ref_pos=lf_ref_pos,
            rf_ref_pos=rf_ref_pos,
            lf_pb_pos=lf_pb_pos,
            rf_pb_pos=rf_pb_pos,
            com_ref_pos=com_ref_pos,
            com_pb_pos=com_pb_pos,
            com_pin_pos=com_pin_pos,
            zmp_pb=zmp_pos,
            zmp_ref=zmp_ref_plot,
        )

        plot_contact_forces(t=t, force_rf=rf_forces, force_lf=lf_forces)

        plt.show()

    # Infinite loop to display the ending position
    while True:
        simulator.step()


if __name__ == "__main__":
    main()
