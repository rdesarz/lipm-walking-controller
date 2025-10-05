from time import sleep, clock_gettime
import meshcat
import numpy as np
from shapely import Polygon
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.transformations as tf

from lipm_walking_controller.controller import (
    compute_preview_control_matrices,
    compute_zmp_ref,
    update_control,
    PreviewControllerParams,
)

from lipm_walking_controller.foot import compute_feet_path_and_poses, get_active_polygon
from lipm_walking_controller.inverse_kinematic import qp_inverse_kinematics, QPParams
from lipm_walking_controller.model import Talos

if __name__ == "__main__":
    # General parameters
    dt = 0.02  # Delta of time of the model simulation

    # Preview controller parameters
    t_preview = 1.6  # Time horizon used for the preview controller

    # ZMP reference parameters
    t_ss = 1.0  # Single support phase time window
    t_ds = 0.2  # Double support phase time window

    foot_shape = Polygon(((0.11, 0.05), (0.11, -0.05), (-0.11, -0.05), (-0.11, 0.05)))
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

    # Initialize the model position
    talos = Talos(path_to_model="~/projects")
    q = talos.set_and_get_default_pose()

    # Initialize visualizer
    viz = MeshcatVisualizer(talos.model, talos.geom, talos.vis)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    # viz.setCameraPosition(np.array([1.0, 1.0]))

    oMf_rf0 = talos.data.oMf[talos.right_foot_id].copy()
    oMf_lf0 = talos.data.oMf[talos.left_foot_id].copy()

    lf_initial_pose = oMf_lf0.translation
    rf_initial_pose = oMf_rf0.translation
    com_initial_pose = pin.centerOfMass(talos.model, talos.data, q)

    # Build ZMP reference to track
    t, lf_path, rf_path, steps_pose, phases = compute_feet_path_and_poses(
        rf_initial_pose,
        lf_initial_pose,
        n_steps,
        t_ss,
        t_ds,
        l_stride,
        dt,
        max_height_foot,
    )

    zmp_ref = compute_zmp_ref(t, com_initial_pose[0:2], steps_pose, t_ss, t_ds)

    T = len(t)
    u = np.zeros((T, 2))

    zmp_padded = np.vstack(
        [zmp_ref, np.repeat(zmp_ref[-1][None, :], ctrler_params.n_preview_steps, axis=0)]
    )

    x0 = np.array([0.0, com_initial_pose[0], 0.0, 0.0], dtype=float)
    y0 = np.array([0.0, com_initial_pose[1], 0.0, 0.0], dtype=float)
    x = np.zeros((len(zmp_ref) + 1, 4), dtype=float)
    y = np.zeros((len(zmp_ref) + 1, 4), dtype=float)
    x[0] = x0
    y[0] = y0

    sleep(0.5)

    # Simulate
    ctrler_params = QPParams(
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

    for k in range(T):
        start = clock_gettime(0)

        # Get zmp ref horizon
        zmp_ref_horizon = zmp_padded[k + 1 : k + ctrler_params.n_preview_steps]

        u[k], x[k + 1], y[k + 1] = update_control(
            ctrler_mat, zmp_ref[k], zmp_ref_horizon, x[k], y[k]
        )

        com_target = np.array([x[k, 1], y[k, 1], lf_initial_pose[2] + ctrler_params.zc])

        # Alternate between feet
        if phases[k] < 0.0:
            ctrler_params.fixed_foot_frame = talos.right_foot_id
            ctrler_params.moving_foot_frame = talos.left_foot_id

            oMf_lf = pin.SE3(oMf_lf0.rotation, lf_path[k])
            q_new, dq = qp_inverse_kinematics(q, com_target, oMf_lf, ctrler_params)
            q = q_new
        else:
            ctrler_params.fixed_foot_frame = talos.left_foot_id
            ctrler_params.moving_foot_frame = talos.right_foot_id

            oMf_rf = pin.SE3(oMf_rf0.rotation, rf_path[k])
            q_new, dq = qp_inverse_kinematics(q, com_target, oMf_rf, ctrler_params)
            q = q_new

        pin.forwardKinematics(talos.model, talos.data, q)
        pin.updateFramePlacements(talos.model, talos.data)

        # Display the path of the CoM in the viewer
        com = pin.centerOfMass(talos.model, talos.data, q)
        n = viz.viewer[f"world/com_traj/pt_{k:05d}"]
        n.set_object(
            meshcat.geometry.Sphere(0.01),
            meshcat.geometry.MeshLambertMaterial(color=0xFF0000),
        )
        n.set_transform(tf.translation_matrix(com))

        # Update the model visualization
        if viz:
            viz.display(q)

        # Compute the remaining time to render in real time the visualization
        stop = clock_gettime(0)
        elapsed_dt = stop - start
        remaining_dt = dt - elapsed_dt
        sleep(max(0.0, remaining_dt))
