from biped_walking_controller.foot import (
    compute_double_support_polygon,
    compute_single_support_polygon,
)


def plot_steps(axes, steps_pose, step_shape):
    # Plot double support polygon
    for current_step, next_step in zip(steps_pose[:-1], steps_pose[1:]):
        support_polygon = compute_double_support_polygon(current_step, next_step, step_shape)

        x, y = support_polygon.exterior.xy
        axes.plot(x, y, color="blue")  # outline
        axes.fill(x, y, color="lightblue", alpha=0.5)  # filled polygon

    # Plot single support polygon
    for current_step in steps_pose:
        support_polygon = compute_single_support_polygon(current_step, step_shape)

        x, y = support_polygon.exterior.xy
        axes.plot(x, y, color="red")  # outline
        axes.fill(x, y, color="red", alpha=0.5)  # filled polygon


import numpy as np
import matplotlib.pyplot as plt


def _trim_to_min_len(arrs):
    n = min(a.shape[0] for a in arrs)
    return [a[:n] for a in arrs]


def _finite_mask(*arrs):
    mask = np.isfinite(arrs[0]).all(axis=1)
    for a in arrs[1:]:
        mask &= np.isfinite(a).all(axis=1)
    return mask


def plot_feet_and_com(
    t,
    lf_pin_pos,
    rf_pin_pos,
    lf_ref_pos,
    rf_ref_pos,
    lf_pb_pos,
    rf_pb_pos,
    com_ref_pos,
    com_pb_pos,
    com_pin_pos,
    zmp_pb,
    zmp_ref,
    title_prefix="Feet and CoM",
):
    # Trim to common length
    (
        t,
        lf_pin_pos,
        rf_pin_pos,
        lf_ref_pos,
        rf_ref_pos,
        lf_pb_pos,
        rf_pb_pos,
        com_ref_pos,
        com_pb_pos,
        com_pin_pos,
        zmp_pb,
        zmp_ref,
    ) = _trim_to_min_len(
        [
            t,
            lf_pin_pos,
            rf_pin_pos,
            lf_ref_pos,
            rf_ref_pos,
            lf_pb_pos,
            rf_pb_pos,
            com_ref_pos,
            com_pb_pos,
            com_pin_pos,
            zmp_pb,
            zmp_ref,
        ]
    )

    # Drop NaNs/Infs consistently
    mask = np.isfinite(t) & _finite_mask(
        lf_pin_pos,
        rf_pin_pos,
        lf_ref_pos,
        rf_ref_pos,
        lf_pb_pos,
        rf_pb_pos,
        com_ref_pos,
        com_pb_pos,
        com_pin_pos,
        zmp_pb,
        zmp_ref,
    )
    t = t[mask]
    lf_pin_pos = lf_pin_pos[mask]
    rf_pin_pos = rf_pin_pos[mask]
    lf_ref_pos = lf_ref_pos[mask]
    rf_ref_pos = rf_ref_pos[mask]
    lf_pb_pos = lf_pb_pos[mask]
    rf_pb_pos = rf_pb_pos[mask]
    com_ref_pos = com_ref_pos[mask]
    com_pb_pos = com_pb_pos[mask]
    com_pin_pos = com_pin_pos[mask]
    zmp_pb = zmp_pb[mask]
    zmp_ref = zmp_ref[mask]

    # -------- Time plots (x,z,y in meters) --------
    fig, axes = plt.subplots(2, sharex=True, layout="constrained", figsize=(8, 8))

    series = [
        ("LF ref", lf_ref_pos),
        ("LF pb", lf_pb_pos),
        ("RF ref", rf_ref_pos),
        ("RF pb", rf_pb_pos),
        ("ZMP ref", zmp_ref),
        ("ZMP pb", zmp_pb),
    ]
    coord_labels = ["x [m]", "y [m]"]
    coord_idx = [0, 1]  # match your original order

    linestyles = {
        "pin": "-",
        "ref": "--",
        "pb": "-",
    }

    for ax, j in zip(axes, coord_idx):
        for name, arr in series:
            # Pick linestyle by source keyword in label
            key = "ref" if "ref" in name.lower() else ("pb" if "pb" in name.lower() else "pin")
            ax.plot(t, arr[:, j], linestyle=linestyles[key], label=name)
        ax.set_ylabel(coord_labels[coord_idx.index(j)])
        ax.grid(True)
        ax.legend(loc="center left")

    axes[0].set_ylim(0.0, 0.6)
    axes[1].set_ylim(-0.12, 0.12)

    axes[-1].set_xlabel("t [s]")

    # One combined legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(f"ZMP and feet trajectories (with improvements)")

    # -------- Plan view (x vs y) --------
    plt.rcParams.update({"font.size": 14})

    fig2, ax2 = plt.subplots(1, figsize=(12, 8), layout="constrained")
    fig2.canvas.manager.set_window_title(f"{title_prefix} — plan view (x–y)")

    def traj2d(arr):  # drop last 2 samples as in your code
        if arr.shape[0] > 2:
            return arr[:-2, 0], arr[:-2, 1]
        return arr[:, 0], arr[:, 1]

    for name, arr in [
        ("Left foot position (Pinocchio)", lf_pin_pos),
        ("Left foot position (reference)", lf_ref_pos),
        ("Left foot position (PyBullet)", lf_pb_pos),
        ("Right foot position (Pinocchio)", rf_pin_pos),
        ("Right foot position (reference)", rf_ref_pos),
        ("Right foot position (PyBullet)", rf_pb_pos),
        ("CoM position (Pinocchio)", com_pin_pos),
        ("CoM position (reference)", com_ref_pos),
        ("CoM position (PyBullet)", com_pb_pos),
        ("ZMP position (PyBullet)", zmp_pb),
        ("ZMP reference (PyBullet)", zmp_ref),
    ]:
        x, y = traj2d(arr)
        key = "ref" if "ref" in name.lower() else ("pb" if "pb" in name.lower() else "pin")
        ax2.plot(x, y, linestyle=linestyles[key], label=name)

    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.grid(True)
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax2.set_title(f"{title_prefix} — plan view (x–y)")


def plot_contact_forces(t, force_rf, force_lf, title="Contact force Fx"):
    t = np.asarray(t).ravel()
    force_rf = np.asarray(force_rf).ravel()
    force_lf = np.asarray(force_lf).ravel()
    assert t.size == force_rf.size == force_lf.size

    plt.figure(figsize=(12, 8), layout="constrained")
    plt.plot(t, force_rf, label="right foot")
    plt.plot(t, force_lf, label="left foot")
    plt.xlabel("t [s]")
    plt.ylabel("Normal force [N]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
