from lipm_walking_controller.foot import (
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
    zmp_pos,
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
        zmp_pos,
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
            zmp_pos,
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
        zmp_pos,
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
    zmp_pos = zmp_pos[mask]

    # -------- Time plots (x,z,y in meters) --------
    fig, axes = plt.subplots(3, sharex=True, layout="constrained", figsize=(12, 8))

    series = [
        ("LF pin", lf_pin_pos),
        ("LF ref", lf_ref_pos),
        ("LF pb", lf_pb_pos),
        ("RF pin", rf_pin_pos),
        ("RF ref", rf_ref_pos),
        ("RF pb", rf_pb_pos),
        ("CoM pin", com_pin_pos),
        ("CoM pb", com_pb_pos),
        ("CoM ref", com_ref_pos),
        ("ZMP pos", zmp_pos),
    ]
    coord_labels = ["x [m]", "z [m]", "y [m]"]
    coord_idx = [0, 2, 1]  # match your original order

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

    axes[-1].set_xlabel("t [s]")

    # One combined legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=5, frameon=False)
    fig.suptitle(f"{title_prefix} — time profiles")

    # -------- Plan view (x vs y) --------
    fig2, ax2 = plt.subplots(1, figsize=(10, 8))

    def traj2d(arr):  # drop last 2 samples as in your code
        if arr.shape[0] > 2:
            return arr[:-2, 0], arr[:-2, 1]
        return arr[:, 0], arr[:, 1]

    for name, arr in [
        ("LF pos (Pinocchio)", lf_pin_pos),
        ("LF pos (reference)", lf_ref_pos),
        ("LF pos (PyBullet)", lf_pb_pos),
        ("RF pos (Pinocchio)", rf_pin_pos),
        ("RF pos (reference)", rf_ref_pos),
        ("RF pos (PyBullet)", rf_pb_pos),
        ("CoM pos (Pinocchio)", com_pin_pos),
        ("CoM pos (reference)", com_ref_pos),
        ("CoM pos (PyBullet)", com_pb_pos),
        ("ZMP pos (PyBullet)", zmp_pos),
    ]:
        x, y = traj2d(arr)
        key = "ref" if "ref" in name.lower() else ("pb" if "pb" in name.lower() else "pin")
        ax2.plot(x, y, linestyle=linestyles[key], label=name)

    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.grid(True)
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig2.tight_layout()
    ax2.set_title(f"{title_prefix} — plan view (x–y)")

    plt.show()
