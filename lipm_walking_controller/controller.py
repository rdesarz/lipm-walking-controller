from dataclasses import dataclass

import numpy as np

from scipy.linalg import solve_discrete_are


def compute_zmp_ref(t, com_initial_pose, steps, ss_t, ds_t):
    T = len(t)
    zmp_ref = np.zeros([T, 2])

    # Step on the first foot
    mask = t < ds_t
    alpha = t[mask] / ds_t
    zmp_ref[mask, :] = (1 - alpha)[:, None] * com_initial_pose + alpha[:, None] * steps[0]

    # Alternate between foot
    for idx, (current_step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
        # Compute current time range
        t_start = ds_t + idx * (ss_t + ds_t)

        # Add single support phase
        zmp_ref[(t >= t_start) & (t < t_start + ss_t)] = current_step

        # Add double support phase
        mask = (t >= t_start + ss_t) & (t < t_start + ss_t + ds_t)
        alpha = (t[mask] - (t_start + ss_t)) / ds_t
        zmp_ref[mask, :] = (1 - alpha)[:, None] * current_step + alpha[:, None] * next_step

    # Last phase is single support at last foot pose
    mask = t >= ds_t + (len(steps) - 1) * (ss_t + ds_t)
    zmp_ref[mask, :] = steps[-1]

    return zmp_ref


@dataclass
class PreviewControllerMatrices:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Gi: np.ndarray
    Gx: np.ndarray
    Gd: np.ndarray


@dataclass
class PreviewControllerParams:
    zc: float  # Height of the COM
    g: float # Gravity
    Qe: np.ndarray # Cost on the integral error of the ZMP reference
    Qx: np.ndarray # Cost on the state vector variation.
    R: np.ndarray # Cost on the input command u(t) magnitude
    n_preview_steps: int # Number of steps to preview on the ZMP reference

def compute_preview_control_matrices(params: PreviewControllerParams, dt: float):
    # Discrete cart-table model with jerk input
    A = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], dtype=float)
    B = np.array([[dt**3 / 6.0], [dt**2 / 2.0], [dt]], dtype=float)
    C = np.array([[1.0, 0.0, -params.zc / params.g]], dtype=float)

    A1 = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
    B1 = np.vstack((C @ B, B))
    I1 = np.vstack((np.array([1]), np.zeros((3, 1))))
    F = np.vstack((C @ A, A))

    Q = np.block([[params.Qe, np.zeros((1, 3))], [np.zeros((3, 1)), params.Qx]])

    # Compute K by solving Ricatti equation
    R = params.R
    K = solve_discrete_are(A1, B1, Q, R)

    # Compute Gi and Gx
    Gi = float(np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ I1)
    Gx = np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ F

    # Compute Gd
    Ac = A1 - B1 @ np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ A1
    X1 = Ac.T @ K @ I1
    X = X1
    Gd = np.zeros((params.n_preview_steps - 1))
    Gd[0] = -Gi
    for l in range(params.n_preview_steps - 2):
        Gd[l + 1] = np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ X
        X = Ac.T @ X

    return PreviewControllerMatrices(A=A, B=B, C=C, Gi=Gi, Gx=Gx, Gd=Gd)


def update_control(ctrl_mat: PreviewControllerMatrices, current_zmp, zmp_ref, x, y):
    u = np.zeros(2)
    u[0] = -ctrl_mat.Gi * x[0] - ctrl_mat.Gx @ x[1:] + ctrl_mat.Gd.T @ zmp_ref[:, 0]
    u[1] = -ctrl_mat.Gi * y[0] - ctrl_mat.Gx @ y[1:] + ctrl_mat.Gd.T @ zmp_ref[:, 1]

    # Compute integrated error
    x_next = np.zeros(len(ctrl_mat.A) + 1)
    y_next = np.zeros(len(ctrl_mat.A) + 1)
    x_next[0] = x[0] + (ctrl_mat.C @ x[1:] - current_zmp[0])
    y_next[0] = y[0] + (ctrl_mat.C @ y[1:] - current_zmp[1])

    x_next[1:] = ctrl_mat.A @ x[1:] + ctrl_mat.B.ravel() * u[0]
    y_next[1:] = ctrl_mat.A @ y[1:] + ctrl_mat.B.ravel() * u[1]

    return u, x_next, y_next
