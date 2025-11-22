import typing
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from biped_walking_controller.state_machine import WalkingState, Foot


def linear_interpolation(t: np.ndarray, pos_begin: np.ndarray, pos_end: np.ndarray) -> np.ndarray:
    return (1 - t)[:, None] * pos_begin + t[:, None] * pos_end


def cubic_spline_interpolation(
    t: np.ndarray, pos_begin: np.ndarray, pos_end: np.ndarray
) -> np.ndarray:
    if isinstance(t, float):
        return (
            pos_begin
            + 3.0 * (pos_end - pos_begin) * np.square(t)
            - 2.0 * (pos_end - pos_begin) * np.pow(t, 3)
        )

    return (
        pos_begin
        + 3.0 * (pos_end - pos_begin) * np.square(t)[:, None]
        - 2.0 * (pos_end - pos_begin) * np.pow(t, 3)[:, None]
    )


def compute_zmp_ref(
    t, com_initial_pose, steps, ss_t, ds_t, t_init, t_final, interp_fn=cubic_spline_interpolation
):
    """
    Build a piecewise ZMP reference on the ground plane from footsteps. The ZMP reference starts at com_initial_pose.
    Then during t_init period of time, the ZMP shift to the right foot. It then goes from a step to another. Then it
    goes back between the final position of the feet during t_final.

    Parameters
    ----------
    t : array-like, shape (T,)
        Time samples (seconds), monotonically increasing.
    com_initial_pose : array-like, shape (2,)
        Initial ZMP target under the CoM at startup (x, y).
    steps : array-like, shape (N, 2)
        Footstep sequence on the ground (x, y) per step.
        The last element is the final midpoint target.
    ss_t : float
        Duration of single support per step.
    ds_t : float
        Duration of double support between steps.
    t_init : float
        Ramp time from initial CoM ZMP to the first step.
    t_final : float
        Final blending time to bring ZMP to the midpoint of the last two feet.

    Returns
    -------
    zmp_ref : ndarray, shape (T, 2)
        ZMP reference trajectory on the ground plane.

    Notes
    -----
    - For each step: hold current foot during SS, then linearly blend to the next
      foot during DS.
    - Last phase: SS on the penultimate foot, then blend to the average of the
      last two foot positions during `t_final`.
    """
    T = len(t)
    zmp_ref = np.zeros([T, 2])

    # Step on the first foot
    mask = t < t_init
    alpha = t[mask] / t_init
    zmp_ref[mask, :] = interp_fn(alpha, com_initial_pose, steps[0])

    # Alternate between foot
    for idx, (current_step, next_step) in enumerate(zip(steps[:-2], steps[1:-1])):
        # Compute current time range
        t_start = t_init + idx * (ss_t + ds_t)

        # Add single support phase
        zmp_ref[(t >= t_start) & (t < t_start + ss_t)] = current_step

        # Add double support phase
        mask = (t >= t_start + ss_t) & (t < t_start + ss_t + ds_t + t[1])
        alpha = (t[mask] - (t_start + ss_t)) / ds_t
        zmp_ref[mask, :] = interp_fn(alpha, current_step, next_step)

    # Last phase: SS on last-but-one, then blend to midpoint of last two
    t_start = t_init + (len(steps) - 2) * (ss_t + ds_t)
    zmp_ref[(t >= t_start) & (t < t_start + ss_t)] = steps[-2]

    mask = (t >= t_start + ss_t) & (t < t_start + ss_t + t_final)
    alpha = (t[mask] - (t_start + ss_t)) / t_final
    zmp_ref[mask, :] = interp_fn(alpha, steps[-2], (steps[-1] + steps[-2]) / 2.0)

    return zmp_ref


@dataclass
class PreviewControllerParams:
    """
    Hyperparameters for preview control gain synthesis.

    Attributes
    ----------
    zc : float
        CoM height (meters).
    g : float
        Gravity magnitude (m/s^2).
    Qe : ndarray, shape (1, 1)
        Weight on integrated ZMP tracking error.
    Qx : ndarray, shape (3, 3)
        Weight on state [x, x_dot, x_ddot].
    R : ndarray, shape (1, 1)
        Weight on jerk input magnitude.
    n_preview_steps : int
        Number of preview items P used to build Gd (uses P-1 gains).
    """

    zc: float
    g: float
    Qe: np.ndarray
    Qx: np.ndarray
    R: np.ndarray
    n_preview_steps: int


@dataclass
class PreviewControllerMatrices:
    """
    Structure that contains the Discrete LIPM-with-jerk preview-control matrices. Serves as input for the update of the
    control command.

    Attributes
    ----------
    A : ndarray, shape (3, 3)
        State transition of [x, x_dot, x_ddot] (per-axis).
    B : ndarray, shape (3, 1)
        Input matrix for jerk u.
    C : ndarray, shape (1, 3)
        Output mapping from state to ZMP.
    Gi : float or ndarray, shape (1, 1)
        Integral gain on ZMP tracking error.
    Gx : ndarray, shape (1, 4)
        State-feedback gain on [e_int, x, x_dot, x_ddot].
    Gd : ndarray, shape (P-1,)
        Preview gains for future ZMP references over P-1 steps.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Gi: np.ndarray
    Gx: np.ndarray
    Gd: np.ndarray


def compute_preview_control_matrices(params: PreviewControllerParams, dt: float):
    """
    Construct preview-control gains for the discrete LIPM with jerk input.

    Parameters
    ----------
    params : PreviewControllerParams
        Model and cost weights.
    dt : float
        Discrete time step.

    Returns
    -------
    PreviewControllerMatrices
        Controller matrices (A, B, C, Gi, Gx, Gd) reused at runtime.

    Method
    ------
    - Build augmented system with integral of ZMP error.
    - Solve discrete-time algebraic Riccati equation for K.
    - Derive integral gain Gi, state gain Gx, and preview gains Gd for P-1 future
      reference samples using the closed-loop matrix `Ac`.

    References
    ----------
    - Kajita et al., 2003. Biped walking pattern generation by using preview control.
    - Katayama et al., 1985. Design of an optimal preview controller.
    """
    # Discrete cart-table model with jerk input
    A = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], dtype=float)
    B = np.array([[dt**3 / 6.0], [dt**2 / 2.0], [dt]], dtype=float)
    C = np.array([[1.0, 0.0, -params.zc / params.g]], dtype=float)

    A1 = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
    B1 = np.vstack((C @ B, B))
    I1 = np.vstack((np.array([1]), np.zeros((3, 1))))
    F = np.vstack((C @ A, A))

    Q = np.block([[params.Qe, np.zeros((1, 3))], [np.zeros((3, 1)), params.Qx]])

    # Compute K by solving Riccati equation
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
    """
    One-step preview control update for x and y axes.

    Parameters
    ----------
    ctrl_mat : PreviewControllerMatrices
        Precomputed controller matrices.
    current_zmp : array-like, shape (2,)
        Measured ZMP at current step [zx, zy].
    zmp_ref : ndarray, shape (P-1, 2)
        Future ZMP references used with Gd for both axes. First element is the
        next sample after the current time.
    x, y : ndarray, shape (4,)
        Augmented states per axis: [e_int, pos, vel, acc].

    Returns
    -------
    u : ndarray, shape (2,)
        Jerk command for x and y.
    x_next, y_next : ndarray, shape (4,)
        Next augmented states after applying u.

    Notes
    -----
    - Integral state is updated with ZMP output error, then state is propagated
      with (A, B) and jerk input.
    - Uses the same scalar Gi and vector Gx for both axes.
    """
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


def build_zmp_horizon(
    com_initial_target,
    t_horizon: float,
    t_state: float,
    state: WalkingState,
    delta_t: float,
    current_step_idx: int,
    steps_sequence: np.ndarray,
    steps_foot: typing.List[Foot],
    ss_t: float,
    ds_t: float,
    t_init: float,
    t_end: float,
    interp_fn=cubic_spline_interpolation,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a ZMP reference over a preview horizon based on the current walking state.

    Parameters
    ----------
    t_horizon : float
        Length of the preview horizon [s].
    t_state : float
        Time elapsed in the current state [s].
    state : WalkingState
        Current walking state (INIT, DS, SS_LEFT, SS_RIGHT, END).
    delta_t : float
        Sampling period of the preview horizon [s].
    current_step_idx : int
        Index of the current step in `steps_sequence`. Interpreted as:
        - In SS: index of the current support foot.
        - In DS: index of the *target* foot (previous is current_step_idx - 1).
    steps_sequence : np.ndarray
        Sequence of footsteps. Shape (N, 2) or (N, >=2).
        Only the (x, y) components are used as ZMP targets.
    ss_t : float
        Duration of a single-support phase [s].
    ds_t : float
        Duration of a double-support phase [s].
    t_init : float
        Duration of the INIT phase [s].
    t_end : float
        Duration of the END phase [s].
    interp_fn : callable
        Interpolation function used in double support.
        Expected signature: interp_fn(alpha, p0, p1)
        where alpha in [0, 1], p0, p1 are 2D numpy arrays.

    Returns
    -------
    t_samples : np.ndarray, shape (N,)
        Relative time samples over the horizon, starting at 0.
    zmp_horizon : np.ndarray, shape (N, 2)
        ZMP reference (x, y) at each time sample.

    Notes
    -----
    - This function does NOT enforce continuity with the *previous* ZMP reference.
      At each call, the horizon is built from scratch from the current state.
    - State-transition logic is a simple cyclic model:
      INIT -> DS -> SS -> DS -> SS -> ... -> END
      and the step index increments when leaving an SS state.
    """
    # Sanity checks and normalization
    steps = np.asarray(steps_sequence, dtype=float)
    if steps.ndim != 2 or steps.shape[0] == 0:
        raise ValueError("steps_sequence must be a non-empty (N, D) array")

    # Use only x, y
    steps_xy = steps[:, :2]
    n_steps = steps_xy.shape[0]

    step_idx = int(np.clip(current_step_idx, 0, n_steps - 1))

    def state_duration(s: WalkingState) -> float:
        if s == WalkingState.INIT:
            return t_init
        if s == WalkingState.DS:
            return ds_t
        if s in (WalkingState.SS_LEFT, WalkingState.SS_RIGHT):
            return ss_t
        if s == WalkingState.END:
            return t_end
        raise ValueError(f"Unknown state: {s}")

    def next_state_and_step(
        s: WalkingState, idx: int, steps_foot: typing.List[Foot]
    ) -> tuple[WalkingState, int]:
        """Simple progression model over footsteps.

        INIT -> DS(step 0)
        DS(k) -> SS(k)
        SS(k) -> DS(k+1) (clamped to last step)
        END stays END
        """
        if s == WalkingState.DS or s == WalkingState.INIT:
            # Land on target step
            if idx < n_steps - 1:
                return (
                    WalkingState.SS_LEFT if steps_foot[idx] is Foot.LEFT else WalkingState.SS_RIGHT
                ), idx
            else:
                return WalkingState.END, idx

        if s in (WalkingState.SS_LEFT, WalkingState.SS_RIGHT):
            # Move to next DS between current and next step
            if idx < n_steps - 1:
                return WalkingState.DS, idx + 1

        if s == WalkingState.END:
            return WalkingState.END, idx

        raise ValueError(f"Unknown state: {s}")

    def zmp_for_state(s: WalkingState, idx: int, t_in_state: float) -> np.ndarray:
        """Return ZMP position (x, y) for a given state and time in state."""
        # Clamp index
        idx = int(np.clip(idx, 0, n_steps - 1))

        if s == WalkingState.INIT:
            # Keep ZMP on the first support
            p0 = com_initial_target
            p1 = steps_xy[0]

            alpha = np.clip(t_in_state / t_init, 0.0, 1.0)
            return interp_fn(alpha, p0, p1)

        if s in (WalkingState.SS_LEFT, WalkingState.SS_RIGHT):
            # ZMP fixed at current support foot
            return steps_xy[idx]

        if s == WalkingState.DS:
            # ZMP moves between previous and current step
            if idx <= 0:
                p0 = com_initial_target
                p1 = steps_xy[0]
            else:
                p0 = steps_xy[idx - 1]
                p1 = steps_xy[idx]

            alpha = np.clip(t_in_state / max(ds_t, 1e-6), 0.0, 1.0)
            return interp_fn(alpha, p0, p1)

        if s == WalkingState.END:
            # Keep ZMP on last support
            return (steps_xy[-2] + steps_xy[-1]) / 2.0

        raise ValueError(f"Unknown state: {s}")

    # Allocate horizon
    n_samples = int(np.floor(t_horizon / delta_t))
    t_samples = np.arange(n_samples, dtype=float) * delta_t
    zmp_horizon = np.zeros((n_samples, 2), dtype=float)

    # Simulated state over the horizon
    sim_state = state
    time_in_state = t_state
    sim_step_idx = step_idx

    for k in range(n_samples):
        # Advance state machine if needed (except END which just saturates)
        while sim_state != WalkingState.END and time_in_state >= state_duration(sim_state):
            time_in_state -= state_duration(sim_state)
            sim_state, sim_step_idx = next_state_and_step(sim_state, sim_step_idx, steps_foot)

        # Compute ZMP for current simulated state
        zmp_horizon[k] = zmp_for_state(sim_state, sim_step_idx, time_in_state)

        # Advance local time
        time_in_state += delta_t

    return t_samples, zmp_horizon


class CentroidalPlanner:
    def __init__(
        self,
        dt: float,
        com_initial_target: np.ndarray,
        params: PreviewControllerParams,
    ):
        self.params = params
        self.dt = dt
        self.ctrler_mat = compute_preview_control_matrices(params, dt)
        self.x = np.array([0.0, com_initial_target[0], 0.0, 0.0], dtype=float)
        self.y = np.array([0.0, com_initial_target[1], 0.0, 0.0], dtype=float)
        self.steps_sequence = None
        self.steps_foot = None
        self.step_idx = 0
        self.zmp_ref_horizon = None

    def set_steps_sequence(self, steps_sequence: np.ndarray, steps_foot: typing.List[Foot]):
        # We assign the new sequence and reset the step counter
        self.steps_sequence = steps_sequence
        self.steps_foot = steps_foot
        self.step_idx = 0

    def update(
        self,
        com_initial_target,
        state: WalkingState,
        step_idx: int,
        t_state: float,
        t_init: float,
        t_end: float,
        t_ss: float,
        t_ds: float,
    ):
        # Update control
        _, self.zmp_ref_horizon = build_zmp_horizon(
            com_initial_target,
            t_horizon=(self.params.n_preview_steps - 1) * self.dt,
            t_state=t_state,
            state=state,
            delta_t=self.dt,
            current_step_idx=step_idx,
            steps_sequence=self.steps_sequence,
            steps_foot=self.steps_foot,
            ss_t=t_ss,
            ds_t=t_ds,
            t_init=t_init,
            t_end=t_end,
            interp_fn=cubic_spline_interpolation,
        )

        _, self.x, self.y = update_control(
            self.ctrler_mat,
            self.zmp_ref_horizon[0],
            self.zmp_ref_horizon,
            self.x.copy(),
            self.y.copy(),
        )

    def get_com_pos(self) -> typing.Tuple[float, float]:
        return self.x[1], self.y[1]

    def get_ref_horizon(self):
        return self.zmp_ref_horizon
