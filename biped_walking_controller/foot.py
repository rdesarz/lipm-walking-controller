"""
Footstep geometry and trajectories utilities.

Provides:

* Polygon operations to compute single/double-support regions.
* ZMP clamp to the current support polygon.
* Deterministic swing-foot trajectories and step poses.
"""

import math

import numpy as np
import shapely
from shapely import Polygon, Point, affinity, union
from shapely.ops import nearest_points

from biped_walking_controller.state_machine import WalkingStateMachineParams, Foot


def bezier_quintic(P: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Evaluate a quintic (degree-5) Bézier curve at parameter values ``s``.

    A quintic Bézier curve is:
        B(s) = Σ_{i=0..5} C(5,i) (1-s)^{5-i} s^i P_i

    Parameters
    ----------
    P : np.ndarray
        Control points of shape ``(6, 3)``. Rows are ``P0..P5``.
        Columns are Cartesian coordinates ``[x, y, z]``.
    s : np.ndarray
        1D array of parameter values in ``[0, 1]`` of shape ``(N,)``.

    Returns
    -------
    np.ndarray
        Curve samples of shape ``(N, 3)``.

    Notes
    -----
    With ``P0=P1=P2`` and ``P3=P4=P5``, the curve has zero velocity and
    acceleration at both ends, which is the typical “minimum-jerk” boundary
    condition used for swing-foot profiles.
    """
    # (6, N) Bernstein basis for degree 5
    B = np.array([math.comb(5, i) * ((1.0 - s) ** (5 - i)) * (s**i) for i in range(6)])
    return B.T @ P  # (N, 3)


class BezierCurveFootPathGenerator:
    """
    Swing-foot path generator using a quintic Bézier with zero vel/acc at endpoints.

    This generator fixes the first three control points at the start pose and
    the last three at the end pose, then sets the vertical components of the
    interior points to reach a prescribed apex height. The vertical “shape”
    parameter ``alpha`` is searched once at construction, then reused per call.

    Parameters
    ----------
    foot_height : float
        Desired apex height of the swing foot above the line segment connecting
        start and end poses.

    Attributes
    ----------
    alpha : float
        Internal vertical shaping parameter calibrated so that the curve height
        at ``s=0.5`` equals ``foot_height``.

    Notes
    -----
    - The constructor performs a simple grid search over ``alpha`` in
      ``[0, 2*foot_height]`` to match the apex within ``1e-3``.
    - Endpoints have zero velocity and acceleration due to repeated control
      points: ``P0=P1=P2`` and ``P3=P4=P5``.
    """

    def __init__(self, foot_height: float):
        P = np.zeros((6, 3))
        P[0] = P[1] = P[2] = np.array([0.0, 0.0, 0.0])
        P[3] = P[4] = P[5] = np.array([0.3, 0.0, 0.0])

        # Brute-force search for alpha that achieves the desired apex height. Could be optimized in the future if necessary.
        element = np.linspace(0.0, 2.0 * foot_height, num=3000)
        self.alpha = 0.0  # default in case foot_height == 0

        for alpha in element:
            P[1][2] = alpha / 2.0
            P[2][2] = alpha
            P[3][2] = alpha
            P[4][2] = alpha / 2.0

            apex = bezier_quintic(P, np.array([0.5]))  # (1, 3)
            if abs(apex[:, 2] - foot_height) < 1e-3:
                self.alpha = alpha
                break

    def __call__(self, p_start: np.ndarray, p_end: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Generate a swing-foot trajectory between two poses.

        Parameters
        ----------
        p_start : np.ndarray
            Start foot position ``(3,)`` as ``[x, y, z]``.
        p_end : np.ndarray
            End foot position ``(3,)`` as ``[x, y, z]``.
        s : np.ndarray
            1D array of parameter values in ``[0, 1]`` of shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Sampled path of shape ``(N, 3)``.

        Examples
        --------
        >>> gen = BezierCurveFootPathGenerator(foot_height=0.06)
        >>> s = np.linspace(0.0, 1.0, 101)
        >>> p = gen(np.array([0,0,0]), np.array([0.3,0,0]), s)  # (101, 3)
        """
        P = np.zeros((6, 3))
        P[0] = P[1] = P[2] = p_start
        P[3] = P[4] = P[5] = p_end

        # Set vertical shape using calibrated alpha
        P[1][2] = self.alpha / 2.0
        P[2][2] = self.alpha
        P[3][2] = self.alpha
        P[4][2] = self.alpha / 2.0

        return bezier_quintic(P, s)


class SinusoidFootPathGenerator:
    """
    Swing-foot path generator with sinusoidal vertical profile.

    The horizontal motion is linear from start to end along x, with y held
    constant at the start value. The vertical component follows:
        z(s) = foot_height * sin(pi * s)

    Parameters
    ----------
    foot_height : float
        Maximum height at mid-swing.

    Notes
    -----
    This profile is C1 at the endpoints (zero velocity) but not C2 like the
    Bézier construction. Use the Bézier for smoother contact transitions.
    """

    def __init__(self, foot_height: float):
        self.foot_height = foot_height

    def __call__(self, p_start: np.ndarray, p_end: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Generate a sinusoidal swing-foot trajectory.

        Parameters
        ----------
        p_start : np.ndarray
            Start foot position ``(3,)`` as ``[x, y, z]``.
        p_end : np.ndarray
            End foot position ``(3,)`` as ``[x, y, z]``.
        s : np.ndarray
            1D array of parameter values in ``[0, 1]`` of shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Sampled path of shape ``(N, 3)``.
        """
        path = np.zeros((len(s), 3))
        theta = s * math.pi
        path[:, 2] = np.sin(theta) * self.foot_height
        path[:, 1] = p_start[1]
        path[:, 0] = (1.0 - s) * p_start[0] + s * p_end[0]
        return path


def compute_steps_sequence(
    rf_initial_pose: np.ndarray,
    lf_initial_pose: np.ndarray,
    n_steps: int,
    l_stride: float,
):
    steps_pose = np.zeros((n_steps + 2, 3))
    steps_pose[0] = rf_initial_pose
    steps_ids = [Foot.RIGHT]
    for i in range(1, n_steps + 1):
        sign = -1.0 if i % 2 == 0 else 1.0
        steps_ids.append(Foot.RIGHT if i % 2 == 0 else Foot.LEFT)
        steps_pose[i] = np.array(
            [i * l_stride, sign * math.fabs(lf_initial_pose[1]), rf_initial_pose[2]]
        )

    # Add a last step to have both feet at the same level
    steps_pose[-1] = steps_pose[-2]
    steps_pose[-1][1] = steps_pose[-1][1] * -1.0

    return steps_pose, steps_ids


def compute_onplace_steps_sequence(
    rf_initial_pose: np.ndarray,
    lf_initial_pose: np.ndarray,
    n_steps: int,
):
    steps_pose = np.zeros((n_steps + 2, 3))
    steps_pose[0] = rf_initial_pose
    steps_ids = [Foot.RIGHT]
    for i in range(1, n_steps + 1):
        steps_ids.append(Foot.RIGHT if i % 2 == 0 else Foot.LEFT)
        steps_pose[i] = rf_initial_pose if i % 2 == 0 else lf_initial_pose

    # Add a last step to have both feet at the same level
    steps_pose[-1] = steps_pose[-2]
    steps_pose[-1][1] = steps_pose[-1][1] * -1.0

    return steps_pose, steps_ids


def compute_time_vector(t_ss, t_ds, t_init, t_final, n_steps, dt):
    total_time = t_init + n_steps * (t_ss + t_ds) + (t_ss + t_final)
    N = int(total_time / dt)
    t = np.arange(N) * dt

    return t


def compute_feet_trajectories(
    rf_initial_pose,
    lf_initial_pose,
    n_steps,
    steps_pose,
    t_ss,
    t_ds,
    t_init,
    t_final,
    dt,
    traj_generator=BezierCurveFootPathGenerator(foot_height=0.1),
):
    """
    Generate swing trajectories and step poses for alternating feet.

    Scenario
    --------
    Start with a DS phase of duration `t_init` to shift CoM.
    Then for `n_steps`, alternate SS (duration `t_ss`) and DS (duration `t_ds`)
    with forward stride `l_stride`. Finish with one SS to align both feet,
    then a final DS of duration `t_final` to center CoM.

    Parameters
    ----------
    rf_initial_pose : array-like, shape (3,)
        Initial right foot pose proxy [x, y, z].
    lf_initial_pose : array-like, shape (3,)
        Initial left foot pose proxy [x, y, z].
    n_steps : int
        Number of forward steps.
    t_ss : float
        Single-support duration per step.
    t_ds : float
        Double-support duration between steps.
    t_init : float
        Initial DS duration before stepping.
    t_end : float
        Final DS duration to center the CoM.
    l_stride : float
        Step length along +x for each new foothold.
    dt : float
        Time discretization step.
    max_height_foot : float
        Peak swing height for the swinging foot.

    Returns
    -------
    t : ndarray, shape (N,)
        Time samples.
    lf_path : ndarray, shape (N, 3)
        Left foot trajectory [x, y, z].
    rf_path : ndarray, shape (N, 3)
        Right foot trajectory [x, y, z].
    steps_pose : ndarray, shape (n_steps + 2, 2)
        Planned foothold XY positions. Last row mirrors penultimate y to align feet.
    phases : ndarray, shape (N,)
        Phase flag over time: -1 for left swing, +1 for right swing, 1 during right-stance,
        -1 during left-stance, unchanged during DS.

    Notes
    -----
    - Swing z uses a half-sine profile with peak `max_height_foot`.
    - X is linearly interpolated during SS, then held constant until the next step.
    - Y remains fixed to the initial lateral offsets.
    """
    # The sequence is the following:
    # Start with a double support phase to switch CoM on right foot
    # Then n_steps, for each step there is a single support phase and a
    # double support phase. The length of the step is given by l_stride.
    # At the last step, we add a single support step to join both feet at
    # the same level and a double support step to  place the CoM in the
    # middle of the feet

    t = compute_time_vector(t_ss, t_ds, t_init, t_final, n_steps, dt)
    N = len(t)

    # Initialize path
    rf_path = np.ones([N, 3]) * rf_initial_pose
    lf_path = np.ones([N, 3]) * lf_initial_pose
    phases = np.ones(N)

    # Compute motion of left foot
    for k in range(0, n_steps, 2):
        t_begin = t_init + k * (t_ss + t_ds)
        t_end = t_init + k * (t_ss + t_ds) + t_ss
        mask = (t >= t_begin) & (t < t_end)
        sub_time = t[mask] - (t_init + k * (t_ss + t_ds))

        phases[mask] = -1.0

        # Compute motion on every axis
        if k == 0:
            alpha = sub_time / t_ss
            lf_path[mask] = traj_generator(lf_initial_pose, steps_pose[k + 1], alpha)
        else:
            alpha = sub_time / t_ss
            lf_path[mask] = traj_generator(steps_pose[k - 1], steps_pose[k + 1], alpha)

        # # Add constant part till the next step
        t_begin = t_init + k * (t_ss + t_ds) + t_ss
        t_end = t[-1]
        mask = (t >= t_begin) & (t < t_end)
        lf_path[mask, 0] = steps_pose[k + 1][0]

    # Compute motion of right foot
    for k in range(1, n_steps + 1, 2):
        t_begin = t_init + k * (t_ss + t_ds)
        t_end = t_init + k * (t_ss + t_ds) + t_ss
        mask = (t > t_begin) & (t < t_end)
        sub_time = t[mask] - (t_init + k * (t_ss + t_ds))

        phases[mask] = 1.0

        # Compute motion on x-axis
        if k == 1:
            alpha = sub_time / t_ss
            rf_path[mask] = traj_generator(rf_initial_pose, steps_pose[k + 1], alpha)
        else:
            alpha = sub_time / t_ss
            rf_path[mask] = traj_generator(steps_pose[k - 1], steps_pose[k + 1], alpha)

        # # Add constant part till the next step
        t_begin = t_init + k * (t_ss + t_ds) + t_ss
        t_end = t[-1]
        mask = (t >= t_begin) & (t < t_end)
        rf_path[mask, 0] = steps_pose[k + 1][0]

    return t, lf_path, rf_path, phases


def clamp_to_polygon(pnt: np.ndarray, poly: Polygon):
    """
    Project a 2D point to the closest location inside a polygon.

    Parameters
    ----------
    pnt : ndarray, shape (2,)
        Input point [x, y].
    poly : Polygon
        Support polygon in world coordinates.

    Returns
    -------
    ndarray, shape (2,)
        Point inside `poly`. Equals `pnt` if already inside.

    Notes
    -----
    Uses the nearest point on the polygon exterior if outside.
    """
    p = Point(pnt[0], pnt[1])
    if poly.contains(p):
        return pnt

    # nearest point on boundary
    q = nearest_points(poly.exterior, p)[0]

    return np.array([q.x, q.y])


def compute_double_support_polygon(foot_pose_a, foot_pose_b, foot_shape: shapely.Polygon):
    """
    Compute the convex support region for double support.

    Parameters
    ----------
    foot_pose_a : array-like, shape (2,) or (3,)
        World pose proxy for foot A: uses [x, y].
    foot_pose_b : array-like, shape (2,) or (3,)
        World pose proxy for foot B: uses [x, y].
    foot_shape : shapely.Polygon
        Foot contact shape in the foot frame (centered).

    Returns
    -------
    shapely.Polygon
        Convex hull of the union of both translated foot polygons.
    """
    curent_foot = affinity.translate(foot_shape, xoff=foot_pose_a[0], yoff=foot_pose_a[1])
    next_foot = affinity.translate(foot_shape, xoff=foot_pose_b[0], yoff=foot_pose_b[1])

    return union(curent_foot, next_foot).convex_hull


def compute_single_support_polygon(foot_pose, foot_shape: shapely.Polygon):
    """
    Compute the support region for single support.

    Parameters
    ----------
    foot_pose : array-like, shape (2,) or (3,)
        World pose proxy for the stance foot: uses [x, y].
    foot_shape : shapely.Polygon
        Foot contact shape in the foot frame (centered).

    Returns
    -------
    shapely.Polygon
        Translated foot polygon in world coordinates.
    """
    return affinity.translate(foot_shape, xoff=foot_pose[0], yoff=foot_pose[1])


def get_active_polygon(t: float, steps_pose, t_ss: float, t_ds: float, foot_shape: shapely.Polygon):
    """
    Select the active support polygon at time `t`.

    Parameters
    ----------
    t : float
        Elapsed time since start.
    steps_pose : array-like, shape (K, 2)
        Footstep sequence in world XY. Index 0 is the first stance (right or left).
    t_ss : float
        Single-support duration per step.
    t_ds : float
        Double-support duration between steps.
    foot_shape : shapely.Polygon
        Foot contact shape in the foot frame (centered).

    Returns
    -------
    shapely.Polygon
        Single- or double-support polygon active at time `t`.

    Notes
    -----
    - Step period t_step = t_ss + t_ds.
    - For the last period, returns single support on the last foot.
    """
    t_step = t_ss + t_ds
    i = int(t / t_step)
    i = min(i, len(steps_pose) - 2)
    t_in = t - i * t_step

    if t_in < t_ss:
        return compute_single_support_polygon(steps_pose[i], foot_shape)
    elif t >= (len(steps_pose) - 1) * t_step:
        return compute_single_support_polygon(steps_pose[-1], foot_shape)
    else:
        return compute_double_support_polygon(steps_pose[i], steps_pose[i + 1], foot_shape)


def compute_swing_foot_pose(
    t_state: float,
    params: WalkingStateMachineParams,
    step_start: np.ndarray,
    step_target: np.ndarray,
    touchdown_extension_vel: float,
    path_generator,
) -> np.ndarray:
    """
    Compute swing foot pose for current state time with late touchdown extension.

    - For t_state in [0, t_ss]: nominal min-jerk swing between step_start and step_target.
    - For t_state > t_ss and no contact: keep moving the *commanded* foot down
      below the nominal ground height at constant velocity.
      The physics/contact solver will clamp penetration.
    - As soon as contact_force > force_threshold: freeze at step_target.
    """
    t_ss = params.t_ss

    if t_state <= t_ss:
        # Normal swing phase
        s = np.clip(t_state / t_ss, 0.0, 1.0)

        return path_generator(step_start, step_target, s)

    # Late-touchdown extension phase
    # We assume step_target[2] is the nominal ground height.
    # After nominal end of swing, keep commanding the foot below that plane.
    dt = t_state - t_ss
    pos = step_target.copy()
    pos[2] = step_target[2] - touchdown_extension_vel * dt
    return pos
