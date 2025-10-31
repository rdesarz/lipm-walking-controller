import typing
from dataclasses import dataclass
from typing import Any

import numpy as np

import pinocchio as pin
from qpsolvers import solve_qp


@dataclass
class InvKinSolverParams:
    """
    Parameters for inverse-kinematics QP with CoM, feet, and torso tasks.

    Attributes
    ----------
    fixed_foot_frame : int
        Pinocchio frame id of the stance foot (hard equality).
    moving_foot_frame : int
        Pinocchio frame id of the swing foot (soft task).
    torso_frame : int
        Pinocchio frame id for torso orientation task (angular only).
    model : pin.Model
        Pinocchio kinematic model.
    data : Any
        Pinocchio data buffer associated with `model`.
    w_torso : float
        Weight of torso angular task in the QP cost.
    w_com : float
        Weight of CoM task in the QP cost.
    w_mf : float
        Weight of moving-foot 6D task in the QP cost.
    mu : float
        Damping on joint velocities in the QP (Tikhonov term).
    dt : float
        Time step used by caller; not used here directly but kept for API symmetry.
    locked_joints : list[int] | None
        Optional list of joints or velocity indices to lock.
        Each element can be:
          - a Pinocchio joint id j in [0, model.njoints), which locks its velocity span
          - or a direct velocity index v in [0, model.nv)
    """

    fixed_foot_frame: int
    moving_foot_frame: int
    torso_frame: int
    model: pin.Model
    data: Any
    w_torso: float
    w_com: float
    w_mf: float
    mu: float
    dt: float
    locked_joints: typing.Optional[typing.List[int]] = None


def se3_task_error_and_jacobian(model, data, q, frame_id, M_des):
    """
    Compute 6D right-invariant pose error and task Jacobian for a frame.

    Parameters
    ----------
    model : pin.Model
        Pinocchio model.
    data : pin.Data
        Pinocchio data (assumed up to date for frame placements if needed).
    q : ndarray, shape (nq,)
        Configuration vector.
    frame_id : int
        Target frame id in `model`.
    M_des : pin.SE3
        Desired frame pose in world.

    Returns
    -------
    e6 : ndarray, shape (6,)
        Right-invariant pose residual in the LOCAL frame.
        Order: angular (rx, ry, rz), linear (vx, vy, vz).
    Jtask : ndarray, shape (6, nv)
        Task Jacobian that maps generalized velocity `dq` to residual rate.

    Notes
    -----
    Uses LOCAL frame convention and Pinocchio's `Jlog6` to map spatial velocity
    to se(3) log-space.
    """
    # Pose of frame i in world; LOCAL frame convention (right differentiation)
    oMi = data.oMf[frame_id]  # requires updateFramePlacements()
    iMd = oMi.actInv(M_des)  # ^i M_d  = oMi^{-1} * oMdes
    e6 = pin.log(iMd).vector  # right-invariant pose error in LOCAL frame

    # Compute frame jacobian in local reference frame
    Jb = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)

    # Right Jacobian of the log map (Pinocchio’s Jlog6)
    Jl = pin.Jlog6(iMd)  # maps LOCAL spatial vel -> d(log) in se(3)

    # Task Jacobian
    Jtask = Jl @ Jb  # minus sign per right-invariant residual

    return e6, Jtask


def _joint_vel_span(j, model):
    """
    Return the velocity-index span for joint `j`.

    Parameters
    ----------
    j : int
        Pinocchio joint id.
    model : pin.Model
        Model containing `idx_v` mapping.

    Returns
    -------
    range
        Range of velocity indices covered by joint `j`.
    """
    i = model.idx_v[j]
    nvj = (model.idx_v[j + 1] - i) if j + 1 < model.njoints else (model.nv - i)
    return range(i, i + nvj)


def solve_inverse_kinematics(
    q,
    com_target,
    oMf_fixed_foot,
    oMf_moving_foot,
    oMf_torso,
    params: InvKinSolverParams,
):
    """
    One-step inverse kinematics via QP with a hard stance-foot constraint.

    Problem
    -------
    Minimize
        w_com || J_com dq - e_com ||^2
      + w_torso || J_torso dq - e_torso ||^2
      + w_mf || J_mf dq - e_mf ||^2
      + mu || dq ||^2
    subject to
        J_ff dq = e_ff        (fixed-foot 6D equality)

    Parameters
    ----------
    q : ndarray, shape (nq,)
        Current configuration.
    com_target : ndarray, shape (3,)
        Desired CoM position in world.
    oMf_fixed_foot : pin.SE3
        Desired world pose of stance foot.
    oMf_moving_foot : pin.SE3
        Desired world pose of swing foot.
    oMf_torso : pin.SE3
        Desired world pose of torso (only angular part is used).
    params : InvKinSolverParams
        Weights, model/data, damping, and optional locked joints.

    Returns
    -------
    q_next : ndarray, shape (nq,)
        Integrated configuration `integrate(model, q, dq)`.
    dq : ndarray, shape (nv,)
        Generalized velocity solution (zeros for locked indices).

    Notes
    -----
    - Builds a reduced QP on active velocity indices if `locked_joints` is set.
    - CoM task uses Pinocchio `jacobianCenterOfMass`.
    - Torso task selects angular rows via S = [0 I; 0 0].
    - QP is solved with `qpsolvers.solve_qp(..., solver="osqp")`.
    """
    model, data = params.model, params.data

    # ---------- Kinematics ----------
    nv = model.nv

    # ---------- Build locked velocity-index set ----------
    locked_v_idx = set()
    if params.locked_joints:
        # Accept either Pinocchio joint IDs or direct velocity indices.
        for j in params.locked_joints:
            if 0 <= j < model.njoints:
                i0 = model.idx_vs[j]
                i1 = model.idx_vs[j + 1] if j + 1 < model.njoints else nv
                locked_v_idx.update(range(i0, i1))
            elif 0 <= j < nv:
                locked_v_idx.add(j)
            else:
                raise ValueError(f"locked joint/index {j} out of range")
    active_idx = np.array(sorted(set(range(nv)) - locked_v_idx), dtype=int)

    def red(M):
        return M[:, active_idx] if M is not None else None

    # CoM
    pin.computeCentroidalMap(model, data, q)
    com = pin.centerOfMass(model, data, q)
    Jcom = pin.jacobianCenterOfMass(model, data, q)
    e_com = com_target - com

    # Fixed foot
    e_ff, J_ff = se3_task_error_and_jacobian(
        model, data, q, params.fixed_foot_frame, oMf_fixed_foot
    )

    # Moving foot
    e_mf, J_mf = se3_task_error_and_jacobian(
        model, data, q, params.moving_foot_frame, oMf_moving_foot
    )

    # Torso (only angular part)
    e_torso6, J_torso6 = se3_task_error_and_jacobian(model, data, q, params.torso_frame, oMf_torso)
    S = np.zeros((3, 6))
    S[0, 3] = S[1, 4] = S[2, 5] = 1.0
    e_torso = S @ e_torso6
    J_torso = S @ J_torso6

    Jcom_r = red(Jcom)
    J_ff_r = red(J_ff)
    J_mf_r = red(J_mf)
    J_torso_r = red(J_torso)
    nav = active_idx.size

    Aeq = J_ff_r
    beq = e_ff

    H = (
        (Jcom_r.T @ (np.eye(3) * params.w_com) @ Jcom_r)
        + (J_torso_r.T @ (np.eye(3) * params.w_torso) @ J_torso_r)
        + (J_mf_r.T @ (np.eye(6) * params.w_mf) @ J_mf_r)
        + np.eye(nav) * params.mu
    )
    g = (
        (-Jcom_r.T @ (np.eye(3) * params.w_com) @ e_com)
        + (-J_torso_r.T @ (np.eye(3) * params.w_torso) @ e_torso)
        + (-J_mf_r.T @ (np.eye(6) * params.w_mf) @ e_mf)
    )

    H = 0.5 * (H + H.T)  # symmetrize

    dq_r = solve_qp(P=H, q=g, A=Aeq, b=beq, solver="osqp")

    dq = np.zeros(nv)
    dq[active_idx] = dq_r

    q_next = pin.integrate(model, q, dq)

    return q_next, dq
