import typing
from dataclasses import dataclass
from typing import Any

import numpy as np

import pinocchio as pin
from qpsolvers import solve_qp


@dataclass
class InvKinSolverParams:
    fixed_foot_frame: int
    moving_foot_frame: int
    torso_frame: int
    model: pin.Model
    data: Any
    w_torso: float
    w_com: float
    w_mf: float
    w_ff: float
    mu: float
    dt: float
    locked_joints: typing.Optional[typing.List[int]] = None


def se3_task_error_and_jacobian(model, data, q, frame_id, M_des):
    # Pose of frame i in world; LOCAL frame convention (right differentiation)
    oMi = data.oMf[frame_id]  # requires updateFramePlacements()
    iMd = oMi.actInv(M_des)  # ^i M_d  = oMi^{-1} * oMdes
    e6 = pin.log(iMd).vector  # right-invariant pose error in LOCAL frame

    # Geometric Jacobian in LOCAL frame
    Jb = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)

    # Right Jacobian of the log map (Pinocchio’s Jlog6)
    Jl = pin.Jlog6(iMd)  # maps LOCAL spatial vel -> d(log) in se(3)

    # Task Jacobian
    Jtask = Jl @ Jb  # minus sign per right-invariant residual

    return e6, Jtask


def joint_vel_span(j, model):
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
    model, data = params.model, params.data

    # ---------- Kinematics ----------
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
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

    # ---------- Tasks ----------
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

    # ---------- Reduce Jacobians to active DoFs ----------
    Jcom_r = red(Jcom)
    J_ff_r = red(J_ff)
    J_mf_r = red(J_mf)
    J_torso_r = red(J_torso)
    nav = active_idx.size

    # ---------- Quadratic cost on reduced variables dq_r ----------
    H = (
        (Jcom_r.T @ (np.eye(3) * params.w_com) @ Jcom_r)
        + (J_torso_r.T @ (np.eye(3) * params.w_torso) @ J_torso_r)
        + (J_mf_r.T @ (np.eye(6) * params.w_mf) @ J_mf_r)
        + (J_ff_r.T @ (np.eye(6) * params.w_ff) @ J_ff_r)
        + np.eye(nav) * params.mu
    )
    g = (
        (-Jcom_r.T @ (np.eye(3) * params.w_com) @ e_com)
        + (-J_torso_r.T @ (np.eye(3) * params.w_torso) @ e_torso)
        + (-J_mf_r.T @ (np.eye(6) * params.w_mf) @ e_mf)
        + (-J_ff_r.T @ (np.eye(6) * params.w_ff) @ e_ff)
    )

    H = 0.5 * (H + H.T)  # symmetrize

    # ---------- Solve reduced QP ----------
    dq_r = solve_qp(P=H, q=g, solver="osqp")  # soft contact via w_ff

    # ---------- Lift back to full dq ----------
    dq = np.zeros(nv)
    dq[active_idx] = dq_r

    q_next = pin.integrate(model, q, dq)

    return q_next, dq
