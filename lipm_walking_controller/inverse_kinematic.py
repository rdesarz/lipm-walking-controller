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
    mu: float
    dt: float


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
    q, com_target, oMf_fixed_foot, oMf_moving_foot, oMf_torso, params: InvKinSolverParams
):
    pin.forwardKinematics(params.model, params.data, q)
    pin.updateFramePlacements(params.model, params.data)

    nv = params.model.nv

    # -------- CoM task (Euclidean) --------
    pin.computeCentroidalMap(params.model, params.data, q)
    com = pin.centerOfMass(params.model, params.data, q)
    Jcom = pin.jacobianCenterOfMass(params.model, params.data, q)  # (3,nv)
    e_com = com_target - com

    # -------- Fixed-foot hard contact (6D) --------
    # Drive residual to zero at velocity level: J_ff * dq = -Kc * e_ff
    e_ff, J_ff = se3_task_error_and_jacobian(
        params.model,
        params.data,
        q,
        params.fixed_foot_frame,
        oMf_fixed_foot,
    )

    # -------- Moving-foot soft pose task (6D) --------
    e_mf, J_mf = se3_task_error_and_jacobian(
        params.model, params.data, q, params.moving_foot_frame, oMf_moving_foot
    )

    # -------- Torso roll/pitch soft task --------
    # Select angular part of e_torso and corresponding Jacobian rows
    e_torso6, J_torso6 = se3_task_error_and_jacobian(
        params.model,
        params.data,
        q,
        params.torso_frame,
        # keep current yaw: project desired as same yaw in world
        oMf_torso,
    )
    S = np.zeros((3, 6))
    S[0, 3] = 1.0
    S[1, 4] = 1.0
    S[2, 5] = 1.0
    e_torso = S @ e_torso6
    J_torso = S @ J_torso6

    i_start = 18
    i_end = 50
    J_torso[:, i_start:i_end] = np.zeros((J_torso.shape[0], i_end - i_start))
    Jcom[:, i_start:i_end] = np.zeros((Jcom.shape[0], i_end - i_start))
    J_mf[:, i_start:i_end] = np.zeros((J_mf.shape[0], i_end - i_start))
    J_ff[:, i_start:i_end] = np.zeros((J_ff.shape[0], i_end - i_start))
    I = np.eye(nv)
    I[:, i_start:i_end] = np.zeros((I.shape[0], i_end - i_start))

    # -------- Quadratic cost --------
    H = (
        (Jcom.T @ (np.eye(3) * params.w_com) @ Jcom)
        + (J_torso.T @ (np.eye(3) * params.w_torso) @ J_torso)
        + (J_mf.T @ (np.eye(6) * params.w_mf) @ J_mf)
        + np.eye(nv) * params.mu
    )

    g = (
        (-Jcom.T @ (np.eye(3) * params.w_com) @ e_com)
        - (J_torso.T @ (np.eye(3) * params.w_torso) @ e_torso)
        - (J_mf.T @ (np.eye(6) * params.w_mf) @ e_mf)
    )

    # Symmetrization of the cost matrix
    H = 0.5 * (H + H.T)

    # Use equality constraint for the position of the fixed foot
    A_eq = J_ff
    b_eq = e_ff

    dq = solve_qp(P=H, q=g, A=A_eq, b=b_eq, solver="osqp")

    q_next = pin.integrate(params.model, q, dq)

    return q_next, dq
