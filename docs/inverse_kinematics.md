# Differential Inverse Kinematics

## Introduction

In the context of this project, the goal of the inverse kinematics task is to compute joint velocities to track
the desired position of the Center of Mass and foot poses. Each of this goal is called a task. Walking requires several
tasks to be handled at the same time:

* Keeping the stand foot exactly at a desired 6D pose
* Tracking the swing foot trajectory
* Tracking the CoM trajectory
* Maintaining the torso straight (postural task)
* Locking some joints as the LIPM model relies on some assumptions on the upper body restrained motion.
* Constraints due to the joints properties (position and/or velocity limits)

Those tasks might be concurrent. Therefore, the inverse kinematics is well-suited to be seen as an optimization
problem with tasks being either a cost to reduce or a constraint to fulfill. A way to solve this is by using a
hierarchial
Quadratic Program where each task is assigned with a priority (tracking the CoM is more important that postural task for
instance). In
practice it is hard to find an open-source solver for this kind of optimization problem. Therefore it is modelled as a
QP with different weights for each task. A task with high priority will have a higher weight than a low priority task.
This is the approach used for instance by the [pink](https://stephane-caron.github.io/pink/index.html) library from
which this part is inspired.

## Kinematic task

Let:

- $q \in R^{n_q}$, the configuration vector of the controlled robot where $n_q$ is the number of configurations.
- $dq \in R^{n_v}$ the velocity vector of the controlled robot where $n_v$ is the number of velocities.
- $M \in SE(3)$, a pose.

The goal of a task is to reduce the error (residual) between the desired pose and the actual pose of a frame on the
robot.

One can compute this error,

For frame `i` with current world pose `oMf[i]` and desired pose `oMd`, define the **right-invariant residual** in the
local frame:

\[
e = \log\!\big( \, ^i\!M_d \big), \qquad
^i\!M_d = (oM_f)^{-1} \, oM_d .
\]

`e` encodes the smallest twist that brings `oMf` toward `oMd`, ordered as `[rx, ry, rz, vx, vy, vz]`.

The velocity relationship is:

\[
\dot e = J_{\log}\, V^i, \quad V^i = J_{\text{frame,LOCAL}}\, dq,
\]

giving the **task Jacobian**:

\[
J_{\text{task}} = J_{\log}\, J_{\text{frame,LOCAL}} \in \mathbb{R}^{6\times nv}.
\]

For the implementation, we rely on the Pinocchio library to compute Jacobian and CoM position.

---

## Tasks

We detail here each specific task that we need to fulfill when solving the inverse kinematics problem.

### 1. Fixed (Stance) Foot 

The stance foot is a hard equality constraint in the QP:

\[
J_{\text{ff}}\, dq = e_{\text{ff}}
\]

This constraint ensures the stance foot remains fixed in the world frame. It enforces a no-slip contact at velocity
level.

### 2. Swing (Moving) Foot 

Residual and Jacobian $e_{mf}, J_{mf}$ enter the cost as a weighted least squares term, allowing the swing foot to track
its desired trajectory.

### 3. Torso Orientation

We want to maintain the torso orientation. We select only the angular part of the Jacobian and residual with the matrix `S = [0, I]`:

\[
e_{\text{torso}} = S\, e_{\text{torso6}}, \quad
J_{\text{torso}} = S\, J_{\text{torso6}}.
\]

This aligns the torso without constraining its position.

### 4. Center of Mass

The center of mass is only constrained by its position. This is modelled as a cost in the optimization problem. The 
jacobian $J_{\text{com}}$ is the jacobian of the center of mass. The residual is defined below:

\[
e_{\text{com}} = (x_{\text{com}}^{\star} - x_{\text{com}}), \quad
\]

This task pulls the CoM toward the desired position.

---

## Optimization Problem

Solve for $\Delta q$ (over active velocity indices) the QP:

\begin{equation}
\begin{aligned}
\min_{\Delta q}\quad & \tfrac{1}{2}\,\Delta q^\top H \Delta q + g^\top \Delta q \\
\text{s.t.}\quad & A \Delta q = b
\end{aligned}
\label{eq:qp_eq_only}
\end{equation}

where

\begin{align}
H &= 
J_{\mathrm{com}}^\top W_{\mathrm{com}} J_{\mathrm{com}}
+ J_{\mathrm{torso}}^\top W_{\mathrm{torso}} J_{\mathrm{torso}}
+ J_{\mathrm{mf}}^\top W_{\mathrm{mf}} J_{\mathrm{mf}}
+ \mu I, \\[12pt]
g &= 
- J_{\mathrm{com}}^\top W_{\mathrm{com}} e_{\mathrm{com}}
- J_{\mathrm{torso}}^\top W_{\mathrm{torso}} e_{\mathrm{torso}}
- J_{\mathrm{mf}}^\top W_{\mathrm{mf}} e_{\mathrm{mf}},
\end{align}

and

\begin{align}
A &= J_{ff}  &  b &= e_{ff}
\end{align} 

- The term $\lambda I$ provides Tikhonov damping to improve numerical stability

This is solved with `qpsolvers.solve_qp` (OSQP backend). The solution is then integrated using Pinocchio to compute the 
desired joints position.

---

## Example

Inverse kinematics are computed to track CoM and foot trajectories using the Talos model.

This produces a full kinematic walking sequence without dynamic simulation:

```bash
docker run --rm -it -p 7000:7000 -p 6000:6000 biped-walking-controller python examples/example_3_walk_inverse_kinematic.py --path-talos-data "/"
```

The result should look like this:

<p align="center">
  <img src="../img/inverse_kinematic.gif" />
</p>

## References

- Caron, S.  
  *Jacobian of a kinematic task and derivatives on manifolds.*  
  Available online
  at  https://scaron.info/robotics/differential-inverse-kinematics.html,
  accessed 2025.  
  (Detailed explanations and examples for frame kinematics, Jacobian computation, and task-space control using
  Pinocchio.)

- Caron, S.  
  *Differential inverse kinematics.*  
  Available online
  at https://scaron.info/robotics/jacobians.html,
  accessed 2025.  
  (General introduction to differential inverse kinematics.)

- Caron, S.  
  *Pink: Python inverse kinematics based on Pinocchio*   
  License Apache-2.0   
  Available at https://github.com/stephane-caron/pink 
  2025

# Code API

::: biped_walking_controller.inverse_kinematic
    options:
        members_order: source
        heading_level: 2