# LIPM Walking Controller

[![Build and tests](https://github.com/rdesarz/lipm-walking-controller/actions/workflows/build.yml/badge.svg)](https://github.com/rdesarz/lipm-walking-controller/actions/workflows/build.yml)

This repository presents an open-source implementation of the **Linear Inverted Pendulum Model (LIPM)** walking pattern 
generator based on **preview control of the Zero-Moment Point (ZMP)**, following the formulation introduced 
by _Kajita et al., “Biped Walking Pattern Generation by Using Preview Control of the Zero-Moment Point,” ICRA 2003_.

---

## Introduction

Humanoid walking requires the generation of dynamically stable trajectories of the Center of Mass (CoM) with 
respect to the Zero-Moment Point (ZMP). This project implements the **discrete-time LIPM** dynamics and 
the associated **optimal preview control law**, reproducing the approach used in model-based humanoid locomotion 
control. The framework includes visualization and inverse kinematics modules, enabling reproducible experiments 
on trajectory generation and tracking. 

---

## Overview

The objective is to reproduce and analyze the ZMP preview control pipeline:

- Model the robot’s CoM using the 3D LIPM  
- Compute optimal CoM trajectories given a reference ZMP sequence using preview control 
- Generate and visualize corresponding foot trajectories
- Apply inverse kinematics to produce consistent joint motions  

The implementation prioritizes **clarity** and **experimental reproducibility**, making it suitable for education
purpose.

---

## Methodology

### Linear Inverted Pendulum Model

The CoM motion is modeled by the discrete-time linearized dynamics of the inverted pendulum:

\[
\ddot{x}_c = \frac{g}{z_c} (x_c - x_z)
\]

where \( x_z \) denotes the ZMP, \( x_c \) the CoM projection, and \( z_c \) the constant CoM height.

### Preview Control

The control input minimizes a quadratic cost over a finite horizon:

\[
J = \sum_{k=0}^{\infty} \left( Q_e e_k^2 + x_k^T Q_x x_k + R \Delta u_k^2 \right)
\]

yielding a feedback + integral + preview law.  
The resulting controller anticipates future ZMP references, ensuring stable walking trajectories.

---

## Features

- Discrete-time 3D LIPM formulation  
- Full preview controller (state feedback, integral, preview gain)  
- Configurable parameters: preview horizon, \( Q_e, Q_x, R \)  
- Visualization of CoM/ZMP trajectories and foot motion  
- Inverse kinematics tracking using the Talos humanoid model  

---

## 4. Experiments

### 4.1 Preview Control Demonstration

```bash
git clone https://github.com/rdesarz/lipm-walking-controller.git
cd lipm-walking-controller
pip install ".[dev]"
python examples/step_2_lipm_preview_control.py
