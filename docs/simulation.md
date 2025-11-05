# Integration in physical simulator

## Introduction

The simulation module validates the entire walking control pipeline, from ZMP-based pattern generation to joint-space 
motion execution. It provides a reproducible environment to test the controller under realistic dynamics before 
deployment on hardware.
The robot is simulated in PyBullet, which integrates forward dynamics and computes contact forces during each support 
phase. The module reproduces the walking sequence defined by the preview controller, including single and double support 
phases, ZMP transitions, and foot exchanges. The reference trajectories of the center of mass (CoM) and feet are generated 
by the LIPM + preview control model, then projected into joint space through the inverse-kinematics solver based on Pinocchio.

At each timestep, the Pinocchio kinematic model is updated using the robot state from PyBullet. The desired joint 
positions are then computed and applied through position control, ensuring consistency between the kinematic model and the simulated robot.

## Example

In this example, the controller and inverse-kinematics modules are integrated into PyBullet to test the walking pattern in a simulated environment:

```bash
xhost +
docker run --rm -it \
  --env DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  --device /dev/dri:/dev/dri \
  biped-walking-controller \
  python examples/example_4_physics_simulation.py \
  --path-talos-data --launch-gui --plot-results
```

You should get a visualization of the simulation such as the one below:

<p align="center">
  <img src="../img/physics_simulation.gif" />
</p>

At the end of the walking sequence, two plots are displayed to compare reference and simulated trajectories.



# Code API

The simulation module provides all components required to initialize and manage the physics simulation. It simplifies 
interaction with the robot, including data extraction and joint configuration. The Simulator class currently implements 
this interface using PyBullet as the physics backend.

::: biped_walking_controller.simulation
    options:
      members_order: source
      heading_level: 2