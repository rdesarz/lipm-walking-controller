# Integration in physical simulator

## Introduction

We currently use Pybullet as the simulator 

## Example

In this example we integrate the controller and inverse kinematic modules inside Pybullet to test the walking pattern in
a simulated environment:

```bash
xhost +
docker run --rm -it --env DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro --device /dev/dri:/dev/dri lipm-walking-controller python examples/example_4_physics_simulation.py --path-talos-data --launch-gui"/"
```

<p align="center">
  <img src="../img/physics_simulation.gif" />
</p>


# Code API

The simulation module provides all components required to initialize and manage the physics simulation. It simplifies 
interaction with the robot, including data extraction and joint configuration. The Simulator class currently implements 
this interface using PyBullet as the physics backend.

::: lipm_walking_controller.simulation
    options:
      members_order: source
      heading_level: 2