## Zero Moment Point and Balance Criterion

The Zero Moment Point (ZMP) is the point on the ground where the resultant contact forces between the feet and the
ground produce no moment about the horizontal axes. To maintain balance, the ZMP must remain within the robot’s support
polygon, defined as the convex hull of the contact areas of the feet. Intuitively, this ensures that the ground reaction
forces can generate a counteracting moment to keep the feet flat and prevent tipping, maintaining dynamic equilibrium.
For a more thorough explanation I
recommend [this blog post](https://scaron.info/robotics/zero-tilting-moment-point.html) by Stéphane Caron.

## Linear Inverted Pendulum Model

The first step of the controller is to define a reference ZMP trajectory, alternating from one foot to the other at each
step. This reference is generated using a cubic spline that interpolates the position of each foot. 
The objective is to establish a relationship between the position of this reference ZMP and the robot’s Center of Mass (
CoM).
This relationship can be derived from a simplified model of the robot’s dynamics known as the **Linear Inverted Pendulum
Model (LIPM)**.

The LIPM is derived under the following assumptions:

* The mass of the body is concentrated at a single point, the Center of Mass (CoM).
* Legs are massless and do not contribute to the system dynamics.
* The CoM moves on a horizontal plane at a constant height, eliminating vertical motion coupling.
* No angular momentum is generated about the CoM, meaning the upper body remains still to avoid producing additional
  moments.

Under these assumptions and for small angles, the inverted pendulum dynamics can be linearized, leading to the following
second-order linear equation:

$$
\ddot{x}_c = \frac{g}{z_c} (x_c - x_z)
$$

where $x_z$ denotes the ZMP, $x_c$ the CoM projection, and $z_c$ the constant CoM height.

## Preview Control

In Kajita's paper, the idea is to use a preview control in order to track and anticipate the ZMP reference change.
The control input minimizes a quadratic cost over a finite horizon:

$$
J = \sum_{k=0}^{\infty} \left( Q_e e_k^2 + x_k^T Q_x x_k + R \Delta u_k^2 \right)
$$

yielding a feedback + integral + preview law.  
The resulting controller anticipates future ZMP references, ensuring stable walking trajectories.

The result of the preview controller can be observed on the figure below. The upper-left figure shows the trajectory of
the CoM in red over time, the generated reference ZMP in blue and the support polygon in green. The upper-right and
lower-right figures show the trajectory of the ZMP and COM over time for x and y pos. Finally, the lower-left figure
shows the preview gains that are computed.

<p align="center">
  <img src="../img/preview_control.gif" />
</p>

## Example

You can reproduce the example displayed on the figure by launching the script `example_1_lipm_preview_control.py`. We
recommend you to use Docker as explained in the installation part:

```bash
xhost +local:root
docker run --rm -it \
  --env DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  --device /dev/dri:/dev/dri \
  ghcr.io/rdesarz/biped-walking-controller \
  python examples/example_1_lipm_preview_control.py 
```

The parameters used in this script are the following:

```python
dt = 0.005  # Delta of time of the model simulation
t_preview = 1.6  # Time horizon used for the preview controller
t_ss = 0.6  # Single support phase time window
t_ds = 0.4  # Double support phase time window
t_init = 2.0  # Initialization phase (transition from still position to first step)
t_end = 1.0  # Final phase (transition from walking to standstill position)
foot_shape = Polygon(
    ((0.11, 0.05), (0.11, -0.05), (-0.11, -0.05), (-0.11, 0.05)))  # Shape of the foot for support polygon computation
n_steps = 5  # Number of steps
l_stride = 0.3  # Length of the stride
```

## References

- Kajita, S., Kanehiro, F., Kaneko, K., Fujiwara, K., Harada, K., Yokoi, K., & Hirukawa, H.  
  *Biped Walking Pattern Generation by Using Preview Control of Zero-Moment Point.*  
  *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2003.*

- Katayama, T., Ohki, T., Inoue, T., & Kato, T.  
  *Design of an Optimal Controller for a Discrete-Time System Subject to Previewable Demand.*  
  *International Journal of Control*, vol. 41, no. 3, pp. 677–699, 1985.

- Caron, S.  
  *Zero-tilting moment point.*  
  Available online
  at  [https://scaron.info/robotics/zero-tilting-moment-point.html](https://scaron.info/robotics/zero-tilting-moment-point.html),
  accessed 2025.  
  (Detailed explanations and examples for Zero-tilting moment point)

# Code API

::: biped_walking_controller.preview_control
options:
members_order: source
heading_level: 2