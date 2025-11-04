# Differential Inverse Kinematics

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
problem with tasks being either a cost to reduce or a constraint to fulfill. A way to solve this is by using a hierarchial
Quadratic Program where each task is assigned with a priority (tracking the CoM is more important that postural task for instance). In 
practice it is hard to find an open-source solver for this kind of optimization problem. Therefore it is modelled as a
QP with different weights for each task. A task with high priority will have a higher weight than a low priority task.
This is the approach used for instance by the [pink](https://stephane-caron.github.io/pink/index.html) library from which this
part is inspired.



## References



@software{pink,
  title = {{Pink: Python inverse kinematics based on Pinocchio}},
  author = {Caron, Stéphane and De Mont-Marin, Yann and Budhiraja, Rohan and Bang, Seung Hyeon and Domrachev, Ivan and Nedelchev, Simeon and peterd-NV, github user and Vaillant, Joris},
  license = {Apache-2.0},
  url = {https://github.com/stephane-caron/pink},
  version = {3.4.0},
  year = {2025}
}

# Code API

::: lipm_walking_controller.inverse_kinematic
    options:
      members_order: source
      heading_level: 2