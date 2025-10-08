import pybullet as pb
import pybullet_data
import os

from lipm_walking_controller.model import Talos, print_joints


def track_q_position(robot_id, pin_model, q, name_to_bid, kp=80.0, kd=2.0, fmax=200.0):
    # 1) Base: choose ONE option
    # A) During early tests, fix base:
    #   pb.resetBasePositionAndOrientation(robot_id, q[0:3], q[3:7])
    #   # Load robot with useFixedBase=True to prevent collapse.
    # B) Or constrain a stance foot to the world (see lock_foot below).

    # 2) Joints: drive each to q_des with PD
    idx = 7  # after free-flyer
    bids, qdes = [], []
    for j in pin_model.joints[1:]:  # skip universe
        if j.nq == 1:
            jname = pin_model.names[j.id]
            if jname in name_to_bid:
                bids.append(name_to_bid[jname])
                qdes.append(float(q[idx]))
            idx += j.nq
        else:
            idx += j.nq

    if bids:
        pb.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=bids,
            controlMode=pb.POSITION_CONTROL,
            targetPositions=qdes,
            positionGains=[kp] * len(bids),
            velocityGains=[kd] * len(bids),
            forces=[fmax] * len(bids),
        )


def apply_position(q_des, j_to_q_idx):
    for j_id, q_id in j_to_q_idx.items():
        pb.setJointMotorControl2(
            robot,
            j_id,
            pb.POSITION_CONTROL,
            targetPosition=q_des[q_id],
            positionGain=0.4,
            velocityGain=1.0,
            force=200,
        )


if __name__ == "__main__":
    dt = 1.0 / 240.0
    cid = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=50, erp=0.2, contactERP=0.2)

    plane = pb.loadURDF("plane.urdf")
    PKG_PARENT = os.path.expanduser(os.environ.get("PKG_PARENT", "~/projects"))
    URDF = os.path.join(PKG_PARENT, "talos_data/urdf/talos_full.urdf")
    robot = pb.loadURDF(
        URDF, [0, 0, 1.1], [0, 0, 0, 1], useFixedBase=False, flags=pb.URDF_MERGE_FIXED_LINKS
    )

    talos = Talos(path_to_model="~/projects", reduced=False)

    map_joint_idx_to_q_idx = {}
    for j in range(pb.getNumJoints(robot)):
        joint_name = pb.getJointInfo(robot, j)[1]
        map_joint_idx_to_q_idx[j] = talos.get_joint_id(joint_name)

    q = talos.set_and_get_default_pose()

    k = 0
    while True:
        apply_position(q_des=q, j_to_q_idx=map_joint_idx_to_q_idx)
        pb.stepSimulation()
        k += 1
