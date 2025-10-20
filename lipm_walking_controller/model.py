from pathlib import Path

import numpy as np
import pinocchio as pin


def print_joints(model):
    for j_id, j_name in enumerate(model.names):
        print(j_id, j_name, model.joints[j_id].shortname(), model.joints[j_id].nq)


def print_frames(model):
    for i, frame in enumerate(model.frames):
        print(i, frame.name, frame.parent, frame.type)


def set_joint(q, model, joint_name, val):
    jid = model.getJointId(joint_name)
    if jid > 0 and model.joints[jid].nq == 1:
        q[model.joints[jid].idx_q] = val


def q_from_base_and_joints(q, oMb):
    R = oMb.rotation
    p = oMb.translation
    quat = pin.Quaternion(R)  # xyzw
    q_out = q.copy()
    q_out[:3] = p
    q_out[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])
    return q_out


class Talos:
    def __init__(self, path_to_model: Path, reduced=True):
        # Load full model
        path_to_urdf = path_to_model / "talos_data" / "urdf" / "talos_full.urdf"
        full_model, full_col_model, full_vis_model = pin.buildModelsFromUrdf(
            path_to_urdf, str(path_to_model), pin.JointModelFreeFlyer()
        )

        q = pin.neutral(full_model)

        # Position the arms
        # We lock joints of the upper body since there are not meant to move with LIPM model
        set_joint(q, full_model, "leg_left_4_joint", 0.0)
        set_joint(q, full_model, "leg_right_4_joint", 0.0)
        set_joint(q, full_model, "arm_right_4_joint", -1.5)
        set_joint(q, full_model, "arm_left_4_joint", -1.5)

        # We build a reduced model by locking the specificied joints if needed
        self.reduced = reduced
        if self.reduced:
            joints_to_lock = list(range(14, 45))
            self.model, self.geom = pin.buildReducedModel(
                full_model, full_col_model, joints_to_lock, q
            )
            _, self.vis = pin.buildReducedModel(full_model, full_vis_model, joints_to_lock, q)
        else:
            self.model = full_model
            self.geom = full_col_model
            self.vis = full_vis_model

        self.data = self.model.createData()

        upper_v_idx = {}
        for j in self.model.joints:
            if j.nv == 0 or j.idx_q == -1:
                continue

            name = self.model.names[j.id]  # joint name from URDF
            # if upper_rx.match(name):
            start = j.idx_v  # first velocity index for this joint
            upper_v_idx[name] = (start, j.idx_q)

        self.left_foot_id = self.model.getFrameId("left_sole_link")
        self.right_foot_id = self.model.getFrameId("right_sole_link")
        self.torso_id = self.model.getFrameId("torso_1_link")

    def set_and_get_default_pose(self):
        # Initialize reduced model
        q = pin.neutral(self.model)

        if not self.reduced:
            set_joint(q, self.model, "leg_left_4_joint", 0.0)
            set_joint(q, self.model, "leg_right_4_joint", 0.0)
            set_joint(q, self.model, "arm_right_4_joint", -1.5)
            set_joint(q, self.model, "arm_left_4_joint", -1.5)

        # Initialize left leg position
        set_joint(q, self.model, "leg_left_1_joint", 0.0)
        set_joint(q, self.model, "leg_left_2_joint", 0.0)
        set_joint(q, self.model, "leg_left_3_joint", -0.5)
        set_joint(q, self.model, "leg_left_4_joint", 1.0)
        set_joint(q, self.model, "leg_left_5_joint", -0.6)

        # Initialize right leg position
        set_joint(q, self.model, "leg_right_1_joint", 0.0)
        set_joint(q, self.model, "leg_right_2_joint", 0.0)
        set_joint(q, self.model, "leg_right_3_joint", -0.5)
        set_joint(q, self.model, "leg_right_4_joint", 1.0)
        set_joint(q, self.model, "leg_right_5_joint", -0.6)

        # Update position of the model and the data
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        return q

    def get_joint_id(self, name):
        jid = self.model.getJointId(name)

        n_joints = len(self.model.joints)

        return self.model.joints[jid].idx_q if jid < n_joints else None

    def get_locked_joints_idx(self):
        return range(18, 50)
