import meshcat
import meshcat.transformations as tf

import numpy as np
from pinocchio.visualize import MeshcatVisualizer


class Visualizer:
    def __init__(self, robot_model, open_viewer=True):
        self.viz = MeshcatVisualizer(robot_model.model, robot_model.geom, robot_model.vis)
        self.viz.initViewer(open=open_viewer)
        self.viz.loadViewerModel()

        self.remove_grid()
        self.set_background_color([0.10, 0.10, 0.10], [0.02, 0.02, 0.02])

        size = np.array([20.0, 20.0, 0.01])  # X,Y size, thickness
        floor = meshcat.geometry.Box(size)
        mat = meshcat.geometry.MeshPhongMaterial(color=0x888888, shininess=10.0)

        left_foot_tf = robot_model.data.oMf[robot_model.left_foot_id].copy()

        self.viz.viewer["scene/ground"].set_object(floor, mat)
        self.viz.viewer["scene/ground"].set_transform(
            tf.translation_matrix([0, 0, left_foot_tf.translation[2] - size[2] / 2])
        )

    def remove_grid(self):
        self.viz.viewer["/Grid"].set_property("visible", False)

    def set_background_color(self, top_color, bottom_color):
        self.viz.viewer["/Background"].set_property("top_color", top_color)
        self.viz.viewer["/Background"].set_property("bottom_color", bottom_color)

    def update_display(self, q):
        self.viz.display(q)

    def point_camera_at_robot(self, robot_model, camera_offset):
        target = robot_model.data.oMf[robot_model.torso_id].translation
        target[1] = 0.0
        self.viz.setCameraTarget(target)
        self.viz.setCameraPosition(target + camera_offset)
