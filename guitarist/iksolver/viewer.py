from raygllib import ui
from raygllib.viewer import Viewer
from raygllib.model import Scene
from . import JointConfig, IKSolver
import numpy as np

__all__ = ['Controller', 'IKViewer']

class Controller:
    def __init__(self, scene, model_name, angle_ranges={}):
        armatured_model = scene.get_model(model_name)
        joint_configs = []
        pi = np.pi
        for joint in armatured_model.joints:
            pos = joint.matrix[0:3, 3] / joint.matrix[3, 3]
            joint_configs.append(JointConfig(
                parent_name = joint.parent.name if joint.parent else None,
                name = joint.name,
                position = pos,
                orient = joint.matrix[0:3, 1],
                axis = joint.matrix[0:3, 2],
                min_angle = angle_ranges[joint.name][0]\
                    if joint.name in angle_ranges else -pi,
                max_angle = angle_ranges[joint.name][1]\
                    if joint.name in angle_ranges else pi,
                range_weight = 1.,
            ))
        self.solver = IKSolver(joint_configs)
        self.joints = armatured_model.joints

    def update_model_joints(self):
        solver = self.solver
        for joint in self.joints:
            joint.angle = solver.get_angle_by_name(joint.name)


class IKViewer(Viewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controllers = []

    def add_controller(self, controller):
        self._controllers.append(controller)

    def update(self, dt):
        for controller in self._controllers:
            controller.update_model_joints()
        super().update(dt)
