import numpy as np
from .iksolver import JointConfig, IKSolver, TargetUnreachable

def _get_hand_joint_configs():
    # Configure names
    sections = ('000', '001', '002', '003', 'tip')
    names = [
        ['joint_{}_{}'.format(finger, section) for section in sections]
        for finger in range(5)
    ]
    configs = {}
    pi = np.pi
    for finger in (1, 2, 3, 4):
        configs[names[finger][0]] = [('z', -pi / 6, pi / 3, 1.)]
        configs[names[finger][1]] = [('z', -pi / 6, pi / 3, 1.), ('x', -pi / 8, pi / 8, 1.)]
        configs[names[finger][2]] = [('z', 0, pi / 2, 1.)]
        configs[names[finger][3]] = [('z', 0, pi / 2, 1.)]
    configs[names[0][0]] = [('z', -pi / 3, 0, 1.), ('x', 0, pi / 2, 1.)]
    configs[names[0][1]] = [('z', 0, pi / 2, 1.), ('x', -pi / 2, 0, 1.)]
    configs[names[0][2]] = [('z', 0, pi / 2, 1.)] 
    configs[names[0][3]] = [('z', 0, pi / 2, 1.)] 
    configs['arm_upper'] = [('z', -pi, pi, 1.), ('x', -pi, pi, 1.), ('y', -pi, pi, 1.)]
    configs['arm_lower'] = [('z', 0, pi * 4 / 5, 1.)]
    configs['arm_wrist'] = [('z', -pi / 2, - pi / 5, 1.), ('x', -pi / 5, pi / 5, 1.)]

    return configs


class Hand:
    def __init__(self, scene, name):
        model = scene.get_model(name)
        joints = {joint.name: joint for joint in model.joints}
        basic_configs  = _get_hand_joint_configs()
        joint_configs = []
        for name, joint in joints.items():
            parent = joint.parent
            # TODO

        # self.solver = IKSolver(joint_configs)


class Guitar:
    def __init__(self, scene, name):
        pass

    def get_grid_pos(self, fret, string):
        pass


class Performer:
    def __init__(self, scene):
        self.hands = [Hand(scene, 'hand_left'), Hand(scene, 'hand_right')]
        self.guitar = Guitar(scene, 'guitar')

    def set_sheet(self, sheet, arranger):
        pass

    def sync_to_time(self, time):
        pass
