import unittest
import numpy as np
import os
from raygllib.utils import timeit_context

from guitarist.iksolver import JointConfig, IKSolver, TargetUnreachable
from guitarist.iksolver.viewer import IKViewer, Controller

def get_resource_path(*sub_paths):
    return os.path.join(os.path.dirname(__file__), *sub_paths)

class TestIKSolver(unittest.TestCase):
    def test_solve_3_joints(self):
        configs = [
            # parent_name, name, position, orient, axis, min_angle, max_angle, range_weight
            JointConfig(
                None, 'joint-1', (0, 0, 0), (1, 0, 0), (0, 0, -1), -np.pi/2., np.pi/2, 1.),
            JointConfig(
                'joint-1', 'joint-2', (0, 2, 0), (0, 1, 0), (0, 0, 1), -np.pi/2., np.pi/2, 1.),
            JointConfig(
                'joint-2', 'joint-3', (0, 2, 0), (0, 1, 0), (0, 0, 1), 0., np.pi/2, 1.),
        ]
        solver = IKSolver(configs)
        joint2_id = solver.get_joint_id('joint-2')
        joint3_id = solver.get_joint_id('joint-3')
        solver.set_target_pos(joint3_id, (2, 2, 0))
        with timeit_context('solve'):
            solver.solve()
        assert np.allclose(solver.get_angle(joint2_id), -np.pi / 2)

    @unittest.skip
    def test_solve_5_joints(self):
        configs = [
            # parent_name, name, position, orient, axis, min_angle, max_angle, range_weight
            JointConfig(
                None, 'joint-1', (0, 0, 0), (1, 0, 0), (0, 0, -1), -np.pi/2., np.pi/2, 1.),

            JointConfig(
                'joint-1', 'joint-2', (0, 2, 0), (0, 1, 0), (0, 0, 1), -np.pi/2., np.pi/2, 1.),
            JointConfig(
                'joint-2', 'joint-3', (0, 2, 0), (0, 1, 0), (0, 0, 1), 0., np.pi/2, 1.),

            JointConfig(
                'joint-1', 'joint-4', (0, 0, 2), (0, 0, 1), (0, -1, 0), -np.pi/2., np.pi/2, 1.),
            JointConfig(
                'joint-4', 'joint-5', (0, 0, 2), (0, 0, 1), (0, -1, 1), 0., np.pi/2, 1.),
        ]
        solver = IKSolver(configs)
        solver.set_target_pos(solver.get_joint_id('joint-3'), (2, 2, 0))
        solver.set_target_pos(solver.get_joint_id('joint-5'), (0, 2, -2))
        with timeit_context('solve'):
            solver.solve()
        assert np.allclose(solver.get_angle(solver.get_joint_id('joint-2')), np.pi / 2)
        assert np.allclose(solver.get_angle(solver.get_joint_id('joint-4')), np.pi / 2)

    def test_update_spaces_random(self):
        spaces = np.zeros((3, 4, 4), dtype=np.double)
        from guitarist.iksolver.iksolver import _update_spaces
        for t in range(100):
            spaces[0] = spaces[1] = spaces[2] = np.eye(4)
            angles = np.random.rand(3)
            _update_spaces(spaces, angles)

            for i in range(len(spaces)):
                space = spaces[i]
                x = space[0:3, 0]
                y = space[0:3, 1]
                z = space[0:3, 2]
                assert np.allclose(np.dot(x, y), 0, 1e-8, 1e-8)
                assert np.allclose(np.dot(y, z), 0, 1e-8, 1e-8)
                assert np.allclose(np.dot(z, x), 0, 1e-8, 1e-8)
                assert np.allclose(np.dot(x, x), 1, 1e-8, 1e-8)
                assert np.allclose(np.dot(y, y), 1, 1e-8, 1e-8)
                assert np.allclose(np.dot(z, z), 1, 1e-8, 1e-8)

    def test_update_spaces_1(self):
        from guitarist.iksolver.iksolver import _update_spaces
        spaces = np.zeros((1, 4, 4), dtype=np.double)
        spaces[0] = [
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        angles = [np.pi / 2]
        _update_spaces(spaces, angles)
        assert np.allclose(spaces[0], [
            [0, -1, 0, 2],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], 1e-5, 1e-8)

    def test_controller(self):
        window = IKViewer()
        window.load_scene(get_resource_path('models', 'segments.dae'))
        controller1 = Controller(window.scene, 'pipe-1')
        target = window.scene.get_model('Icosphere')
        targetPos = target.matrix[0:3, 3] / target.matrix[3, 3]
        print(target.matrix)
        controller1.solver.set_target_pos_by_name('joint-3', targetPos)
        controller1.solver.solve()
        controller1.update_model_joints()
        # window.add_controller(controller1)
        window.start()

    def test_controller_long(self):
        window = IKViewer()
        window.load_scene(get_resource_path('models', 'segments-long.dae'))
        angle_ranges = {'joint_{:03d}'.format(i): (-np.pi / 4, np.pi / 4)
            for i in range(1, 20)}
        controller1 = Controller(window.scene, 'pipe-1', angle_ranges)
        target = window.scene.get_model('target')
        targetPos = target.matrix[0:3, 3] / target.matrix[3, 3]
        controller1.solver.set_target_pos_by_name('tip', targetPos)
        try:
            controller1.solver.solve()
        except TargetUnreachable:
            pass
        controller1.update_model_joints()
        # window.add_controller(controller1)
        window.start()

    def test_controller_hand(self):
        window = IKViewer()
        window.load_scene(get_resource_path('models', 'hand3.dae'))
        angle_ranges = {}
        for finger in '01234':
            section = '000'
            name = 'finger_{}_{}'.format(finger, section)
            angle_ranges[name] = (-np.pi / 6, np.pi / 6)
            for section in ('001', '002', '003'):
                name = 'finger_{}_{}'.format(finger, section)
                angle_ranges[name] = (0, np.pi / 1.5)
        # angle_ranges = {'joint_{:03d}'.format(i): (-np.pi / 4, np.pi / 4)
        #     for i in range(1, 20)}
        controller1 = Controller(window.scene, 'hand', angle_ranges)
        #
        target = window.scene.get_model('target')
        targetPos = target.matrix[0:3, 3] / target.matrix[3, 3]
        controller1.solver.set_target_pos_by_name(
            'finger_1_tip', targetPos)
        #
        target = window.scene.get_model('target_001')
        targetPos = target.matrix[0:3, 3] / target.matrix[3, 3]
        controller1.solver.set_target_pos_by_name(
            'finger_2_tip', targetPos)
        #
        target = window.scene.get_model('target_002')
        targetPos = target.matrix[0:3, 3] / target.matrix[3, 3]
        controller1.solver.set_target_pos_by_name(
            'finger_3_tip', targetPos)
        #
        target = window.scene.get_model('target_003')
        targetPos = target.matrix[0:3, 3] / target.matrix[3, 3]
        controller1.solver.set_target_pos_by_name(
            'finger_0_tip', targetPos)
        try:
            with timeit_context('solve'):
                controller1.solver.solve()
        except TargetUnreachable:
            pass
        controller1.update_model_joints()
        # window.add_controller(controller1)
        window.start()

if __name__ == '__main__':
    import crash_on_ipy
    TestIKSolver().test_controller_hand()
