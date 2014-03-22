from fractions import Fraction

import numpy as np
from scipy.interpolate import interp1d
from raygllib.utils import timeit_context

from .iksolver import JointConfig, IKSolver, FABRIKSolver, TargetUnreachable, apply_angles

__all__ = ['Performer']

def _length(v):
    return np.sqrt(v.dot(v))

def _assert_matrix(matrix):
    assert np.allclose(_length(matrix[0:3, 0]), 1)
    assert np.allclose(_length(matrix[0:3, 1]), 1)
    assert np.allclose(_length(matrix[0:3, 2]), 1)

def _get_hand_joint_configs():
    # Configure names
    sections = ('000', '001', '002', '003', 'tip')
    names = [
        ['finger_{}_{}'.format(finger, section) for section in sections]
        for finger in range(5)
    ]
    configs = {}
    pi = np.pi
    for finger in (1, 2, 3, 4):
        configs[names[finger][0]] = [('z', 0, pi / 80 * finger, 1.)]
        configs[names[finger][1]] = [
            ('z', -pi / 6, pi / 4, 1.), ('x', -pi / 8, pi / 8, 1.)]
        configs[names[finger][2]] = [('z', 0, pi / 2, 1.)]
        configs[names[finger][3]] = [('z', 0, pi / 2, 1.)]
        configs[names[finger][4]] = [('z', -1, 1, 0.)]
    # configs[names[0][0]] = [('z', -pi / 3, 0, 1.), ('x', 0, pi / 2, 1.)]
    configs[names[0][0]] = [('x', 0, pi / 2, 1.)]
    configs[names[0][1]] = [('z', 0, pi / 2, 1.), ('x', -pi / 2, 0, 1.)]
    # configs[names[0][0]] = [('z', -pi, pi, 1.), ('x', -pi, pi, 1.)]
    # configs[names[0][1]] = [('z', -pi, pi, 1.), ('x', -pi, pi, 1.)]
    configs[names[0][2]] = [('z', 0, pi / 2, 1.)] 
    configs[names[0][3]] = [('z', 0, pi / 2, 1.)] 
    configs[names[0][4]] = [('z', -1, 1, 0.)]
    # configs['arm_upper'] = [('z', -pi, pi, 1.), ('x', -pi, pi, 1.), ('y', -pi, pi, 1.)]
    configs['arm_upper'] = [('z', 0, pi / 3, 1.), ('y', -pi, pi, 1.)]
    configs['arm_lower'] = [('z', 0, pi * 4 / 5, 1.), ('x', 0, pi, 1.)]
    configs['arm_wrist'] = [('z', -pi / 2, - pi / 5, 1.)] #, ('x', -pi / 5, pi / 5, 1.)]

    return configs

_SELECTORS = {
    'x': np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.double),
    'y': np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.double),
    'z': np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.double)
}

class SubJoint:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.id = None
        self.axis = None
        self.global_matrix = None

class Hand:
    """
    public attributes:
        subjoints
        fretboard
        init_spaces
        key_subjoint_ids
        events
        intervals
        angle_interp
        arranger
    """
    N_FINGERS = 4
    KEY_SUBJOINTS = [
        'finger_0_tip_z', 'finger_1_tip_z', 'finger_2_tip_z', 'finger_3_tip_z',
        'finger_4_tip_z', 'finger_1_003_z',
    ]

    def __init__(self, scene, name, fretboard):
        self.arranger = None
        self.fretboard = fretboard
        self.scene = scene
        model = scene.get_model(name)
        self._model_joints = model.joints
        # subjoints[joint.name] = [subjoint1, subjoint2, ..]
        self.subjoints = subjoints = {}

        basic_configs = _get_hand_joint_configs()
        # Collect children
        joint_children = {joint.name: [] for joint in model.joints}
        for joint in model.joints:
            if joint.parent:
                joint_children[joint.parent.name].append(joint)

        joint_configs = []

        def build_subjoints(joint, parent, parent_matrix):
            base_matrix = parent_matrix.dot(joint.matrix)
            subjoints[joint.name] = []
            for axis, min_angle, max_angle, weight in basic_configs[joint.name]:
                subjoint = SubJoint(parent, joint.name + '_' + axis)
                subjoint.global_matrix = base_matrix.dot(_SELECTORS[axis])
                subjoint.axis = axis
                subjoints[joint.name].append(subjoint)
                if parent:
                    matrix = np.linalg.solve(parent.global_matrix, subjoint.global_matrix)
                else:
                    matrix = subjoint.global_matrix
                _assert_matrix(matrix)
                joint_configs.append(JointConfig(
                    parent_name=parent.name if parent else None,
                    name=subjoint.name,
                    position=matrix[0:3, 3] / matrix[3, 3],
                    orient=matrix[0:3, 1], axis=matrix[0:3, 2],
                    min_angle=min_angle, max_angle=max_angle, range_weight=weight,
                ))
                parent = subjoint

            for child in joint_children[joint.name]:
                build_subjoints(child, parent, base_matrix)

        root_joint = next(joint for joint in model.joints if joint.parent is None)
        build_subjoints(root_joint, None, np.eye(4, dtype=np.double))

        self.solver = FABRIKSolver(joint_configs)
        # self.solver = IKSolver(joint_configs)
        # Get id for each subjoint
        for subjoints1 in subjoints.values():
            for subjoint in subjoints1:
                subjoint.id = self.solver.get_joint_id(subjoint.name)
        self.init_spaces = self.solver.get_init_spaces()
        self._flat_subjoints = [
            subjoint for subjoint1 in self.subjoints.values() for subjoint in subjoint1]
        self._flat_subjoints.sort(key=lambda subjoint: subjoint.id)

    def start_arranging(self):
        self.key_subjoint_ids = [
            self.solver.get_joint_id(subjoint_name) 
            for subjoint_name in self.KEY_SUBJOINTS
        ]
        # events[i] = [(t0, pos0), (t1, pos1), ... ]
        self.events = [[] for i in range(len(self.KEY_SUBJOINTS))]
        # intervals[i] = [(t_start, t_end), ...]
        self.intervals = [[] for i in range(len(self.KEY_SUBJOINTS))]
        for i, subjoint_id in enumerate(self.key_subjoint_ids):
            matrix = self.solver.get_global_matrix(subjoint_id)
            t1 = Fraction(-2)
            t2 = Fraction(-1)
            self.events[i].append((t1, matrix[0:3, 3] / matrix[3, 3]))
            self.events[i].append((t2, matrix[0:3, 3] / matrix[3, 3]))
            self.intervals[i].append((t1, t2))
        self.set_init_gesture()

    def set_init_gesture(self):
        pass

    def has_event_at(self, id, time): 
        for t_start, t_end in self.intervals[id]:
            if t_start <= time <= t_end:
                return True
        return False

    def end_arranging(self):
        n_subjoints = len(self.init_spaces)
        key_times = set()
        for i in range(self.N_FINGERS):
            self.events[i].sort()
            for t, pos in self.events[i]:
                key_times.add(t)
            self.intervals[i].sort()

        # Replace events with its interpolation
        for i in range(len(self.KEY_SUBJOINTS)):
            ts = np.array([t for t, _ in self.events[i]], dtype=np.double)
            ps = np.array([p for _, p in self.events[i]], dtype=np.double)
            self.events[i] = interp1d(ts, ps.T)

        key_times = list(key_times)
        key_times.sort()
        # key_angles[i, j] is the angle of subjoint i at key_times[j]
        key_angles = np.zeros((n_subjoints, len(key_times)), dtype=np.double)
        solver = self.solver
        # progress.info = 'Generating for key frames'
        # progress.total = len(key_times)
        for j, time in enumerate(key_times):
            # Set all targets at this time.
            print(j, len(key_times))
            solver.clear()
            for i in range(len(self.KEY_SUBJOINTS)):
                if self.has_event_at(i, time):
                    solver.set_target_pos(
                        self.key_subjoint_ids[i], self.events[i](float(time)))
                    # print(time, self.KEY_SUBJOINTS[i], self.key_subjoint_ids[i],
                    #     self.events[i](time))
            # solver.add_constraint_pos_ref(
            #     solver.get_joint_id('arm_lower_z'),
            #     2.,
            #     self.scene.get_empty_node_pos('elbow_left')
            # )
            # Solver for all subjoint angles.
            with timeit_context('solve'):
                try:
                    solver.solve(max_iteration=30)
                except TargetUnreachable:
                    pass
                # solver._dump_state()
                # raise
            # Store the angles for later interpolation.
            for i in range(n_subjoints):
                key_angles[i, j] = solver.get_angle(i)
            # progress.current = j + 1

        # del self.events
        # del self.intervals
        self.angle_interp = interp1d(
            np.array(key_times, dtype=np.double), key_angles,
            bounds_error=False, fill_value=key_angles[:, -1])

    def update_model_joints(self, time):
        subjoint_angles = self.angle_interp(float(time))

        matrices = apply_angles(self.init_spaces, subjoint_angles)
        for subjoint in self._flat_subjoints:
            if subjoint.parent:
                matrices[subjoint.id] = \
                    matrices[subjoint.parent.id].dot(matrices[subjoint.id])
        # matrices = self.init_spaces
        for joint in self._model_joints:
            last_subjoint = self.subjoints[joint.name][-1]
            selector = _SELECTORS[last_subjoint.axis]
            joint.matrix = matrices[last_subjoint.id].dot(selector.T)
            _assert_matrix(joint.matrix)

        for joint in reversed(self._model_joints):
            if joint.parent:
                joint.matrix = np.linalg.solve(joint.parent.matrix, joint.matrix)


    def arrange_interval(self, t1, t2, frame1, frame2, state1, state2):
        pass


class LeftHand(Hand):
    N_FINGERS = 5
    KEY_SUBJOINTS = [
        'finger_0_003_z', 'finger_0_tip_z', 'finger_1_tip_z', 'finger_2_tip_z', 'finger_3_tip_z',
        'finger_4_tip_z', 'finger_1_003_z',
    ]

    @staticmethod
    def is_pressed(state, finger):
        return state.strings[finger] != -1

    SUBJOINT_ID_MAP = {
        0: KEY_SUBJOINTS.index('finger_1_tip_z'),
        1: KEY_SUBJOINTS.index('finger_2_tip_z'),
        2: KEY_SUBJOINTS.index('finger_3_tip_z'),
        3: KEY_SUBJOINTS.index('finger_4_tip_z'),
        (4, 0): KEY_SUBJOINTS.index('finger_0_tip_z'),
        (4, 1): KEY_SUBJOINTS.index('finger_0_003_z'),
    }

    def start_arranging(self):
        super().start_arranging()
        # Calculate index finger length
        matrix = self.subjoints['finger_1_001'][0].global_matrix
        p1 = matrix[0:3, 3] / matrix[3, 3]
        matrix = self.subjoints['finger_1_tip'][0].global_matrix
        p2 = matrix[0:3, 3] / matrix[3, 3]
        self._index_length = _length(p1 - p2)
        # Calculate thumb length
        matrix = self.subjoints['finger_0_tip'][0].global_matrix
        p1 = matrix[0:3, 3] / matrix[3, 3]
        matrix = self.subjoints['finger_0_003'][0].global_matrix
        p2 = matrix[0:3, 3] / matrix[3, 3]
        self._thumb_length = _length(p1 - p2)

    def set_init_gesture(self):
        solver = self.solver
        solver.clear()
        solver.set_target_pos_by_name('arm_lower_z', self.scene.get_empty_node_pos('elbow_left'))
        try:
            solver.solve()
        except TargetUnreachable:
            pass

    def arrange_interval(self, t1, t2, frame1, frame2, state1, state2):
        t1 += Fraction(1, 10000)
        t2 -= Fraction(1, 10000)
        get_grid_pos = self.fretboard.get_grid_pos
        fretboard = self.fretboard
        above = fretboard.matrix.dot([0, 0, fretboard.neck_height / 2, 0])[:3]
        below = fretboard.matrix.dot([0, 0, - fretboard.neck_height, 0])[:3]
        thumb_bar = fretboard.matrix.dot([self._thumb_length, 0, 0, 0])[:3]

        def get_grid(state, finger):
            return (state.frets[finger], state.strings[finger])

        matched1, _, _ = state1.match(self.arranger.fretboard, frame1)
        # matched2, _, _ = state2.match(self.arranger.fretboard, frame2)
        MIN_RELEASE_TIME = 0.1
        MIN_PRESS_TIME = 0.1

        def get_end_time(matched, finger):
            if matched[finger]:
                if isinstance(matched[finger], list):
                    end_time = max(note.end for _, note in matched[finger])
                else:
                    end_time = matched[finger][1].end
                end_time = min(t2, end_time)
            else:
                end_time = t2
            return end_time

        for finger in (0, 1, 2, 3):
            p1 = self.is_pressed(state1, finger)
            p2 = self.is_pressed(state2, finger)
            if p1:
                grid_pos1 = get_grid_pos(*get_grid(state1, finger))
            if p2:
                grid_pos2 = get_grid_pos(*get_grid(state2, finger))
            subjoint_id = self.SUBJOINT_ID_MAP[finger]

            if p1 and p2:
                if get_grid(state1, finger) != get_grid(state2, finger):
                    # Jump
                    self.events[subjoint_id].append((t1, grid_pos1))
                    self.events[subjoint_id].append(
                        ((t1 + t2) / 2, (grid_pos1 + grid_pos2) / 2 + above))
                    self.events[subjoint_id].append((t2, grid_pos2))
                    self.intervals[subjoint_id].append((t1, t2))
            elif p1 and not p2:
                # Release
                end_time = max(
                    t1, min(get_end_time(matched1, finger), t2 - MIN_RELEASE_TIME))
                self.events[subjoint_id].append((end_time, grid_pos1))
                self.events[subjoint_id].append((t2, grid_pos1 + above))
                self.intervals[subjoint_id].append((end_time, t2))
            elif not p1 and p2:
                # Press
                start_time = max(t1, t2 - MIN_PRESS_TIME)
                self.events[subjoint_id].append((start_time, grid_pos2 + above))
                self.events[subjoint_id].append((t2, grid_pos2))
                self.intervals[subjoint_id].append((start_time, t2))

        subjoint_id = self.SUBJOINT_ID_MAP[4, 0]
        for t, state in ((t1, state1), (t2, state2)):
            thumb_pos = get_grid_pos(state.frets[0], 0) + below * 1.2
            self.events[subjoint_id].append((t, thumb_pos))
        self.intervals[subjoint_id].append((t1, t2))

        subjoint_id = self.SUBJOINT_ID_MAP[4, 1]
        for t, state in ((t1, state1), (t2, state2)):
            thumb_pos = get_grid_pos(state.frets[0], 0) + below * 1.5 - thumb_bar
            self.events[subjoint_id].append((t, thumb_pos))
        self.intervals[subjoint_id].append((t1, t2))


class RightHand(Hand):
    N_FINGERS = 4
    KEY_SUBJOINTS = [
        'finger_0_tip_z', 'finger_1_tip_z', 'finger_2_tip_z', 'finger_3_tip_z',
    ]

    def arrange_interval(self, t1, t2, frame1, frame2, state1, state2):
        pass

class Fretboard:
    """
    public attributes:
        matrix
        neck_height
    """
    # private attributes:
    #     _grid_pos

    MAX_FRET = 17
    N_STRINGS = 6

    def __init__(self, scene):
        p1 = scene.get_empty_node_pos('corner_fret0_string1')
        p2 = scene.get_empty_node_pos('corner_fret0_string6')
        p3 = scene.get_empty_node_pos('corner_fret12_string1')
        p4 = scene.get_empty_node_pos('corner_fret12_string6')
        p5 = scene.get_empty_node_pos('height')
        x = (p2 - p1)
        x /= _length(x)
        y = (p3 + p4 - (p2 + p1)) / 2
        L = _length(y)
        y /= L
        L *= 2
        matrix = np.eye(4, dtype=np.double)
        matrix[0:3, 0] = x
        matrix[0:3, 1] = y
        matrix[0:3, 2] = np.cross(x, y)
        matrix[0:3, 3] = (p1 + p2) / 2
        self.neck_height = _length((p1 + p2) / 2 - p5)
        self._grid_pos = np.zeros((self.MAX_FRET + 1, self.N_STRINGS, 3), dtype=np.double)
        ys = (1 - 2 ** (np.arange(0, self.MAX_FRET + 1) * (-1 / 12))) * L
        top_len = _length(p2 - p1)
        bottom_len = _length(p4 - p3)
        for string in range(self.N_STRINGS):
            k = string / (self.N_STRINGS - 1)
            x_top = 0 * (1 - k) + top_len * k
            x_bottom = 0 * (1 - k) + bottom_len * k
            for fret in range(self.MAX_FRET + 1):
                if fret > 0:
                    k = 0.7
                    y = ys[fret - 1] * (1 - k) + ys[fret] * k
                else:
                    y = ys[fret]
                k = y / L
                x = x_top * (1 - k) + x_bottom * k
                pos = matrix.dot([x, y, 0, 1])
                self._grid_pos[fret, string] = pos[0:3] / pos[3]
        self.matrix = matrix

    def get_grid_pos(self, fret, string):
        """
        fret: 0 <= fret <= self.MAX_FRET
        string: 0 <= string < self.N_STRINGS
        """
        return self._grid_pos[fret, string]


class Performer:
    def __init__(self, scene):
        self.fretboard = Fretboard(scene)
        self.hands = [
            LeftHand(scene, 'hand_left', self.fretboard),
            RightHand(scene, 'hand_right', self.fretboard),
        ]
        self._arranged = False

    def set_sheet(self, sheet, arranger):
        ts = arranger.timePoints
        frames = [arranger.frames[t] for t in ts]
        states = arranger.states

        for hand in self.hands:
            hand.arranger = arranger
            hand.start_arranging()
        for i in range(len(ts) - 1):
            for hand in self.hands:
                hand.arrange_interval(ts[i], ts[i + 1], frames[i], frames[i + 1],
                    states[i], states[i + 1])
        for hand in self.hands:
            hand.end_arranging()
        # progress.info = 'Animation arranged.'

        self._arranged = True

    def sync_to_time(self, time):
        if not self._arranged:
            return
        for hand in self.hands:
            hand.update_model_joints(time)
