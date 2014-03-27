"""
Inverse Kinematic Solver, supporting 3 kinds of constraints.
"""

from collections import namedtuple
import numpy as np
# import pyximport; pyximport.install()
# from ._iksolver import calc_jacobian_matrix

__all__ = [
    'JointConfig', 'IKSolver', 'TargetUnreachable', 'FABRIKSolver',
    'apply_angles',
]

class TargetUnreachable(Exception):
    pass

JointConfig = namedtuple('JointConfig', [
    'parent_name',
    'name',
    'position',
    'orient',
    'axis',
    'min_angle',
    'max_angle',
    'range_weight',
])

Constraint = namedtuple('Constraint', [
    'type',
    'size',  # Number of constrainted variables.
    'weight',
    'joint_id',
    'args',
])

class ConstraintType:
    # args = (min, max)
    ANGLE_RANGE = 0
    # args = (ref_angle,)
    ANGLE_REF = 1
    # args = (ref_pos_x, ref_pos_y, ref_pos_z)
    POS_REF = 2

dot = np.dot

# Thresholds
STOP_THRESHOLD = .002
STOP_THRESHOLD_LOW = .01
ERROR_CLAMP_VALUE = .05
REF_POS_CLAMP = .01
REF_POS_MIN_DISTANCE = .005

def _normalized(p):
    # p = np.array(p, dtype=np.double)
    length = p.dot(p)
    if length < 1e-10:
        return p
    return p / np.sqrt(length)

def _length(p):
    return np.sqrt(p.dot(p))

def _sort_configs(configs):
    " Sort configs by preorder traversal. "
    children = {config.name: [] for config in configs}
    root_candidates = {config.name for config in configs}
    for config in configs:
        if config.parent_name is not None:
            children[config.parent_name].append(config.name)
            root_candidates.remove(config.name)
    root_name = root_candidates.pop()
    names = []

    def preorder_traverse(name):
        names.append(name)
        for child_name in children[name]:
            preorder_traverse(child_name)

    preorder_traverse(root_name)
    configs.sort(key=lambda config: names.index(config.name))

def _calc_relation_matrix(configs):
    " In a relation matrix, entry (i, j) = 1 means item i is the ancestor of item j. "
    name_to_id = {config.name: id for id, config in enumerate(configs)}
    n = len(configs)
    matrix = np.zeros((n, n), dtype=np.bool8)
    for j in range(n):
        name = configs[j].parent_name
        while name is not None:
            i = name_to_id[name]
            matrix[i, j] = True
            name = configs[i].parent_name
    return matrix

def _calc_spaces(configs):
    """
    Calculate joint space matrices from config.
    return: An array with shape (len(configs), 4, 4)

    A space matrix layouts like this:
        [x y z p]
        [0 0 0 1]
    where y is the orientation, x is the up direction, z is the axis,
    p is the position. All of x, y, z is normalized.
    """
    spaces = np.zeros((len(configs), 4, 4), dtype=np.double)
    for i, config in enumerate(configs):
        spaces[i] = np.eye(4)
        # orientation as the first
        spaces[i, 0:3, 1] = _normalized(np.array(config.orient, dtype=np.double))
        spaces[i, 0:3, 2] = _normalized(np.array(config.axis, dtype=np.double))
        spaces[i, 0:3, 0] = np.cross(spaces[i, 0:3, 1], spaces[i, 0:3, 2])
        spaces[i, 0:3, 3] = config.position
    return spaces

def apply_angles(spaces, angles):
    " Update spaces by angles. "
    n = len(angles)
    cos = np.cos(angles)
    sin = np.sin(angles)
    R = np.zeros((4, n), dtype=np.double)
    R[0] = R[3] = cos; R[1] = -sin; R[2] = sin
    R = R.T.reshape((n, 2, 2))
    spaces = spaces.copy()
    # spaces[:, 0:3, (0, 1)] = spaces[:, 0:3, (0, 1)].dot(R)
    for i in range(len(angles)):
        slice_ = (i, slice(0, 3), slice(0, 2))
        spaces[slice_] = spaces[slice_].dot(R[i])
    return spaces


class IKSolver:
    def __init__(self, configs):
        """
        configs: A list of JointConfig.
        """
        _sort_configs(configs)
        self.n_joints = len(configs)
        self.configs = configs

        # Aux members
        self._name_to_id = {config.name: id for id, config in enumerate(configs)}
        self._parent_id = [
            self._name_to_id.get(config.parent_name, None) for config in configs]
        self._is_ancestor_of = _calc_relation_matrix(configs)
        # Current joint angles.
        self._angles = np.array([0 for config in configs], dtype=np.double)
        # Current joint positions.
        self._spaces_init = _calc_spaces(configs)
        self._update_spaces()
        # Each item of _targets is a (joint_id, joint_pos) tuple.
        self._targets = []
        # Will be initialize after calling _prepare_targets. Both _target_positions 
        # and _positions is flattened. And their length is (n_targets * 3)
        self.n_targets = 0
        self._target_positions = None
        self._positions = None
        # Constraints
        self._constraints = []

    def reset_angles(self):
        for i in range(self.n_joints):
            self._angles[i] = 0.
        self._update_spaces()

    def get_joint_id(self, name):
        return self._name_to_id[name]

    def add_constraint_angle_ref(self, joint_id, weight, angle):
        constraint = Constraint(
            type=ConstraintType.ANGLE_REF,
            size=1,
            weight=weight,
            joint_id=joint_id,
            args=(angle,),
        )
        self._constraints.append(constraint)

    def add_constraint_pos_ref(self, joint_id, weight, pos):
        constraint = Constraint(
            type=ConstraintType.POS_REF,
            size=3,
            weight=weight,
            joint_id=joint_id,
            args=tuple(pos),
        )
        self._constraints.append(constraint)

    def set_target_pos(self, joint_id, pos):
        self._targets.append((joint_id, pos))

    def set_target_pos_by_name(self, joint_name, pos):
        self._targets.append((self._name_to_id[joint_name], pos))

    # @profile
    def solve(self, max_iteration=500):
        self._update_spaces()
        self._prepare_targets()
        self._prepare_constraints()

        if len(self._target_positions) == 0:
            return

        ENABLE_LOW = True
        STEP_LENGTH = 0.2 * self.n_joints
        DAMPING = 10

        # Error between target and current position
        error = STOP_THRESHOLD + 1
        error_low = STOP_THRESHOLD_LOW + 1
        iter_count = 0
        while not (np.alltrue(error < STOP_THRESHOLD) and \
                np.alltrue(error_low < STOP_THRESHOLD_LOW)):
            # print('<<<<<<<<<<<<<<<<<< new iter<<<<<<<<<<<<<<<<<<<<<', iter_count) 
            error = self._clamp_error(self._target_positions - self._positions)
            # print('error', error)
            # Calculate delta angle.
            J = self._calc_jacobian_matrix()
            # J = calc_jacobian_matrix(self)
            Jp = np.linalg.pinv(J)
            delta_angle = dot(Jp, error)
            # Clamp step
            angle_step = _length(delta_angle)
            if angle_step > STEP_LENGTH:
                delta_angle *= STEP_LENGTH / angle_step
            if ENABLE_LOW:
                T = np.eye(Jp.shape[0]) - dot(Jp, J)
                # Some constraint may be resolved so that they will not be included
                # in constraint_ids.
                error_low, constraint_ids = self._calc_error_low()
                # print('error_low', error_low)
                # print(len(error_low), len(constraint_ids), constraint_ids)
                # for constraint_id in constraint_ids:
                #     constraint = self._constraints[constraint_id]
                #     print(self.configs[constraint.joint_id].name, self._constraints[constraint_id], self._angles[constraint.joint_id])
                if constraint_ids:
                    JL = self._calc_jacobian_matrix_low(constraint_ids)
                    d_error_low = error_low - dot(JL, delta_angle)
                    # Some magic math operations.
                    S = dot(JL, T)
                    W = dot(S, S.T) + DAMPING * np.eye(S.shape[0])
                    y = dot(S.T, np.linalg.solve(W, d_error_low))
                    delta_angle += dot(T, y)

                # Clamp step
                angle_step = _length(delta_angle)
                if angle_step > STEP_LENGTH:
                    delta_angle *= STEP_LENGTH / angle_step
            else:
                error_low = 0
            self._angles += delta_angle
            self._update_spaces()
            self._update_positions()
            # self._dump_state()
            iter_count += 1
            if iter_count > max_iteration:
                raise TargetUnreachable()
        print('iter_count', iter_count)

    def clear(self):
        self._targets.clear()
        self._constraints.clear()

    def _clamp_error(self, error):
        for i in range(0, self.n_targets * 3, 3):
            w = error[i: i + 3]
            w_len = _length(w)
            if w_len > ERROR_CLAMP_VALUE:
                error[i: i + 3] = w * (ERROR_CLAMP_VALUE / w_len)
        return error

    def get_angle(self, joint_id):
        return self._angles[joint_id]

    def get_angle_by_name(self, joint_name):
        return self._angles[self._name_to_id[joint_name]]

    def get_global_matrix(self, joint_id):
        return self._spaces[joint_id]

    def get_global_matrix_by_name(self, joint_name):
        return self._spaces[self._name_to_id[joint_name]]

    def get_init_spaces(self):
        " return: Initial local space matrices for each joint. "
        return self._spaces_init.copy()

    def _calc_jacobian_matrix(self):
        """
        Calculate a m x n Jacobian matrix, where m is the number of target joints multiply
        by 3, and n is the number of joints(which equals to number of angles).

        J(i, j) = Deriv(position(i), angle(j)) = axis(j) x (position(i) - position(j))
        """
        m = 3 * self.n_targets
        n = self.n_joints
        J = np.zeros((m, n), dtype=np.double)
        is_ancestor_of = self._is_ancestor_of
        positions = self._positions
        spaces = self._spaces
        for i in range(0, m, 3):
            target_id = self._targets[i // 3][0]
            for j in range(n):
                if is_ancestor_of[j, target_id]:
                    J[i:i + 3, j] = np.cross(
                        spaces[j, 0:3, 2],
                        positions[i:i + 3] - spaces[j, 0:3, 3],
                    )
        return J

    def _calc_jacobian_matrix_low(self, constraint_ids):
        """
        There 3 kinds of constraints, which are:
            Angle range
            Angle reference
            Position reference

        The return result is the low priority (constraint) Jacobian matrix.
        JL(i, j) = Deriv(metric[i], angle(j))
        """
        m = sum(self._constraints[i].size for i in constraint_ids)
        n = self.n_joints
        JL = np.zeros((m, n), dtype=np.double)
        i = 0
        get_joint_pos = self._get_joint_global_pos3
        for id in constraint_ids:
            c = self._constraints[id]
            if c.type == ConstraintType.ANGLE_RANGE:
                JL[i, c.joint_id] = c.weight
            elif c.type == ConstraintType.ANGLE_REF:
                JL[i, c.joint_id] = c.weight
            elif c.type == ConstraintType.POS_REF:
                for j in range(self.n_joints):
                    if self._is_ancestor_of[j, c.joint_id]:
                        deriv = c.weight * np.cross(
                            self._get_joint_axis3(j),
                            get_joint_pos(c.joint_id) - get_joint_pos(j))
                        JL[i:i + 3, j] = deriv
            i += c.size
        return JL

    def _calc_error_low(self):
        """
        return: (error_low, constraint_ids)
        constraint_ids tells all active constraints.
        """
        error_low = []
        constraint_ids = []
        # DAMPING = 0.1
        DAMPING = 1.
        for i, c in enumerate(self._constraints):
            if c.type == ConstraintType.ANGLE_RANGE:
                angle = self._angles[c.joint_id]
                min_angle, max_angle = c.args
                if angle < min_angle:
                    error_low.append(c.weight * DAMPING * (min_angle - angle))
                    constraint_ids.append(i)
                elif angle > max_angle:
                    error_low.append(c.weight * DAMPING * (max_angle - angle))
                    constraint_ids.append(i)
            elif c.type == ConstraintType.ANGLE_REF:
                angle = self._angle[c.joint_id]
                ref_angle = c.args[0]
                if abs(angle - ref_angle) > 1e-3:
                    error_low.append(c.weight * DAMPING * (ref_angle - angle))
                    constraint_ids.append(i)
            elif c.type == ConstraintType.POS_REF:
                # pos = self._positions[3 * c.joint_id:3 * c.joint_id + 3]
                pos = self._spaces[c.joint_id, 0:3, 3]
                ref_pos = c.args
                d = ref_pos - pos
                d_len = _length(d)
                if d_len > REF_POS_MIN_DISTANCE:
                    if d_len > REF_POS_CLAMP:
                        d *= REF_POS_CLAMP / d_len
                    error_low.extend(d)
                    constraint_ids.append(i)
        error_low = np.array(error_low, dtype=np.double)
        return error_low, constraint_ids

    def _get_joint_axis3(self, joint_id):
        return self._spaces[joint_id, 0:3, 2]

    def _get_joint_global_pos3(self, joint_id):
        matrix = self._spaces[joint_id]
        return matrix[0:3, 3] / matrix[3, 3]

    def _update_spaces(self):
        " Update joint positions after some joints' angle changed. "
        self._spaces = spaces = apply_angles(self._spaces_init, self._angles)
        for i in range(self.n_joints):
            if self._parent_id[i] is not None:
                spaces[i] = spaces[self._parent_id[i]].dot(spaces[i])

    def _update_positions(self):
        for i, (joint_id, target_pos) in enumerate(self._targets):
            self._positions[i * 3:i * 3 + 3] = self._get_joint_global_pos3(joint_id)

    def _dump_state(self):
        # print('===================new dump==================')
        print('angles:', self._angles, self._angles * 180 / 3.14)
        print('positions:')
        for i, (target_id, _) in enumerate(self._targets):
            slice_ = slice(i * 3, i * 3 + 3)
            print('  {}: current:{} target:{}'.format(
                self.configs[target_id].name,
                tuple(self._positions[slice_]), tuple(self._target_positions[slice_])))
        print('spaces:')
        for i in range(self.n_joints):
            print(self.configs[i].name)
            print(self._spaces[i])

    def _prepare_targets(self):
        self.n_targets = len(self._targets)
        self._target_positions = np.zeros((self.n_targets, 3), dtype=np.double)
        self._positions = np.zeros_like(self._target_positions)
        for i, (joint_id, target_pos) in enumerate(self._targets):
            self._target_positions[i] = target_pos
            self._positions[i] = self._get_joint_global_pos3(joint_id)
        self._target_positions = self._target_positions.flatten()
        self._positions = self._positions.flatten()

    def _prepare_constraints(self):
        for joint_id, config in enumerate(self.configs):
            for j in range(self.n_joints):
                if self._is_ancestor_of[joint_id, j]:
                    self._constraints.append(Constraint(
                        type=ConstraintType.ANGLE_RANGE,
                        size=1,
                        weight=config.range_weight,
                        joint_id=joint_id,
                        args=(config.min_angle, config.max_angle),
                    ))
                    break

def normalized(a):
    d = np.dot(a, a) ** .5
    return a / (d if d != 0 else 1)

def rotate(axis, pos, angle):
    n = normalized(axis[:3])
    I3 = np.eye(3, dtype=np.double)
    T = np.array([
        (0, -n[2], n[1]),
        (n[2], 0, -n[0]),
        (-n[1], n[0], 0),
    ], dtype=np.double)
    c = np.cos(angle)
    R = (c * I3 + (1 - c) * np.outer(n, n)) + np.sin(angle) * T
    mat = np.eye(4, dtype=np.double)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = (I3 - R).dot(pos[:3])
    return mat


class ConstraintArc:
    def __init__(self, pos, min_angle, max_angle):
        px, py, pz = pos
        c1 = np.cos(min_angle)
        s1 = np.sin(min_angle)
        c2 = np.cos(max_angle)
        s2 = np.sin(max_angle)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.start_pos = np.array([px * c1 - py * s1, px * s1 + py * c1, pz])
        self.end_pos = np.array([px * c2 - py * s2, px * s2 + py * c2, pz])
        # self.norm1 = np.array([px * (-s1) - py * c1, px * c1 + py * (-s1), 0])
        # self.norm2 = np.array([px * s2 - py * (-c2), px * (-c2) + py * s2, 0])
        self.pos = pos
        self.length_xy = np.sqrt(px * px + py * py)

    def closest(self, pos):
        # import pdb; pdb.set_trace()
        px, py = pos[0:2]
        length_xy = np.sqrt(px * px + py * py)
        if self.length_xy < 1e-8 or length_xy < 1e-8:
            return self.pos.copy()
        px0, py0, pz0 = self.pos
        rel_angle = np.arccos(
            np.clip((px0 * px + py0 * py) / self.length_xy / length_xy, -1, 1))
        if px0 * py - py0 * px < 0:
            rel_angle *= -1
        if rel_angle < self.min_angle:
            return self.start_pos
        if rel_angle > self.max_angle:
            return self.end_pos
        k = self.length_xy / length_xy
        return np.array([px * k, py * k, pz0])

    def closest_angle(self, pos):
        px, py = pos[0:2]
        length_xy = np.sqrt(px * px + py * py)
        if self.length_xy < 1e-8 or length_xy < 1e-8:
            return (self.min_angle + self.max_angle) / 2
        px0, py0, pz0 = self.pos
        rel_angle = np.arccos((px0 * px + py0 * py) / self.length_xy / length_xy)
        if px0 * py - py0 * px < 0:
            rel_angle *= -1
        return np.clip(rel_angle, self.min_angle, self.max_angle)


class FABRIKSolver(IKSolver):

    def __init__(self, configs):
        super().__init__(configs)
        self._min_angles = [joint_config.min_angle for joint_config in self.configs]
        self._max_angles = [joint_config.max_angle for joint_config in self.configs]
        self._spaces_init_inv = self._spaces_init.copy()
        for i in range(self.n_joints):
            self._spaces_init_inv[i] = np.linalg.inv(self._spaces_init_inv[i])
        self._parent_ids = [
            self._name_to_id.get(joint_config.parent_name, -1)
            for joint_config in self.configs
        ]
        self._children = [[] for i in range(self.n_joints)]
        for i in range(self.n_joints):
            if self._parent_ids[i] >= 0:
                self._children[self._parent_ids[i]].append(i)


    # @profile
    def solve(self, max_iteration=500):
        if len(self._targets) == 0:
            return

        STOP_THRESHOLD = .002
        ERROR_CLAMP_DISTANCE = 0.02
        D_STOP_THRESHOLD = 1e-5

        target_positions = np.zeros((len(self._targets), 3), dtype=np.double)
        joint_targets = [None] * self.n_joints
        for i, (joint_id, target_pos) in enumerate(self._targets):
            target_positions[i] = target_pos
            joint_targets[joint_id] = target_pos
        target_selector = np.array([joint_id for joint_id, _ in self._targets])
        last_positions = self._spaces[:, 0:3, 3].copy()
        spaces_init = self._spaces_init
        spaces_init_inv = self._spaces_init_inv

        configs = self.configs
        backward_arcs =[ConstraintArc(
            spaces_init_inv[j, 0:3, 3],
            - configs[j].max_angle,
            - configs[j].min_angle,
        ) for j in range(self.n_joints)]

        forward_arcs = {}
        for i in range(self.n_joints):
            for j in self._children[i]:
                forward_arcs[i, j] = ConstraintArc(
                    spaces_init[j, 0:3, 3],
                    configs[i].min_angle,
                    configs[i].max_angle,
                )

        for iter_count in range(max_iteration):
            error = np.absolute(last_positions[target_selector] - target_positions)
            # print('=====iter {}========================='.format(iter_count))
            # print('error', error)
            # print('last_positions', last_positions)
            # print('target_positions', target_positions)
            if np.alltrue(error < STOP_THRESHOLD):
                print('iter_count', iter_count)
                break
            # Backward phase
            for i in reversed(range(self.n_joints)):
                pos4 = self._spaces[i, :, 3]
                new_poss = []
                if joint_targets[i] is not None:
                    dpos = joint_targets[i] - pos4[:3]
                    if _length(dpos) > ERROR_CLAMP_DISTANCE:
                        dpos *= ERROR_CLAMP_DISTANCE / _length(dpos)
                    new_poss.append(pos4[:3] + dpos)
                # collect all target position and take centroid
                for j in self._children[i]:
                    child_global_matrix = self._spaces[j]
                    # current joint i position relative to child j.
                    # pos_rel_child = np.linalg.solve(child_global_matrix, pos4)[:3]
                    # closest_rel_pos = backward_arcs[j].closest(pos_rel_child)
                    # new_pos = child_global_matrix.dot(np.hstack([closest_rel_pos, 1]))[:3]
                    new_pos = child_global_matrix.dot(spaces_init_inv[j, :, 3])[:3]
                    new_poss.append(new_pos)
                # print('i', i, 'new_poss', new_poss)
                if new_poss:
                    new_pos = np.average(new_poss, 0)
                    # self._drag_to(i, new_pos)
                    self._spaces[i, 0:3, 3] = new_pos
                # print('i', i, 'space\n', (self._spaces[i]))
                # yield True

            # Forward phase
            for i in range(self.n_joints):
                angles = []
                parent_id = self._parent_ids[i]
                if parent_id >= 0:
                    global_matrix = self._spaces[parent_id].dot(spaces_init[i])
                else:
                    global_matrix = spaces_init[i]
                global_matrix_inv = np.linalg.inv(global_matrix)
                for j in self._children[i]:
                    child_rel_pos = global_matrix_inv.dot(self._spaces[j, :, 3])[:3]
                    angles.append(forward_arcs[i, j].closest_angle(child_rel_pos))
                if angles:
                    angle = np.average(angles)
                    cos = np.cos(angle)
                    sin = np.sin(angle)
                    self._spaces[i] = global_matrix
                    self._spaces[i, 0:3, 0:2] = self._spaces[i, 0:3, 0:2].dot(
                        [[cos, -sin], [sin, cos]])
                else:
                    angle = 0.
                    self._spaces[i] = global_matrix
                # for j in self._children[i]:
                #     self._spaces[j] = self._spaces[i].dot(spaces_init[j])
                self._angles[i] = angle
                # yield True
                # print(i, self.configs[i].name, angle)

            positions = self._spaces[:, 0:3, 3]
            if np.alltrue(np.absolute(positions - last_positions) < D_STOP_THRESHOLD):
                raise TargetUnreachable('System went static')
            last_positions = positions.copy()
            # yield True
        else:
            raise TargetUnreachable('Max iteration exceeded.')

    def _drag_to(self, joint_id, new_pos):
        parent_id = self._parent_ids[joint_id]
        if parent_id >= 0:
            parent_pos = self._spaces[parent_id, 0:3, 3]
            pos = self._spaces[joint_id, 0:3, 3]
            arm1 = pos - parent_pos
            arm2 = new_pos - parent_pos
            axis = np.cross(arm1, arm2)
            if _length(axis) > 1e-8:
                angle = np.arcsin(
                    _length(axis) / (_length(arm1) * _length(arm2)))
                R4 = rotate(axis, parent_pos, angle)
                self._spaces[joint_id] = R4.dot(self._spaces[joint_id])
        self._spaces[joint_id, 0:3, 3] = new_pos
