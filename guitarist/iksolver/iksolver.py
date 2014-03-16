"""
Inverse Kinematic Solver, supporting 3 kinds of constraints.
"""

from collections import namedtuple
import numpy as np
# import pyximport; pyximport.install()
# from ._iksolver import calc_jacobian_matrix

__all__ = ['JointConfig', 'IKSolver', 'TargetUnreachable']

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

def _update_spaces(spaces, angles):
    " Update spaces by angles. Inplace operation. "
    n = len(angles)
    cos = np.cos(angles)
    sin = np.sin(angles)
    R = np.zeros((4, n), dtype=np.double)
    R[0] = R[3] = cos; R[1] = -sin; R[2] = sin
    R = R.T.reshape((n, 2, 2))
    # spaces[:, 0:3, (0, 1)] = spaces[:, 0:3, (0, 1)].dot(R)
    for i in range(len(angles)):
        slice_ = (i, slice(0, 3), slice(0, 2))
        spaces[slice_] = spaces[slice_].dot(R[i])


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
        # Current joint positions.
        self._spaces_init = _calc_spaces(configs)
        self._spaces = self._spaces_init.copy()
        # Current joint angles.
        self._angles = np.array([0 for config in configs], dtype=np.double)
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

    def clear_constraints(self):
        self._constraints.clear()

    def clear_targets(self):
        self._targets.clear()

    def set_target_pos(self, joint_id, pos):
        self._targets.append((joint_id, pos))

    def set_target_pos_by_name(self, joint_name, pos):
        self._targets.append((self._name_to_id[joint_name], pos))

    # @profile
    def solve(self, max_iteration=500):
        self._update_spaces()
        self._prepare_targets()
        self._prepare_constraints()

        STEP_LENGTH = 0.2 * self.n_joints
        DAMPING = 5

        # Error between target and current position
        error = STOP_THRESHOLD + 1
        error_low = STOP_THRESHOLD_LOW + 1
        iter_count = 0
        while not (np.alltrue(error < STOP_THRESHOLD) and \
                np.alltrue(error_low < STOP_THRESHOLD_LOW)):
            # print('<<<<<<<<<<<<<<<<<< new iter<<<<<<<<<<<<<<<<<<<<<', iter_count) 
            error = self._clamp_error(self._target_positions - self._positions)
            # print(error, self._target_positions, self._positions)
            # Calculate delta angle.
            J = self._calc_jacobian_matrix()
            # J = calc_jacobian_matrix(self)
            Jp = np.linalg.pinv(J)
            delta_angle = dot(Jp, error)
            # Clamp step
            angle_step = _length(delta_angle)
            if angle_step > STEP_LENGTH:
                delta_angle *= STEP_LENGTH / angle_step
            T = np.eye(Jp.shape[0]) - dot(Jp, J)
            # Some constraint may be resolved so that they will not be included
            # in constraint_ids.
            error_low, constraint_ids = self._calc_error_low()
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
            self._angles += delta_angle
            self._update_spaces()
            self._update_positions()
            # self._dump_state()
            iter_count += 1
            if iter_count >= max_iteration:
                raise TargetUnreachable()
        print('iter_count', iter_count)

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
        """
        m = sum(self._constraints[i].size for i in constraint_ids)
        n = self.n_joints
        JL = np.zeros((m, n), dtype=np.double)
        start = 0
        angles = self._angles
        positions = self._positions
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
                pos = self._positions[3 * c.joint_id:3 * c.joint_id + 3]
                ref_pos = c.args
                d = ref_pos - pos
                d_len = _length(d)
                if d_len > REF_POS_MIN_DISTANCE:
                    if d_len > REF_POS_CLAMP:
                        d *= REF_POS_CLAMP / d_len
                    error_low.extends(d)
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
        self._spaces = spaces = self._spaces_init.copy()
        _update_spaces(spaces, self._angles)
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
