import numpy as np

def calc_jacobian_matrix(self):
    """
    Calculate a m x n Jacobian matrix, where m is the number of target joints multiply
    by 3, and n is the number of joints(which equals to number of angles).

    J(i, j) = Deriv(position(i), angle(j)) = axis(j) x (position(i) - position(j))
    """
    cdef:
        int m, n, i, j, target_id
        # double[:] positions
        # double[:, :] J
        # double[:, :, :] spaces
        int[:, :] is_ancestor_of

    m = 3 * self.n_targets
    n = self.n_joints
    J = Ja = np.zeros((m, n), dtype=np.double)
    is_ancestor_of = self._is_ancestor_of.astype(np.int32)
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
    return Ja
