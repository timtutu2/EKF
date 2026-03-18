import numpy as np


def mat2euler(mat: np.ndarray) -> tuple[float, float, float]:
    """
    Minimal replacement for transforms3d.euler.mat2euler with default sxyz axes.
    """
    mat = np.asarray(mat, dtype=float)

    sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
    singular = sy < 1e-9

    if not singular:
        x = np.arctan2(mat[2, 1], mat[2, 2])
        y = np.arctan2(-mat[2, 0], sy)
        z = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])
        y = np.arctan2(-mat[2, 0], sy)
        z = 0.0

    return x, y, z
