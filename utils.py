import numpy as np


def zero_init(dist_class, num_atoms, v_mins, v_maxs):
    """Initialize a distribution with zero values.

    Args:
        dist_class (class): The distribution class.
        num_atoms (ndarray): The number of atoms per dimension.
        v_mins (ndarray): The minimum values.
        v_maxs (ndarray): The maximum values.

    Returns:
        Dist: A distribution.
    """
    return delta_dist(dist_class, num_atoms, v_mins, v_maxs, np.zeros(len(num_atoms)))


def delta_dist(dist_class, num_atoms, v_mins, v_maxs, value):
    """Create a delta distribution.

    Args:
        dist_class (class): The distribution class.
        num_atoms (ndarray): The number of atoms per dimension.
        v_mins (ndarray): The minimum values.
        v_maxs (ndarray): The maximum values.
        value (ndarray): The value of the delta distribution.

    Returns:
        Dist: A distribution.
    """
    dist = dist_class(num_atoms, v_mins, v_maxs)
    dist.static_update([value], [1])
    return dist
