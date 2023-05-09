import numpy as np
from collections import defaultdict


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
    dist = dist_class(num_atoms, v_mins, v_maxs, vecs=[value], probs=[1])
    return dist


def remove_dists(dists_to_remove, all_dists):
    """Remove a list of distributions from another list of distributions.

    Args:
        dists_to_remove (List[MultivariateCategoricalDistribution]): A list of distributions to remove.
        all_dists (List[MultivariateCategoricalDistribution]): The list of all distributions.

    Returns:
        List[MultivariateCategoricalDistribution]: A list of distributions.
    """
    if not isinstance(dists_to_remove, list):  # Make sure the distributions to remove is a list.
        dists_to_remove = [dists_to_remove]

    dists = []

    for dist in all_dists:
        keep_dist = True

        for dist_rem in dists_to_remove:
            if np.all(dist.dist == dist_rem.dist):
                keep_dist = False
                break

        if keep_dist:
            dists.append(dist)

    return dists


def create_mixture_distribution(dists, probs, dist_class, num_atoms, v_mins, v_maxs):
    """Create a mixture distribution.

    Args:
        dists (list): A list of distributions.
        probs (list): A probability for each distribution.
        dist_class (class): The class of the mixture distribution
        num_atoms (ndarray): The number of atoms per dimension.
        v_mins (ndarray): The minimum values.
        v_maxs (ndarray): The maximum values.

    Returns:
        Dist: A distribution.
    """
    vecs_probs = defaultdict(lambda: 0)

    for dist, dist_prob in zip(dists, probs):
        for vec, vec_prob in dist.nonzero_vecs_probs():
            vecs_probs[tuple(vec)] += vec_prob * dist_prob

    vecs = np.array(list(vecs_probs.keys()))
    probs = np.array(list(vecs_probs.values()))
    new_dist = dist_class(num_atoms, v_mins, v_maxs, vecs=vecs, probs=probs)
    return new_dist
