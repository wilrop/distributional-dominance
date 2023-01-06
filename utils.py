import json
import os
from collections import defaultdict

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
    new_dist = zero_init(dist_class, num_atoms, v_mins, v_maxs)
    new_dist.static_update(vecs, probs)
    return new_dist


def save_dists(dists, dir_path):
    """Save a list of distributions to a file.

    Args:
        dists (list): A list of distributions.
        dir_path (str): The name of the directory.
    """
    for i, dist in enumerate(dists):
        dist.save(dir_path, file_name=f'dist_{i}')


def load_dists(dir_path, dist_class):
    """Load a list of distributions from a file.

    Args:
        dir_path (str): The name of the directory.

    Returns:
        list: A list of distributions.
    """
    dists = []

    for file_name in os.listdir(dir_path):
        with open(os.path.join(dir_path, file_name), 'r') as f:
            dist_data = json.load(f)

        num_atoms = np.array(dist_data['num_atoms'])
        v_mins = np.array(dist_data['v_mins'])
        v_maxs = np.array(dist_data['v_maxs'])
        name = dist_data['name']
        dist = dist_class(num_atoms, v_mins, v_maxs, name=name)

        vecs = []
        probs = []
        for vec, prob in dist_data['dist'].items():
            vecs.append(vec)
            probs.append(prob)
        dist = dist.static_update(vecs, probs)
        dists.append(dist)

    return dists


def save_momdp(momdp, dir_path, file_name):
    file_name += '.json'
    config = momdp.get_config()
    with open(os.path.join(dir_path, file_name), 'w') as f:
        json.dump(config, f)
