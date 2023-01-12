import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


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


def save_dists(dists, dir_path):
    """Save a list of distributions to a file.

    Args:
        dists (list): A list of distributions.
        dir_path (str): The name of the directory.
    """
    os.makedirs(dir_path, exist_ok=True)
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
        if file_name.startswith('dist_'):
            with open(os.path.join(dir_path, file_name), 'r') as f:
                dist_data = json.load(f)

        num_atoms = np.array(dist_data['num_atoms'])
        v_mins = np.array(dist_data['v_mins'])
        v_maxs = np.array(dist_data['v_maxs'])
        name = dist_data['name']
        vecs = dist_data['dist']['vecs']
        probs = dist_data['dist']['probs']
        dist = dist_class(num_atoms, v_mins, v_maxs, name=name, vecs=vecs, probs=probs)
        dists.append(dist)

    return dists


def save_momdp(momdp, dir_path, file_name):
    os.makedirs(dir_path, exist_ok=True)
    file_name += '.json'
    config = momdp.get_config()
    with open(os.path.join(dir_path, file_name), 'w') as f:
        json.dump(config, f)


def save_alg(alg, dds_size, duration, dir_path, file_name):
    os.makedirs(dir_path, exist_ok=True)
    file_name += '.json'
    config = alg.get_config()
    config['dds_size'] = dds_size
    config['duration'] = duration

    with open(os.path.join(dir_path, file_name), 'w') as f:
        json.dump(config, f)


def save_results(results, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(dir_path, 'results.csv'), index=False)
