import numpy as np
from pymoo.indicators.hv import HV
from sklearn.cluster import DBSCAN


def dist_hypervolume(ref_point, dists):
    """Compute the hypervolume of the expected values from a list of distributions.

    Args:
        ref_point (List[float]): The reference point.
        dists (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        float: The hypervolume.
    """
    points = [dist.expected_value() for dist in dists]
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def max_inter_distance(dist_lst):
    """Compute the maximum distance between distributions in a list.

    Args:
        dist_lst (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        float: The maximum distance.
    """
    dist_matrix = js_distances(dist_lst)
    return np.max(dist_matrix)


def linear_utility(dist_lst):
    """Compute the linear utility of a list of distributions.

    Args:
        dist_lst (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        float: The linear utility.
    """
    return sum(sum(dist.expected_value()) for dist in dist_lst)


def js_distances(dists):
    """Compute the Jensen-Shannon distances of a list of distributions.

    Args:
        dists (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        ndarray: A matrix of distances. Conceptually this is a graph where each node is a distribution.
    """
    distance_matrix = np.zeros((len(dists), len(dists)))
    for i, dist1 in enumerate(dists):
        for j, dist2 in zip(range(i + 1, len(dists)), dists[i + 1:]):
            distance = dist1.js_distance(dist2)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def get_best(dist_lst, max_dists=10):
    """Get the best distributions from a list of distributions.

    Args:
        dist_lst (List[MultivariateCategoricalDistribution]): The distributions.
        max_dists (int): The maximum number of distributions to return.

    Returns:
        List[MultivariateCategoricalDistribution]: The best distributions.
    """
    l_eps = 0.
    r_eps = 1.
    min_samples = 5

    dist_matrix = js_distances(dist_lst)

    while len(dist_lst) > max_dists:
        eps = (l_eps + r_eps) / 2
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
        clustering.fit(dist_matrix)

        cores = clustering.core_sample_indices_
        outliers = np.squeeze(np.argwhere(clustering.labels_ == -1))

        if len(outliers) > 0:
            l_eps = eps
        elif len(cores) == len(dist_lst):
            r_eps = eps
        else:
            l_eps = 0.
            r_eps = 1.
            dist_lst = [dist_lst[i] for i in cores]
            dist_matrix = js_distances(dist_lst)

    return dist_lst
