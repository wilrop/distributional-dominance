import numpy as np
from pymoo.indicators.hv import HV


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


def js_distances(dists):
    """Compute the Jensen-Shannon distances of a list of distributions.

    Args:
        dists (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        List[float]: The Jensen-Shannon distances for each distribution to the others.
    """
    distribution_distances = []
    for dist1 in dists:
        distances = []

        for dist2 in dists:
            distances.append(dist1.js_distance(dist2))

        distribution_distances.append(distances)
    return dists
