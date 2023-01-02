import numpy as np
from pymoo.indicators.hv import HV


def dist_hypervolume(ref_point, dists):
    """Compute the hypervolume of a list of distributions.
    Args:
        ref_point (List[float]): The reference point.
        dists (List[MultivariateCategoricalDistribution]): The distributions.
    Returns:
        float: The hypervolume.
    """
    points = [dist.expected_value for dist in dists]
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)
