import numpy as np


def strict_stochastic_dominance(dist1, dist2):
    """Check if dist1 strictly first-order stochastic dominates dist2.

    Args:
        dist1 (MultivariateDistribution): A first multivariate distribution.
        dist2 (MultivariateDistribution): A second multivariate distribution.

    Returns:
        bool: Whether dist1 strictly first-order stochastic dominates dist2.
    """
    return stochastic_dominance(dist1, dist2) and np.any(dist1.get_cdf() < dist2.get_cdf())


def stochastic_dominance(dist1, dist2):
    """Check if dist1 first-order stochastic dominates dist2.

    Args:
        dist1 (MultivariateDistribution): A first multivariate distribution.
        dist2 (MultivariateDistribution): A second multivariate distribution.

    Returns:
        bool: Whether dist1 first-order stochastic dominates dist2.
    """
    return np.all(dist1.get_cdf() <= dist2.get_cdf())
