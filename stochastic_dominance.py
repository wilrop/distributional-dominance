import numpy as np


def strict_stochastic_dominance(dist1, dist2):
    return stochastic_dominance(dist1, dist2) and np.any(dist1.cdf < dist2.cdf)


def stochastic_dominance(dist1, dist2):
    return np.all(dist1.cdf <= dist2.cdf)
