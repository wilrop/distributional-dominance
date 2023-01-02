import numpy as np

from stochastic_dominance import stochastic_dominance, strict_stochastic_dominance


def distributionally_dominates(dist1, dist2):
    if stochastic_dominance(dist1, dist2):
        for dim in range(dist1.num_dims):
            if strict_stochastic_dominance(dist1.marginal(dim), dist2.marginal(dim)):
                return True

    return False


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


def dd_prune(candidates):
    """Prune the distributions which are ESR dominated from a list of candidates.

    Args:
        candidates (List[MultivariateCategoricalDistribution]): A list of distributions.

    Returns:
        List[MultivariateCategoricalDistribution]: A list of distributions which are not pairwise distributionally
            dominated.
    """
    dd_set = []

    while candidates:
        dist = candidates.pop()
        dists_to_remove = [dist]

        for alternative in candidates:
            if distributionally_dominates(alternative, dist):
                dist = alternative
                dists_to_remove.append(alternative)

        candidates = remove_dists(dists_to_remove, candidates)
        dd_set.append(dist)

    return dd_set
