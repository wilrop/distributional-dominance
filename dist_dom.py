from stochastic_dominance import stochastic_dominance, strict_stochastic_dominance
from utils import remove_dists


def distributionally_dominates(dist1, dist2):
    """Check if distribution dist1 distributionally dominates dist2.

    Args:
        dist1 (MultivariateCategoricalDistribution): The first distribution.
        dist2 (MultivariateCategoricalDistribution): The second distribution.

    Returns:
        bool: Whether dist1 distributionally dominates dist2.
    """
    if stochastic_dominance(dist1, dist2):
        for dim in range(dist1.num_dims):
            if strict_stochastic_dominance(dist1.marginal(dim), dist2.marginal(dim)):
                return True

    return False


def dd_prune(candidates):
    """Prune the distributions which are dominated from a list of candidates.

    Args:
        candidates (List[MultivariateCategoricalDistribution]): A list of distributions.

    Returns:
        List[MultivariateCategoricalDistribution]: A list of distributions which are not pairwise distributionally
            dominated.
    """
    dd_set = []

    while candidates:
        dist = candidates[0]
        idx_to_remove = [0]

        for alt_idx, alternative in enumerate(candidates[1:], 1):
            if distributionally_dominates(alternative, dist):
                dist = alternative
                idx_to_remove.append(alt_idx)
            elif distributionally_dominates(dist, alternative):
                idx_to_remove.append(alt_idx)

        candidates = [dist for idx, dist in enumerate(candidates) if idx not in idx_to_remove]
        dd_set.append(dist)

    return dd_set
