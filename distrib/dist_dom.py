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
            if strict_stochastic_dominance(dist1.get_marginal(dim), dist2.get_marginal(dim)):
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
        dist = candidates.pop()
        dists_to_remove = [dist]

        for alternative in candidates:
            if distributionally_dominates(alternative, dist):
                dist = alternative
                dists_to_remove.append(alternative)
            elif distributionally_dominates(dist, alternative):
                dists_to_remove.append(alternative)

        candidates = remove_dists(dists_to_remove, candidates)
        dd_set.append(dist)

    return dd_set
