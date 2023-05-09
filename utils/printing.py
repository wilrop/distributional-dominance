from copy import deepcopy

from classic_dominance import p_prune, c_prune
from dist_dom import dd_prune
from convex_dist_dom import cdd_prune


def print_dists(dists, subsets=('DDS', 'CDUS', 'PF', 'CH'), print_all=False, print_name=False, print_evs=False):
    print(f'Starting size of the list of distributions: {len(dists)}')

    for subset in subsets:
        dists_copy = deepcopy(dists)
        if subset == 'DDS':
            pruned = dd_prune(dists_copy)
        elif subset == 'CDUS':
            pruned = cdd_prune(dists_copy)
        elif subset == 'PF':
            pruned = p_prune(dists_copy)
        elif subset == 'CH':
            pruned = c_prune(dists_copy)
        else:
            raise ValueError(f"Invalid pruning method: {subset}")

        print(f'Size of the {subset}: {len(pruned)}')

        if print_all or print_name or print_evs:
            print(f'Distributions in the pruned subset: ')

            for dist in pruned:
                if print_name:
                    print(dist.name)
                if print_evs:
                    print(f'Expected value: {dist.get_expected_value()}')
                if print_all:
                    print(f'Non zero vectors and their probabilities: ')
                    print(dist.nonzero_vecs_probs())
