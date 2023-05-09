import numpy as np

from distrib.convex_dist_dom import cdd_prune
from distrib.dist_dom import dd_prune
from distrib.multivariate_categorical_distribution import MCD
from distrib.stochastic_dominance import stochastic_dominance, strict_stochastic_dominance

n_atoms = np.array([10, 10])
v_maxs = np.array([9, 9])
v_mins = np.array([0, 0])

dista = MCD(n_atoms, v_mins, v_maxs, name='a')
distb = MCD(n_atoms, v_mins, v_maxs, name='b')
distc = MCD(n_atoms, v_mins, v_maxs, name='c')
distd = MCD(n_atoms, v_mins, v_maxs, name='d')
diste = MCD(n_atoms, v_mins, v_maxs, name='e')

z_a = [np.array([1, 4]), np.array([4, 1])]
probs_a = [1 / 2, 1 / 2]
dista.static_update(z_a, probs_a)

z_b = [np.array([1, 5])]
probs_b = [1]
distb.static_update(z_b, probs_b)

z_c = [np.array([5, 1])]
probs_c = [1]
distc.static_update(z_c, probs_c)

z_d = [np.array([1, 3]), np.array([3, 1])]
probs_d = [1 / 2, 1 / 2]
distd.static_update(z_d, probs_d)

z_e = [np.array([1, 4]), np.array([4, 1])]
probs_e = [1 / 2, 1 / 2]
diste.static_update(z_e, probs_e)


def test_dd_prune():
    candidates = [dista, distb, distc, distd]
    print(f'Size of the candidates: {len(candidates)}')

    dd_set = dd_prune(candidates)
    print(f'Size of the DD set: {len(dd_set)}')
    print(f'Distributions in the DD set:')
    for dist in dd_set:
        print(f'- Distribution {dist.name}')


def test_cdd_prune_joint():
    candidates = [dista, distb, distc, distd]
    print(f'Size of the candidates: {len(candidates)}')

    cdd_set = cdd_prune(candidates, lp='joint')
    print(f'Size of the convex DD set: {len(cdd_set)}')
    print(f'Distributions in the convex DD set:')
    for dist in cdd_set:
        print(f'- Distribution {dist.name}')


def test_cdd_prune_marginals():
    candidates = [dista, distb, distc, distd]
    print(f'Size of the candidates: {len(candidates)}')

    cdd_set = cdd_prune(candidates, lp='marginals')
    print(f'Size of the convex DD set: {len(cdd_set)}')
    print(f'Distributions in the convex DD set:')
    for dist in cdd_set:
        print(f'- Distribution {dist.name}')


def test_mvcd():
    added_dist = dista + distb
    print(f'Added distribution: {added_dist.dist}')

    multiplied_dist = dista * 0.5
    print(f'Multiplied distribution: {multiplied_dist.dist}')


def test_sd():
    dom_ab = stochastic_dominance(dista, distb)
    dom_ad = stochastic_dominance(dista, distd)
    dom_ae = stochastic_dominance(dista, diste)
    print(f'Dominance result for a and b: {dom_ab}')
    print(f'Dominance result for a and d: {dom_ad}')
    print(f'Dominance result for a and e: {dom_ae}')

    strict_dom_ae = strict_stochastic_dominance(dista, diste)
    strict_dom_ad = strict_stochastic_dominance(dista, distd)
    print(f'Dominance result for a and e: {strict_dom_ae}')
    print(f'Dominance result for a and d: {strict_dom_ad}')


def test_distance():
    print(f'Distance between a and b: {dista.js_distance(distb)}')
    print(f'Distance between a and c: {dista.js_distance(distc)}')
    print(f'Distance between a and d: {dista.js_distance(distd)}')
    print(f'Distance between a and e: {dista.js_distance(diste)}')


if __name__ == '__main__':
    test_dd_prune()
    test_cdd_prune_joint()
    test_cdd_prune_marginals()
    test_mvcd()
    test_sd()
    test_distance()
