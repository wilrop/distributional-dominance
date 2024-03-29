import argparse
import os
import time
from copy import deepcopy

import numpy as np

from distrib.classic_dominance import p_prune, c_prune
from distrib.convex_dist_dom import cdd_prune
from distrib.dist_dom import dd_prune
from distrib.multivariate_categorical_distribution import MCD
from utils.data import load_dists, save_pruning_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='logs', help='The directory to save the logs.')
    parser.add_argument("--seed", type=int, nargs='+', default=[1],
                        help="The seed for random number generation.")
    parser.add_argument("--env", type=str, nargs='+', default=["small"],
                        help="The environments to run experiments on.")
    parser.add_argument("--alg", type=str, nargs='+', default=['DIMOQ'], help="The algorithm to use.")
    parser.add_argument("--prune", type=str, nargs='+', default=['dds', 'cdds', 'pf', 'ch'],
                        help="The pruning methods to use.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    print(f'Starting pruning at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    for env_name in args.env:
        for seed in args.seed:
            for alg in args.alg:
                dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
                dists = load_dists(dists_dir, MCD)

                for i, dist in enumerate(dists):  # Set name for reference.
                    dist.name = i

                evs = [dist.get_expected_value() for dist in dists]  # Get expected values for reference.
                results = {f'ev_{i}': [ev[i] for ev in evs] for i in range(len(evs[0]))}  # Save expected values.
                durations = {}

                for prune in args.prune:
                    dists_copy = deepcopy(dists)

                    start = time.time()
                    if prune == 'dds':
                        pruned = dd_prune(dists_copy)
                    elif prune == 'cdds':
                        pruned = cdd_prune(dists_copy)
                    elif prune == 'pf':
                        pruned = p_prune(dists_copy)
                    elif prune == 'ch':
                        pruned = c_prune(dists_copy)
                    else:
                        raise ValueError(f"Invalid pruning method: {prune}")

                    end = time.time()
                    durations[prune] = end - start
                    prune_results = np.zeros(len(dists))
                    in_set = [d.name for d in pruned]
                    prune_results[in_set] = 1
                    results[prune] = prune_results

                save_pruning_results(results, durations, dists_dir)

    print(f'Finished pruning at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
