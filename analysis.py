import os
import json
import argparse
import time
import pandas as pd
import numpy as np
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='logs', help='The directory to save the logs.')
    parser.add_argument("--seed", type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help="The seed for random number generation.")
    parser.add_argument("--env", type=str, nargs='+', default=["small",],
                        help="The environments to run experiments on.")
    parser.add_argument("--alg", type=str, nargs='+', default=['DIMOQ'], help="The algorithm to use.")
    args = parser.parse_args()
    return args


def get_subset_sizes(env_name, alg):
    subset_sizes = defaultdict(list)

    for seed in args.seed:
        dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
        df = pd.read_csv(os.path.join(dists_dir, 'pruning_results.csv'))
        subset_sizes['dds'].append(len(df[df['dds'] == 1]))
        subset_sizes['cdds'].append(len(df[(df['dds'] == 1) & (df['cdds'] == 1)]))
        subset_sizes['pf'].append(len(df[(df['dds'] == 1) & (df['pf'] == 1)]))
        subset_sizes['ch'].append(len(df[(df['dds'] == 1) & (df['ch'] == 1)]))

    subset_sizes = {k: np.array(v) for k, v in subset_sizes.items()}
    return subset_sizes


def check_pf_subset(env_name, alg):
    for seed in args.seed:
        dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
        df = pd.read_csv(os.path.join(dists_dir, 'pruning_results.csv'))
        pf_not_cdus = df[(df['dds'] == 1) & (df['cdds'] == 0) & (df['pf'] == 1)]
        print(f"Number of policies in the PF and not in the CDUS: {len(pf_not_cdus)}")


def get_percentages():
    subset_sizes = get_subset_sizes(env_name, alg)
    print(f'Mean and std of start DDS: {np.mean(subset_sizes["dds"])} +- {np.std(subset_sizes["dds"])}')

    for subset, sizes in subset_sizes.items():
        percentage = sizes / subset_sizes['dds'] * 100
        mean = np.mean(percentage)
        std = np.std(percentage)
        print(f'Fraction {subset}: {mean:.2f}% +- {std:.2f}%')


def get_alg_stats():
    alg_durations = []

    for seed in args.seed:
        dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
        with open(os.path.join(dists_dir, f'{alg}.json'), 'r') as f:
            alg_duration = json.load(f)
            alg_durations.append(alg_duration['duration'])

    alg_mean = np.mean(alg_durations)
    alg_std = np.std(alg_durations)
    alg_min = np.min(alg_durations)
    alg_max = np.max(alg_durations)

    alg_mean = time.strftime("%H:%M:%S", time.gmtime(alg_mean))
    alg_std = time.strftime("%H:%M:%S", time.gmtime(alg_std))
    alg_min = time.strftime("%H:%M:%S", time.gmtime(alg_min))
    alg_max = time.strftime("%H:%M:%S", time.gmtime(alg_max))

    print(f'Algorithm: {alg}')
    print(f'Minimum found for seed: {args.seed[np.argmin(alg_durations)]}')
    print(f'Maximum found for seed: {args.seed[np.argmax(alg_durations)]}')
    print(f'Algorithm results: mean = {alg_mean}, std = {alg_std}, min = {alg_min}, max = {alg_max}')


def get_prune_stats():
    prune_durations = defaultdict(list)

    for seed in args.seed:
        dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
        with open(os.path.join(dists_dir, 'pruning_durations.json'), 'r') as f:
            prune_res = json.load(f)
            for prune, durations in prune_res.items():
                prune_durations[prune].append(durations)

    for prune, durations in prune_durations.items():
        prune_mean = np.mean(durations)
        prune_std = np.std(durations)
        prune_min = np.min(durations)
        prune_max = np.max(durations)
        print(
            f'Pruning results - {prune}: mean = {prune_mean}, std = {prune_std}, min = {prune_min}, max = {prune_max}')


if __name__ == "__main__":
    args = parse_args()

    print(f'Analysing results')
    for env_name in args.env:
        for alg in args.alg:
            get_alg_stats()
            print('---------------------------')
            get_prune_stats()
            print('---------------------------')
            get_percentages()
            print('---------------------------')
            check_pf_subset(env_name, alg)
            print('--------------------------------------------------------------------------')
    print(f'Finished analysing results')
