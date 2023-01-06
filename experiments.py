import argparse
from copy import deepcopy

import mo_gym
import numpy as np
from mo_gym.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP
from ramo.pareto.verify import p_prune

from convex_dist_dom import cdd_prune
from dimoq import DIMOQ
from dist_dom import dd_prune
from modvi import MODVI
from random_momdp import RandomMOMDP
from space_traders import SpaceTraders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3, help="The seed for random number generation.")
    parser.add_argument("--env", type=str, nargs='+', default=["small"], help="The environments to run experiments on.")
    parser.add_argument("--augment-env", action='store_true', default=True, help="Whether to augment the environment.")
    parser.add_argument("--alg", type=str, default='DIMOQ', help="The algorithm to use.")
    parser.add_argument("--num-episodes", type=int, default=2000, help="The number of episodes to run.")
    parser.add_argument('--max-timesteps', type=int, default=3, help="The maximum number of timesteps per episode.")
    parser.add_argument('--gamma', type=float, default=1., help='The discount factor.')
    parser.add_argument('--log', action='store_true', default=False, help='Whether to log the results.')
    parser.add_argument('--log-every', type=int, default=100, help='The number of episodes between logging.')
    parser.add_argument('--save', action='store_true', default=False, help='Whether to save the results.')
    parser.add_argument('--warmup', type=int, default=10000, help='The number of warmup episodes.')
    args = parser.parse_args()
    return args


def print_dds(dds, print_all=True, sanity_check=False):
    print(f'Size of the DDS: {len(dds)}')
    if print_all:
        for dist in dds:
            print(dist.expected_value())
            print(dist.nonzero_vecs_probs())
            print('------------------')

    if sanity_check:
        set_copy = deepcopy(dds)
        dds = dd_prune(set_copy)
        print(f'Size of the DDS after sanity check: {len(dds)}')

    set_copy = deepcopy(dds)
    cdds = cdd_prune(set_copy)
    print(f'Size of the CDDS: {len(cdds)}')

    if sanity_check:
        set_copy = deepcopy(dds)
        cdds = cdd_prune(set_copy)
        print(f'Size of the CDDS after sanity check: {len(cdds)}')

    set_copy = deepcopy(dds)
    candidates = set([tuple(dist.expected_value()) for dist in set_copy])
    pf = p_prune(candidates)
    print(f'Size of the Pareto front: {len(pf)}')


def deep_sea_treasure(args):
    """Run an experiment on the Deep Sea Treasure environment using DIMOQ.

    Args:
        args (Dict): The arguments.

    Returns:
        List[Dist]: A list of distributionally non-dominated distributions.
    """
    env = mo_gym.make('deep-sea-treasure-v0', dst_map=CONCAVE_MAP)
    ref_point = np.array([0, -25])
    initial_epsilon = 1.0
    epsilon_decay = 0.9975
    final_epsilon = 0.2
    num_atoms = (125, 21)
    v_mins = (0., -20.)
    v_maxs = (124., 0.)
    project_name = "Distributional-Dominance"
    experiment_name = "DIMOQ-DST"

    if args.alg == 'DIMOQ':
        dimoq = DIMOQ(env, ref_point, args.gamma, initial_epsilon, epsilon_decay, final_epsilon, num_atoms, v_mins,
                      v_maxs, seed=args.seed, project_name=project_name, experiment_name=experiment_name, log=args.log)
        dimoq.train(num_episodes=args.num_episodes, log_every=args.log_every)
        dds = dimoq.get_local_dds()
    elif args.alg == 'MODVI':
        modvi = MODVI(env, args.gamma, num_atoms, v_mins, v_maxs)
        dds = modvi.get_dds(num_iters=args.num_episodes)
    else:
        raise ValueError(f'Unknown algorithm {args.alg}')

    print_dds(dds)

    return dds


def space_traders(args):
    """Run an experiment on the Deep Sea Treasure environment using DIMOQ.

    Args:
        args (Dict): The arguments.

    Returns:
        List[Dist]: A list of distributionally non-dominated distributions.
    """
    env = SpaceTraders(seed=args.seed)
    ref_point = np.array([0, -30])
    initial_epsilon = 1.0
    epsilon_decay = 0.99
    final_epsilon = 0.1
    num_atoms = (21, 23)
    v_mins = (0., -22.)
    v_maxs = (1., 0.)
    project_name = "Distributional-Dominance"
    experiment_name = "DIMOQ-ST"

    if args.alg == 'DIMOQ':
        dimoq = DIMOQ(env, ref_point, args.gamma, initial_epsilon, epsilon_decay, final_epsilon, num_atoms, v_mins,
                      v_maxs, seed=args.seed, project_name=project_name, experiment_name=experiment_name, log=args.log)
        dimoq.train(num_episodes=args.num_episodes, log_every=args.log_every, warmup_time=args.warmup)
        dds = dimoq.get_local_dds()
    elif args.alg == 'MODVI':
        modvi = MODVI(env, args.gamma, num_atoms, v_mins, v_maxs)
        dds = modvi.get_dds(num_iters=args.num_episodes)
    else:
        raise ValueError(f'Unknown algorithm {args.alg}')

    print_dds(dds)

    return dds


def small_momdp(args):
    """Run an experiment on the Small MOMDP environment using DIMOQ.

    Args:
        args (Dict): The arguments.

    Returns:
        List[Dist]: A list of distributionally non-dominated distributions.
    """
    # Environment variables
    num_states = 5
    num_objectives = 2
    num_actions = 2
    num_next_states = 2
    num_terminal_states = 1
    reward_min = np.zeros(num_objectives)
    reward_max = np.ones(num_objectives) * 5
    start_state = 0
    env = RandomMOMDP(num_states, num_objectives, num_actions, num_next_states, num_terminal_states, reward_min,
                      reward_max, reward_dist='discrete', start_state=start_state, max_timesteps=args.max_timesteps,
                      seed=args.seed, augment_state=args.augment_env)

    # DIMOQ variables
    ref_point = np.array([0, 0])
    initial_epsilon = 1.
    epsilon_decay = 1.
    final_epsilon = 0.2
    num_atoms = (21, 21)
    v_mins = (0., 0.)
    v_maxs = (20., 20.)
    project_name = "Distributional-Dominance"
    experiment_name = "DIMOQ-SmallMOMDP"

    if args.alg == 'DIMOQ':
        dimoq = DIMOQ(env, ref_point, args.gamma, initial_epsilon, epsilon_decay, final_epsilon, num_atoms, v_mins,
                      v_maxs, seed=args.seed, project_name=project_name, experiment_name=experiment_name, log=args.log)
        dimoq.train(num_episodes=args.num_episodes, log_every=args.log_every, warmup_time=args.warmup)
        dds = dimoq.get_local_dds()
    elif args.alg == 'MODVI':
        modvi = MODVI(env, args.gamma, num_atoms, v_mins, v_maxs)
        dds = modvi.get_dds(num_iters=args.num_episodes)
    else:
        raise ValueError(f'Unknown algorithm {args.alg}')

    print_dds(dds)

    return dds


if __name__ == "__main__":
    args = parse_args()
    if "dst" in args.env:
        deep_sea_treasure(args)

    if "small" in args.env:
        small_momdp(args)

    if "space-traders" in args.env:
        space_traders(args)
