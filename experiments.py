import argparse
import mo_gym
import numpy as np
from mo_gym.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP
from random_momdp import RandomMOMDP
from space_traders import SpaceTraders
from dimoq import DIMOQ
from modvi import MODVI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The seed for random number generation.")
    parser.add_argument("--env", type=str, nargs='+', default=["space-traders"], help="The environments to run experiments on.")
    args = parser.parse_args()
    return args


def deep_sea_treasure(args):
    """Run an experiment on the Deep Sea Treasure environment using DIMOQ.

    Args:
        args (Dict): The arguments.

    Returns:
        List[Dist]: A list of distributionally non-dominated distributions.
    """
    env = mo_gym.make('deep-sea-treasure-v0', dst_map=CONCAVE_MAP)
    ref_point = np.array([0, -25])
    gamma = 1.
    initial_epsilon = 1.0
    epsilon_decay = 0.9975
    final_epsilon = 0.2
    num_atoms = (125, 21)
    v_mins = (0., -20.)
    v_maxs = (124., 0.)
    seed = args.seed
    project_name = "Distributional-Dominance"
    experiment_name = "DIMOQ-DST"
    log = False
    dimoq = DIMOQ(env, ref_point, gamma, initial_epsilon, epsilon_decay, final_epsilon, num_atoms, v_mins, v_maxs, seed,
                  project_name, experiment_name, log)

    dimoq.train(num_episodes=3000, log_every=10)

    distributional_set = dimoq.get_local_dcs()
    for dist in distributional_set:
        print(dist.expected_value())
        print('---------')

    return distributional_set


def space_traders(args):
    """Run an experiment on the Deep Sea Treasure environment using DIMOQ.

    Args:
        args (Dict): The arguments.

    Returns:
        List[Dist]: A list of distributionally non-dominated distributions.
    """
    env = SpaceTraders(seed=args.seed)

    ref_point = np.array([0, -25])
    gamma = 1.
    initial_epsilon = 1.0
    epsilon_decay = 0.9975
    final_epsilon = 0.2
    num_atoms = (20, 26)
    v_mins = (0., -22.)
    v_maxs = (1., 0.)
    seed = args.seed
    project_name = "Distributional-Dominance"
    experiment_name = "DIMOQ-DST"
    log = False
    dimoq = DIMOQ(env, ref_point, gamma, initial_epsilon, epsilon_decay, final_epsilon, num_atoms, v_mins, v_maxs, seed,
                  project_name, experiment_name, log)

    modvi = MODVI(env, gamma, num_atoms, v_mins, v_maxs)
    dds = modvi.get_dds()

    for dist in dds:
        print(dist.expected_value())
        print(dist.nonzero_vecs())
        print('---------')

    print(f'Size of DDS: {len(dds)}')

    """dimoq.train(num_episodes=3000, log_every=10)

    distributional_set = dimoq.get_local_dcs()
    for dist in distributional_set:
        print(dist.expected_value())
        print('---------')"""

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
    seed = args.seed
    env = RandomMOMDP(num_states, num_objectives, num_actions, num_next_states, num_terminal_states, reward_min, reward_max, reward_dist='discrete', start_state=start_state, seed=seed)

    print(env._transition_function)
    print(env._reward_function)
    print(env._terminal_states)
    # DIMOQ variables
    ref_point = np.array([0, 0])
    gamma = 1.
    initial_epsilon = 0.2
    epsilon_decay = 1.
    final_epsilon = 0.2
    num_atoms = (21, 21)
    v_mins = (0., 0.)
    v_maxs = (20., 20.)
    project_name = "Distributional-Dominance"
    experiment_name = "DIMOQ-SmallMOMDP"
    log = False

    modvi = MODVI(env, gamma,num_atoms, v_mins, v_maxs)
    dimoq = DIMOQ(env, ref_point, gamma, initial_epsilon, epsilon_decay, final_epsilon, num_atoms, v_mins, v_maxs, seed,
                  project_name, experiment_name, log)

    dds = modvi.get_dds()
    for dist in dds:
        print(dist.expected_value())
        print(dist.nonzero_vecs())
        print('---------')

    raise Exception
    dimoq.train(num_episodes=3000, log_every=500)

    distributional_set = dimoq.get_local_dcs()
    for dist in distributional_set:
        print(dist.expected_value())
        print('---------')

    return distributional_set


if __name__ == "__main__":
    args = parse_args()
    if "dst" in args.env:
        deep_sea_treasure(args)

    if "small" in args.env:
        small_momdp(args)

    if "space-traders" in args.env:
        space_traders(args)