import mo_gym
import numpy as np
from mo_gym.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP

from dimoq import DIMOQ


def deep_sea_treasure():
    """Run an experiment on the Deep Sea Treasure environment using DIMOQ.

    Returns:
        List[Dist]: A list of distributionally non-dominated distributions.
    """
    env = mo_gym.make('deep-sea-treasure-v0', dst_map=CONCAVE_MAP)
    ref_point = np.array([0, -25])
    gamma = 1.
    initial_epsilon = 1.0
    epsilon_decay = 0.99
    final_epsilon = 0.1
    num_atoms = (125, 21)
    v_mins = (0., -20.)
    v_maxs = (124., 0.)
    seed = 1
    project_name = "Distributional-Dominance"
    experiment_name = "DIMOQ"
    log = False
    dimoq = DIMOQ(env, ref_point, gamma, initial_epsilon, epsilon_decay, final_epsilon, num_atoms, v_mins, v_maxs, seed,
                  project_name, experiment_name, log)

    dimoq.train(num_episodes=3000, log_every=50)

    distributional_set = dimoq.get_local_dcs()
    for dist in distributional_set:
        print(dist.expected_value())
        print(dist.cdf)
        print('---------')

    return distributional_set


if __name__ == "__main__":
    deep_sea_treasure()
