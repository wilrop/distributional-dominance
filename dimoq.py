from itertools import product

import numpy as np
from gym.spaces import Box

from count_based_mcd import CountBasedMCD
from dist_dom import dd_prune
from dist_metrics import dist_hypervolume, get_best, max_inter_distance, linear_utility
from multivariate_categorical_distribution import MultivariateCategoricalDistribution
from utils import create_mixture_distribution, zero_init


class DIMOQ:
    """
    Distributional Dominance Multi-Objective Q-learning
    """

    def __init__(
            self,
            env,
            ref_point: np.ndarray,
            gamma: float = 0.8,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0.99,
            final_epsilon: float = 0.1,
            num_atoms: tuple = (51, 51),
            v_mins: tuple = (0., 0.),
            v_maxs: tuple = (1., 1.),
            max_dists: int = 20,
            seed: int = None,
            project_name: str = "MORL-baselines",
            experiment_name: str = "Pareto Q-Learning",
            log: bool = True,
    ):
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Algorithm setup
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ref_point = ref_point
        self.num_atoms = num_atoms
        self.v_mins = v_mins
        self.v_maxs = v_maxs
        self.max_dists = max_dists

        # Internal Q-learning variables
        self.env = env
        self.num_actions = self.env.action_space.n
        if isinstance(self.env.observation_space, Box):
            low_bound = self.env.observation_space.low
            high_bound = self.env.observation_space.high
            self.env_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1)
            self.num_states = np.prod(self.env_shape)
        else:
            self.num_states = self.env.observation_space.n
            self.env_shape = (self.num_states,)
        self.num_objectives = self.env.reward_space.shape[0]
        self.non_dominated = self._init_zero_dists(MultivariateCategoricalDistribution, squeeze=False)
        self.reward_dists = self._init_zero_dists(CountBasedMCD, squeeze=True)
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

    def _init_zero_dists(self, dist_class, squeeze=True):
        """Initialize the distributional Q-set for all state-action-next pairs."""
        zero_dists = []
        for _ in range(self.num_states):
            state_dists = []
            for _ in range(self.num_actions):
                state_action_dists = []
                for _ in range(self.num_states):
                    if squeeze:
                        dist = zero_init(dist_class, self.num_atoms, self.v_mins, self.v_maxs)
                    else:
                        dist = [zero_init(dist_class, self.num_atoms, self.v_mins, self.v_maxs)]
                    state_action_dists.append(dist)
                state_dists.append(state_action_dists)
            zero_dists.append(state_dists)

        return zero_dists

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
            "num_atoms": list(self.num_atoms),
            "v_mins": list(self.v_mins),
            "v_maxs": list(self.v_maxs),
            "seed": self.seed
        }

    def score_hypervolume(self, state):
        """Compute the action scores based upon the hypervolume metric for the expected values.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_dists_lst = [self.get_q_dists(state, action) for action in range(self.num_actions)]
        action_scores = [dist_hypervolume(self.ref_point, q_dists) for q_dists in q_dists_lst]
        return action_scores

    def score_inter_distance(self, state):
        """Compute the action scores based upon the maximum inter-distribution distance metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_dists_lst = [self.get_q_dists(state, action) for action in range(self.num_actions)]
        action_scores = [max_inter_distance(q_dists) for q_dists in q_dists_lst]
        return action_scores

    def score_linear_utility(self, state):
        """Compute the action scores based upon the linear utility metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_dists_lst = [self.get_q_dists(state, action) for action in range(self.num_actions)]
        action_scores = [linear_utility(q_dists) for q_dists in q_dists_lst]
        return action_scores

    def get_q_dists(self, state, action):
        """Compute the distributional Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            List[Dist]: A list of Q distributions.
        """
        transition_function = self.transitions[state, action] / max(1, np.sum(self.transitions[state, action]))
        mixture_dists = []
        probs = []

        for next_state in range(self.num_states):
            prob = transition_function[next_state]

            if prob > 0:
                probs.append(prob)
                next_state_set = []

                for nd_dist in self.non_dominated[state][action][next_state]:
                    return_dist = self.reward_dists[state][action][next_state] + self.gamma * nd_dist
                    next_state_set.append(return_dist)

                mixture_dists.append(next_state_set)

        if not mixture_dists:
            return [zero_init(MultivariateCategoricalDistribution, self.num_atoms, self.v_mins, self.v_maxs)]

        q_dists = []

        for dists in product(*mixture_dists):
            mixture = create_mixture_distribution(dists, probs, MultivariateCategoricalDistribution, self.num_atoms,
                                                  self.v_mins, self.v_maxs)
            q_dists.append(mixture)

        return q_dists

    def select_action(self, state, score_func):
        """Select an action in the current state.

        Args:
            state (int): The current state.
            score_func (callable): A function that returns a score per action.

        Returns:
            int: The selected action.
        """
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.rng.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())

    def calc_non_dominated(self, state):
        """Get the distributionally non-dominated distributions in a given state.

        Args:
            state (int): The current state.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        q_dists_lst = [self.get_q_dists(state, action) for action in range(self.num_actions)]
        q_dists = [q_dist for q_dists in q_dists_lst for q_dist in q_dists]
        non_dominated = dd_prune(q_dists)
        return non_dominated

    def _flatten_state(self, state):
        """Flatten a state.

        Args:
            state (int): The current state.

        Returns:
            int: The flattened state.
        """
        if isinstance(self.env.observation_space, Box):
            return np.ravel_multi_index(state, self.env_shape)
        else:
            return state

    def warmup(self, num_episodes, score_func):
        """Run a number of episodes to warm up the agent.

        Args:
            num_episodes (int): The number of episodes to run.
        """
        print(f'Warming up for {num_episodes} episodes...')
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = self._flatten_state(state)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.select_action(state, score_func)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._flatten_state(next_state)
                self.transitions[state, action, next_state] += 1
                state = next_state

    def train(self, num_episodes=3000, warmup_time=1000, learn_model=False, log_every=100, action_eval='linear'):
        """Learn the distributional dominance set.

        Args:
            num_episodes (int, optional): The number of episodes to train for.
            log_every (int, optional): Log the results every number of episodes. (Default value = 100)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')

        Returns:
            List: The final set of non-dominated distributions.
        """
        if action_eval == 'hypervolume':
            score_func = self.score_hypervolume
        elif action_eval == 'distance':
            score_func = self.score_inter_distance
        elif action_eval == 'linear':
            score_func = self.score_linear_utility
        else:
            raise Exception('No other method implemented yet')

        if warmup_time is not None:
            self.warmup(warmup_time, score_func)

        for episode in range(num_episodes):
            if episode % log_every == 0:
                print(f'Training episode {episode}')

            state, _ = self.env.reset()
            state = self._flatten_state(state)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.select_action(state, score_func)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._flatten_state(next_state)

                new_nd = self.calc_non_dominated(next_state)
                self.non_dominated[state][action][next_state] = get_best(new_nd, max_dists=self.max_dists, rng=self.rng)
                self.reward_dists[state][action][next_state].update(reward)
                if learn_model:
                    self.transitions[state, action, next_state] += 1
                state = next_state

            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

            if episode % log_every == 0:
                dds = self.get_local_dds(state=0, keep_best=False)
                print(f'Size of the DDS: {len(dds)}')

        return self.get_local_dds(state=0)

    def get_local_dds(self, state=0, keep_best=False):
        """Collect the local DCS in a given state.

        Args:
            state (int, optional): The state to get a local DCS for. (Default value = 0)

        Returns:
            Set: A set of distributional optimal vectors.
        """
        q_dists = [self.get_q_dists(state, action) for action in range(self.num_actions)]
        candidates = [q_dist for q_dist_list in q_dists for q_dist in q_dist_list]
        dds = dd_prune(candidates)
        if keep_best:
            return get_best(dds, max_dists=self.max_dists, rng=self.rng)
        else:
            return dds
