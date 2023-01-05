from itertools import product

from dist_dom import dd_prune
from multivariate_categorical_distribution import MultivariateCategoricalDistribution
from utils import zero_init, delta_dist, create_mixture_distribution


class MODVI:
    """
    Implement the Multi-Objective Distributional Value Iteration (MODVI) algorithm.
    """

    def __init__(self, env, gamma, num_atoms, v_mins, v_maxs):
        self.env = env
        self.gamma = gamma
        self.num_atoms = num_atoms
        self.v_mins = v_mins
        self.v_maxs = v_maxs

        try:
            self.start_state = self.env._start_state
        except AttributeError:
            self.start_state = 0

        self.finite_horizon = self.env.finite_horizon
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.transition_function = self.env._transition_function

        self.reward_dists = self._init_reward_dists()
        self.q_dists = self._init_q_dists()
        self.return_dists = self._init_return_dists()

    def _init_reward_dists(self):
        """Initialize the reward distributions for each state, action, and next state."""
        reward_dists = []

        for state in range(self.num_states):
            state_dists = []

            for action in range(self.num_actions):
                action_dists = []

                for next_state in range(self.num_states):
                    reward = self.env._reward_function[state, action, next_state]
                    reward_dist = delta_dist(MultivariateCategoricalDistribution, self.num_atoms, self.v_mins,
                                             self.v_maxs, reward)
                    action_dists.append(reward_dist)
                state_dists.append(action_dists)
            reward_dists.append(state_dists)

        return reward_dists

    def _init_q_dists(self):
        return [[[zero_init(MultivariateCategoricalDistribution, self.num_atoms, self.v_mins, self.v_maxs)] for _ in
                 range(self.num_actions)] for _ in range(self.num_states)]

    def _init_return_dists(self):
        if self.finite_horizon:
            return_dists = []
            for _ in range(self.env.max_timesteps + 1):
                return_dists.append(
                    [[zero_init(MultivariateCategoricalDistribution, self.num_atoms, self.v_mins, self.v_maxs)] for _ in
                     range(self.num_states)])
        else:
            return_dists = [[zero_init(MultivariateCategoricalDistribution, self.num_atoms, self.v_mins, self.v_maxs)]
                            for _ in range(self.num_states)]
        return return_dists

    def _cross_sum(self, list_of_dists, probs):
        """Compute the c

        Args:
            list_of_dists (List[List[Dist]]): List of lists of distributions.
            probs (List[float]): List of probabilities.

        Returns:
            List[Dist]: A list of distributions.
        """
        res = []
        for dists in product(*list_of_dists):
            mixture = create_mixture_distribution(dists, probs, MultivariateCategoricalDistribution, self.num_atoms,
                                                  self.v_mins, self.v_maxs)
            res.append(mixture)
        return res

    def get_dds_fh(self, num_iters):
        for i in range(num_iters):
            print(f"Iteration {i}")

            for t in range(self.env.max_timesteps):
                new_return_dists = [[]] * self.num_states

                for state in range(self.num_states):
                    for action in range(self.num_actions):
                        q_dists = []
                        probs = []

                        for next_state in range(self.num_states):
                            prob = self.transition_function[state, action, next_state]

                            if prob > 0:
                                probs.append(prob)
                                next_state_set = []

                                for return_dist in self.return_dists[t + 1][next_state]:
                                    q_dist = self.reward_dists[state][action][next_state] + self.gamma * return_dist
                                    next_state_set.append(q_dist)

                                q_dists.append(next_state_set)

                        self.q_dists[state][action] = self._cross_sum(q_dists, probs)

                    candidates = [dist for action in range(self.num_actions) for dist in self.q_dists[state][action]]
                    new_return_dists[state] = dd_prune(candidates)

                self.return_dists[t] = new_return_dists

        return self.return_dists[0][self.start_state]

    def get_dds_ih(self, num_iters):
        for i in range(num_iters):
            print(f"Iteration {i}")
            new_return_dists = [[]] * self.num_states

            for state in range(self.num_states):
                for action in range(self.num_actions):
                    q_dists = []
                    probs = []

                    for next_state in range(self.num_states):
                        prob = self.transition_function[state, action, next_state]

                        if prob > 0:
                            probs.append(prob)
                            next_state_set = []

                            for return_dist in self.return_dists[next_state]:
                                q_dist = self.reward_dists[state][action][next_state] + self.gamma * return_dist
                                next_state_set.append(q_dist)

                            q_dists.append(next_state_set)

                    self.q_dists[state][action] = self._cross_sum(q_dists, probs)

                candidates = [dist for action in range(self.num_actions) for dist in self.q_dists[state][action]]
                new_return_dists[state] = dd_prune(candidates)

            self.return_dists = new_return_dists

        return self.return_dists[self.start_state]

    def get_dds(self, num_iters=10):
        """Compute the distributionally non-dominated distributions.

        Args:
            num_iters (int): The number of iterations to run MODVI for.

        Returns:
            List[Dist]: A list of distributionally non-dominated distributions.
        """
        if self.env.finite_horizon:
            return self.get_dds_fh(num_iters)
        else:
            return self.get_dds_ih(num_iters)
