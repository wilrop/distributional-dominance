from dist_dom import dd_prune
from multivariate_categorical_distribution import MultivariateCategoricalDistribution
from utils import zero_init, delta_dist
from itertools import product


class MODVI:
    def __init__(self, env, gamma, num_atoms, v_mins, v_maxs):
        self.env = env
        self.gamma = gamma
        self.num_atoms = num_atoms
        self.v_mins = v_mins
        self.v_maxs = v_maxs

        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.transition_function = self.env._transition_function
        self.reward_dists = self._init_reward_dists()
        self.q_dists = [[[zero_init(MultivariateCategoricalDistribution, num_atoms, v_mins, v_maxs)] for _ in
                         range(self.num_actions)] for _ in range(self.num_states)]
        self.return_dists = [[zero_init(MultivariateCategoricalDistribution, num_atoms, v_mins, v_maxs)] for _ in
                             range(self.num_states)]

    def _init_reward_dists(self):
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

    def _cross_sum(self, set_of_dists):
        res = []
        for dists in product(*set_of_dists):
            res.append(sum(dists, zero_init(MultivariateCategoricalDistribution, self.num_atoms, self.v_mins, self.v_maxs)))
        return res

    def get_dds(self, num_iters=10):
        for i in range(num_iters):
            print(f"Iteration {i}")

            for state in range(self.num_states):

                for action in range(self.num_actions):
                    q_dists = []

                    for next_state in range(self.num_states):
                        prob = self.transition_function[state, action, next_state]

                        if prob > 0:
                            prob_set = []
                            for return_dist in self.return_dists[next_state]:
                                q_dist = prob * (self.reward_dists[state][action][next_state] + self.gamma * return_dist)
                                prob_set.append(q_dist)

                            q_dists.append(prob_set)

                    self.q_dists[state][action] = self._cross_sum(q_dists)

                candidates = [dist for action in range(self.num_actions) for dist in self.q_dists[state][action]]
                self.return_dists[state] = dd_prune(candidates)

        return self.return_dists[0]
