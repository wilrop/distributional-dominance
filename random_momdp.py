import gym
import numpy as np

from gym import spaces


class RandomMOMDP(gym.Env):
    """A class to generate random MOMDPs."""

    def __init__(self, num_states, num_objectives, num_actions, num_next_states, num_terminal_states, reward_min,
                 reward_max, reward_dist='uniform', start_state=0, max_timesteps=None, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.num_states = num_states
        self.num_objectives = num_objectives
        self.num_actions = num_actions
        self.num_next_states = num_next_states
        self.num_terminal_states = num_terminal_states
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.reward_dist = reward_dist
        self.start_state = start_state

        self._reward_function = self._init_reward_function()
        self._terminal_states = self._init_terminal_states()
        self._transition_function = self._init_transition_function()

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)
        self.reward_space = spaces.Box(low=reward_min, high=reward_max)

        self.finite_horizon = max_timesteps is not None
        self.max_timesteps = max_timesteps
        self._state = start_state
        self._timestep = 0

    def _init_terminal_states(self):
        """Initialize the terminal states.

        Returns:
            ndarray: A list of terminal states.
        """
        if self.num_terminal_states > self.num_states:
            raise ValueError('The number of terminal states cannot be greater than the number of states')
        elif self.start_state is None:
            candidates = np.arange(self.num_states)
        else:
            candidates = np.delete(np.arange(self.num_states), self.start_state)

        return self.rng.choice(candidates, size=self.num_terminal_states, replace=False)

    def _init_reward_function(self):
        """Generate a reward function with rewards drawn from a given distribution."""
        if self.reward_dist == 'uniform':
            return self.rng.uniform(low=self.reward_min, high=self.reward_max,
                                    size=(self.num_states, self.num_actions, self.num_states, self.num_objectives))
        elif self.reward_dist == 'discrete':
            return self.rng.integers(low=self.reward_min,
                                     high=self.reward_max,
                                     size=(self.num_states,
                                           self.num_actions,
                                           self.num_states,
                                           self.num_objectives))
        else:
            raise ValueError("Invalid reward distribution")

    def _init_transition_function(self):
        """Initialize the transition function.

        Returns:
            ndarray: A transition function.
        """
        transition_function = np.zeros((self.num_states, self.num_actions, self.num_states))
        for state in range(self.num_states):
            if state in self._terminal_states:
                probs = np.zeros((self.num_actions, self.num_states))
                probs[:, state] = 1
                transition_function[state] = probs
            else:
                for action in range(self.num_actions):
                    next_states = self.rng.choice(self.num_states, size=self.num_next_states, replace=False)
                    probs = np.random.dirichlet(np.ones(self.num_next_states))
                    for next_state, prob in zip(next_states, probs):
                        transition_function[state, action, next_state] = prob
        return transition_function

    def get_config(self):
        """Get the configuration of the environment."""
        return {
            "num_states": self.num_states,
            "num_objectives": self.num_objectives,
            "num_actions": self.num_actions,
            "num_next_states": self.num_next_states,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "start_state": self.start_state,
            "seed": self.seed
        }

    def reset(self, seed=None, start_state=None):
        """Reset the environment.

        Args:
            seed (int, optional): The seed to use. (Default value = None)
            start_state (int, optional): The state to start in. (Default value = None)

        Returns:
            int, Dict: The initial state and no info.
        """
        # Pick an initial state at random
        self._state = start_state if start_state is not None else self.start_state
        self._timestep = 0

        return self._state, {}

    def step(self, action):
        """Take a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            int, ndarray, bool, bool, dict: The next state, the reward, whether the episode is done, whether the episode
                is truncated and no info.
        """
        next_state = self.rng.choice(self.num_states, p=self._transition_function[self._state, action])
        rewards = self._reward_function[self._state, action, next_state]
        self._state = next_state
        self._timestep += 1

        return self._state, rewards, self._state in self._terminal_states, self._timestep == self.max_timesteps, {}
