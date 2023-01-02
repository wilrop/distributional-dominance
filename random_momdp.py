import gym
import numpy as np

from gym import spaces


class RandomMOMDP(gym.Env):

    def __init__(self, num_states, num_objectives, num_actions, num_next_states, num_terminal_states, reward_min,
                 reward_max, start_state=0, seed=None):
        self.rng = np.random.default_rng(seed)

        self.num_states = num_states
        self.num_objectives = num_objectives
        self.num_actions = num_actions
        self.num_next_states = num_next_states
        self.num_terminal_states = num_terminal_states
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.start_state = start_state

        self._reward_function = self.rng.uniform(low=reward_min, high=reward_max,
                                                 size=(num_states, num_actions, num_next_states, num_objectives))
        self._terminal_states = self.rng.choice(num_states, size=num_terminal_states, replace=False)
        self._transition_function = self.__init_transition_function()

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self._state = start_state
        self._timestep = 0

    def __init_transition_function(self):
        transition_function = np.zeros((self.num_states, self.num_actions, self.num_states))
        for state in range(self.num_states):
            if state in self._terminal_states:
                transition_function[state] = np.zeros((self.num_actions, self.num_states))
            else:
                for action in range(self.num_actions):
                    next_states = self.rng.choice(self.num_states, size=self.num_next_states, replace=False)
                    for next_state in next_states:
                        self._transition_function[state, action, next_state] = 1 / self.num_next_states
        return transition_function

    def get_config(self):
        return {
            "num_states": self.num_states,
            "num_objectives": self.num_objectives,
            "num_actions": self.num_actions,
            "num_next_states": self.num_next_states,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "start_state": self.start_state,
        }

    def reset(self, seed=None, start_state=None):
        """ Reset the environment and return the initial state number
        """
        # Pick an initial state at random
        self._state = start_state if start_state is not None else self.start_state
        self._timestep = 0

        return self._state, None

    def step(self, action):
        # Change state using the transition function
        next_state = self.rng.choice(self.num_states, p=self._transition_function[self._state, action])
        rewards = self._reward_function[self._state, action, next_state]
        self._state = next_state
        self._timestep += 1

        # Return the current state, a reward and whether the episode terminates
        return self._state, rewards, self._state in self._terminal_states, self._timestep == 50, {}
