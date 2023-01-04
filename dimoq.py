import numpy as np
import wandb
from gym.spaces import Box
from torch.utils.tensorboard import SummaryWriter

from count_based_mcd import CountBasedMCD
from dist_dom import dd_prune
from dist_metrics import dist_hypervolume
from multivariate_categorical_distribution import MultivariateCategoricalDistribution
from utils import zero_init


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
        self.non_dominated = [[[zero_init(MultivariateCategoricalDistribution, num_atoms, v_mins, v_maxs)] for _ in
                               range(self.num_actions)] for _ in range(self.num_states)]
        self.reward_dists = [[zero_init(CountBasedMCD, num_atoms, v_mins, v_maxs) for _ in range(self.num_actions)] for
                             _ in range(self.num_states)]

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            self.setup_wandb()

    def setup_wandb(self):
        """Set up the wandb logging."""
        self.experiment_name = self.experiment_name

        wandb.init(
            project=self.project_name,
            sync_tensorboard=True,
            config=self.get_config(),
            name=self.experiment_name,
            monitor_gym=False,
            save_code=True,
        )
        self.writer = SummaryWriter(f"/tmp/{self.experiment_name}")

    def close_wandb(self):
        """Close the wandb logging."""
        self.writer.close()
        wandb.finish()

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
        q_dists = [self.get_q_dists(state, action) for action in range(self.num_actions)]
        action_scores = [dist_hypervolume(self.ref_point, q_dist) for q_dist in q_dists]
        return action_scores

    def get_q_dists(self, state, action):
        """Compute the distributional Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            List[Dist]: A list of Q distributions.
        """
        nd_dists = self.non_dominated[state][action]
        reward_dist = self.reward_dists[state][action]
        q_dists = [reward_dist + nd_dist * self.gamma for nd_dist in nd_dists]
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
        q_dists = [self.get_q_dists(state, action) for action in range(self.num_actions)]
        q_dists = [q_dist for q_dist_list in q_dists for q_dist in q_dist_list]
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

    def train(self, num_episodes=3000, log_every=100, action_eval='hypervolume'):
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
        else:
            raise Exception('No other method implemented yet')

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

                self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                self.reward_dists[state][action].update(reward)
                state = next_state

            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

            if episode % log_every == 0:
                pf = self.get_local_dcs(state=0)
                evs = [dist.expected_value() for dist in pf]
                print(f'Expected values: {evs}')
                value = dist_hypervolume(self.ref_point, pf)
                print(f'Hypervolume after episode {episode}: {value}')
                if self.log:
                    self.writer.add_scalar("train/hypervolume", value, episode)

        return self.get_local_dcs(state=0)

    def track_policy(self, vec):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector.
        """
        target = np.array(vec)
        state, _ = self.env.reset()
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)

        while not (terminated or truncated):
            state = self._flatten_state(state)
            new_target = False

            for action in range(self.num_actions):
                im_rew = self.reward_dists[state][action].expected_value()
                nd = self.non_dominated[state][action]
                for q in nd:
                    q = np.array(q.expected_value())
                    if np.all(self.gamma * q + im_rew == target):
                        state, reward, terminated, truncated, _ = self.env.step(action)
                        total_rew += reward
                        target = q
                        new_target = True
                        break

                if new_target:
                    break

        return total_rew

    def get_local_dcs(self, state=0):
        """Collect the local DCS in a given state.

        Args:
            state (int, optional): The state to get a local DCS for. (Default value = 0)

        Returns:
            Set: A set of distributional optimal vectors.
        """
        q_dists = [self.get_q_dists(state, action) for action in range(self.num_actions)]
        candidates = [q_dist for q_dist_list in q_dists for q_dist in q_dist_list]
        return dd_prune(candidates)
