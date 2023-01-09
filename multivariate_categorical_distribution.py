import json
import math
import os
from collections import defaultdict
from itertools import product

import numpy as np
import ot
import scipy


class MultivariateCategoricalDistribution:
    """A class to represent a multivariate categorical distribution."""

    def __init__(self, num_atoms, v_mins, v_maxs, decimals=3, name=None):
        # Check if the num atoms is a number and if so wrap it in a list and numpy array.
        if isinstance(num_atoms, (int, float, np.number)):
            num_atoms = np.array([num_atoms])
        else:
            num_atoms = np.array(num_atoms)

        # Check if the mins and maxs are numbers and if so wrap them in a list and numpy array.
        if isinstance(v_mins, (int, float, np.number)):
            v_mins = np.array([v_mins])
        else:
            v_mins = np.array(v_mins)

        if isinstance(v_maxs, (int, float, np.number)):
            v_maxs = np.array([v_maxs])
        else:
            v_maxs = np.array(v_maxs)

        self.num_atoms = num_atoms
        self.num_dims = len(num_atoms)
        self.v_mins = v_mins
        self.v_maxs = v_maxs
        self.decimals = decimals
        self.name = name

        self.gaps = (v_maxs - v_mins) / (num_atoms - 1)
        self.thetas = self._init_thetas()
        self.dist = np.full(num_atoms, 1 / np.prod(num_atoms))  # Uniform distribution.
        self.cdf = self.get_cdf()

    @staticmethod
    def _get_min_theta_idx(val, thetas):
        """Get the index of the minimum theta that is greater than the value.

        Args:
            val (float): The value to compare to the thetas.
            thetas (np.ndarray): The thetas to compare the value to.

        Returns:
            int: The index of the minimum theta that is greater than the value.
        """
        diff = val - thetas
        return diff[diff >= 0].argmin()

    def _clip_vec(self, vec):
        """Clip the vector to the bounds of the distribution.

        Args:
            vec (np.ndarray): The vector to clip.

        Returns:
            np.ndarray: The clipped vector.
        """
        return [min(v_max, max(v_min, item)) for v_min, v_max, item in zip(self.v_mins, self.v_maxs, vec)]

    def _init_thetas(self):
        """Initialize the thetas for each dimension of the distribution.

        Returns:
            list: A list of thetas for each dimension of the distribution.
        """
        thetas = []
        for num_atom, min_val, max_val in zip(self.num_atoms, self.v_mins, self.v_maxs):
            dim_thetas = np.linspace(min_val, max_val, num_atom)
            dim_thetas = np.append(dim_thetas, float('inf'))
            thetas.append(dim_thetas)
        return thetas

    def _cast_in_range(self, vec):
        """Cast a vector in the atom range determined by the minimum and maximum vectors."""
        return [(item - v_min) / gap for item, v_min, gap in zip(vec, self.v_mins, self.gaps)]

    def _vec_to_idx(self, vec):
        """Get the index of the vector in the distribution.

        Args:
            vec (np.ndarray): The vector to get the index of.

        Returns:
            tuple: The index of the vector in the distribution.
        """
        atom = self._cast_in_range(vec)
        return tuple([round(val) for val in atom])

    def _idx_to_vec(self, idx):
        """Get the vector of the index in the distribution.

        Args:
            idx (tuple): The index to get the vector of.

        Returns:
            np.ndarray: The vector of the index in the distribution.
        """
        return np.array([dim_thetas[i] for i, dim_thetas in zip(idx, self.thetas)])

    def get_vecs(self):
        """Get the vectors of the distribution."""
        return [np.array(vec) for vec in product(*[dim_thetas[:-1] for dim_thetas in self.thetas])]

    def dist_categories(self):
        """Get the categories of the distribution."""
        dim_thetas = []
        for atoms, min_val, max_val in zip(self.num_atoms, self.v_mins, self.v_maxs):
            dim_thetas.append(np.linspace(min_val, max_val, atoms))

        thetas_shape = tuple(self.num_atoms) + (self.num_dims,)
        thetas = np.zeros(thetas_shape)

        for idx, theta in zip(np.ndindex(*self.num_atoms), product(*dim_thetas)):
            thetas[idx] = theta
        return thetas

    def static_update(self, vecs, probs):
        """Do a static update of the distribution.

        Args:
            vecs (list): A list of the vectors to update the distribution with.
            probs (array_like): A list of the probabilities of the vectors.
        """
        self.dist = np.zeros(self.num_atoms)
        for vec, prob in zip(vecs, probs):
            delta_vecs_probs = [(delta_vec, prob * delta_prob) for delta_vec, delta_prob in self.project_vec(vec)]
            for delta_vec, delta_prob in delta_vecs_probs:
                if delta_prob > 0:
                    idx = self._vec_to_idx(delta_vec)
                    self.dist[idx] += delta_prob

        self.dist /= np.sum(self.dist)  # Normalize the distribution.
        if self.decimals is not None:
            self.dist = np.around(self.dist, decimals=self.decimals)  # Round the distribution
        self.cdf = self.get_cdf()

    def get_cdf(self):
        """Compute the cumulative distribution function of the distribution."""
        cdf = np.copy(self.dist)
        for i in range(self.num_dims):
            cdf = np.cumsum(cdf, axis=i)
        return cdf

    def expected_value(self):
        """Compute the expected value of the distribution."""
        expectation = np.zeros(self.num_dims)
        for idx in np.ndindex(*self.num_atoms):
            expectation += self.dist[idx] * self._idx_to_vec(idx)
        return expectation

    def p(self, vec):
        """Get the probability of a given vector in the distribution.

        Args:
            vec (np.ndarray): The vector to get the probability of.

        Returns:
            float: The probability of the vector.
        """
        idx = self._vec_to_idx(vec)
        return self.dist[idx]

    def nonzero_vecs(self):
        """Get the nonzero vectors of the distribution."""
        return [self._idx_to_vec(idx) for idx in np.argwhere(self.dist > 0)]

    def nonzero_vecs_probs(self):
        """Get the nonzero vectors of the distribution with their probabilities."""
        return [(self._idx_to_vec(idx), self.dist[tuple(idx)]) for idx in np.argwhere(self.dist > 0)]

    def marginal(self, dim):
        """Get the marginal distribution of a given dimension.

        Args:
            dim (int): The dimension to get the marginal distribution of.

        Returns:
            Distribution: The marginal distribution of the given dimension.
        """
        marginal_dist = MultivariateCategoricalDistribution(self.num_atoms[dim], self.v_mins[dim], self.v_maxs[dim])
        vecs = self.thetas[dim][:-1].reshape(-1, 1)
        probs = np.zeros(len(vecs))
        for idx in np.ndindex(*self.num_atoms):
            marginal_idx = idx[dim]
            probs[marginal_idx] += self.dist[idx]
        marginal_dist.static_update(vecs, probs)
        return marginal_dist

    def js_distance(self, other):
        """Get the Jensen-Shannon distance between two distributions."""
        return scipy.spatial.distance.jensenshannon(self.dist.flatten(), other.dist.flatten())

    def wasserstein_distance(self, other, lambd=1e-1):
        """Get the Wasserstein distance between two distributions."""
        M = ot.dist(np.array(self.get_vecs()))
        if lambd > 0:
            dist = ot.sinkhorn2(self.dist, other.dist, M, lambd)
        else:
            dist = ot.emd2(self.dist, other.dist, M)
        return dist

    def spawn(self):
        """Spawn a new distribution with the same parameters."""
        return MultivariateCategoricalDistribution(self.num_atoms, self.v_mins, self.v_maxs)

    def project_vec(self, vec, method='nearest'):
        """Project a vector onto the distribution.

        Args:
            vec (np.ndarray): The vector to project onto the distribution.
            method (str, optional): What kind of projection to use for the vector. By default the vector is projected
                to the nearest bin.

        Returns:
            list: A list of tuples of the projected vector and the probability of the vector.
        """
        if method == 'direct':
            return [(vec, 1.)]
        elif method == 'nearest':
            return [(self._idx_to_vec(self._vec_to_idx(vec)), 1.)]
        elif method == 'deterministic':
            vec = self._clip_vec(vec)
            b = self._cast_in_range(vec)
            lower = []
            upper = []
            lower_ratios = []
            upper_ratios = []

            for coord in b:
                low = math.floor(coord)
                up = math.ceil(coord)
                lower.append(low)
                upper.append(up)
                lower_ratios.append(up - coord)
                upper_ratios.append(coord - low)

            delta_vecs_probs = []
            for points, probs in zip(product(*zip(lower, upper)), product(*zip(lower_ratios, upper_ratios))):
                delta_vecs_probs.append((np.array(points), math.prod(probs)))
            return delta_vecs_probs
        else:
            raise ValueError('Invalid projection method.')

    def __add__(self, other):
        """Add two distributions together.

        Args:
            other (Distribution): The other distribution to add to this one.

        Returns:
            Distribution: The sum of the two distributions.
        """
        vec_probs = defaultdict(lambda: 0)

        for vec1, prob1 in self.nonzero_vecs_probs():
            for vec2, prob2 in other.nonzero_vecs_probs():
                vec = self._clip_vec(vec1 + vec2)
                vec_probs[tuple(vec)] += prob1 * prob2

        new_dist = MultivariateCategoricalDistribution(self.num_atoms, self.v_mins, self.v_maxs)
        new_dist.static_update(list(vec_probs.keys()), list(vec_probs.values()))
        return new_dist

    def __mul__(self, scalar):
        """Multiply the distribution by a scalar.

        Args:
            scalar (float): The scalar to multiply the distribution by.

        Returns:
            Distribution: The distribution multiplied by the scalar.
        """
        vecs = []
        probs = []
        for vec, prob in self.nonzero_vecs_probs():
            vecs.append(np.array(vec) * scalar)
            probs.append(prob)

        new_dist = MultivariateCategoricalDistribution(self.num_atoms, self.v_mins, self.v_maxs)
        new_dist.static_update(vecs, probs)
        return new_dist

    def __rmul__(self, scalar):
        """Multiply the distribution by a scalar.

        Args:
            scalar (float): The scalar to multiply the distribution by.

        Returns:
            Distribution: The distribution multiplied by the scalar.
        """
        return self.__mul__(scalar)

    def get_config(self):
        return {
            'num_atoms': self.num_atoms.tolist(),
            'v_mins': self.v_mins.tolist(),
            'v_maxs': self.v_maxs.tolist(),
            'decimals': self.decimals,
            'name': self.name,
        }

    def save(self, dir_path, file_name=None):
        """Save the distribution to a file.

        Args:
            dir_path (str): The directory to save the distribution to.
        """
        save_data = self.get_config()
        save_data['dist'] = {'vecs': [], 'probs': []}
        for vec, prob in self.nonzero_vecs_probs():
            save_data['dist']['vecs'].append(vec.tolist())
            save_data['dist']['probs'].append(prob)

        if file_name is None:
            if self.name is None:
                file_name = 'dist'
            else:
                file_name = self.name

        file_name += '.json'
        with open(os.path.join(dir_path, file_name), 'w') as f:
            json.dump(save_data, f)

    def load(self, path):
        """Load the distribution from a file.

        Args:
            path (str): The path to load the distribution from.
        """
        with open(path, 'r') as f:
            dist_data = json.load(f)
        self.static_update(dist_data['dist']['vecs'], dist_data['dist']['probs'])
