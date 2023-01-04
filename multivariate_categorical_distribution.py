from collections import defaultdict
from itertools import product

import numpy as np
import scipy


class MultivariateCategoricalDistribution:
    """A class to represent a multivariate categorical distribution."""

    def __init__(self, num_atoms, v_mins, v_maxs, name=None):
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
        return np.clip(vec, self.v_mins, self.v_maxs)

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

    def _get_vecs(self):
        """Get the vectors of the distribution."""
        return [np.array(vec) for vec in product(*[dim_thetas[:-1] for dim_thetas in self.thetas])]

    def _vec_to_idx(self, vec):
        """Get the index of the vector in the distribution.

        Args:
            vec (np.ndarray): The vector to get the index of.

        Returns:
            tuple: The index of the vector in the distribution.
        """
        idx = (vec - self.v_mins) / self.gaps
        return tuple([round(val) for val in idx])

    def _idx_to_vec(self, idx):
        """Get the vector of the index in the distribution.

        Args:
            idx (tuple): The index to get the vector of.

        Returns:
            np.ndarray: The vector of the index in the distribution.
        """
        return np.array([dim_thetas[i] for i, dim_thetas in zip(idx, self.thetas)])

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

    def spawn(self):
        """Spawn a new distribution."""
        return MultivariateCategoricalDistribution(self.num_atoms, self.v_mins, self.v_maxs)

    def project_vec(self, vec, method='nearest'):
        """Project a vector onto the distribution.

        Args:
            vec (np.ndarray): The vector to project onto the distribution.
            method (str, optional): What kind of projection to use for the vector. By default the vector is assumed to
                be directly indexable in the distribution.

        Returns:
            list: A list of tuples of the projected vector and the probability of the vector.
        """
        if method == 'direct':
            return [(vec, 1.)]
        elif method == 'nearest':
            return [(self._idx_to_vec(self._vec_to_idx(vec)), 1.)]
        elif method == 'deterministic':
            vec = self._clip_vec(vec)
            min_indices = [self._get_min_theta_idx(vec[idx], self.thetas[idx]) for idx in range(self.num_dims)]
            min_sups = np.array([self.thetas[dim][idx] for dim, idx in enumerate(min_indices)])
            max_sups = np.array([self.thetas[dim][idx + 1] for dim, idx in enumerate(min_indices)])
            zetas = (vec - min_sups) / (max_sups - min_sups)

            edges = []

            for min_sup, max_sup, zeta in zip(min_sups, max_sups, zetas):
                dim_edges = [(min_sup, 1 - zeta), (max_sup, zeta)]
                edges.append(dim_edges)

            delta_vecs_probs = []
            for edge in product(*edges):
                delta_vec = []
                delta_prob = 1
                for point, point_prob in edge:
                    delta_vec.append(point)
                    delta_prob *= point_prob

                delta_vecs_probs.append((np.array(delta_vec), delta_prob))
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

        for vec1 in self.nonzero_vecs():
            for vec2 in other.nonzero_vecs():
                vec = self._clip_vec(vec1 + vec2)
                vec_probs[tuple(vec)] += self.p(vec1) * other.p(vec2)

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
        vecs = self.nonzero_vecs()
        probs = [self.p(vec) for vec in vecs]
        vecs = list(np.array(vecs) * scalar)
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
