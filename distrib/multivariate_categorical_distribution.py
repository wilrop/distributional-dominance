import json
import math
import os
from collections import defaultdict
from itertools import product

import numpy as np
import ot
import scipy


class MCD:
    """A class to represent a multivariate categorical distribution."""

    def __init__(self, num_atoms, v_mins, v_maxs, vecs=None, probs=None, decimals=3, name=None):
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
        self.coordinates = np.array(list(product(*[range(atoms) for atoms in self.num_atoms])))
        self.hist_shape = tuple(self.num_atoms) + (self.num_dims,)

        self.dist = None
        self.expected_value = None
        self.cdf = None
        self.marginals = [None] * self.num_dims
        self._init_dist(vecs, probs)

    def _init_dist(self, vecs, probs):
        """Initialize the distribution.

        Args:
            vecs (array_like): The vectors of the distribution.
            probs (array_like): The probabilities of the vectors of the distribution.
        """
        if vecs is None or probs is None:
            self.static_update([np.zeros(self.num_dims)], [1])
        else:
            self.static_update(vecs, probs)

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
        """Initialize the thetas for each dimension of the distribution."""
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

    def get_marginal(self, dim):
        """Get the marginal distribution of a specific dimension.

        Args:
            dim (int): The dimension to get the marginal of.

        Returns:
            MCD: The marginal distribution of the dimension.
        """
        if self.marginals[dim] is None:
            self.set_marginal(dim)

        return self.marginals[dim]

    def get_expected_value(self):
        """Get the expected value of the distribution."""
        if self.expected_value is None:
            self.set_expected_value()

        return self.expected_value

    def get_cdf(self):
        """Get the cumulative distribution function of the distribution."""
        if self.cdf is None:
            self.set_cdf()

        return self.cdf

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
            vecs (array_like): A list of the vectors to update the distribution with.
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

    def set_cdf(self):
        """Compute the cumulative distribution function of the distribution."""
        cdf = self.dist
        for i in range(self.num_dims):
            cdf = np.cumsum(cdf, axis=i)

        self.cdf = cdf

    def set_expected_value(self):
        """Compute the expected value of the distribution."""
        summed = self.coordinates.reshape(self.hist_shape) * self.gaps
        scaled = summed * np.expand_dims(self.dist, axis=-1)
        self.expected_value = scaled.sum(axis=tuple(range(self.num_dims)))

    def set_marginal(self, dim):
        """Compute the marginal distribution of a given dimension.

        Args:
            dim (int): The dimension to get the marginal distribution of.

        Returns:
            MCD: The marginal distribution of the given dimension.
        """
        vecs = self.thetas[dim][:-1].reshape(-1, 1)
        probs = np.zeros(len(vecs))
        for idx in np.ndindex(*self.num_atoms):
            marginal_idx = idx[dim]
            probs[marginal_idx] += self.dist[idx]
        marginal_dist = MCD(self.num_atoms[dim], self.v_mins[dim], self.v_maxs[dim], vecs=vecs, probs=probs)

        self.marginals[dim] = marginal_dist

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

    def js_distance(self, other):
        """Get the Jensen-Shannon distance between two distributions."""
        return scipy.spatial.distance.jensenshannon(self.dist.flatten(), other.dist.flatten())

    def wasserstein_distance(self, other, lambd=0, smoothing=1e-5):
        """Get the Wasserstein distance between two distributions."""
        return self.ot(other, metric='euclidean', lambd=lambd, smoothing=smoothing)

    def ot(self, other, metric='sqeuclidean', lambd=0, smoothing=1e-5):
        """Get the optimal transport between two distributions."""
        M = ot.dist(self.coordinates, metric=metric)
        distribution1 = self.dist.flatten()
        distribution2 = other.dist.flatten()
        if lambd > 0:
            for distribution in [distribution1, distribution2]:
                distribution += smoothing
                distribution /= np.sum(distribution)
            dist = ot.sinkhorn2(distribution1, distribution2, M, lambd)
        else:
            distribution1 /= np.sum(distribution1)
            distribution2 /= np.sum(distribution2)
            dist = ot.emd2(distribution1, distribution2, M)
        return dist

    def expected_utility(self, u_func):
        """Compute the expected utility of the distribution.

        Args:
            u_func (function): The utility function to use.

        Returns:
            float: The expected utility of the distribution.
        """
        return np.sum(u_func(self.coordinates) * self.dist.flatten())

    def spawn(self):
        """Spawn a new distribution with the same parameters."""
        return MCD(self.num_atoms, self.v_mins, self.v_maxs)

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

    def get_config(self):
        """Get the configuration of the distribution."""
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

    def __add__(self, other):
        """Add two distributions together.

        Args:
            other (Dist): The other distribution to add to this one.

        Returns:
            Dist: The sum of the two distributions.
        """
        vec_probs = defaultdict(lambda: 0)

        for vec1, prob1 in self.nonzero_vecs_probs():
            for vec2, prob2 in other.nonzero_vecs_probs():
                vec = self._clip_vec(vec1 + vec2)
                vec_probs[tuple(vec)] += prob1 * prob2

        vecs = list(vec_probs.keys())
        probs = list(vec_probs.values())
        new_dist = MCD(self.num_atoms, self.v_mins, self.v_maxs, vecs=vecs, probs=probs)
        return new_dist

    def __mul__(self, scalar):
        """Multiply the distribution by a scalar.

        Args:
            scalar (float): The scalar to multiply the distribution by.

        Returns:
            Dist: The distribution multiplied by the scalar.
        """
        vecs = []
        probs = []
        for vec, prob in self.nonzero_vecs_probs():
            vecs.append(self._clip_vec(np.array(vec) * scalar))
            probs.append(prob)

        new_dist = MCD(self.num_atoms, self.v_mins, self.v_maxs, vecs=vecs, probs=probs)
        return new_dist

    def __rmul__(self, scalar):
        """Multiply the distribution by a scalar.

        Args:
            scalar (float): The scalar to multiply the distribution by.

        Returns:
            Dist: The distribution multiplied by the scalar.
        """
        return self.__mul__(scalar)
