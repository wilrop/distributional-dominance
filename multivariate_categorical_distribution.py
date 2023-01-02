from itertools import product

import numpy as np


class MultivariateCategoricalDistribution:
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
        self.thetas = self.__init_thetas()
        self.dist = np.full(num_atoms, 1 / np.prod(num_atoms))  # Uniform distribution.
        self.cdf = self.get_cdf()

    def __init_thetas(self):
        thetas = []
        for num_atom, min_val, max_val in zip(self.num_atoms, self.v_mins, self.v_maxs):
            dim_thetas = np.linspace(min_val, max_val, num_atom)
            dim_thetas = np.append(dim_thetas, float('inf'))
            thetas.append(dim_thetas)
        return thetas

    def __get_vecs(self):
        return list(product(*[dim_thetas[:-1] for dim_thetas in self.thetas]))

    def dist_categories(self):
        dim_thetas = []
        for atoms, min_val, max_val in zip(self.num_atoms, self.v_mins, self.v_maxs):
            dim_thetas.append(np.linspace(min_val, max_val, atoms))

        thetas_shape = tuple(self.num_atoms) + (self.num_dims,)
        thetas = np.zeros(thetas_shape)

        for idx, theta in zip(np.ndindex(*self.num_atoms), product(*dim_thetas)):
            thetas[idx] = theta
        return thetas

    def __vec_to_idx(self, vec):
        idx = (vec - self.v_mins) / self.gaps
        return tuple([int(val) for val in idx])

    def __idx_to_vec(self, idx):
        return np.array([dim_thetas[i] for i, dim_thetas in zip(idx, self.thetas)])

    def static_update(self, vecs, probs):
        self.dist = np.zeros(self.num_atoms)
        for vec, prob in zip(vecs, probs):
            delta_vecs_probs = [(delta_vec, prob * delta_prob) for delta_vec, delta_prob in self.project_vec(vec)]
            for delta_vec, delta_prob in delta_vecs_probs:
                if delta_prob > 0:
                    idx = self.__vec_to_idx(delta_vec)
                    self.dist[idx] += delta_prob

        self.cdf = self.get_cdf()

    def get_cdf(self):
        cdf = np.copy(self.dist)
        for i in range(self.num_dims):
            cdf = np.cumsum(cdf, axis=i)
        return cdf

    def expected_value(self):
        """Compute the expected value of the distribution.

        Returns:
            np.ndarray: The expected value of the distribution.
        """
        expectation = np.zeros(self.num_dims)
        for idx in np.ndindex(*self.num_atoms):
            expectation += self.dist[idx] * np.array([dim_thetas[i] for dim_thetas, i in zip(self.thetas, idx)])
        return expectation

    def __add__(self, other):
        vecs1 = self.__get_vecs()
        vecs2 = other.__get_vecs()
        vecs = vecs1 + vecs2
        vec_probs = {tuple(vec): 0 for vec in vecs}

        for vec in vec_probs.keys():
            prob = 0
            for vec1 in self.__get_vecs():
                vec1 = np.array(vec1)
                proj_diff = self.project_vec(vec - vec1)
                for proj_vec, proj_prob in proj_diff:
                    if proj_prob > 0:
                        prob += self.p(vec1) * other.p(proj_vec) * proj_prob
            vec_probs[vec] = prob

        new_dist = MultivariateCategoricalDistribution(self.num_atoms, self.v_mins, self.v_maxs)
        new_dist.static_update(list(vec_probs.keys()), list(vec_probs.values()))
        return new_dist

    def __mul__(self, scalar):
        vecs = self.__get_vecs()
        vecs = list(np.array(vecs) * scalar)
        new_dist = MultivariateCategoricalDistribution(self.num_atoms, self.v_mins, self.v_maxs)
        new_dist.static_update(vecs, self.dist.flatten())
        return new_dist

    def p(self, vec):
        idx = self.__vec_to_idx(vec)
        return self.dist[idx]

    def nonzero_vecs(self):
        return [self.__idx_to_vec(idx) for idx in np.argwhere(self.dist > 0)]

    def marginal(self, dim):
        marginal_dist = MultivariateCategoricalDistribution(self.num_atoms[dim], self.v_mins[dim], self.v_maxs[dim])
        vecs = self.thetas[dim][:-1].reshape(-1, 1)
        probs = np.zeros(len(vecs))
        for idx in np.ndindex(*self.num_atoms):
            marginal_idx = idx[dim]
            probs[marginal_idx] += self.dist[idx]
        marginal_dist.static_update(vecs, probs)
        return marginal_dist

    @staticmethod
    def __get_min_theta_idx(val, thetas):
        diff = val - thetas
        return diff[diff >= 0].argmin()

    def _clip_vec(self, vec):
        return np.clip(vec, self.v_mins, self.v_maxs)

    def project_vec(self, vec):
        vec = self._clip_vec(vec)
        min_indices = [self.__get_min_theta_idx(vec[idx], self.thetas[idx]) for idx in range(self.num_dims)]
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
