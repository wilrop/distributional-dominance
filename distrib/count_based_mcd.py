import numpy as np

from distrib.multivariate_categorical_distribution import MCD


class CountBasedMCD(MCD):
    def __init__(self, num_atoms, v_mins, v_maxs, vecs=None, probs=None, decimals=3, name=None):
        super().__init__(num_atoms, v_mins, v_maxs, vecs=vecs, probs=probs, decimals=decimals, name=name)
        self.counts = np.zeros(self.num_atoms)

    def update(self, vec):
        """Update the distribution with a new vector.

        Args:
            vec (ndarray): The vector to update the distribution with.
        """
        projected_vecs = self.project_vec(vec)
        for projected_vec, prob in projected_vecs:
            self.counts[self._vec_to_idx(projected_vec)] += prob
        self.dist = self.counts / np.sum(self.counts)
