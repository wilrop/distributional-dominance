import numpy as np

from multivariate_categorical_distribution import MultivariateCategoricalDistribution


class CountBasedMCD(MultivariateCategoricalDistribution):
    def __init__(self, num_atoms, v_mins, v_maxs, name=None):
        super().__init__(num_atoms, v_mins, v_maxs, name=name)
        self.counts = np.zeros(self.num_atoms)

    def update(self, vec):
        projected_vecs = self.project_vec(vec)
        for projected_vec, prob in projected_vecs:
            self.counts[self.__vec_to_idx(projected_vec)] += prob
        self.dist = self.counts / np.sum(self.counts)
        self.cdf = self.get_cdf()
