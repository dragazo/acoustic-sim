import numpy as np
import typing

class ClusterFilter:
    def __init__(self, max_clusters: int, max_weight: int, embedding_size: int, thresh: float, distance_metric: str = 'euclidean'):
        """
        Initializes the ClusterFilter with the given parameters.

        :param max_clusters: Maximum number of clusters to maintain.
        :param max_weight: Maximum weight for a cluster.
        :param embedding_size: Size of the embedding for each cluster.
        :param thresh: Threshold for the distance to consider a point as part of a cluster.
        """
        self.means = np.zeros((max_clusters, embedding_size))
        self.weights = np.zeros((max_clusters,))
        self.max_weight = max_weight
        self.max_clusters = max_clusters
        self.base_radius = thresh
        self.distance_metric = distance_metric

    def insert(self, mean: np.ndarray) -> bool:
        mean = np.array(mean)

        assert mean.shape == self.means.shape[1:] and self.means.shape[0] == self.max_clusters and self.weights.shape == (self.max_clusters,)

        if self.distance_metric == 'euclidean':
            distance = np.sqrt(np.sum((self.means - mean) ** 2, axis = 1))
        elif self.distance_metric == 'cosine':
            distance = 1 - np.sum(self.means * mean, axis=1) / (np.linalg.norm(self.means, axis=1) * np.linalg.norm(mean) + 1e-8)

        close = distance <= np.sqrt(self.weights) * self.base_radius

        if np.any(close):
            # If the mean is close to an existing cluster, update that cluster
            center = (np.sum((self.means[close].T * self.weights[close]).T, axis = 0) + mean) / (np.sum(self.weights[close]) + 1)
            self.means = np.concatenate([
                np.zeros((self.means.shape[0] - (np.sum(~close) + 1), self.means.shape[1])),
                self.means[~close],
                [ center ],
            ])

            weight = min(np.sum(self.weights[close]) + 1, self.max_weight)
            self.weights = np.concatenate([
                np.zeros((self.weights.shape[0] - (np.sum(~close) + 1),)),
                self.weights[~close],
                [ weight ],
            ])
            return False
        else:
            # If the mean is not close to any existing cluster, add it as a new cluster, removing the oldest one
            self.means = np.concatenate([
                self.means[1:],
                [ mean ],
            ])
            self.weights = np.concatenate([
                self.weights[1:],
                [ 1 ],
            ])
            return True

if __name__ == '__main__':
    f = ClusterFilter(3, 5, 2, 0.5)
    def check(x, y):
        if not np.all(np.abs(f.means - x) < 0.0001) or not np.all(np.abs(f.weights - y) < 0.0001):
            raise RuntimeError(f'filter error:\n\n{f.means}\n\n{f.weights}')
    check([(0, 0), (0, 0), (0, 0)], [0, 0, 0])

    assert f.insert([1, 2]) == True
    check([(0, 0), (0, 0), (1, 2)], [0, 0, 1])

    assert f.insert([-3, 2]) == True
    check([(0, 0), (1, 2), (-3, 2)], [0, 1, 1])

    assert f.insert([-3.2, 2.1]) == False
    check([(0, 0), (1, 2), (-3.1, 2.05)], [0, 1, 2])

    assert f.insert([-4, 1]) == True
    check([(1, 2), (-3.1, 2.05), (-4, 1)], [1, 2, 1])

    assert f.insert([-4, 1.8]) == True
    check([(-3.1, 2.05), (-4, 1), (-4, 1.8)], [2, 1, 1])

    assert f.insert([-3.4, 2.2]) == False
    check([(-4, 1), (-4, 1.8), (-3.2, 2.1)], [1, 1, 3])

    assert f.insert([-4.1, 1.41]) == False
    check([(0, 0), (-3.2, 2.1), (-4.033333, 1.4033333)], [0, 3, 3])

    assert f.insert([-3, 2]) == False
    check([(0, 0), (-4.033333, 1.4033333), (-3.15, 2.075)], [0, 3, 4])

    assert f.insert([-3.59, 1.74]) == False
    check([(0, 0), (0, 0), (-3.53624875, 1.78125)], [0, 0, 5])

    print('passed all tests (no output means good)')
