import numpy as np

class ClusterFilter2:
    def __init__(self, max_clusters: int, max_weight: int, embedding_size: int, thresh: float):
        """
        Initializes the ClusterFilter with the given parameters.

        :param max_clusters: Maximum number of clusters to maintain.
        :param max_weight: Maximum weight for a cluster.
        :param embedding_size: Size of the embedding for each cluster.
        :param thresh: Threshold for the distance to consider a point as part of a cluster.
        """
        self.means = np.zeros((max_clusters, embedding_size))
        self.weights = np.zeros((max_clusters,))
        self.staleness = np.zeros((max_clusters,))
        self.max_weight = max_weight
        self.max_clusters = max_clusters
        self.base_radius = thresh
        self.MAX_STALENESS = 1000
        self.staleness.fill(self.MAX_STALENESS)  # Initialize staleness to a high value

    def insert(self, mean: np.ndarray) -> bool:
        mean = np.array(mean)

        assert mean.shape == self.means.shape[1:] and self.means.shape[0] == self.max_clusters and self.weights.shape == (self.max_clusters,)

        l2_norm = np.linalg.norm(self.means - mean, axis=1)
        close = l2_norm <= np.sqrt(self.weights) * self.base_radius

        # increment staleness for all clusters
        self.staleness += 1

        if np.any(close):
            # If the mean is close to existing clusters, update those clusters and merge them
            center = (np.sum((self.means[close].T * self.weights[close]).T, axis = 0) + mean) / (np.sum(self.weights[close]) + 1)
            weight = min(np.sum(self.weights[close]) + 1, self.max_weight)

            # Get indices of all clusters participating in the merge
            close_indices = np.where(close)[0]
            
            # Designate the first one as the "survivor"
            survivor_idx = close_indices[0]

            self.means[survivor_idx] = center
            self.weights[survivor_idx] = weight
            self.staleness[survivor_idx] = 0

            if len(close_indices) > 1:
                # If there are multiple clusters being merged, we need to handle them
                for idx in close_indices[1:]:
                    self.means[idx] = np.zeros_like(self.means[idx])
                    self.weights[idx] = 0
                    self.staleness[idx] = self.MAX_STALENESS  # mark as very stale so new clusters will overwrite these first

            return False
        else:
            most_stale = np.argmax(self.staleness)

            # If the mean is not close to any existing cluster, add it as a new cluster, removing the most stale one
            self.means[most_stale] = mean
            self.weights[most_stale] = 1
            self.staleness[most_stale] = 0

            return True

if __name__ == '__main__':
    f = ClusterFilter2(3, 5, 2, 0.5)
    def check(x, y):
        if not np.all(np.abs(f.means - x) < 0.0001) or not np.all(np.abs(f.weights - y) < 0.0001):
            raise RuntimeError(f'filter error:\n\n{f.means}\n\n{f.weights}')
    check([(0, 0), (0, 0), (0, 0)], [0, 0, 0])

    assert f.insert([1, 2]) == True
    check([(1, 2), (0, 0), (0, 0)], [1, 0, 0])

    assert f.insert([-3, 2]) == True
    check([(1, 2), (-3, 2), (0, 0)], [1, 1, 0])

    assert f.insert([-3.2, 2.1]) == False
    check([(1, 2), (-3.1, 2.05),  (0, 0)], [1, 2, 0])

    assert f.insert([-4, 1]) == True
    check([(1, 2), (-3.1, 2.05), (-4, 1)], [1, 2, 1])

    assert f.insert([-4, 1.8]) == True
    check([(-4, 1.8), (-3.1, 2.05), (-4, 1)], [1, 2, 1])

    assert f.insert([-3.4, 2.2]) == False
    check([(-4, 1.8), (-3.2, 2.1),  (-4, 1)], [1, 3, 1])

    assert f.insert([-4.1, 1.41]) == False
    check([(-4.03333, 1.403333), (-3.2, 2.1), (0, 0)], [3, 3, 0])

    assert f.insert([-3, 2]) == False
    check([(-4.033333, 1.4033333), (-3.15, 2.075), (0, 0)], [3, 4, 0])

    assert f.insert([-3.59, 1.74]) == False
    check([(-3.53624875, 1.78125), (0, 0), (0, 0)], [5, 0, 0])

    print('passed all tests (no output means good)')
