import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    """
    K-nearest neighbors implementation using euclidean distance.
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # using helper function to handle predictions for each test point
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # calculating the distance to each sample in the training set
        distances = [euclidean_distance(x, x_train)
                     for x_train in self.X_train]

        # indices of the k-nearest samples
        k_indices = np.argsort(distances)[:self.k]

        # labels of the k-nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, most frequent label in the k-nearest neighbors
        majority_vote = Counter(k_nearest_labels).most_common(1)

        return majority_vote[0][0]
