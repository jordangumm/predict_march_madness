"""

1. KNN as a feature engine: Practical Statistics for Data Science, 2017

"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics   import log_loss
from tqdm              import tqdm


def k_neighbor_probs(train_X, train_y, test_X, k=24, optimize=False):
    """Calculate locality-based probabilities.

    Args:
        train_X:  matrix of examples by row
        train_y:  vector of example outcomes
        test_X:   matrix of test examples by row
        n:        number of closest neighbors to use to predict probability
        optimize: whether or not to go through expensive cross-validation

    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_X, train_y)

    preds = neigh.predict_proba(test_X)[:,1]
    return preds.reshape(len(preds), 1)
