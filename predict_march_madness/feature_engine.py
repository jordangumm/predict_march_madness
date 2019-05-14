"""

1. KNN as a feature engine: Practical Statistics for Data Science, 2017

"""
import numpy as np
import pandas as pd

from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics         import log_loss
from tqdm                    import tqdm


def estimate_best_ksize(X, y, splits=10, kmin=10, kmax=140, step=10) -> int:
    """Estimate best ksize for sklearn KNeighborsClassifier using cross validation.

    Args:
        X:      training data set features
        y:      training data set labels
        splits: number of splitting iterations to use in the cross-validator
        kmin:   ksize search starting point
        kmin:   ksize search stoping point
        step:   size of increase in ksize between trials

    Returns:
        Estimated best ksize.

    """
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=splits)

    best_loss = 100
    best_ksize = kmin
    for k in tqdm(range(kmin, kmax, step)):
        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            probs = k_neighbor_probs(X_train, y_train, X_test, k, False)
            probs = [x if x != 1.0 else 0.99 for x in probs]
            probs = [x if x != 0.0 else 0.01 for x in probs]

            loss = log_loss(y_test, probs)
            if loss < best_loss:
                best_loss  = loss
                best_ksize = k

    return best_ksize


def k_neighbor_probs(train_X, train_y, test_X, k=None, optimize=True):
    """Calculate locality-based probabilities.

    Args:
        train_X:  matrix of examples by row
        train_y:  vector of example outcomes
        test_X:   matrix of test examples by row
        n:        number of closest neighbors to use to predict probability
        optimize: whether or not to go through expensive cross-validation

    """
    if optimize:
        best_ksize = estimate_best_ksize(train_X, train_y)
    else:
        best_ksize = k

    neigh = KNeighborsClassifier(n_neighbors=best_ksize)
    neigh.fit(train_X, train_y)

    preds = neigh.predict_proba(test_X)[:,1]
    preds = preds.reshape(len(preds), 1)
    return [x[0] for x in preds]


def empirical_probs(train_X, train_y, test_X):
    """Generate LogisticRegression probabilities.

    Assumes that train_X is a one-dimensional array

    """
    from sklearn.linear_model import LogisticRegression

    train_X = train_X.values.reshape(-1, 1)

    clf = LogisticRegression(solver='lbfgs').fit(train_X, train_y)


    diffs = [x for x in test_X]
    probs = []
    for diff in diffs:
        probs.append(clf.predict_proba([[diff],])[0][1])

    probs = pd.Series(probs)
    #print(probs.describe())
    return pd.Series(probs)

