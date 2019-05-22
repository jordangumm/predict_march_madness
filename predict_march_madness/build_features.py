"""Build features from initial training set.

There may be something more to the KNeighbors feature engine.

"""
import pandas as pd
import numpy  as np

from sklearn.metrics import log_loss

from predict_march_madness.feature_engine import k_neighbor_probs


def build_features(X: pd.DataFrame, y: pd.Series):
    """Build out features from initial set.

    """
    for feature in X.columns:
        total_probs = pd.DataFrame()
        for k in range(1, 20):
            newfeature = X[feature].values.reshape(-1, 1)
            probs = k_neighbor_probs(newfeature, y, newfeature, k)
            total_probs[f'{k}'] = probs
            
            # predict with probs
            loss1 = log_loss(y, probs)

            # predict with mean total_probs
            mean_probs = [np.mean(x) for x in total_probs.values]
            loss2 = log_loss(y, probs)

            # predict with total_preds via Logistic Regression
            

            print(f'{k}: {loss1} - {loss2}')
