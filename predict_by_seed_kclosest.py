import click
import pandas as pd
import numpy as np

from typing import Dict, Iterable

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics         import log_loss, accuracy_score
from tqdm                    import tqdm

from cbbstats.data import load_team_seeds
from cbbstats.game import get_tournament_games
from cbbstats.team import get_team_seed
from datautil      import get_train_examples, get_test_examples

from feature_engine import k_neighbor_probs


SEEDS = load_team_seeds()


def seed_matchup_results(season: int):
    """

    
    """
    X, y = [], []
    games = get_tournament_games(season)
    for game in games:
        wseed = get_team_seed(SEEDS, season, game['WTeamID'])
        lseed = get_team_seed(SEEDS, season, game['LTeamID'])

        X.append([wseed, lseed])
        y.append(1)

        X.append([lseed, wseed])
        y.append(0)
    return X, y



def estimate_best_ksize(X, y, splits=10, kmin=10, kmax=140, step=1) -> int:
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

            probs = k_neighbor_probs(X_train, y_train, X_test, k=k)
            probs = [x if x != 1.0 else 0.99 for x in probs]
            probs = [x if x != 0.0 else 0.01 for x in probs]

            loss = log_loss(y_test, probs)
            if loss < best_loss:
                best_loss  = loss
                best_ksize = k

    return best_ksize


def seed_matchup_probs(train_X, train_y, test_X):
    """Calculate probabilities given 

    """
    best_ksize = estimate_best_ksize(train_X, train_y)
    print(f'best ksize: {best_ksize}')

    probs = k_neighbor_probs(train_X, train_y, test_X, k=best_ksize)
    probs = [x if x != 1.0 else 0.99 for x in probs]
    probs = [x if x != 0.0 else 0.01 for x in probs]

    return probs


@click.command()
@click.option('--verbose/--not-verbose', default=False)
def main(verbose):

    for season in range(2010, 2019):
        print(f'\n\n{season}')

        train_X, train_y = [], []
        for s in range(1985, season):
            X, y = seed_matchup_results(s)
            train_X += X
            train_y += y

        test_X, test_y = seed_matchup_results(season)

        probs = seed_matchup_probs(train_X, train_y, test_X)
        acc  = accuracy_score(test_y, [round(prob) for prob in probs])
        loss = log_loss(test_y, probs)
        print(f'{season} acc: {acc}')
        print(f'{season} loss: {loss}')
        print('')


if __name__ == '__main__':
    main()
