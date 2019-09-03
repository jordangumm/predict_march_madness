import os

import click
import collections
import numpy           as np
import pandas          as pd
import xgboost         as xgb

from automaxout.models.maxout   import Maxout
from automaxout.model_selection import GeneticSelector
from cbbstats.data              import load_team_seeds
from cbbstats.game              import get_tournament_results, get_regular_results
from scipy.interpolate          import UnivariateSpline
from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import accuracy_score, log_loss
from sklearn.model_selection    import KFold

from predict_march_madness.datautil       import get_train_examples, get_test_examples
from predict_march_madness.feature_engine import k_neighbor_probs, empirical_probs
from predict_march_madness.build_features import build_features


@click.group()
def cli():
    pass


def load_examples(season: int):
    train_y, train_X = get_train_examples(season)
    test_y, test_X   = get_test_examples(season)

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X  = scaler.transform(test_X)
    train_y = train_y.values
    test_y  = test_y.values

    return train_X, train_y, test_X, test_y


@cli.command()
@click.option('--season', '-s', default=2016)
def logreg(season: int) -> None:
    """Predict games using Logistic Regression.

    Args:
        season: tournament year to build model for and predict

    """
    train_X, train_y, test_X, test_y = load_examples(season)

    train_y, train_X = get_train_examples(season)
    test_y, test_X   = get_test_examples(season)

    train_X = train_X.values
    train_y = train_y.values

    test_X = test_X.values
    test_y = test_y.values

    clf = LogisticRegression(solver='lbfgs').fit(train_X, train_y)

    probs = clf.predict_proba(test_X)
    acc  = clf.score(test_X, test_y)
    loss = log_loss(test_y, probs)
    print(f'{season}  acc: {acc}')
    print(f'{season} loss: {loss}')
    print('')


@cli.command()
@click.option('--season', '-s', default=2016)
@click.option('--verbose', is_flag=True)
def select_maxout(season: int, verbose: bool) -> None:
    train_X, train_y, test_X, test_y = load_examples(season)

    selector = GeneticSelector(
        Maxout,
        train_X,
        train_y,
        node_min=2,
        node_max=200,
        layer_min=1,
        layer_max=2,
        dropout_min=0.0,
        dropout_max=0.5,
        stop_min=2,
        stop_max=5,
        ngen=10,
    )

    num_nodes, num_layers, dropout, early_stop = selector.select_best_model()

    num_nodes, num_layers, dropout, early_stop = 20, 2, 0.1, 3

    clf = Maxout(
        len(test_X[0]),
        num_nodes=num_nodes,
        num_layers=num_layers,
        dropout_rate=dropout,
        early_stop=early_stop,
        verbose=verbose,
    )

    probs = clf.fit(train_X, train_y, test_X, test_y)

    acc  = accuracy_score(test_y, [np.rint(x[1]) for x in probs])
    loss = log_loss(test_y, [x[1] for x in probs])
    print(f'{season}  acc: {acc}')
    print(f'{season} loss: {loss}')


@cli.command()
@click.option('--season', '-s', default=2016)
@click.option('--verbose', is_flag=True)
def maxout(season: int, verbose: bool) -> None:
    """Predict tournament games using Maxout Network.

    Args:
       season:  tournmanet year to build model for and predict
       verbose: whether or not to display epoch information (default=False)

    """
    train_X, train_y, test_X, test_y = load_examples(season)

    num_nodes, num_layers, dropout, early_stop = 10, 1, 0.9, 4

    clf = Maxout(
        len(test_X[0]),
        num_nodes=num_nodes,
        num_layers=num_layers,
        dropout_rate=dropout,
        early_stop=early_stop,
        verbose=verbose,
    )

    # FIXME: integrate bagging procedure
    probs = clf.fit(train_X, train_y, test_X, test_y, test_X)

    acc  = accuracy_score(test_y, [np.rint(x[1]) for x in probs])
    loss = log_loss(test_y, [x[1] for x in probs])
    print(f'{season}  acc: {acc}')
    print(f'{season} loss: {loss}')


if __name__ == '__main__':
    cli()
