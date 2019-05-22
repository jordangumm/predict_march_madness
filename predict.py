import click
import pandas as pd
import numpy as np

from automaxout.models.maxout import Maxout
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import log_loss

from predict_march_madness.datautil       import get_train_examples, get_test_examples
from predict_march_madness.feature_engine import k_neighbor_probs, empirical_probs
from predict_march_madness.build_features import build_features


@click.group()
def cli():
    pass


@cli.command()
def logreg():
    for season in range(2011, 2019):

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
@click.option('--verbose', is_flag=True)
def maxout(verbose: bool):
    for season in range(2011, 2019):

        train_y, train_X = get_train_examples(season)
        test_y, test_X   = get_test_examples(season)

        scaler = StandardScaler()
        scaler.fit(train_X)

        train_X = scaler.transform(train_X)
        test_X  = scaler.transform(test_X)
        train_y = train_y.values
        test_y  = test_y.values

        clf = Maxout(
            len(test_X[0]),
            num_layers=2,
            num_nodes=70,
            dropout_rate=0.2,
            learning_rate=1e-4,
            weight_decay=None,
            verbose=verbose,
        )

        probs = clf.fit(train_X, train_y, test_X, test_y)

        acc  = clf.score(test_X, test_y)
        loss = log_loss(test_y, probs)
        print(f'{season}  acc: {acc}')
        print(f'{season} loss: {loss}')


if __name__ == '__main__':
    cli()
