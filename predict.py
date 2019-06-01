import click
import pandas as pd
import numpy as np

from automaxout.models.maxout   import Maxout
from automaxout.model_selection import GeneticSelector
from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import log_loss

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

        selector = GeneticSelector(
            Maxout,
            train_X,
            train_y,
            node_min=2,
            node_max=100,
            layer_min=1,
            layer_max=2,
            dropout_min=0.0,
            dropout_max=0.5,
            ngen=10,
        )

        num_nodes, num_layers, dropout, early_stop = selector.select_best_model()

        clf = Maxout(
            len(test_X[0]),
            num_nodes=num_nodes,
            num_layers=num_layers,
            dropout_rate=dropout,
            early_stop=early_stop,
            verbose=verbose,
        )

        probs = clf.fit(train_X, train_y, test_X, test_y)

        acc  = clf.score(test_X, test_y)
        loss = log_loss(test_y, probs)
        print(f'{season}  acc: {acc}')
        print(f'{season} loss: {loss}')


if __name__ == '__main__':
    cli()
