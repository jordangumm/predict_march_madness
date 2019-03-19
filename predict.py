import click
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import log_loss

from datautil             import get_train_examples, get_test_examples
from feature_engine       import k_neighbor_probs

from models.seed import predict_by_seed


@click.command()
def main():
    for season in range(2011, 2019):

        train_y, train_X = get_train_examples(season)
        test_y, test_X   = get_test_examples(season)

        train_column = k_neighbor_probs(train_X, train_y, train_X)
        test_column  = k_neighbor_probs(train_X, train_y, test_X)

        train_X = np.append(train_X, train_column, axis=1)
        test_X  = np.append(test_X, test_column, axis=1)

        clf = LogisticRegression(solver='lbfgs').fit(train_X, train_y)

        probs = clf.predict_proba(test_X)
        acc  = clf.score(test_X, test_y)
        loss = log_loss(test_y, probs)
        print(f'{season}  acc: {acc}')
        print(f'{season} loss: {loss}')
        print('')


if __name__ == '__main__':
    main()
