import click
import pandas as pd
import numpy as np

from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import log_loss

from datautil       import get_train_examples, get_test_examples
from feature_engine import k_neighbor_probs
from build_features import build_features


@click.command()
def main():
    for season in range(2011, 2019):

        train_y, train_X = get_train_examples(season)
        test_y, test_X   = get_test_examples(season)

        train_column = k_neighbor_probs(train_X, train_y, train_X, 24, False)
        test_column  = k_neighbor_probs(train_X, train_y, test_X, 24, False)

        train_X['npred'] = train_column
        test_X['npred']  = test_column

        train_X = build_features(train_X, train_y)  # extending training features
        
        import sys
        sys.exit()

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


if __name__ == '__main__':
    main()
