from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import log_loss

import pandas as pd
import numpy as np


df         = pd.read_csv('tourney_games.tsv', sep='\t', header=None)
df_regular = pd.read_csv('regular_games.tsv', sep='\t', header=None)


for season in range(2011, 2019):
    test_examples = df.loc[df.iloc[:,-1] == season]
    test_examples.drop(test_examples.columns[len(test_examples.columns)-1], axis=1, inplace=True)
    test_y = test_examples.iloc[:,0].values
    test_X = test_examples.iloc[:,1:].values

    train_examples = df.loc[df.iloc[:,-1] < season]
    train_examples.drop(train_examples.columns[len(train_examples.columns)-1], axis=1, inplace=True)
    train_y = train_examples.iloc[:,0].values
    train_X = train_examples.iloc[:,1:].values

    train_examples = df_regular.loc[df_regular.iloc[:,-1] <= season]
    train_examples.drop(train_examples.columns[len(train_examples.columns)-1], axis=1, inplace=True)
    train_y = np.concatenate((train_y, train_examples.iloc[:,0].values))
    train_X = np.concatenate((train_X, train_examples.iloc[:,1:].values), axis=0)


    clf = LogisticRegression(solver='lbfgs').fit(train_X, train_y)

    probs = clf.predict_proba(test_X)
    print(probs)
    acc  = clf.score(test_X, test_y)
    loss = log_loss(test_y, probs)
    print(f'{season}  acc: {acc}')
    print(f'{season} loss: {loss}')
