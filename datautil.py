import pandas as pd
import numpy  as np


tourney_games = pd.read_csv('tourney_games.tsv', sep='\t')
regular_games = pd.read_csv('regular_games.tsv', sep='\t')

def get_train_examples(season):
    examples = tourney_games.loc[tourney_games['season'] < season]
    examples = examples.append(regular_games.loc[regular_games['season'] <= season])
    examples = examples.drop('season', axis=1)

    train_y = examples['class']
    train_X = examples[[x for x in examples.columns.values if x != 'class']]

    return train_y, train_X


def get_test_examples(season):
    examples = tourney_games.loc[tourney_games['season'] == season]
    examples = examples.drop('season', axis=1)

    test_y = examples['class']
    test_X = examples[[x for x in examples.columns.values if x != 'class']]

    return test_y, test_X

