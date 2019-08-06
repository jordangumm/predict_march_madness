import os

from pathlib import Path

import pandas as pd
import numpy  as np


dirpath = Path(os.path.dirname(os.path.realpath(__file__)))
tourney_games = pd.read_csv(dirpath / 'predict_march_madness/tourney_games.tsv', sep='\t')
regular_games = pd.read_csv(dirpath / 'predict_march_madness/regular_games.tsv', sep='\t')

tourney_scoring_games = pd.read_csv(dirpath / 'predict_march_madness/tourney_games_scoring.tsv', sep='\t')

def get_train_examples(season):
    examples = tourney_games.loc[tourney_games['season'] < season]
    examples = examples.append(regular_games.loc[regular_games['season'] <= season])
    examples.drop('season', axis=1, inplace=True)
    examples.drop('WTeamID', axis=1, inplace=True)
    examples.drop('LTeamID', axis=1, inplace=True)

    train_y = examples['class']
    train_X = examples[[x for x in examples.columns.values if x != 'class']]

    return train_y, train_X


def get_test_examples(season):
    examples = tourney_games.loc[tourney_games['season'] == season]
    examples = examples.drop('season', axis=1)
    examples.drop('WTeamID', axis=1, inplace=True)
    examples.drop('LTeamID', axis=1, inplace=True)

    test_y = examples['class']
    test_X = examples[[x for x in examples.columns.values if x != 'class']]

    return test_y, test_X


def get_tournament_results(compact: bool = False) -> pd.DataFrame:
    if compact:
        gamesfile = dirpath / 'data/original/NCAATourneyCompactResults.csv'
    else:
        gamesfile = dirpath / 'data/original/NCAATourneyDetailedResults.csv'

    return pd.read_csv(gamesfile)


def get_regular_results(compact: bool = False) -> pd.DataFrame:
    if compact:
        gamesfile = dirpath / 'data/original/RegularSeasonCompactResults.csv'
    else:
        gamesfile = dirpath / 'data/original/RegularSeasonDetailedResults.csv'

    return pd.read_csv(gamesfile)


def load_team_seeds() -> pd.DataFrame:
    return pd.read_csv(dirpath / 'data/original/NCAATourneySeeds.csv')
