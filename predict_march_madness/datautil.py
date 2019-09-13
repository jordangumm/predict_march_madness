import os

from pathlib import Path
from typing  import Tuple

import pandas as pd
import numpy  as np


dirpath = Path(os.path.dirname(os.path.realpath(__file__)))
tourney_games = pd.read_csv(dirpath / 'data/tourney_games.tsv', sep='\t')
regular_games = pd.read_csv(dirpath / 'data/regular_games.tsv', sep='\t')

tourney_scoring_games = pd.read_csv(dirpath / 'data/tourney_games_nodups.tsv', sep='\t')

def get_train_examples(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    examples = tourney_games.loc[tourney_games['season'] < season]
    examples = examples.append(regular_games.loc[regular_games['season'] <= season])
    examples.drop('season', axis=1, inplace=True)
    examples.drop('WTeamID', axis=1, inplace=True)
    examples.drop('LTeamID', axis=1, inplace=True)

    train_y = examples['class']
    train_X = examples[[x for x in examples.columns.values if x != 'class']]

    return train_y, train_X


def add_team_info(team_info: pd.DataFrame) -> pd.DataFrame:
    """Add extra team information to DataFrame.

    Args:
        team_info: games with team identifiers and season

    Returns:
        Expanded team info.

    """
    return team_info


def get_test_examples(season: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Query test examples for season.

    Args:
        season: tournament year

    Returns:
        class, features, team info

    """
    examples  = tourney_scoring_games.loc[tourney_scoring_games['season'] == season]
    team_info = add_team_info(examples[['season', 'WTeamID', 'LTeamID']])

    examples.drop('season',  axis=1, inplace=True)
    examples.drop('WTeamID', axis=1, inplace=True)
    examples.drop('LTeamID', axis=1, inplace=True)

    test_y = examples['class']
    test_X = examples[[x for x in examples.columns.values if x != 'class']]

    return test_y, test_X, team_info
