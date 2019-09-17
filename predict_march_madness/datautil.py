import os

from pathlib import Path
from typing  import Tuple

import pandas as pd
import numpy  as np

from cbbstats.data import load_team_seeds, load_team_rankings
from cbbstats.team import get_team_seed, get_team_ranking

from predict_march_madness.predict_by_seed_diff import seed_probs


dirpath = Path(os.path.dirname(os.path.realpath(__file__)))
tourney_games = pd.read_csv(dirpath / 'data/tourney_games.tsv', sep='\t')
regular_games = pd.read_csv(dirpath / 'data/regular_games.tsv', sep='\t')
SEEDS         = load_team_seeds()
RANKINGS      = load_team_rankings('WLK', daynum=133)  # final pre-tournament rankings

tourney_scoring_games = pd.read_csv(dirpath / 'data/tourney_games_nodups.tsv', sep='\t')

tourney_margin_games = pd.read_csv(dirpath / 'data/tourney_games_margin.tsv', sep='\t')
tourney_scoring_margin_games = pd.read_csv(dirpath / 'data/tourney_games_margin_nodups.tsv', sep='\t')


# FIXME: test if this is a useful feature
def append_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append features for one season."""
    season = int(df['season'].unique()[0])
    probs, _ = seed_probs(df, season)
    df.loc[:, 'seed_probs'] = probs

    return df


def get_train_examples(season: int, binary: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if binary:
        examples = tourney_games.loc[tourney_games['season'] < season]
    else:
        examples = tourney_margin_games.loc[tourney_margin_games['season'] < season]

    # FIXME either incorporate filler seed for teams that did not qualify OR incorporate games of only tourney teams
    # examples = examples.append(regular_games.loc[regular_games['season'] <= season])

    # FIXME: utlizing raw seeds for now
    #for season in examples['season'].unique():
    #    examples.loc[examples['season'] == season, 'seed_probs'] = append_features(examples.loc[examples['season'] == season])['seed_probs']

    examples.drop('season', axis=1, inplace=True)
    examples.drop('TeamOne', axis=1, inplace=True)
    examples.drop('TeamTwo', axis=1, inplace=True)

    train_y = examples['class']
    train_X = examples[[x for x in examples.columns.values if x != 'class']]

    return train_y, train_X


def get_test_examples(season: int, binary: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Query test examples for season.

    Args:
        season:      tournament year

    Returns:
        class, features, team info

    """
    if binary:
        examples = tourney_scoring_games.loc[tourney_scoring_games['season'] == season]
    else:
        examples = tourney_scoring_margin_games.loc[tourney_scoring_margin_games['season'] == season]
    
    if season == 2010:
        # remove the play-in game
        examples = examples.iloc[1:, :]
    else:
        # remove first four play-in games
        examples = examples.iloc[4:, :]

    # Add info for scoring purposes
    team_info = examples[['class', 'season', 'TeamOne', 'TeamOneSeed', 'TeamOneRank', 'TeamTwo', 'TeamTwoSeed', 'TeamTwoRank']]
    team_info.loc[:, 'WTeamID']   = team_info.apply(lambda x: x['TeamOne'] if x['class'] == 1 else x['TeamTwo'], axis=1)
    team_info.loc[:, 'WTeamSeed'] = team_info.apply(lambda x: x['TeamOneSeed'] if x['class'] == 1 else x['TeamTwoSeed'], axis=1)
    team_info.loc[:, 'WTeamRank'] = team_info.apply(lambda x: x['TeamOneRank'] if x['class'] == 1 else x['TeamTwoRank'], axis=1)
    team_info.loc[:, 'LTeamID']   = team_info.apply(lambda x: x['TeamTwo'] if x['class'] == 1 else x['TeamOne'], axis=1)
    team_info.loc[:, 'LTeamSeed'] = team_info.apply(lambda x: x['TeamTwoSeed'] if x['class'] == 1 else x['TeamOneSeed'], axis=1)
    team_info.loc[:, 'LTeamRank'] = team_info.apply(lambda x: x['TeamTwoRank'] if x['class'] == 1 else x['TeamOneRank'], axis=1)
    team_info = team_info[['season', 'WTeamID', 'WTeamSeed', 'WTeamRank', 'LTeamID', 'LTeamSeed', 'LTeamRank']]

    examples.drop('season',  axis=1, inplace=True)
    examples.drop('TeamOne', axis=1, inplace=True)
    examples.drop('TeamTwo', axis=1, inplace=True)

    test_y = examples['class']
    test_X = examples[[x for x in examples.columns.values if x != 'class']]

    return test_y, test_X, team_info
