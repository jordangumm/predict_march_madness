from typing import Dict, Union

import pandas as pd

from tqdm import tqdm

from cbbstats.data import load_team_stats
from cbbstats.game import get_tournament_games, get_regular_games
from cbbstats.team import get_team_stats


stats = load_team_stats()


def write_games(filename: str, gamesfunc):
    with open(filename, 'w') as output:
        for season in tqdm(range(2010, 2019)):
            for game in gamesfunc(season):

                wteam = get_team_stats(stats, season, game['WTeamID'])
                lteam = get_team_stats(stats, season, game['LTeamID'])

                g = '1\t'
                g += '\t'.join([str(x) for x in [wteam['eFG%'], wteam['opp_eFG%'], wteam['TO%'], wteam['opp_TO%'], wteam['OR%'], wteam['DR%'], wteam['FTR'], wteam['opp_FTR'], lteam['eFG%'], lteam['opp_eFG%'], lteam['TO%'], lteam['opp_TO%'], lteam['OR%'], lteam['DR%'], lteam['FTR'], lteam['opp_FTR']]])
                g += f'\t{season}'
                g += '\n'

                output.write(g)

                g = '0\t'
                g += '\t'.join([str(x) for x in [lteam['eFG%'], lteam['opp_eFG%'], lteam['TO%'], lteam['opp_TO%'], lteam['OR%'], lteam['DR%'], lteam['FTR'], lteam['opp_FTR'], wteam['eFG%'], wteam['opp_eFG%'], wteam['TO%'], wteam['opp_TO%'], wteam['OR%'], wteam['DR%'], wteam['FTR'], wteam['opp_FTR']]])
                g += f'\t{season}'
                g += '\n'

                output.write(g)


write_games('tourney_games.tsv', get_tournament_games)
write_games('regular_games.tsv', get_regular_games)
