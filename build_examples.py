from typing import Dict, Union

import pandas as pd

from tqdm import tqdm

from cbbstats.data import load_team_stats, load_team_seeds
from cbbstats.game import get_tournament_games, get_regular_games
from cbbstats.team import get_team_stats


stats = load_team_stats()
seeds = load_team_seeds()


def write_examples(filename: str, gamesfunc, test=False):
    """Write appropriate examples for train or test.

    If test, only write game examples with larget kaggle_id coming first (as they score it).
    
    """
    with open(filename, 'w') as output:

        output.write('class\teFG%\topp_eFG%\tTO%\topp_TO%\tOR%\topp_OR%\tDR%\topp_DR%\tFTR\topp_FTR\t')
        output.write('_eFG%\t_opp_eFG%\t_TO%\t_opp_TO%\t_OR%\t_opp_OR%\t_DR%\t_opp_DR%\t_FTR\t_opp_FTR\tseason\n')

        for season in tqdm(range(2010, 2019)):
            for game in gamesfunc(season):

                wteam = get_team_stats(stats, season, game['WTeamID'])
                lteam = get_team_stats(stats, season, game['LTeamID'])

                if not test or int(game['WTeamID']) > int(game['LTeamID']):
                    g = '1\t'
                    g += '\t'.join([str(x) for x in [wteam['eFG%'], wteam['opp_eFG%'], wteam['TO%'], wteam['opp_TO%'],
                        wteam['OR%'], wteam['opp_OR%'], wteam['DR%'], wteam['opp_DR%'], wteam['FTR'], wteam['opp_FTR'],
                        lteam['eFG%'], lteam['opp_eFG%'], lteam['TO%'], lteam['opp_TO%'], lteam['OR%'], lteam['opp_OR%'],
                        lteam['DR%'], lteam['opp_DR%'], lteam['FTR'], lteam['opp_FTR']]])
                    g += f'\t{season}'
                    g += '\n'
                    output.write(g)

                if not test or int(game['LTeamID']) > int(game['WTeamID']):
                    g = '0\t'
                    g += '\t'.join([str(x) for x in [lteam['eFG%'], lteam['opp_eFG%'], lteam['TO%'], lteam['opp_TO%'],
                        lteam['OR%'], lteam['opp_OR%'], lteam['DR%'], lteam['opp_DR%'], lteam['FTR'], lteam['opp_FTR'],
                        wteam['eFG%'], wteam['opp_eFG%'], wteam['TO%'], wteam['opp_TO%'], wteam['OR%'], wteam['opp_OR%'],
                        wteam['DR%'], wteam['opp_DR%'], wteam['FTR'], wteam['opp_FTR']]])
                    g += f'\t{season}'
                    g += '\n'
                    output.write(g)


#write_examples('tourney_games.tsv', get_tournament_games)
write_examples('regular_games.tsv', get_regular_games)
write_examples('tourney_games.tsv', get_tournament_games, True)
