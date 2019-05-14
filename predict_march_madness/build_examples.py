from typing import Dict, Union

import pandas as pd

from tqdm import tqdm

from cbbstats.data import load_team_stats, load_team_seeds
from cbbstats.game import get_tournament_games, get_regular_games
from cbbstats.team import get_team_stats


stats = load_team_stats()
seeds = load_team_seeds()

VARS = ['eFG%', 'opp_eFG%', 'TO%', 'opp_TO%', 'OR%', 'DR%', 'FTR']


def write_examples(filename: str, gamesfunc, test=False):
    """Write appropriate examples for train or test.

    If test, only write game examples with larget kaggle_id coming first (as they score it).
    
    """
    with open(filename, 'w') as output:
        output.write('class\t')
        output.write('\t'.join(VARS))
        output.write('\t')
        output.write('\t'.join([f'_{s}' for s in VARS]))
        output.write('\tseason\tWTeamID\tLTeamID')  # for compatibility sake only
        output.write('\n')

        for season in tqdm(range(2010, 2019)):
            for game in gamesfunc(season):

                wteam = get_team_stats(stats, season, game['WTeamID'])
                lteam = get_team_stats(stats, season, game['LTeamID'])

                if not test or int(game['WTeamID']) > int(game['LTeamID']):
                    g = '1\t'
                    g += '\t'.join([str(wteam[s]) for s in VARS])
                    g += '\t'
                    g += '\t'.join([str(lteam[s]) for s in VARS])
                    g += f'\t{season}\t{game["WTeamID"]}\t{game["LTeamID"]}'
                    g += '\n'
                    output.write(g)

                if not test or int(game['LTeamID']) > int(game['WTeamID']):
                    g = '0\t'
                    g += '\t'.join([str(lteam[s]) for s in VARS])
                    g += '\t'
                    g += '\t'.join([str(wteam[s]) for s in VARS])
                    g += f'\t{season}\t{game["LTeamID"]}\t{game["WTeamID"]}'
                    g += '\n'
                    output.write(g)


write_examples('tourney_games.tsv', get_tournament_games)
write_examples('regular_games.tsv', get_regular_games)
write_examples('tourney_games_scoring.tsv', get_tournament_games, test=True)
write_examples('regular_games_nodups.tsv', get_regular_games, test=True)
