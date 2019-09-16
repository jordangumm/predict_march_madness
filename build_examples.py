from pathlib import Path
from typing  import Dict, Union

import numpy  as np
import pandas as pd

from tqdm import tqdm

from cbbstats.data import load_team_stats, load_team_seeds, load_team_rankings
from cbbstats.game import get_tournament_games, get_regular_games
from cbbstats.team import get_team_stats, get_team_seed, get_team_ranking


STATS = load_team_stats()
SEEDS = load_team_seeds()
RANKS = load_team_rankings('WLK', daynum=133)

VARS = ['eFG%', 'opp_eFG%', 'TO%', 'opp_TO%', 'OR%', 'DR%', 'FTR']


def calc_average_win_margin(team_id, season):
    score_diffs = []
    for game in get_regular_games(season):
        if team_id == game['WTeamID'] or team_id == game['LTeamID']:
            margin = game['WScore'] - game['LScore'] if team_id == game['WTeamID'] else game['LScore'] - game['WScore']
            score_diffs.append(margin)
    return np.mean(score_diffs)


def write_examples(filename: str, gamesfunc, test=False):
    """Write appropriate examples for train or test.

    If test, only write game examples with larget kaggle_id coming first (as they score it).
    
    """
    with open(filename, 'w') as output:
        output.write('class\t')
        output.write('\t'.join(VARS))
        output.write('\t')
        output.write('\t'.join([f'_{s}' for s in VARS]))
        output.write('\tseason\tTeamOne\tTeamOneSeed\tTeamOneRank\tTeamOneMargin\tTeamTwo\tTeamTwoSeed\tTeamTwoRank\tTeamTwoMargin')
        output.write('\n')

        for season in tqdm(range(2010, 2020)):
            cache = {}
            for game in gamesfunc(season):
                wteam = int(game['WTeamID'])
                lteam = int(game['LTeamID'])

                if wteam in cache:
                    wteam_stats  = cache[wteam]['stats']
                    wteam_seed   = cache[wteam]['seed']
                    wteam_rank   = cache[wteam]['rank']
                    wteam_margin = cache[wteam]['margin']
                else:
                    wteam_stats  = get_team_stats(STATS, season, wteam)
                    wteam_seed   = get_team_seed(SEEDS, season, wteam)
                    wteam_rank   = get_team_ranking(RANKS, season, wteam)
                    wteam_margin = calc_average_win_margin(wteam, season)
                    cache[wteam] = {}
                    cache[wteam]['stats']  = wteam_stats
                    cache[wteam]['seed']   = wteam_seed
                    cache[wteam]['rank']   = wteam_rank
                    cache[wteam]['margin'] = wteam_margin

                if lteam in cache:
                    lteam_stats  = cache[lteam]['stats']
                    lteam_seed   = cache[lteam]['seed']
                    lteam_rank   = cache[lteam]['rank']
                    lteam_margin = cache[lteam]['margin']
                else:
                    lteam_stats = get_team_stats(STATS, season, lteam)
                    lteam_seed  = get_team_seed(SEEDS, season, lteam)
                    lteam_rank  = get_team_ranking(RANKS, season, lteam)
                    lteam_margin = calc_average_win_margin(lteam, season)
                    cache[lteam] = {}
                    cache[lteam]['stats']  = lteam_stats
                    cache[lteam]['seed']   = lteam_seed
                    cache[lteam]['rank']   = lteam_rank
                    cache[lteam]['margin'] = lteam_margin

                if not test or wteam > lteam:
                    g = '1\t'
                    g += '\t'.join([str(wteam_stats[s]) for s in VARS])
                    g += '\t'
                    g += '\t'.join([str(lteam_stats[s]) for s in VARS])
                    g += f'\t{season}'
                    g += f'\t{wteam}\t{wteam_seed}\t{wteam_rank}\t{wteam_margin}'
                    g += f'\t{lteam}\t{lteam_seed}\t{lteam_rank}\t{lteam_margin}'
                    g += '\n'
                    output.write(g)

                if not test or lteam > wteam:
                    g = '0\t'
                    g += '\t'.join([str(lteam_stats[s]) for s in VARS])
                    g += '\t'
                    g += '\t'.join([str(wteam_stats[s]) for s in VARS])
                    g += f'\t{season}'
                    g += f'\t{lteam}\t{lteam_seed}\t{lteam_rank}\t{lteam_margin}'
                    g += f'\t{wteam}\t{wteam_seed}\t{wteam_rank}\t{wteam_margin}'
                    g += '\n'
                    output.write(g)

data_path = Path('predict_march_madness/data')
if not data_path.exists():
    data_path.mkdir()

write_examples('predict_march_madness/data/tourney_games.tsv', get_tournament_games)
#write_examples('predict_march_madness/data/regular_games.tsv', get_regular_games)
write_examples('predict_march_madness/data/tourney_games_nodups.tsv', get_tournament_games, test=True)
#write_examples('predict_march_madness/data/regular_games_nodups.tsv', get_regular_games, test=True)
