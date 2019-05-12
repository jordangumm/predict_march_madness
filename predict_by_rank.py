import click
import pandas as pd
import numpy as np

from itertools import chain
from typing    import Dict, Iterable

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics         import log_loss, accuracy_score
from tqdm                    import tqdm

from cbbstats.data import load_team_rankings
from cbbstats.game import get_tournament_games, get_regular_games
from cbbstats.team import get_team_ranking
from datautil      import get_train_examples, get_test_examples

from feature_engine import k_neighbor_probs, estimate_best_ksize


RANKINGS = load_team_rankings('WLK', daynum=133)  # final pre-tournament rankings


def ranking_matchup_results(season: int, training: bool):
    """
    
    """
    X, y = [], []
    
    games = get_tournament_games(season)

    for game in games:
        try:
            wranking = get_team_ranking(RANKINGS, season, game['WTeamID'])
            lranking = get_team_ranking(RANKINGS, season, game['LTeamID'])
        except:
            continue

        if training:
            X.append([wranking-lranking])
            y.append(1)

            X.append([lranking-wranking])
            y.append(0)
        else:
            if int(game['WTeamID']) > int(game['LTeamID']):
                X.append([wranking-lranking])
                y.append(1)
            else:
                X.append([lranking-wranking])
                y.append(0)
    return X, y


def ranking_matchup_probs(train_X, train_y, test_X):
    """Calculate probabilities given 

    """
    best_ksize = estimate_best_ksize(train_X, train_y, kmax=140)
    print(f'best ksize: {best_ksize}')

    probs = k_neighbor_probs(train_X, train_y, test_X, k=best_ksize)
    probs = [x if x != 1.0 else 0.99 for x in probs]
    probs = [x if x != 0.0 else 0.01 for x in probs]

    return probs


@click.command()
@click.option('--verbose/--not-verbose', default=False)
def main(verbose):

    for season in range(2010, 2019):
        print(f'\n\n{season}')

        train_X, train_y = [], []
        print('generating training data...')
        for s in tqdm(range(2003, season)):
            X, y = ranking_matchup_results(s, training=True)
            train_X += X
            train_y += y

        test_X, test_y = ranking_matchup_results(season, training=False)

        probs = ranking_matchup_probs(train_X, train_y, test_X)
        acc  = accuracy_score(test_y, [round(prob) for prob in probs])
        loss = log_loss(test_y, probs)
        print(f'{season} acc: {acc}')
        print(f'{season} loss: {loss}')
        print('')


if __name__ == '__main__':
    main()
