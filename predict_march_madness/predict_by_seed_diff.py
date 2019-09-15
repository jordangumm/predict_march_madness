import pandas as pd
import numpy as np

from typing import Dict, Iterable

from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import log_loss, accuracy_score

from cbbstats.data import load_team_seeds
from cbbstats.game import get_tournament_games
from cbbstats.team import get_team_seed


SEEDS = load_team_seeds()


def seed_matchup_results(season: int) -> Dict[str, float]:
    """Produce empirical seed matchup probabilities."""
    outcomes = {}
    for season in range(1985, season):
        games = get_tournament_games(season)
        for game in games:
            wseed = get_team_seed(SEEDS, season, game['WTeamID'])
            lseed = get_team_seed(SEEDS, season, game['LTeamID'])

            result = f'{wseed - lseed}'
            if result not in outcomes:
                outcomes[result] = 1
                continue
            outcomes[result] += 1
    return outcomes


def seed_probs(games: pd.DataFrame, season: int):
    """Add empirical seed difference win probabilities.

    Args:
        games:       DataFrame of games
        season:      test season

    Returns:
        Empirical game probabilities.

    """
    outcomes = seed_matchup_results(season)

    test_y = []
    probs = []
    for _, game in games.iterrows():
        wteam  = game['WTeamID']
        lteam  = game['LTeamID']
        season = int(game['season'])
        if game['WTeamID'] > game['LTeamID']:
            hseed = get_team_seed(SEEDS, season, wteam)
            lseed = get_team_seed(SEEDS, season, lteam)
            #if hseed == 16 and lseed == 16:
            #    continue
            test_y.append(1)
        else:
            lseed = get_team_seed(SEEDS, season, wteam)
            hseed = get_team_seed(SEEDS, season, lteam)
            #if hseed == 16 and lseed == 16:
            #    continue
            test_y.append(0)

        hline = f'{hseed - lseed}'
        lline = f'{lseed - hseed}'

        if hline not in outcomes or lline not in outcomes:
            if lline in outcomes and hline not in outcomes:
                probs.append(0.01)
            elif hline in outcomes and lline not in outcomes:
                probs.append(0.99)
            else:
                probs.append(0.5)
            continue

        hocc = outcomes[hline]
        locc = outcomes[lline]

        prob = hocc / (hocc + locc)
        probs.append(prob)

    return probs, test_y


def main():
    from datautil import get_test_examples

    for season in range(2010, 2019):
        test_y, test_X, team_info = get_test_examples(season)  # calls seed_probs internally
        probs = test_X['seed_probs'].values

        #if season == 2010:
            # remove the play-in game
        #    probs  = probs[1:]
        #    test_y = test_y[1:]
        #else:
            # remove first four play-in games
        #    probs  = probs[4:]
        #    test_y = test_y[4:]

        acc  = accuracy_score(test_y, [round(prob) for prob in probs])
        loss = log_loss(test_y, probs)
        print(f'{season}  acc: {acc}')
        print(f'{season} loss: {loss}')
        print('')


if __name__ == '__main__':
    main()
