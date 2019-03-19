import click
import pandas as pd
import numpy as np

from typing import Dict, Iterable

from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import log_loss, accuracy_score

from cbbstats.data import load_team_seeds
from cbbstats.game import get_tournament_games
from cbbstats.team import get_team_seed
from datautil      import get_train_examples, get_test_examples


SEEDS = load_team_seeds()

def seed_matchup_results(season: int) -> Dict[str, float]:
    """

    
    """
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


@click.command()
@click.option('--verbose/--not-verbose', default=False)
def main(verbose):

    for season in range(2010, 2019):

        outcomes = seed_matchup_results(season)

        if verbose:
            for i in range(1, 17):
                for j in range(i, 17):
                    hline = f'{j - i}'
                    lline = f'{i - j}'
                    if hline not in outcomes:
                        if lline not in outcomes:
                            prob = 0.5
                        else:
                            prob = 0.01
                    elif lline not in outcomes:
                        if hline not in outcomes:
                            prob = 0.5
                        else:
                            prob = 0.99
                    else:
                        hocc = outcomes[hline]
                        locc = outcomes[lline]
                        prob = hocc / (hocc + locc)
                    print(f'{j} over {i}: {prob}')

        test_y = []
        probs = []
        for game in get_tournament_games(season):
            wteam = game['WTeamID']
            lteam = game['LTeamID']
            if game['WTeamID'] > game['LTeamID']:
                hseed = get_team_seed(SEEDS, season, wteam)
                lseed = get_team_seed(SEEDS, season, lteam)
                if hseed == 16 and lseed == 16:
                    continue
                test_y.append(1)
            else:
                lseed = get_team_seed(SEEDS, season, wteam)
                hseed = get_team_seed(SEEDS, season, lteam)
                if hseed == 16 and lseed == 16:
                    continue
                test_y.append(0)

            hline = f'{hseed - lseed}'
            lline = f'{lseed - hseed}'

            if verbose:
                print(hline)
                print(game)
            
            if hline not in outcomes or lline not in outcomes:
                if lline in outcomes and hline not in outcomes:
                    if verbose:
                        print(0.01)
                    probs.append(0.01)
                elif hline in outcomes and lline not in outcomes:
                    if verbose:
                        print(0.99)
                    probs.append(0.99)
                else:
                    if verbose:
                        print(0.5)
                    probs.append(0.5)
                continue

            hocc = outcomes[hline]
            locc = outcomes[lline]

            prob = hocc / (hocc + locc)
            if verbose:
                print(prob)
            probs.append(prob)


        acc  = accuracy_score(test_y, [round(prob) for prob in probs])
        loss = log_loss(test_y, probs)
        print(f'{season}  acc: {acc}')
        print(f'{season} loss: {loss}')
        print('')


if __name__ == '__main__':
    main()
