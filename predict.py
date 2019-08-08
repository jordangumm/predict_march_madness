import os

import click
import collections
import numpy           as np
import pandas          as pd
import statsmodels.api as sm
import tensorflow      as tf
import xgboost         as xgb

#from automaxout.models.maxout   import Maxout
from automaxout.model_selection import GeneticSelector
from cbbstats.data              import load_team_seeds
from cbbstats.game              import get_tournament_results, get_regular_results
from scipy.interpolate          import UnivariateSpline
from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import accuracy_score, log_loss
from sklearn.model_selection    import KFold
from tensorflow                 import keras

from predict_march_madness.datautil       import get_train_examples, get_test_examples
from predict_march_madness.feature_engine import k_neighbor_probs, empirical_probs
from predict_march_madness.build_features import build_features


@click.group()
def cli():
    pass


def load_examples(season: int):
    train_y, train_X = get_train_examples(season)
    test_y, test_X   = get_test_examples(season)

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X  = scaler.transform(test_X)
    train_y = train_y.values
    test_y  = test_y.values

    return train_X, train_y, test_X, test_y


@cli.command()
def logreg():
    for season in range(2011, 2019):
        train_X, train_y, test_X, test_y = load_examples(season)

        train_y, train_X = get_train_examples(season)
        test_y, test_X   = get_test_examples(season)

        train_X = train_X.values
        train_y = train_y.values

        test_X = test_X.values
        test_y = test_y.values

        clf = LogisticRegression(solver='lbfgs').fit(train_X, train_y)

        probs = clf.predict_proba(test_X)
        acc  = clf.score(test_X, test_y)
        loss = log_loss(test_y, probs)
        print(f'{season}  acc: {acc}')
        print(f'{season} loss: {loss}')
        print('')


def prepare_data(df):
    dfswap = df[[
        'Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
        'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
        'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR',
        'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'
    ]]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'    

    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output.location=='N','location'] = '0'
    output.loc[output.location=='H','location'] = '1'
    output.loc[output.location=='A','location'] = '-1'
    output.location = output.location.astype(int)

    output['PointDiff'] = output['T1_Score'] - output['T2_Score']

    return output


def team_quality(regular_season_effects, season):
    print('team_quality:', season)
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(
        formula = formula,
        data    = regular_season_effects.loc[regular_season_effects.Season==season,:], 
        family  = sm.families.Binomial()
    ).fit()

    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)

    print(quality)

    return quality


@cli.command()
@click.option('--season', '-s', default=2016)
@click.option('--verbose', is_flag=True)
def raddar(season: int, verbose: bool) -> None:
    """Raddar 2018 WMM winner, also used to win MMM 2019 by different user.

    2019 winner added select regular season data.
    https://github.com/salmatfq/KaggleMarchMadnessFirstPlace/blob/master/win_ncaa_men.R

    Python implementation rewrite by original author raddar.
    https://www.kaggle.com/raddar/paris-madness

    Quote from author:
        My model is a single xgboost model.  I only used tournament data as my
        training set.  I modeled the win margin using MAE metric together with cauchy 
        objective function, which is suitable for MAE optimization.  The score
        difference predictions were used as inputs for smoothing splines GAM
        model to transform them to probabilities.
        https://www.kaggle.com/c/womens-machine-learning-competition-2018/discussion/53597

    """
    tourney_results = get_tournament_results(compact=False)
    regular_results = get_regular_results(compact=False)
    seeds = load_team_seeds()

    regular_data = prepare_data(regular_results)
    tourney_data = prepare_data(tourney_results)

    boxscore_cols = [
            'T1_Score', 'T2_Score', 'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_FTM',
            'T1_FTA', 'T1_OR', 'T1_DR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_Blk', 'T1_PF', 
            'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_FTM', 'T2_FTA', 'T2_OR',
            'T2_DR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk', 'T2_PF', 'PointDiff'
    ]

    funcs = [np.mean]

    season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()
    season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]

    season_statistics_T1 = season_statistics.copy()
    season_statistics_T2 = season_statistics.copy()

    season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
    season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]
    season_statistics_T1.columns.values[0] = "Season"
    season_statistics_T2.columns.values[0] = "Season"

    tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']]
    tourney_data = pd.merge(tourney_data, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

    last14days_stats_T1 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
    last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
    last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

    last14days_stats_T2 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
    last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
    last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

    tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')

    regular_season_effects = regular_data[['Season','T1_TeamID','T2_TeamID','PointDiff']].copy()
    regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
    regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
    regular_season_effects['win'] = np.where(regular_season_effects['PointDiff']>0,1,0)
    march_madness = pd.merge(seeds[['Season','TeamID']],seeds[['Season','TeamID']],on='Season')
    march_madness.columns = ['Season', 'T1_TeamID', 'T2_TeamID']
    march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
    march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
    regular_season_effects = pd.merge(regular_season_effects, march_madness, on = ['Season','T1_TeamID','T2_TeamID'])

    glm_quality = pd.concat([
        team_quality(regular_season_effects, y) for y in range(2010, season+1)
    ]).reset_index(drop=True)

    glm_quality_T1 = glm_quality.copy()
    glm_quality_T2 = glm_quality.copy()
    glm_quality_T1.columns = ['T1_TeamID','T1_quality','Season']
    glm_quality_T2.columns = ['T2_TeamID','T2_quality','Season']

    print(glm_quality_T1)
    print(glm_quality_T1.keys())
    print(tourney_data.keys())
    import sys
    sys.exit('')

    tourney_data = pd.merge(tourney_data, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')

    print(tourney_data)

    import sys
    sys.exit('')

    seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))

    seeds_T1 = seeds[['Season','TeamID','seed']].copy()
    seeds_T2 = seeds[['Season','TeamID','seed']].copy()
    seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
    seeds_T2.columns = ['Season','T2_TeamID','T2_seed']

    tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

    tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]

    y = tourney_data['T1_Score'] - tourney_data['T2_Score']

    features = list(season_statistics_T1.columns[2:999]) + \
               list(season_statistics_T2.columns[2:999]) + \
               list(seeds_T1.columns[2:999]) + \
               list(seeds_T2.columns[2:999]) + \
               list(last14days_stats_T1.columns[2:999]) + \
               list(last14days_stats_T2.columns[2:999]) + \
               ["Seed_diff"] + ["T1_quality","T2_quality"]

    X = tourney_data[features].values
    print(tourney_data[features])
    dtrain = xgb.DMatrix(X, label = y)
    print('got here')



"""
@cli.command()
@click.option('--season', '-s', default=2016)
@click.option('--verbose', is_flag=True)
def select_maxout(season: int, verbose: bool):
    train_X, train_y, test_X, test_y = load_examples(season)

    selector = GeneticSelector(
        Maxout,
        train_X,
        train_y,
        node_min=2,
        node_max=200,
        layer_min=1,
        layer_max=2,
        dropout_min=0.0,
        dropout_max=0.5,
        stop_min=2,
        stop_max=5,
        ngen=10,
    )

    num_nodes, num_layers, dropout, early_stop = selector.select_best_model()

    import sys
    sys.exit('done genetic selecting')
    num_nodes, num_layers, dropout, early_stop = 20, 2, 0.1, 3

    clf = Maxout(
        len(test_X[0]),
        num_nodes=num_nodes,
        num_layers=num_layers,
        dropout_rate=dropout,
        early_stop=early_stop,
        verbose=verbose,
    )

    probs = clf.fit(train_X, train_y, test_X, test_y)

    acc  = accuracy_score(test_y, [np.rint(x[1]) for x in probs])
    loss = log_loss(test_y, [x[1] for x in probs])
    print(f'{season}  acc: {acc}')
    print(f'{season} loss: {loss}')
"""

@cli.command()
@click.option('--season', default=2016)
@click.option('--nodes', type=int, required=True)
@click.option('--layers', type=int)
@click.option('--dropout', type=float)
@click.option('--stop', type=int)
@click.option('--verbose', is_flag=True)
def train_maxout(
    season:  int,
    nodes:   int,
    layers:  int,
    dropout: float,
    stop:    int,
    verbose: bool
) -> None:
    """Train Maxout Model."""
    train_X, train_y, test_X, train_X = load_examples(season)

    model = keras.Sequential([
        keras.layers.Dense(nodes, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(train_X, train_y, epochs=5)


if __name__ == '__main__':
    cli()
