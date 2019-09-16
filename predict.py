import os
import random

import click
import collections
import numpy           as np
import pandas          as pd
import xgboost         as xgb

from automaxout.models.maxout   import Maxout, MaxoutDense, MaxoutResidual
from automaxout.model_selection import GeneticSelector
from cbbstats.data              import load_team_seeds, load_teams
from cbbstats.team              import get_team_seed, get_team_name
from cbbstats.game              import get_tournament_results, get_regular_results
from scipy.interpolate          import UnivariateSpline
from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import accuracy_score, log_loss
from sklearn.model_selection    import KFold

from predict_march_madness.datautil       import get_train_examples, get_test_examples
from predict_march_madness.feature_engine import k_neighbor_probs, empirical_probs


@click.group()
def cli():
    pass


def load_examples(season: int):
    train_y, train_X = get_train_examples(season)
    test_y, test_X, test_info = get_test_examples(season)

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X  = scaler.transform(test_X)
    train_y = train_y.values
    test_y  = test_y.values

    return train_X, train_y, test_X, test_y, test_info


def evaluate(probs, test_y, team_info, season):
    final_probs = np.mean(probs, axis=0)

    acc  = accuracy_score(test_y, [np.rint(x[1]) for x in final_probs])
    loss = log_loss(test_y, [x[1] for x in final_probs])
    print(f'{acc:.6f}\t{loss:.6f}\t{np.var(probs):.6f}')


def print_misses(probs, test_y, team_info, season):
    teams = load_teams()
    final_probs = np.mean(probs, axis=0)
    for i, prob in enumerate(final_probs):
        if prob[1].round() != test_y[i].round():
            info = team_info.iloc[i]

            wseed = info['WTeamSeed']
            lseed = info['LTeamSeed']

            wname = get_team_name(teams, info['WTeamID'])
            lname = get_team_name(teams, info['LTeamID'])

            print(f'{i}: {prob[1]} {wseed}.{wname} over {lseed}.{lname}')


def get_even_match(probs):
    """Find closest to even match in game probability list.

    Args:
        probs: list of game probabilities

    Returns:
        Index of game with most even win probabilities.

    """
    nearest_index   = 0
    current_nearest = 0.5
    for i, prob in enumerate(probs):
        if abs(prob[1] - 0.5) < current_nearest:
            current_nearest = abs(prob[1] - 0.5)
            nearest_index = i
    return nearest_index


def probability_calibration(probs, team_info):
    """Generate two submissions with probabilities calibrated for optimized score."""
    new_probs_one = []
    new_probs_two = []
    for game_index in range(len(probs)):
        info = team_info.iloc[game_index]
        if (info['WTeamSeed'] in (1, 2) and info['LTeamSeed'] in (15,  16)) or (info['LTeamSeed'] in (1, 2) and info['WTeamSeed'] in (15, 16)):
            low_seed  = info['WTeamID'] if info['WTeamSeed'] in (1, 2) else info['LTeamID']
            high_seed = info['LTeamID'] if info['LTeamSeed'] in (15, 16) else info['WTeamID']

            if low_seed > high_seed:
                new_probs_one.append([0.001, 0.999])
                new_probs_two.append(probs[game_index])
            else:
                new_probs_one.append([0.999, 0.001])
                new_probs_two.append(probs[game_index])
            continue

        new_probs_one.append(probs[game_index])
        new_probs_two.append(probs[game_index])

    # nearest_index = get_even_match(probs)
    # new_probs_one[nearest_index] = [0.999, 0.001]
    # new_probs_two[nearest_index] = [0.001, 0.999]
    return new_probs_one, new_probs_two

@cli.command()
@click.option('--season', '-s', default=2016)
def logreg(season: int) -> None:
    """Predict games using Logistic Regression.

    Args:
        season: tournament year to build model for and predict

    """
    train_X, train_y, test_X, test_y, team_info = load_examples(season)

    clf   = LogisticRegression(solver='lbfgs').fit(train_X, train_y)
    probs = clf.predict_proba(test_X)
    
    evaluate([probs], test_y, team_info, season)

    sub_one, sub_two = probability_calibration(probs, team_info)

    print('submission one')
    evaluate([sub_one], test_y, team_info, season)
    print('submission two')
    evaluate([sub_two], test_y, team_info, season)
    print('misses')
    print_misses([probs], test_y, team_info, season)


def subsample(X, y):
    """Subsample for boosted aggregation.

    Args:
        X: two dimensional feature array
        y: class array

    """
    subX = []
    suby = []
    
    while len(subX) < len(X):
        index = random.randint(0, len(X)-1)
        subX.append(X[index])
        suby.append(y[index])

    return subX, suby


def train(season: int, numbags: int, verbose: bool, model: Maxout) -> None:
    """Predict tournament games using Maxout Network.

    Args:
       season:  tournmanet year to predict
       numbags: number of boosted training sessions to aggregate
       verbose: whether or not to display epoch information (default=False)
       model:   maxout model class to use

    """
    num_nodes, num_layers, dropout, early_stop = 20, 2, 0.9, 2

    train_X, train_y, test_X, test_y, team_info = load_examples(season)
    print(f'Predicting {season}')
    print('Accuracy\tLog Loss\tVariance')

    random.seed()
    probs = []
    for _ in range(numbags):
        tmp_X, tmp_y = subsample(train_X, train_y)
        clf = model(
            len(test_X[0]),
            num_nodes=num_nodes,
            num_layers=num_layers,
            dropout_rate=dropout,
            early_stop=early_stop,
            verbose=verbose,
        )
        probs.append(clf.fit(tmp_X, tmp_y, train_X, train_y, test_X))
        evaluate(probs, test_y, team_info, season)
    print_misses(probs, test_y, team_info, season)

    probs = np.mean(probs, axis=0)

    sub_one, sub_two = probability_calibration(probs, team_info)

    print('submission one')
    evaluate([sub_one], test_y, team_info, season)
    print('submission two')
    evaluate([sub_two], test_y, team_info, season)
    print('misses')
    print_misses([probs], test_y, team_info, season)


@cli.command()
@click.option('--season',  '-s', default=2016)
@click.option('--numbags', '-n', default=3)
@click.option('--verbose', is_flag=True)
def maxout(season: int, numbags: int, verbose: bool) -> None:
    """Predict tournament games using Residual Maxout Network.

    Args:
        season:  tournamnet year to predict
        numbags: number of boosted training sessions to aggregate
        verbose: whether to display model training information (default=false)

    """
    train(season, numbags, verbose, Maxout)


@cli.command()
@click.option('--season', '-s', default=2016)
@click.option('--numbags', '-n', default=3)
@click.option('--verbose', is_flag=True)
def residual(season: int, numbags: int, verbose: bool) -> None:
    """Predict tournament games using Residual Maxout Network.

    Args:
        season:  tournamnet year to predict
        numbags: number of boosted training sessions to aggregate
        verbose: whether to display model training information (default=false)

    """
    train(season, numbags, verbose, MaxoutResidual)


@cli.command()
@click.option('--season', '-s', default=2016)
@click.option('--numbags', '-n', default=3)
@click.option('--verbose', is_flag=True)
def dense(season: int, numbags: int, verbose: bool) -> None:
    """Predict tournament games using Dense Maxout Network.

    Args:
        season:  tournamnet year to predict
        numbags: number of boosted training sessions to aggregate
        verbose: whether to display model training information (default=false)

    """
    train(season, numbags, verbose, MaxoutDense)


@cli.command()
@click.option('--season', '-s', default=2016)
def xgboost_spline(season: int) -> None:
    """Predict tournament games using XGBoost.

    Great for producing gain metric to examine feature importance.

    Args:
        season:  tournamnet year to predict
        numbags: number of boosted training sessions to aggregate
        verbose: whether to display model training information (default=false)

    """
    train_X, train_y, test_X, test_y, team_info = load_examples(season)

    dtrain = xgb.DMatrix(train_X, label = train_y)

    def cauchyobj(preds, dtrain):
        labels = dtrain.get_label()
        c = 5000
        x = preds - labels
        grad = x / (x**2/c**2+1)
        hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
        return grad, hess

    param = {
        'eval_metric':      'mae',
        'booster':          'gbtree',
        'eta':               0.02,
        'subsample':         0.35,
        'colsample_bytree':  0.7,
        'num_parallel_tree': 10,
        'min_child_weight':  40,
        'gamma':             10,
        'max_depth':         3,
        'silent':            1,
    }

    xgb_cv = []
    repeat_cv = 3 # change to 10?

    for i in range(repeat_cv):
        print(f'Fold repeater {i}')
        xgb_cv.append(
            xgb.cv(
                params                = param,
                dtrain                = dtrain,
                obj                   = cauchyobj,
                num_boost_round       = 3000,
                folds                 = KFold(n_splits = 5, shuffle = True, random_state = i),
                early_stopping_rounds = 25,
                verbose_eval          = 50,
            )
        )
    iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
    val_mae          = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
    
    print(iteration_counts, val_mae)

    oof_preds = []
    for i in range(repeat_cv):
        print(f'Fold repeater {i}')
        preds = train_y.copy()
        kfold = KFold(n_splits = 5, shuffle = True, random_state = i)
        for train_index, val_index in kfold.split(train_X, train_y):
            dtrain_i = xgb.DMatrix(train_X[train_index], label = train_y[train_index])
            dval_i   = xgb.DMatrix(train_X[val_index],   label = train_y[val_index])
            model = xgb.train(
                params = param,
                dtrain = dtrain_i,
                num_boost_round = iteration_counts[i],
                verbose_eval = 50,
            )
            preds[val_index] = model.predict(dval_i)
        oof_preds.append(np.clip(preds, -30, 30))

    plot_df = pd.DataFrame({'pred': oof_preds[0], 'label':np.where(train_y>0,1,0)})
    print(plot_df)
    return

    spline_model = []

    for i in range(repeat_cv):
        dat = list(zip(oof_preds[i], np.where(train_y>0,1,0)))
        dat = sorted(dat, key = lambda x: x[0])
        datdict = {}
        for k in range(len(dat)):
            datdict[dat[k][0]] = dat[k][1]

        spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
        spline_fit = spline_model[i](oof_preds[i])
        print('logloss of cvsplit {i}: {log_loss(np.where(train_y>0,1,0),spline_fit)}')



@cli.command()
@click.option('--season', '-s', default=2016)
@click.option('--verbose', is_flag=True)
def select_maxout(season: int, verbose: bool) -> None:
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


if __name__ == '__main__':
    cli()
