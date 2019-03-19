import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from scipy.stats          import describe
from sklearn.metrics      import log_loss

from datautil import get_train_examples, get_test_examples


train_y, train_X = get_train_examples(2014)
test_y, test_X   = get_test_examples(2014)

features = train_X.columns.values

t1_features = features[:len(features)//2]
t2_features = features[len(features)//2:]

final_probs = np.array([])
for t1_feature in t1_features:
    for t2_feature in t2_features:
        # calculate logistic regression and therefore probability of win given difference in features
        feat = f'{t1_feature}-{t2_feature}'
        train_X[feat] = train_X[t1_feature] - train_X[t2_feature]

        trX = train_X[t1_feature].values - train_X[t2_feature].values
        teX = test_X[t1_feature].values - test_X[t2_feature].values

        trX = trX.reshape(-1,1)
        teX = teX.reshape(-1,1)
        
        clf = LogisticRegression(solver='lbfgs').fit(trX, train_y)

        probs = clf.predict_proba(teX)[:,1]
        probs = probs.reshape(-1,1)

        if not final_probs.any():
            final_probs = probs
        else:
            ty = train_y.values.reshape(-1,1)
            corr = abs(np.corrcoef(trX, train_y.values, rowvar=False)[0,1])
            if corr > 0.2:
                print(f'added {feat}')
                final_probs = np.append(final_probs, probs, 1)

        loss = log_loss(test_y, probs)

        #print(f'{feat} loss: {loss}')


final_probs = [np.average(x) for x in final_probs]
loss = log_loss(test_y, final_probs)
print(loss)
