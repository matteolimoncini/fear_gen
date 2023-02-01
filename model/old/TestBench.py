import pymc as pm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats
from sklearn.metrics import accuracy_score
import aesara.tensor as at
import sys
sys.path.append('../../')
import extract_correct_csv as functions
import os
from sklearn.preprocessing import StandardScaler
import csv


scaler = StandardScaler()
os.chdir('..')
os.chdir('..')
all_subjects = functions.extract_only_valid_subject()
columns = ['subject', 'type', 'fold', 'train', 'test my accuracy', 'test posterior']
with open('output/Regression.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)

types_ = ['hr', 'eda', 'pupil']

for sub in all_subjects:

    for type_ in types_:

        X = pd.read_csv('data/features_4_2/'+type_+'/'+str(sub)+'.csv')
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X)
        X = X[48:]

        df_ = pd.read_csv('data/LookAtMe_0'+functions.read_correct_subject_csv(sub)+'.csv', sep='\t')
        y = np.array(list([int(d > 2) for d in df_['rating']]))[:, np.newaxis]
        y = pd.DataFrame(y)
        y = y[48:]

        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=123)

        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train = X.iloc[train_index,:]
            y_train = y.iloc[train_index,:]

            X_test = X.iloc[test_index, :]
            y_test = y.iloc[test_index, :]

            with pm.Model() as GLM:
                x_data = pm.MutableData('x_data', X_train)
                y_data = pm.MutableData('y_data', y_train)

                # intercept = pm.Normal("intercept", 0, 1, shape=y_data.shape)
                slope = pm.Normal("slope", shape=(x_data.shape[1], 1))
                likelihood = pm.Bernoulli('likelihood', p=pm.math.sigmoid(at.dot(x_data, slope)), observed=y_data)
                approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
                trace = approx.sample(1000)
                posterior_predictive = pm.sample_posterior_predictive(
                    trace, var_names=["likelihood"], random_seed=123)

            # train accuracy
            prediction_y = posterior_predictive.posterior_predictive['likelihood']
            e_pred_mode_train = np.squeeze(stats.mode(prediction_y[0], keepdims=False)[0])[:, np.newaxis]

            train_accuracy_exp = accuracy_score(y_train, e_pred_mode_train)

            # my posterior
            a_trained = trace.posterior['slope'][0]
            y_new = at.matmul(X_test, np.array(a_trained))
            y_new = np.where(y_new.eval() < 0, 0, 1)
            y_new_pred = stats.mode(y_new, keepdims=False)[0]
            my_accuracy_score = accuracy_score(y_test, y_new_pred)

            # posterior predictive
            with GLM:
                pm.set_data({'x_data': X_test})
                pm.set_data({'y_data': y_test})
                posterior_pred = pm.sample_posterior_predictive(trace, var_names=['likelihood'], random_seed=123,
                                                                predictions=True)

            prediction_y_test = posterior_pred.predictions['likelihood']
            e_pred_mode_test = np.squeeze(stats.mode(prediction_y_test[0], keepdims=False)[0])[:, np.newaxis]

            test_accuracy_exp = accuracy_score(y_test, e_pred_mode_test)

            row = [sub, type_, i, train_accuracy_exp, my_accuracy_score, test_accuracy_exp]

            with open('output/Regression.csv', 'a') as f:
                write = csv.writer(f)
                write.writerow(row)
