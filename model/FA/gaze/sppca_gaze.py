"""
unpooled model
supervised probabilistic principal component analysis using factor analysis implementation

using gaze data to predict pain expectation

division of data into train, test using k-fold cross validation

consider only trials of the fear generalization phase

using data normalized

results:FactorAnalysis_kcrossval_monophysio.py
"""

import pymc as pm
import aesara.tensor as at
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy import stats
from sklearn.metrics import accuracy_score
import csv
from sklearn.metrics import confusion_matrix
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append('../../../.')

import extract_correct_csv
import os

os.chdir('..')
os.chdir('..')

scaler = StandardScaler()
RANDOM_SEED = 31415
rng = default_rng(RANDOM_SEED)

# all valid subjects
all_subject = extract_correct_csv.extract_only_valid_subject()
# all_subject.remove(49)

# all k = {2, 4, 6, 8} for the latent space
valid_k_list = list([2, 6, 10, 12, 15, 20, 23])

# keep only generalization trials
num_trials_to_remove = 48


# functions that creates triangular matrices
def expand_packed_block_triangular(d, k, packed, diag=None, mtype="aesara"):
    # like expand_packed_triangular, but with d > k.
    assert mtype in {"aesara", "numpy"}
    assert d >= k

    def set_(M, i_, v_):
        if mtype == "aesara":
            return at.set_subtensor(M[i_], v_)
        M[i_] = v_
        return M

    out = at.zeros((d, k), dtype=float) if mtype == "aesara" else np.zeros((d, k), dtype=float)
    if diag is None:
        idxs = np.tril_indices(d, m=k)
        out = set_(out, idxs, packed)
    else:
        idxs = np.tril_indices(d, k=-1, m=k)
        out = set_(out, idxs, packed)
        idxs = (np.arange(k), np.arange(k))
        out = set_(out, idxs, diag)
    return out


def makeW(d, k, dim_names, name):
    # make a W matrix adapted to the data shape
    n_od = int(k * d - k * (k - 1) / 2 - k)
    # trick: the cumulative sum of z will be positive increasing
    z = pm.HalfNormal("W_z_" + name, 1.0, dims="latent_columns")
    b = pm.HalfNormal("W_b_" + name, 1.0, shape=(n_od,), dims="packed_dim")
    L = expand_packed_block_triangular(d, k, b, at.ones(k))
    W = pm.Deterministic(name, at.dot(L, at.diag(at.extra_ops.cumsum(z))), dims=dim_names)
    return W


def my_post_predict(trace, feature_val):
    wfeature_ = trace.posterior['W_feature'][0]
    C_val = at.dot(np.linalg.pinv(wfeature_), feature_val.T)
    we_ = trace.posterior['W_e'][0]
    val_label_gen = at.matmul(np.array(we_), C_val.eval())
    label_val = np.where(val_label_gen.eval() < 0, 0, 1)
    label_val = stats.mode(label_val[0], keepdims=False)[0]
    return label_val


# loop features
types_ = ['hr', 'eda', 'pupil']
columns = ['subject', 'k', 'fold', 'feature', 'train', 'test']

output_csv = '../output/FA/gaze/sppca_gaze.csv'

with open(output_csv, 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)

TRAIN_PERC = 0.70
TEST_PERC = 0.3  # 1-TRAIN_PERC

for sub in all_subject:
    # loop within all k
    for k in valid_k_list:

        string_sub = extract_correct_csv.read_correct_subject_csv(sub)

        df_ = pd.read_csv('../data/LookAtMe_old/LookAtMe_0' + string_sub + '.csv', sep='\t')
        df_ = df_[num_trials_to_remove:]
        label = np.array(list([int(d > 2) for d in df_['rating']]))
        E = label[:, np.newaxis]
        E = pd.DataFrame(E)

        gaze = pd.read_csv('../data/mockup.csv')  # TODO INSERT GAZE DATA PATH
        print(gaze)
        feature = gaze[num_trials_to_remove:]

        # features normalization
        feature = scaler.fit_transform(feature)

        # num trials
        N = feature.shape[0]

        # num of trials used in the train set
        N_train = int(N * (TRAIN_PERC))

        # convert features into dataframe and reset index
        feature = pd.DataFrame(feature)
        feature = feature.reset_index().drop(columns=('index'))

        # RANDOM SPLIT of the gaze data
        feature = feature.sample(frac=1, random_state=0)
        feature = feature.reset_index(drop=True).to_numpy()

        # RANDOM SPLIT of the label shock expectancy data
        e_labels = E.sample(frac=1, random_state=0)
        e_labels = e_labels.reset_index(drop=True).to_numpy()

        # convert data into dataframe and reset index
        feature = pd.DataFrame(feature)
        feature = feature.reset_index().drop(columns=('index'))
        e_labels = pd.DataFrame(e_labels)
        e_labels = e_labels.reset_index().drop(columns=('index'))

        # 3 fold cross validation with a test_size=0.2 and training=0.8
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=123)

        for i, (train_index, test_index) in enumerate(sss.split(feature, E)):
            # len train set
            N_train = len(train_index)

            # divide gaze data into train and test
            feature_train = feature.iloc[train_index, :]
            feature_test = feature.iloc[test_index, :]

            # divide label shock prediction data into train and test
            e_labels_train = e_labels.iloc[train_index, :]
            e_labels_test = e_labels.iloc[test_index, :]

            # dimensions of each signal
            d_feature = feature_train.shape[1]
            d_e = e_labels_train.shape[1]

            # model definition
            with pm.Model() as PPCA_identified:
                # model coordinates
                PPCA_identified.add_coord("latent_columns", np.arange(k), mutable=True)
                PPCA_identified.add_coord("rows", np.arange(N_train), mutable=True)
                PPCA_identified.add_coord("observed_gaze", np.arange(d_feature), mutable=False)
                PPCA_identified.add_coord("observed_label", np.arange(d_e), mutable=False)

                feature_data = pm.MutableData("feature_data", feature_train.T, dims=["observed_gaze", "rows"])

                W_feature = makeW(d_feature, k, ("observed_gaze", "latent_columns"), 'W_gaze')

                W_e = pm.Normal("W_e", dims=["observed_gaze", "latent_columns"])
                C = pm.Normal("C", dims=["latent_columns", "rows"])

                psi_feature = pm.HalfNormal("psi_feature", 1.0)
                X_feature = pm.Normal("X_gaze", mu=at.dot(W_feature, C), sigma=psi_feature,
                                      observed=feature_data, dims=["observed_gaze", "rows"])

                X_e = pm.Bernoulli("X_e", p=pm.math.sigmoid(at.dot(W_e, C)), dims=["observed_gaze", "rows"],
                                   observed=e_labels_train.T)

            with PPCA_identified:
                approx = pm.fit(1000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
                trace = approx.sample(1000)

            with PPCA_identified:
                posterior_predictive = pm.sample_posterior_predictive(
                    trace, var_names=["X_e"], random_seed=123)

            # train
            e_pred_train = my_post_predict(trace, feature_train)
            train_accuracy_exp = accuracy_score(e_labels_train, e_pred_train)

            # test
            e_pred_mode_test = my_post_predict(trace, feature_test)
            test_accuracy_exp = accuracy_score(e_labels_test, e_pred_mode_test)

            # save results into csv
            row = [sub, k, i, 'gaze', train_accuracy_exp, test_accuracy_exp]

            with open(output_csv, 'a') as f:
                write = csv.writer(f)
                write.writerow(row)
