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

sys.path.append('../../')

import extract_correct_csv
import os

os.chdir('..')
os.chdir('..')

scaler = StandardScaler()
RANDOM_SEED = 31415
rng = default_rng(RANDOM_SEED)

# all valid subjects
all_subject = extract_correct_csv.extract_only_valid_subject()
all_subject.remove(49)


# all k = {2, 4, 6, 8} for the latent space
valid_k_list = list([2, 6, 10, 12, 15, 20])

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


def my_post_predict(trace, hr_new, eda_new, pupil_new):
    whr_ = trace.posterior['W_hr'][0]
    weda_ = trace.posterior['W_eda'][0]
    wpupil_ = trace.posterior['W_pupil'][0]

    we_ = trace.posterior['W_e'][0]

    C_val_hr = at.dot(np.linalg.pinv(whr_), hr_new.T)
    C_val_eda = at.dot(np.linalg.pinv(weda_), eda_new.T)
    C_val_pupil = at.dot(np.linalg.pinv(wpupil_), pupil_new.T)

    val_hr = at.matmul(np.array(we_), C_val_hr.eval())
    val_eda = at.matmul(np.array(we_), C_val_eda.eval())
    val_pupil = at.matmul(np.array(we_), C_val_pupil.eval())

    val_label_gen = at.concatenate((val_hr, val_eda, val_pupil))

    label_val = np.where(val_label_gen.eval() < 0, 0, 1)
    label_val = stats.mode(label_val[0], keepdims=False)[0]
    return label_val


columns = ['subject', 'k', 'fold', 'train', 'test']
with open('output/FA/FA_new_postpred_cv_norm.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)

for sub in all_subject:
    # loop within all k
    for k in valid_k_list:

        eda = pd.read_csv('data/features_4_2/eda/' + str(sub) + '.csv')
        eda = eda[num_trials_to_remove:]
        eda = scaler.fit_transform(eda)

        # hr data
        hr = pd.read_csv('data/features_4_2/hr/' + str(sub) + '.csv')
        hr = hr[num_trials_to_remove:]
        hr = scaler.fit_transform(hr)

        # pupil data
        pupil = pd.read_csv('data/features_4_2/pupil/' + str(sub) + '.csv')
        pupil = pupil[num_trials_to_remove:]
        pupil = scaler.fit_transform(pupil)

        string_sub = extract_correct_csv.read_correct_subject_csv(sub)

        df_ = pd.read_csv('data/LookAtMe_0' + str(string_sub) + '.csv', sep='\t')
        df_ = df_[num_trials_to_remove:]
        label = np.array(list([int(d > 2) for d in df_['rating']]))
        E = label[:, np.newaxis]
        E = pd.DataFrame(E)

        # num trials
        N = eda.shape[0]

        TRAIN_PERC = 0.70
        VAL_PERC = 0.1
        TEST_PERC = 0.2  # 1-TRAIN_PERC+VAL_PERC
        N_train = int(N * (TRAIN_PERC))
        N_val = int(N * (VAL_PERC))

        eda = eda.reset_index().drop(columns=('index'))
        pupil = pupil.reset_index().drop(columns=('index'))
        hr = hr.reset_index().drop(columns=('index'))
        E = E.reset_index().drop(columns=('index'))

        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=123)
        sss.get_n_splits(eda, E)

        for i, (train_index, test_index) in enumerate(sss.split(eda, E)):
            N_train = len(train_index)

            eda_train = eda.iloc[train_index, :]
            eda_test = eda.iloc[test_index, :]
            hr_train = hr.iloc[train_index, :]
            hr_test = hr.iloc[test_index, :]
            pupil_train = pupil.iloc[train_index, :]
            pupil_test = pupil.iloc[test_index, :]
            e_labels_train = E.iloc[train_index, :]
            e_labels_test = E.iloc[test_index, :]

            # dimensions of each signal
            d_eda = eda_train.shape[1]
            d_hr = hr_train.shape[1]
            d_pupil = pupil_train.shape[1]
            d_e = e_labels_train.shape[1]

            # model definition
            with pm.Model() as PPCA_identified:
                # model coordinates
                PPCA_identified.add_coord("latent_columns", np.arange(k), mutable=True)
                PPCA_identified.add_coord("rows", np.arange(N_train), mutable=True)
                PPCA_identified.add_coord("observed_eda", np.arange(d_eda), mutable=False)
                PPCA_identified.add_coord("observed_hr", np.arange(d_hr), mutable=False)
                PPCA_identified.add_coord("observed_pupil", np.arange(d_pupil), mutable=False)
                PPCA_identified.add_coord("observed_label", np.arange(d_e), mutable=False)

                hr_data = pm.MutableData("hr_data", hr_train.T, dims=["observed_hr", "rows"])
                eda_data = pm.MutableData("eda_data", eda_train.T, dims=("observed_eda", "rows"))
                pupil_data = pm.MutableData("pupil_data", pupil_train.T, dims=("observed_pupil", "rows"))

                W_eda = makeW(d_eda, k, ("observed_eda", "latent_columns"), 'W_eda')
                W_hr = makeW(d_hr, k, ("observed_hr", "latent_columns"), 'W_hr')
                W_pupil = makeW(d_pupil, k, ("observed_pupil", "latent_columns"), 'W_pupil')

                W_e = pm.Normal("W_e", dims=["observed_label", "latent_columns"])
                C = pm.Normal("C", dims=["latent_columns", "rows"])

                psi_eda = pm.HalfNormal("psi_eda", 1.0)
                X_eda = pm.Normal("X_eda", mu=at.dot(W_eda, C), sigma=psi_eda, observed=eda_data,
                                  dims=["observed_eda", "rows"])

                psi_hr = pm.HalfNormal("psi_hr", 1.0)
                X_hr = pm.Normal("X_hr", mu=at.dot(W_hr, C), sigma=psi_hr, observed=hr_data, dims=["observed_hr", "rows"])

                psi_pupil = pm.HalfNormal("psi_pupil", 1.0)
                X_pupil = pm.Normal("X_pupil", mu=at.dot(W_pupil, C), sigma=psi_pupil, observed=pupil_data,
                                    dims=["observed_pupil", "rows"])

                X_e = pm.Bernoulli("X_e", p=pm.math.sigmoid(at.dot(W_e, C)), dims=["observed_label", "rows"],
                                   observed=e_labels_train.T)

            with PPCA_identified:
                approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
                trace = approx.sample(1000)

            with PPCA_identified:
                posterior_predictive = pm.sample_posterior_predictive(
                    trace, var_names=["X_e"], random_seed=123)

            e_pred_train = posterior_predictive.posterior_predictive['X_e']
            e_pred_mode_train = np.squeeze(stats.mode(e_pred_train[0], keepdims=False)[0])[:, np.newaxis]

            train_accuracy_exp = accuracy_score(e_labels_train, e_pred_mode_train)
            conf_mat_train = confusion_matrix(e_labels_train, e_pred_mode_train)
            fig = plt.figure()
            plt.matshow(conf_mat_train)
            plt.title('Confusion Matrix all subjs train k=' + str(k))
            plt.colorbar()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('output/FA/unpooled/confusion_matrix_' + str(k) + 'train_cv_norm.jpg')

            # test
            e_pred_mode_test = my_post_predict(trace, hr_test, eda_test, pupil_test)
            test_accuracy_exp = accuracy_score(e_labels_test, e_pred_mode_test)
            conf_mat_test = confusion_matrix(e_labels_test, e_pred_mode_test)
            fig = plt.figure()
            plt.matshow(conf_mat_test)
            plt.title('Confusion Matrix all subjs test k=' + str(k))
            plt.colorbar()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('output/FA/unpooled/confusion_matrix_' + str(k) + 'test_cv_norm.jpg')

            row = [sub, k, i, train_accuracy_exp, test_accuracy_exp]

            with open('output/FA/FA_new_postpred_cv_norm.csv', 'a') as f:
                write = csv.writer(f)
                write.writerow(row)
