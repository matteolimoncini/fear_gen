import pymc as pm
import aesara.tensor as at
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import csv
import sys

sys.path.append('../../')

import extract_correct_csv
import os

os.chdir('..')
os.chdir('..')

scaler = StandardScaler()
RANDOM_SEED = 31415
rng = default_rng(RANDOM_SEED)

all_subject = extract_correct_csv.extract_only_valid_subject()
all_subject.remove(49)

valid_k_list = list([2, 6, 10, 12, 15, 20])

# keep only generalization trials
num_trials_to_remove = 48

TEST_PERC = 0.2
FILENAME = 'output/FA/FA_kcv_norm_image.csv'
columns = ['subject', 'k', 'fold', 'train', 'test']

with open(FILENAME, 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)


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


def convert_to_matrix(vector):
    """
    Converts a 1-dimensional Python array into a matrix with 6 columns.
    Each row of the matrix corresponds to an element in the input vector, and has a 1 in the column
    corresponding to the value of the element in the vector.

    Parameters:
    vector (list): A 1-dimensional list with values ranging from 1 to 6.

    Returns:
    list: A 2-dimensional list representing the matrix.

    Example:
    >> convert_to_matrix([1, 2, 6, 4, 5, 3, 6, 2])
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 1, 0, 0, 0, 0]]
    """
    num_rows = len(vector)
    matrix = [[0 for j in range(6)] for i in range(num_rows)]
    for i, value in enumerate(vector):
        matrix[i][value - 1] = 1
    return matrix


def my_post_predict(trace, hr_new, eda_new, pupil_new, img_new):
    whr_ = trace.posterior['W_hr'][0]
    weda_ = trace.posterior['W_eda'][0]
    wpupil_ = trace.posterior['W_pupil'][0]
    wimg_ = trace.posterior['W_img'][0]
    we_ = trace.posterior['W_e'][0]

    C_val_hr = at.dot(np.linalg.pinv(whr_), hr_new.T)
    C_val_eda = at.dot(np.linalg.pinv(weda_), eda_new.T)
    C_val_pupil = at.dot(np.linalg.pinv(wpupil_), pupil_new.T)
    C_val_img = at.dot(np.linalg.pinv(wimg_), img_new.T)

    val_hr = at.matmul(np.array(we_), C_val_hr.eval())
    val_eda = at.matmul(np.array(we_), C_val_eda.eval())
    val_pupil = at.matmul(np.array(we_), C_val_pupil.eval())
    val_img = at.matmul(np.array(wimg_), C_val_img.eval())

    val_label_gen = at.concatenate((val_hr, val_eda, val_pupil, val_img))

    label_val = np.where(val_label_gen.eval() < 0, 0, 1)
    label_val = stats.mode(label_val[0], keepdims=False)[0]

    return label_val


def extract_cs(df):
    cs1, cs2 = int(df[df.shock == True].picName.unique()[0][5]), int(df[df.shock == True].picName.unique()[1][5])
    return cs1, cs2


def extract_threat_level(df):
    MORPH_POS = 11
    MORPH_VALUE = 15
    threat_person = []
    cs1_pic, cs2_pic = extract_cs(df)

    for i in df.iterrows():
        nome = i[1].picName
        try:
            if nome[5] == str(cs1_pic) or nome[5] == str(cs2_pic):
                threat_person.append(6)
                continue
            elif int(nome[5]):
                threat_person.append(1)
                continue
        except:
            if (int(nome[MORPH_POS]) == cs1_pic) or (int(nome[MORPH_POS]) == cs2_pic):
                if nome[MORPH_VALUE] == '2':
                    threat_person.append(5)

                elif nome[MORPH_VALUE] == '4':
                    threat_person.append(4)

                elif nome[MORPH_VALUE] == '6':
                    threat_person.append(3)

                elif nome[MORPH_VALUE] == '8':
                    threat_person.append(2)
                    continue
            else:
                if nome[MORPH_VALUE] == '2':
                    threat_person.append(2)

                elif nome[MORPH_VALUE] == '4':
                    threat_person.append(3)

                elif nome[MORPH_VALUE] == '6':
                    threat_person.append(4)
                elif nome[MORPH_VALUE] == '8':
                    threat_person.append(5)
    return threat_person


# loop within all subjects
for sub in all_subject:
    # loop within all k
    for k in valid_k_list:

        # eda data
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
        img = extract_threat_level(df_)

        img = convert_to_matrix(img)

        eda = pd.DataFrame(eda)
        eda = eda.reset_index().drop(columns=('index'))
        pupil = pd.DataFrame(pupil)
        pupil = pupil.reset_index().drop(columns=('index'))
        hr = pd.DataFrame(hr)
        hr = hr.reset_index().drop(columns=('index'))
        img = pd.DataFrame(img)
        img = img.reset_index().drop(columns=('index'))
        E = pd.DataFrame(E)
        E = E.reset_index().drop(columns=('index'))

        sss = StratifiedShuffleSplit(n_splits=3, test_size=TEST_PERC, random_state=123)
        for i, (train_index, test_index) in enumerate(sss.split(eda, E)):
            N_train = len(train_index)

            eda_train = eda.iloc[train_index, :]
            eda_test = eda.iloc[test_index, :]
            hr_train = hr.iloc[train_index, :]
            hr_test = hr.iloc[test_index, :]
            pupil_train = pupil.iloc[train_index, :]
            pupil_test = pupil.iloc[test_index, :]
            img_train = img.iloc[train_index, :]
            img_test = img.iloc[test_index, :]
            e_labels_train = E.iloc[train_index, :]
            e_labels_test = E.iloc[test_index, :]

            # dimensions of each signal
            d_eda = eda_train.shape[1]
            d_hr = hr_train.shape[1]
            d_pupil = pupil_train.shape[1]
            d_e = e_labels_train.shape[1]
            d_img = img_train.shape[1]

            # model definition
            with pm.Model() as PPCA_identified:
                # model coordinates
                PPCA_identified.add_coord("latent_columns", np.arange(k), mutable=True)
                PPCA_identified.add_coord("rows", np.arange(N_train), mutable=True)
                PPCA_identified.add_coord("observed_eda", np.arange(d_eda), mutable=False)
                PPCA_identified.add_coord("observed_hr", np.arange(d_hr), mutable=False)
                PPCA_identified.add_coord("observed_pupil", np.arange(d_pupil), mutable=False)
                PPCA_identified.add_coord("observed_img", np.arange(d_img), mutable=False)
                PPCA_identified.add_coord("observed_label", np.arange(d_e), mutable=False)

                hr_data = pm.MutableData("hr_data", hr_train.T, dims=["observed_hr", "rows"])
                eda_data = pm.MutableData("eda_data", eda_train.T, dims=("observed_eda", "rows"))
                pupil_data = pm.MutableData("pupil_data", pupil_train.T, dims=("observed_pupil", "rows"))
                img_data = pm.MutableData("img_data", img_train.T, dims=("observed_img", "rows"))

                W_eda = makeW(d_eda, k, ("observed_eda", "latent_columns"), 'W_eda')
                W_hr = makeW(d_hr, k, ("observed_hr", "latent_columns"), 'W_hr')
                W_pupil = makeW(d_pupil, k, ("observed_pupil", "latent_columns"), 'W_pupil')
                W_img = pm.Normal("W_img", dims=["observed_img", "latent_columns"])
                W_e = pm.Normal("W_e", dims=["observed_label", "latent_columns"])

                C = pm.Normal("C", dims=["latent_columns", "rows"])

                psi_eda = pm.HalfNormal("psi_eda", 1.0)
                X_eda = pm.Normal("X_eda", mu=at.dot(W_eda, C), sigma=psi_eda, observed=eda_data,
                                  dims=["observed_eda", "rows"])

                psi_hr = pm.HalfNormal("psi_hr", 1.0)
                X_hr = pm.Normal("X_hr", mu=at.dot(W_hr, C), sigma=psi_hr, observed=hr_data,
                                 dims=["observed_hr", "rows"])

                psi_pupil = pm.HalfNormal("psi_pupil", 1.0)
                X_pupil = pm.Normal("X_pupil", mu=at.dot(W_pupil, C), sigma=psi_pupil, observed=pupil_data,
                                    dims=["observed_pupil", "rows"])

                X_img = pm.Categorical('X_img', p=pm.math.softmax(at.dot(W_img, C)), dims=["observed_img", "rows"],
                                       observed=img_data)

                X_e = pm.Bernoulli("X_e", p=pm.math.sigmoid(at.dot(W_e, C)), dims=["observed_label", "rows"],
                                   observed=e_labels_train.T)

            g = pm.model_to_graphviz(PPCA_identified)
            g.view('tmp')

            with PPCA_identified:
                approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
                trace = approx.sample(1000)

            with PPCA_identified:
                posterior_predictive = pm.sample_posterior_predictive(
                    trace, var_names=["X_e"], random_seed=123)

            # train
            e_pred_train = posterior_predictive.posterior_predictive['X_e']
            e_pred_mode_train = np.squeeze(stats.mode(e_pred_train[0], keepdims=False)[0])[:, np.newaxis]
            train_accuracy_exp = accuracy_score(e_labels_train, e_pred_mode_train)

            # test
            e_pred_mode_test = my_post_predict(trace, hr_test, eda_test, pupil_test, img_test)
            test_accuracy_exp = accuracy_score(e_labels_test, e_pred_mode_test)

            row = [sub, k, i, train_accuracy_exp, test_accuracy_exp]

            with open(FILENAME, 'a') as f:
                write = csv.writer(f)
                write.writerow(row)
