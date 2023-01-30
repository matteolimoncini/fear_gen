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
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append('../../')

import extract_correct_csv as functions
import os

scaler = StandardScaler()
RANDOM_SEED = 31415
rng = default_rng(RANDOM_SEED)

os.chdir('..')
os.chdir('..')
# all valid subjects
all_subject = functions.extract_only_valid_subject()
all_subject.remove(49)

# all k = {2, 4, 6, 8} for the latent space
valid_k_list = list(range(2, 10))

# keep only generalization trials
num_trials_to_remove = 48

columns = ['subject', 'k', 'train', 'val', 'test']
with open('output/FA/FA_new_complete.csv', 'w') as f:
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


def makeW(d, k, dim_names, name):
    # make a W matrix adapted to the data shape
    n_od = int(k * d - k * (k - 1) / 2 - k)
    # trick: the cumulative sum of z will be positive increasing
    z = pm.HalfNormal("W_z_" + name, 1.0, dims="latent_columns")
    b = pm.HalfNormal("W_b_" + name, 1.0, shape=(n_od,), dims="packed_dim")
    L = expand_packed_block_triangular(d, k, b, at.ones(k))
    W = pm.Deterministic(name, at.dot(L, at.diag(at.extra_ops.cumsum(z))), dims=dim_names)
    return W


TRAIN_PERC = 0.70
VAL_PERC = 0.1
TEST_PERC = 0.2  # 1-TRAIN_PERC+VAL_PERC
NUM_TRIAL = 160
TRIAL = NUM_TRIAL * len(all_subject)

global_e_labels = []
global_subject = []
for sub in all_subject:
    subj_ = functions.read_correct_subject_csv(sub)
    csv_ = 'data/LookAtMe_0' + str(subj_) + '.csv'
    global_data = pd.read_csv(csv_, sep='\t')
    y = np.array(list([int(d > 2) for d in global_data['rating']]))
    e_labels = y[:, np.newaxis]  # rating > 2
    e_labels = e_labels[num_trials_to_remove:]
    global_e_labels = global_e_labels + e_labels.tolist()
    subject = np.array(list([s for s in global_data['subject']]))[:, np.newaxis]
    subject = subject[num_trials_to_remove:]
    global_subject = global_subject + subject.tolist()

global_e_labels = np.array(global_e_labels)
global_subject = np.array(global_subject)


def populate_array(x, name):
    return name[NUM_TRIAL * (x - 1) + num_trials_to_remove:NUM_TRIAL * x]


hr_temp = np.concatenate([pd.read_csv('data/features_4_2/hr/' + str(x) + '.csv') for x in all_subject])
hr = np.concatenate([populate_array(x, hr_temp) for x in range(1, len(all_subject) + 1)])
hr = pd.DataFrame(scaler.fit_transform(hr))

pupil_temp = np.concatenate([pd.read_csv('data/features_4_2/pupil/' + str(x) + '.csv') for x in all_subject])
pupil = np.concatenate([populate_array(x, pupil_temp) for x in range(1, len(all_subject) + 1)])
pupil = pd.DataFrame(scaler.fit_transform(pupil))

eda_temp = np.concatenate([pd.read_csv('data/features_4_2/eda/' + str(x) + '.csv') for x in all_subject])
eda = np.concatenate([populate_array(x, eda_temp) for x in range(1, len(all_subject) + 1)])
eda = pd.DataFrame(scaler.fit_transform(eda))

N = eda.shape[0]
N_train = int(N * (TRAIN_PERC))
N_val = int(N * (VAL_PERC))

N_pupil = pupil.shape[0]
D_pupil = pupil.shape[1]

N_hr = hr.shape[0]
D_hr = hr.shape[1]

N_eda = eda.shape[0]
D_eda = eda.shape[1]

# RANDOM SPLIT
pupil = pd.DataFrame(pupil).sample(frac=1, random_state=0)
pupil = pupil.reset_index(drop=True).to_numpy()

hr = pd.DataFrame(hr).sample(frac=1, random_state=0)
hr = hr.reset_index(drop=True).to_numpy()

eda = pd.DataFrame(eda).sample(frac=1, random_state=0)
eda = eda.reset_index(drop=True).to_numpy()

e_labels = pd.DataFrame(global_e_labels).sample(frac=1, random_state=0)
e_labels = e_labels.reset_index(drop=True).to_numpy()

hr_train = hr[:N_train]
eda_train = eda[:N_train]
pupil_train = pupil[:N_train]
e_labels_train = e_labels[:N_train]

hr_val = hr[N_train:N_train + N_val]
eda_val = eda[N_train:N_train + N_val]
pupil_val = pupil[N_train:N_train + N_val]
e_labels_val = e_labels[N_train:N_train + N_val]

hr_test = hr[N_train + N_val:]
eda_test = eda[N_train + N_val:]
pupil_test = pupil[N_train + N_val:]
e_labels_test = e_labels[N_train + N_val:]

# dimensions of each signal
d_eda = eda_train.shape[1]
d_hr = hr_train.shape[1]
d_pupil = pupil_train.shape[1]
d_e = e_labels_train.shape[1]

'''print(d_eda, d_hr, d_pupil, d_e)
print(hr_train.shape)
print(hr_val.shape)
print(hr_test.shape)
print(eda_train.shape)
print(eda_val.shape)
print(eda_test.shape)
print(pupil_train.shape)
print(pupil_val.shape)
print(pupil_test.shape)
print(e_labels_train.T.shape)
print(e_labels_val.shape)
print(e_labels_test.shape)'''

# loop within all k
for k in valid_k_list:
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

    # gv = pm.model_to_graphviz(PPCA_identified)
    # gv.view('PPCA factor')

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
    plt.savefig('output/FA/complete/confusion_matrix_' + str(k) + 'train.jpg')

    val_results = my_post_predict(trace, hr_val, eda_val, pupil_val)
    validation_accuracy_exp = accuracy_score(e_labels_val, val_results)

    conf_mat_val = confusion_matrix(e_labels_val, val_results)
    fig = plt.figure()
    plt.matshow(conf_mat_val)
    plt.title('Confusion Matrix all subjs val k=' + str(k))
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('output/FA/complete/confusion_matrix_' + str(k) + 'val.jpg')

    test_results = my_post_predict(trace, hr_test, eda_test, pupil_test)
    test_accuracy_exp = accuracy_score(e_labels_test, test_results)

    conf_mat_test = confusion_matrix(e_labels_test, test_results)
    fig = plt.figure()
    plt.matshow(conf_mat_test)
    plt.title('Confusion Matrix all subjs test k=' + str(k))
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('output/FA/complete/confusion_matrix_' + str(k) + 'test.jpg')

    row = ['allsubj', k, train_accuracy_exp, validation_accuracy_exp, test_accuracy_exp]

    with open('output/FA/FA_new_complete.csv', 'a') as f:
        write = csv.writer(f)
        write.writerow(row)
