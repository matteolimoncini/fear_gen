import pymc as pm
import aesara.tensor as at
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy import stats
from sklearn.metrics import accuracy_score

from fear_gen import extract_correct_csv

RANDOM_SEED = 31415
rng = default_rng(RANDOM_SEED)

# all valid subjects
all_subject = extract_correct_csv.extract_only_valid_subject()
all_subject.remove(49)

# all k = {2, 4, 6, 8} for the latent space
valid_k_list = list(range(2, 10, 2))

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

def makeW(d, k, dim_names,name):
    # make a W matrix adapted to the data shape
    n_od = int(k * d - k * (k - 1) / 2 - k)
    # trick: the cumulative sum of z will be positive increasing
    z = pm.HalfNormal("W_z_"+name, 1.0, dims="latent_columns")
    b = pm.HalfNormal("W_b_"+name, 1.0, shape=(n_od,), dims="packed_dim")
    L = expand_packed_block_triangular(d, k, b, at.ones(k))
    W = pm.Deterministic(name, at.dot(L, at.diag(at.extra_ops.cumsum(z))), dims=dim_names)
    return W


# loop within all subjects
for sub in all_subject:
    # loop within all k
    for k in valid_k_list:

        # eda data
        eda = pd.read_csv('data/features_4_2/eda/'+str(sub)+'.csv').to_numpy()
        eda = eda.T

        hr = pd.read_csv('data/features_4_2/hr/'+str(sub)+'.csv').to_numpy()
        hr = hr.T

        pupil = pd.read_csv('data/features_4_2/pupil/'+str(sub)+'.csv').to_numpy()
        pupil = pupil.T

        df_ = pd.read_csv('data/LookAtMe_002.csv', sep='\t')
        label = np.array(list([int(d>2) for d in df_['rating']]))
        E = label[:,np.newaxis]
        E = np.transpose(E)

        N = eda.shape[1]
        d_eda = eda.shape[0]
        d_hr = hr.shape[0]
        d_e = E.shape[0]
        d_pupil = pupil.shape[0]



        coords = {"latent_columns": np.arange(k),
                  "rows": np.arange(N),
                  "observed_eda": np.arange(d_eda),
                  "observed_label":np.arange(d_e),
                  "observed_hr":np.arange(d_hr),
                  "observed_pupil":np.arange(d_pupil)}


        with pm.Model(coords=coords) as PPCA_identified:
            W_eda = makeW(d_eda, k, ("observed_eda", "latent_columns"),'W_eda')
            W_hr = makeW(d_hr, k, ("observed_hr", "latent_columns"),'W_hr')
            W_pupil = pm.Normal("W_pupil", dims=("observed_pupil", "latent_columns"))

            W_e = pm.Normal("W_e", dims=("observed_label", "latent_columns"))
            C = pm.Normal("C", dims=("latent_columns", "rows"))
            psi_eda = pm.HalfNormal("psi_eda", 1.0)
            X_eda = pm.Normal("X_eda", mu=at.dot(W_eda, C), sigma=psi_eda, observed=eda, dims=("observed_eda", "rows"))

            psi_hr = pm.HalfNormal("psi_hr", 1.0)
            X_hr = pm.Normal("X_hr", mu=at.dot(W_hr, C), sigma=psi_hr, observed=hr, dims=("observed_hr", "rows"))

            psi_pupil = pm.HalfNormal("psi_pupil", 1.0)
            X_pupil = pm.Normal("X_pupil", mu=at.dot(W_pupil, C), sigma=psi_pupil, observed=pupil, dims=("observed_pupil", "rows"))

            X_e = pm.Bernoulli("X_e", p=pm.math.sigmoid(at.dot(W_e, C)), dims=("observed_label", "rows"), observed=E)

        gv = pm.model_to_graphviz(PPCA_identified)
        gv.view('PPCA example')

        with PPCA_identified:
            approx = pm.fit(30000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
            trace = approx.sample(500)

        with PPCA_identified:
            posterior_predictive = pm.sample_posterior_predictive(
                trace, var_names=["X_e"], random_seed=123)

        e_pred_train = posterior_predictive.posterior_predictive['X_e']
        e_pred_mode_train = np.squeeze(stats.mode(e_pred_train[0], keepdims=False)[0])[:, np.newaxis]

        train_accuracy_exp = accuracy_score(E.T, e_pred_mode_train)
