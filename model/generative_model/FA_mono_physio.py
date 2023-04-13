import pymc as pm
import aesara.tensor as at
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy import stats
from sklearn.metrics import accuracy_score
import os
import extract_correct_csv as functions


RANDOM_SEED = 31415
rng = default_rng(RANDOM_SEED)

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


all_subjects = functions.extract_only_valid_subject()

# removing subject 49
all_subjects.remove(49)

# loop latent dimensions
all_k = np.arange(2, 11)

# loop features
types_ = ['hr', 'eda', 'pupil']

columns = ['Subject', 'Latent dimension', 'Feature type', 'Train accuracy', 'Validation Accuracy', 'Test accuracy']
results = pd.DataFrame(columns=columns)

for sub in all_subjects:

    for k in all_k:
        # First view
        string_sub = functions.read_correct_subject_csv(sub)
        df_ = pd.read_csv('data/LookAtMe_0' + string_sub + '.csv', sep='\t')
        df_ = df_[num_trials_to_remove:]
        label = np.array(list([int(d > 2) for d in df_['rating']]))
        E = label[:, np.newaxis]
        E = pd.DataFrame(E)


        # Second view

        for type_ in types_:
            feature = pd.read_csv('data/features_4_2/'+type_+'/' + str(sub) + '.csv')
            feature = feature[num_trials_to_remove:]


            # num trials
            N = feature.shape[0]

            TRAIN_PERC = 0.70
            VAL_PERC = 0.1
            TEST_PERC = 0.2  # 1-TRAIN_PERC+VAL_PERC
            N_train = int(N * TRAIN_PERC)
            N_val = int(N * VAL_PERC)

            # RANDOM SPLIT
            feature = feature.sample(frac=1, random_state=0)
            feature = feature.reset_index(drop=True).to_numpy()

            e_labels = E.sample(frac=1, random_state=0)
            e_labels = e_labels.reset_index(drop=True).to_numpy()

            feature_train = feature[:N_train]
            e_labels_train = e_labels[:N_train]

            feature_val = feature[N_train:N_train + N_val]
            e_labels_val = e_labels[N_train:N_train + N_val]

            feature_test = feature[N_train + N_val:]
            e_labels_test = e_labels[N_train + N_val:]

            # dimensions of each signal
            d_feature = feature_train.shape[1]
            d_e = e_labels_train.shape[1]

            with pm.Model() as PPCA_identified:
                # model coordinates
                PPCA_identified.add_coord("latent_columns", np.arange(k), mutable=True)
                PPCA_identified.add_coord("rows", np.arange(N_train), mutable=True)
                PPCA_identified.add_coord("observed_feature", np.arange(d_feature), mutable=False)
                PPCA_identified.add_coord("observed_label", np.arange(d_e), mutable=False)

                feature_data = pm.MutableData("feature_data", feature_train.T, dims=["observed_feature", "rows"])

                W_feature = makeW(d_feature, k, ("observed_feature", "latent_columns"), 'W_feature')

                W_e = pm.Normal("W_e", dims=["observed_label", "latent_columns"])
                C = pm.Normal("C", dims=["latent_columns", "rows"])

                psi_feature = pm.HalfNormal("psi_feature", 1.0)
                X_feature = pm.Normal("X_feature", mu=at.dot(W_feature, C), sigma=psi_feature, observed=feature_data, dims=["observed_feature", "rows"])

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
            train_accuracy = accuracy_score(e_pred_mode_train, e_labels_train)

            def my_post_predict(trace,feature_val):
                wfeature_ = trace.posterior['W_feature'][0]
                C_val = at.dot(np.linalg.pinv(wfeature_), feature_val.T)
                we_ = trace.posterior['W_e'][0]
                val_label_gen = at.matmul(np.array(we_),C_val.eval())
                label_val = np.where(val_label_gen.eval()<0, 0, 1)
                label_val = stats.mode(label_val[0], keepdims=False)[0]
                return  label_val
            validation_accuracy_exp = accuracy_score(e_labels_val, my_post_predict(trace,feature_val))
            test_accuracy_exp = accuracy_score(e_labels_test, my_post_predict(trace, feature_test))

            dict_ = {'Subject': sub,
                     'Latent dimension': k,
                     'Feature type': type_,
                     'Train accuracy': train_accuracy,
                     'Validation accuracy': validation_accuracy_exp,
                     'Test accuracy':test_accuracy_exp}

            results = pd.concat([results, pd.DataFrame(data=dict_, index=np.arange(1))], ignore_index=True)


results.to_csv('output/FA/ResultsMonoFeatures.csv')
