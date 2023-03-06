if __name__ == "__main__":
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
    import re
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    from multiprocessing import Pool, cpu_count

    sys.path.append('../../..')
    import extract_correct_csv
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    import os

    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    scaler = StandardScaler()
    RANDOM_SEED = 31415
    rng = default_rng(RANDOM_SEED)

    # all valid subjects
    all_subject = extract_correct_csv.extract_only_valid_subject()

    columns = ['subject', 'k', 'fold', 'train', 'test']

    with open('output/sppca_physio_gaze.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(columns)


    def run_sub(sub, valid_k_list, result_modified):
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

        def custom_parse_data(X):
            res_x = []
            for i in range(len(X)):
                string = X[i]
                new_string = re.sub(r'\s+', ',', string)
                string_list = list(new_string)
                string_list[1] = ''
                new_string = ''.join(string_list)
                res_x.append(np.fromstring(new_string.strip('[]'), dtype=float, sep=','))

            return np.array(res_x)

        def makeW(d, k, dim_names, name):
            # make a W matrix adapted to the data shape
            n_od = int(k * d - k * (k - 1) / 2 - k)
            # trick: the cumulative sum of z will be positive increasing
            z = pm.HalfNormal("W_z_" + name, 1.0, dims="latent_columns")
            b = pm.HalfNormal("W_b_" + name, 1.0, shape=(n_od,), dims="packed_dim")
            L = expand_packed_block_triangular(d, k, b, at.ones(k))
            W = pm.Deterministic(name, at.dot(L, at.diag(at.extra_ops.cumsum(z))), dims=dim_names)
            return W

        def my_post_predict(trace, hr_new, eda_new, pupil_new, gaze_new):
            whr_ = trace.posterior['W_hr'][0]
            weda_ = trace.posterior['W_eda'][0]
            wpupil_ = trace.posterior['W_pupil'][0]
            wgaze_ = trace.posterior['W_gaze'][0]

            we_ = trace.posterior['W_e'][0]

            C_val_hr = at.dot(np.linalg.pinv(whr_), hr_new.T)
            C_val_eda = at.dot(np.linalg.pinv(weda_), eda_new.T)
            C_val_pupil = at.dot(np.linalg.pinv(wpupil_), pupil_new.T)
            C_val_gaze = at.dot(np.linalg.pinv(wgaze_), gaze_new.T)

            val_hr = at.matmul(np.array(we_), C_val_hr.eval())
            val_eda = at.matmul(np.array(we_), C_val_eda.eval())
            val_pupil = at.matmul(np.array(we_), C_val_pupil.eval())
            val_gaze = at.matmul(np.array(we_), C_val_gaze.eval())

            val_label_gen = at.concatenate((val_hr, val_eda, val_pupil, val_gaze))

            label_val = np.where(val_label_gen.eval() < 0, 0, 1)
            label_val = stats.mode(label_val[0], keepdims=False)[0]
            return label_val

        df_ = pd.read_csv('data/gaze/joined_fixation.csv')
        # remove index column
        df_ = df_.drop(columns=df_.columns[0])
        # replace NaN with 'other'
        df_['ROI'] = df_['ROI'].replace(np.NaN, 'other')
        # parse fixation feature column to convert this in value
        feat = custom_parse_data(df_['Fixation feature'])
        df_['Fixation feature'] = [x for x in feat]
        # mean features with same ROI
        result = df_.groupby(['Subject', 'Trial', 'ROI'], as_index=False)['Fixation feature'].mean()
        # insert an array of zeros if one subject in one trial has no fixations on that ROI
        result_modified = result.copy(deep=True)
        for sub in result['Subject'].unique():
            df = result[result['Subject'] == sub]
            for trial in df['Trial'].unique():
                if 'mouth_nose' not in list(df[df['Trial'] == trial]['ROI']):
                    result_modified = result_modified.append(
                        {'Subject': sub, 'Trial': trial, 'ROI': 'mouth_nose', 'Fixation feature': [0] * 13},
                        ignore_index=True)
                if 'eye' not in list(df[df['Trial'] == trial]['ROI']):
                    result_modified = result_modified.append(
                        {'Subject': sub, 'Trial': trial, 'ROI': 'eye', 'Fixation feature': [0] * 13}, ignore_index=True)
                if 'other' not in list(df[df['Trial'] == trial]['ROI']):
                    result_modified = result_modified.append(
                        {'Subject': sub, 'Trial': trial, 'ROI': 'other', 'Fixation feature': [0] * 13},
                        ignore_index=True)

        # all k = {2, 4, 6, 8} for the latent space
        valid_k_list = list([2, 4, 6, 8, 10, 12])

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

            df_ = pd.read_csv('data/LookAtMe_old/LookAtMe_0' + str(string_sub) + '.csv', sep='\t')
            df_ = df_[num_trials_to_remove:]
            label = np.array(list([int(d > 2) for d in df_['rating']]))
            E = label[:, np.newaxis]
            E = pd.DataFrame(E)

            # gaze data
            df_sub = result_modified[result_modified['Subject'] == sub]
            # normalize feature and convert from [13][13][13] into [39]
            X1 = df_sub[df_sub['ROI'] == 'eye']
            X1_norm = pd.DataFrame(scaler.fit_transform(list(X1['Fixation feature'])))
            X2 = df_sub[df_sub['ROI'] == 'mouth_nose']
            X2_norm = pd.DataFrame(scaler.fit_transform(list(X2['Fixation feature'])))
            X3 = df_sub[df_sub['ROI'] == 'other']
            X3_norm = pd.DataFrame(scaler.fit_transform(list(X3['Fixation feature'])))
            X_norm = pd.concat([X1_norm, X2_norm, X3_norm], axis=1)
            # remove first 48 learning trials
            X_norm = X_norm[48:]
            X_norm = pd.DataFrame(X_norm)
            X_norm = X_norm.reset_index().drop(columns=('index'))
            gaze = X_norm

            # num trials
            N = eda.shape[0]

            TRAIN_PERC = 0.70
            VAL_PERC = 0.1
            TEST_PERC = 0.2  # 1-TRAIN_PERC+VAL_PERC
            N_train = int(N * (TRAIN_PERC))
            N_val = int(N * (VAL_PERC))

            eda = pd.DataFrame(eda)
            eda = eda.reset_index().drop(columns=('index'))
            pupil = pd.DataFrame(pupil)
            pupil = pupil.reset_index().drop(columns=('index'))
            hr = pd.DataFrame(hr)
            hr = hr.reset_index().drop(columns=('index'))
            E = pd.DataFrame(E)
            E = E.reset_index().drop(columns=('index'))

            # convert features into dataframe and reset index
            gaze = pd.DataFrame(gaze)
            gaze = gaze.reset_index().drop(columns=('index'))
            # RANDOM SPLIT of the gaze data
            gaze = gaze.sample(frac=1, random_state=0)
            gaze = gaze.reset_index(drop=True).to_numpy()
            # convert data into dataframe and reset index
            gaze = pd.DataFrame(gaze)
            gaze = gaze.reset_index().drop(columns=('index'))

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

                gaze_train = gaze.iloc[train_index, :]
                gaze_test = gaze.iloc[test_index, :]

                # dimensions of each signal
                d_eda = eda_train.shape[1]
                d_hr = hr_train.shape[1]
                d_pupil = pupil_train.shape[1]
                d_e = e_labels_train.shape[1]
                d_gaze = gaze_train.shape[1]

                # model definition
                with pm.Model() as PPCA_identified:
                    # model coordinates
                    PPCA_identified.add_coord("latent_columns", np.arange(k), mutable=True)
                    PPCA_identified.add_coord("rows", np.arange(N_train), mutable=True)
                    PPCA_identified.add_coord("observed_eda", np.arange(d_eda), mutable=False)
                    PPCA_identified.add_coord("observed_hr", np.arange(d_hr), mutable=False)
                    PPCA_identified.add_coord("observed_pupil", np.arange(d_pupil), mutable=False)
                    PPCA_identified.add_coord("observed_label", np.arange(d_e), mutable=False)
                    PPCA_identified.add_coord("observed_gaze", np.arange(d_gaze), mutable=False)

                    hr_data = pm.MutableData("hr_data", hr_train.T, dims=["observed_hr", "rows"])
                    eda_data = pm.MutableData("eda_data", eda_train.T, dims=("observed_eda", "rows"))
                    pupil_data = pm.MutableData("pupil_data", pupil_train.T, dims=("observed_pupil", "rows"))
                    gaze_data = pm.MutableData("gaze_data", gaze_train.T, dims=("observed_gaze", "rows"))

                    W_eda = makeW(d_eda, k, ("observed_eda", "latent_columns"), 'W_eda')
                    W_hr = makeW(d_hr, k, ("observed_hr", "latent_columns"), 'W_hr')
                    W_pupil = makeW(d_pupil, k, ("observed_pupil", "latent_columns"), 'W_pupil')
                    W_gaze = makeW(d_gaze, k, ("observed_gaze", "latent_columns"), 'W_gaze')

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

                    psi_gaze = pm.HalfNormal("psi_gaze", 1.0)
                    X_gaze = pm.Normal("X_gaze", mu=at.dot(W_gaze, C), sigma=psi_gaze, observed=gaze_data,
                                       dims=['observed_gaze', 'rows'])

                    X_e = pm.Bernoulli("X_e", p=pm.math.sigmoid(at.dot(W_e, C)), dims=["observed_label", "rows"],
                                       observed=e_labels_train.T)

                with PPCA_identified:
                    approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
                    trace = approx.sample(1000)

                with PPCA_identified:
                    posterior_predictive = pm.sample_posterior_predictive(
                        trace, var_names=["X_e"], random_seed=123)

                e_pred_train = my_post_predict(trace, hr_train, eda_train, pupil_train, gaze_train)

                train_accuracy_exp = accuracy_score(e_labels_train, e_pred_train)

                # test
                e_pred_mode_test = my_post_predict(trace, hr_test, eda_test, pupil_test, gaze_test)
                test_accuracy_exp = accuracy_score(e_labels_test, e_pred_mode_test)
                row = [sub, k, i, train_accuracy_exp, test_accuracy_exp]

                with open('output/sppca_physio_gaze.csv', 'a') as f:
                    write = csv.writer(f)
                    write.writerow(row)


    from multiprocessing import Pool

    pool = Pool()

    n_processes = min(cpu_count(), len(all_subject))
    print('The computation will be parallelized in ', n_processes, ' processes')

    with Pool(n_processes) as p:
        [p.apply_async(run_sub, args=(sub)) for sub in all_subject]
