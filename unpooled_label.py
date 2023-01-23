import logging
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az
import aesara.tensor as at
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import warnings
import extract_correct_csv
import random
import csv

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

scaler = StandardScaler()

all_subject = extract_correct_csv.extract_only_valid_subject()
valid_k_list = list(range(1, 10))
columns = ['subject', 'k', 'train', 'val', 'test']
with open('tmp_csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)

num_trials_to_remove = 48

logging.basicConfig(level=logging.INFO, filename="log/unpooled/unpooled_labelADVI1e6_randomsplit", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

for i in all_subject:

    for k in valid_k_list:
        string_subject = extract_correct_csv.read_correct_subject_csv(i)
        csv_ = 'data/LookAtMe_0' + string_subject + '.csv'
        global_data = pd.read_csv(csv_, sep='\t')
        y = np.array(list([int(d > 2) for d in global_data['rating']]))
        e_labels = y[:, np.newaxis]  # rating > 2
        e_labels = e_labels[num_trials_to_remove:]
        e_labels = pd.DataFrame(e_labels)

        TRIAL = 160

        hr = pd.read_csv('data/features/hr/' + str(i) + '.csv')
        hr = hr[num_trials_to_remove:]

        eda = pd.read_csv('data/features/eda/' + str(i) + '.csv')
        eda = eda[num_trials_to_remove:]

        pupil = pd.read_csv('data/features/pupil/' + str(i) + '.csv')
        pupil = pupil[num_trials_to_remove:]

        TRIAL_DEF = TRIAL - num_trials_to_remove

        TRAIN_PERC = 0.70
        VAL_PERC = 0.1
        TEST_PERC = 0.2  # 1-TRAIN_PERC+VAL_PERC
        N_train = int(len(pupil) * (TRAIN_PERC))
        N_val = int(len(pupil) * (VAL_PERC))

        # RANDOM SPLIT
        pupil = pupil.sample(frac=1, random_state=0)
        pupil = pupil.reset_index(drop=True)

        hr = hr.sample(frac=1, random_state=0)
        hr = hr.reset_index(drop=True)

        eda = eda.sample(frac=1, random_state=0)
        eda = eda.reset_index(drop=True)

        e_labels = e_labels.sample(frac=1, random_state=0)
        e_labels = e_labels.reset_index(drop=True)

        pupil_train = pupil[:N_train]
        hr_train = hr[:N_train]
        eda_train = eda[:N_train]
        e_labels_train = e_labels[:N_train]

        pupil_val = pupil[N_train:N_train + N_val]
        hr_val = hr[N_train:N_train + N_val]
        eda_val = eda[N_train:N_train + N_val]
        e_labels_val = e_labels[N_train:N_train + N_val]

        pupil_test = pupil[N_train+N_val:]
        hr_test = hr[N_train+N_val:]
        eda_test = eda[N_train+N_val:]
        e_test = e_labels[N_train+N_val:]

        N_pupil = pupil_train.shape[0]
        D_pupil = pupil_train.shape[1]

        N_hr = hr_train.shape[0]
        D_hr = hr_train.shape[1]

        N_eda = eda_train.shape[0]
        D_eda = eda_train.shape[1]

        N_e = e_labels_train.shape[0]
        D_e = e_labels_train.shape[1]

        K = k

        # print(N_pupil, D_pupil)
        # print(N_hr, D_hr)
        # print(N_eda, D_eda)
        # print(N_e, D_e)

        with pm.Model() as sPPCA:
            sPPCA.add_coord('physio_n', np.arange(N_hr), mutable=True)
            sPPCA.add_coord('physio_d', np.arange(D_hr), mutable=False)
            sPPCA.add_coord('e_label_d', np.arange(D_e), mutable=True)
            sPPCA.add_coord('K', np.arange(K), mutable=True)
            sPPCA.add_coord('pupil_d', np.arange(D_pupil), mutable=True)

            # dati osservabili
            hr_data = pm.MutableData("hr_data", hr_train.T, dims=['physio_d', 'physio_n'])
            pupil_data = pm.MutableData("pupil_data", pupil_train.T, dims=['pupil_d', 'physio_n'])
            eda_data = pm.MutableData("eda_data", eda_train.T, dims=['physio_d', 'physio_n'])

            # matrici pesi
            Whr = pm.Normal('Whr', mu=0, sigma=2.0 * 1, dims=['physio_d', 'K'])
            Wpupil = pm.Normal('Wpupil', mu=0, sigma=2.0 * 1, dims=['pupil_d', 'K'])

            Weda = pm.Normal('Weda', mu=0, sigma=2.0 * 1, dims=['physio_d', 'K'])

            # weight matrix for pain expectation.
            # check mu,sigma,shape
            We = pm.Normal('W_e', mu=0, sigma=2.0 * 1, dims=['e_label_d', 'K'])

            # latent space
            c = pm.Normal('c', mu=0, sigma=1, dims=['K', 'physio_n'])

            # dati dell'hrv interpretati come una gaussiana
            mu_hr = pm.Normal('mu_hr', at.dot(Whr, c), 1, dims=['physio_d', 'physio_n'])  # hyperprior 1
            sigma_hr = pm.Exponential('sigma_hr', 1)  # hyperprior 2
            x_hr = pm.Normal('x_hr', mu=mu_hr, sigma=sigma_hr, observed=hr_data, dims=['physio_d', 'physio_n'])

            # dati della dilatazione pupille interpretati come una gaussiana
            mu_pupil = pm.Normal('mu_pupil', at.dot(Wpupil, c), 1, dims=['pupil_d', 'physio_n'])  # hyperprior 1
            sigma_pupil = pm.Exponential('sigma_pupil', 1)  # hyperprior 2
            x_pupil = pm.Normal('x_pupil', mu=mu_pupil, sigma=sigma_pupil, dims=['pupil_d', 'physio_n'],
                                observed=pupil_data)

            # eda
            mu_eda = pm.Normal('mu_eda', at.dot(Weda, c), 1, dims=['physio_d', 'physio_n'])  # hyperprior 1
            sigma_eda = pm.Exponential('sigma_eda', 1)  # hyperprior 2
            x_eda = pm.Normal('x_eda', mu=mu_eda, sigma=sigma_eda, dims=['physio_d', 'physio_n'], observed=eda_data)

            # pain expectation. ciò che dovremmo inferire dato c
            # due strade: binary o multiclass (1-4)
            # p = probability of success?
            x_e = pm.Bernoulli('x_e', p=pm.math.sigmoid(at.dot(We, c)), dims=['e_label_d', 'physio_n'],
                               observed=e_labels_train.T)

        name = 'unpooled/advi/randomsplit/k' + str(k) + '_sub' + str(i) + '_'
        trace_file = name + 'trace.nc'
        '''if os.path.exists(trace_file):
            print("loading trace...")
            with sPPCA:
                pm.load_trace(trace_file)
        else:'''
        with sPPCA:
            approx = pm.fit(1000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
            trace = approx.sample(500)

        with sPPCA:
            posterior_predictive = pm.sample_posterior_predictive(
                trace, var_names=["x_e"], random_seed=123)

        e_pred_train = posterior_predictive.posterior_predictive['x_e']
        e_pred_mode_train = np.squeeze(stats.mode(e_pred_train[0], keepdims=False)[0])[:, np.newaxis]

        train_accuracy_exp = accuracy_score(e_labels_train, e_pred_mode_train)

        trace.to_netcdf(trace_file)
        np.save(name + 'approx_hist.npy', approx.hist)

        plt.plot(approx.hist)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        plt.savefig(name + 'elboplot.png')
        plt.show()

        plt.plot(approx.hist)
        plt.ylim(0, 1e5)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        plt.savefig(name + 'elboplot_cutted.png')
        plt.show()


        # az.plot_trace(trace);
        with sPPCA:
            # update values of predictors with validation:
            sPPCA.set_data("hr_data", hr_val.T, coords={'physio_n': range(hr_val.shape[0])})
            sPPCA.set_data("pupil_data", pupil_val.T, coords={'physio_n': range(pupil_val.shape[0])})
            sPPCA.set_data("eda_data", eda_val.T, coords={'physio_n': range(eda_val.shape[0])})
            # use the updated values and predict outcomes and probabilities:

            posterior_predictive = pm.sample_posterior_predictive(
                trace, var_names=["x_e"], random_seed=123, predictions=True)

        e_pred = posterior_predictive.predictions['x_e']
        e_pred_mode = np.squeeze(stats.mode(e_pred[0], keepdims=False)[0])[:, np.newaxis]

        val_accuracy_exp = accuracy_score(e_labels_val, e_pred_mode)

        with sPPCA:
            # update values of predictors with validation:
            sPPCA.set_data("hr_data", hr_test.T, coords={'physio_n': range(hr_test.shape[0])})
            sPPCA.set_data("pupil_data", pupil_test.T, coords={'physio_n': range(pupil_test.shape[0])})
            sPPCA.set_data("eda_data", eda_test.T, coords={'physio_n': range(eda_test.shape[0])})
            # use the updated values and predict outcomes and probabilities:

            posterior_predictive = pm.sample_posterior_predictive(
                trace, var_names=["x_e"], random_seed=123, predictions=True)

        e_pred = posterior_predictive.predictions['x_e']
        e_pred_mode = np.squeeze(stats.mode(e_pred[0], keepdims=False)[0])[:, np.newaxis]

        test_accuracy_exp = accuracy_score(e_test, e_pred_mode)
        row = [i, k, train_accuracy_exp, val_accuracy_exp, test_accuracy_exp]

        with open('tmp_csv', 'a') as f:
            write = csv.writer(f)
            write.writerow(row)
