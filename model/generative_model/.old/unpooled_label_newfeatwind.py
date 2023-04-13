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

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

scaler = StandardScaler()

prova_3_subj = extract_correct_csv.extract_only_valid_subject()
valid_k_list = list(range(1, 10))

num_trials_to_remove = 48

logging.basicConfig(level=logging.INFO, filename="log/unpooled/newfeatwind/unpooled_labelADVI1e6", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

for k in valid_k_list:

    for i in prova_3_subj:
        string_subject = extract_correct_csv.read_correct_subject_csv(i)
        csv_ = 'data/LookAtMe_0' + string_subject + '.csv'
        global_data = pd.read_csv(csv_, sep='\t')
        y = np.array(list([int(d > 2) for d in global_data['rating']]))
        e_labels = y[:, np.newaxis]  # rating > 2
        e_labels = e_labels[num_trials_to_remove:]

        TRIAL = 160

        hr = pd.read_csv('data/features_4_2/hr/' + str(i) + '.csv')
        hr = hr[num_trials_to_remove:]

        eda = pd.read_csv('data/features_4_2/eda/' + str(i) + '.csv')
        eda = eda[num_trials_to_remove:]

        pupil = pd.read_csv('data/features_4_2/pupil/' + str(i) + '.csv')
        pupil = pupil[num_trials_to_remove:]

        TRIAL_DEF = TRIAL - num_trials_to_remove

        # TRAIN_PERC = 0.75
        TEST_PERC = 0.25  # 1-TRAIN_PERC
        N_train = int(len(pupil) * (1 - TEST_PERC))

        pupil_train = pupil[:N_train]
        hr_train = hr[:N_train]
        eda_train = eda[:N_train]
        e_labels_train = e_labels[:N_train]

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

            # e_data = pm.MutableData("e_data", e_labels_train.T)

            # matrici pesi
            Whr = pm.Normal('Whr', mu=0, sigma=2.0 * 1, dims=['physio_d', 'K'])
            Wpupil = pm.Normal('Wpupil', mu=0, sigma=2.0 * 1, dims=['pupil_d', 'K'])

            Weda = pm.Normal('Weda', mu=0, sigma=2.0 * 1, dims=['physio_d', 'K'])

            # weight matrix for pain expectation.
            # check mu,sigma,shape
            We = pm.Normal('W_e', mu=0, sigma=2.0 * 1, dims=['e_label_d', 'K'])

            # latent space
            c = pm.Normal('c', mu=0, sigma=1, dims=['physio_n', 'K'])

            # dati dell'hrv interpretati come una gaussiana
            mu_hr = pm.Normal('mu_hr', Whr.dot(c.T), 1, dims=['physio_d', 'physio_n'])  # hyperprior 1
            sigma_hr = pm.Exponential('sigma_hr', 1)  # hyperprior 2
            x_hr = pm.Normal('x_hr', mu=mu_hr, sigma=sigma_hr, observed=hr_data, dims=['physio_d', 'physio_n'])

            # dati della dilatazione pupille interpretati come una gaussiana
            mu_pupil = pm.Normal('mu_pupil', Wpupil.dot(c.T), 1, dims=['pupil_d', 'physio_n'])  # hyperprior 1
            sigma_pupil = pm.Exponential('sigma_pupil', 1)  # hyperprior 2
            x_pupil = pm.Normal('x_pupil', mu=mu_pupil, sigma=sigma_pupil, dims=['pupil_d', 'physio_n'],
                                observed=pupil_data)

            # eda
            mu_eda = pm.Normal('mu_eda', Weda.dot(c.T), 1, dims=['physio_d', 'physio_n'])  # hyperprior 1
            sigma_eda = pm.Exponential('sigma_eda', 1)  # hyperprior 2
            x_eda = pm.Normal('x_eda', mu=mu_eda, sigma=sigma_eda, dims=['physio_d', 'physio_n'], observed=eda_data)

            # pain expectation. ciò che dovremmo inferire dato c
            # due strade: binary o multiclass (1-4)
            # p = probability of success?
            x_e = pm.Bernoulli('x_e', p=pm.math.sigmoid(We.dot(c.T)), dims=['e_label_d', 'physio_n'],
                               observed=e_labels_train.T)

        with sPPCA:
            approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
            trace = approx.sample(500)

        name = 'unpooled/newfeatwind/k' + str(k) + '_sub' + str(i) + '_'

        trace.to_netcdf(name + 'trace.nc')
        np.save(name + 'approx_hist.npy', approx.hist)

        '''plt.plot(approx.hist)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        plt.savefig(name + 'elboplot.png')

        plt.plot(approx.hist)
        plt.ylim(0, 1e5)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        plt.savefig(name + 'elboplot_cutted.png')'''

        # from xarray import open_dataset

        # posterior = open_dataset('posterior.h5', engine='scipy')

        pupil_test = pupil[N_train:].reset_index().drop(columns=['index'])
        hr_test = hr[N_train:].reset_index().drop(columns=['index'])
        eda_test = eda[N_train:].reset_index().drop(columns=['index'])
        e_test = e_labels[N_train:]

        with sPPCA:
            posterior_pred = pm.sample_posterior_predictive(
                trace, var_names=["x_e"], random_seed=123)

        # az.plot_trace(trace);
        with sPPCA:
            # update values of predictors:
            sPPCA.set_data("hr_data", hr_test.T, coords={'physio_n': range(hr_test.shape[0])})
            sPPCA.set_data("pupil_data", pupil_test.T, coords={'physio_n': range(pupil_test.shape[0])})
            sPPCA.set_data("eda_data", eda_test.T, coords={'physio_n': range(eda_test.shape[0])})
            # use the updated values and predict outcomes and probabilities:

            posterior_predictive = pm.sample_posterior_predictive(
                trace, var_names=["x_e"], random_seed=123, predictions=True)

        e_pred = posterior_predictive.predictions['x_e']
        e_pred_mode = np.squeeze(stats.mode(e_pred[0], keepdims=False)[0])[:, np.newaxis]

        test_accuracy_exp = accuracy_score(e_test, e_pred_mode)

        logging.info("Subj num: " + str(i) + " Test Acc Pain Expect: " + str(test_accuracy_exp) + " script: " +
                     os.path.basename(__file__) + ", ft extr HR and EDA: wavelet" +
                     ', ft extr PUP: mean, lat space dims: ' + str(K))