# -*- coding: utf-8 -*-
"""test_model_temporal.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Rwk0EV8Af2Pa-aKj7FqaKAfE5eMWUpiG
"""

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import aesara.tensor as at
import arviz as az
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from scipy import stats
import scipy
import warnings

from deepemogp import feature_extractor
from deepemogp.signal import physio as physio
from deepemogp import datasets as datasets
from deepemogp.signal import behavior as behavior
import extract_correct_csv

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
scaler = StandardScaler()

import os
import logging

logging.basicConfig(level=logging.INFO, filename="logfile_temporal", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

prova_3_subj = extract_correct_csv.extract_only_valid_subject()

for i in prova_3_subj:

    subj_ = i

    TRIAL = 160
    num_trials_to_remove = 48
    K = 5

    hr = pd.read_csv('data/features/hr/' + str(i) + '.csv')
    hr = hr[num_trials_to_remove:]

    eda = pd.read_csv('data/features/eda/' + str(i) + '.csv')
    eda = eda[num_trials_to_remove:]

    pupil = pd.read_csv('data/features/pupil/' + str(i) + '.csv')
    pupil = pupil[num_trials_to_remove:]

    csv_ = 'data/LookAtMe_0' + str(subj_) + '.csv'
    global_data = pd.read_csv(csv_, sep='\t')
    y = np.array(list([int(d > 2) for d in global_data['rating']]))
    e_labels = y[:, np.newaxis]  # rating > 2
    e_labels = e_labels[num_trials_to_remove:]

    N_pupil = pupil.shape[0]
    D_pupil = pupil.shape[1]

    N_hr = hr.shape[0]
    D_hr = hr.shape[1]

    N_eda = eda.shape[0]
    D_eda = eda.shape[1]

    N_e = e_labels.shape[0]
    D_e = e_labels.shape[1]

    index = np.arange(60)

    # coords_ = {'subject': global_subject_df.subject.unique(), 'tot_trial':np.arange(N_hr),'time':index, 'hr_': hr_trial, 'eda_':eda_trial}
    coords_ = {'time': index}
    with pm.Model(coords=coords_) as rolling:
        # dati osservabili
        hr_data = pm.MutableData("hr_data", hr.T)
        eda_data = pm.MutableData("eda_data", eda.T)

        # matrici pesi
        # Whr = pm.Normal('Whr', mu=at.zeros([D_hr, K]), sigma=2.0 * at.ones([D_hr, K]), shape=[D_hr, K])
        # sigma_whr= pm.Exponential('sigma_whr',50.0)
        Whr = pm.GaussianRandomWalk('Whr', sigma=1, init_dist=pm.Normal.dist(0, 10), shape=[D_hr, K])

        # sigma_weda = pm.Exponential('sigma_weda',50.0)
        Weda = pm.GaussianRandomWalk('Weda', sigma=1, init_dist=pm.Normal.dist(0, 10), shape=[D_eda, K])

        # latent space
        c = pm.Normal('c', mu=at.zeros([N_hr, K]), sigma=at.ones([N_hr, K]), shape=[N_hr, K])

        # dati dell'hrv interpretati come una gaussiana
        mu_hr = pm.Normal('mu_hr', Whr.dot(c.T), at.ones([D_hr, N_hr]))  # ,shape=[N_hr,D_hr]
        sigma_hr = pm.Exponential('sigma_hr', at.ones([D_hr, N_hr]))  # hyperprior 2
        x_hr = pm.Normal('x_hr', mu=mu_hr, sigma=sigma_hr, shape=[D_hr, N_hr], observed=hr_data)

        # eda
        mu_eda = pm.Normal('mu_eda', Weda.dot(c.T), at.ones([D_eda, N_eda]))  # ,shape=[N_eda,D_eda]
        sigma_eda = pm.Exponential('sigma_eda', at.ones([D_eda, N_eda]))  # hyperprior 2
        x_eda = pm.Normal('x_eda', mu=mu_eda, sigma=sigma_eda, shape=[D_eda, N_eda], observed=eda_data)

    with rolling:
        approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
        trace = approx.sample(500)

    with rolling:
        posterior_predictive = pm.sample_posterior_predictive(
            trace, random_seed=123)

    eda_pred = posterior_predictive.posterior_predictive['x_eda']

    eda_pred = np.squeeze(eda_pred.mean('draw', keepdims='false')[0]).to_numpy()

    edapred_ = eda_pred.T

    eda_ = eda.to_numpy()

    corrlist = []
    for i in range(112):
        res = np.corrcoef(eda_[i], edapred_[i])[0][1]
        corrlist.append(res)
        # print('trial '+str(i)+ ' corr: '+str(res.round(3)))

    mean_subj = np.mean(corrlist)

    logging.info("Mean corr coeff eda-hr using subj: " + str(i) + " " + str(mean_subj) + " script: " +
                 os.path.basename(__file__) + "latent space dims: " + str(K))
