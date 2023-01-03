import logging
import os.path

import numpy as np
import pymc as pm
import aesara.tensor as at
import arviz as az
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import warnings
import extract_correct_csv

from deepemogp import feature_extractor
from deepemogp.signal import physio as physio
from deepemogp import datasets as datasets
from deepemogp.signal import behavior as behavior

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

scaler = StandardScaler()

valid_subject = extract_correct_csv.extract_only_valid_subject()

for i in valid_subject:
    string_subject = extract_correct_csv.read_correct_subject_csv(i)
    csv_ = 'data/LookAtMe_0'+ string_subject + '.csv'
    # csv_ = '/home/paolo/matteo/matteo/unimi/tesi_master/code/osfstorage-archive/behavior/LookAtMe_045.csv'
    global_data = pd.read_csv(csv_, sep='\t')
    y = np.array(list([int(d>2) for d in global_data['rating']]))
    e_labels = y[:, np.newaxis]  # rating > 2
    N_e = e_labels.shape[0]
    D_e = e_labels.shape[1]

    show = False
    # definition of the feature extractors to be used later
    f2 = feature_extractor.FE('wavelet', window=(2, 1))
    f3 = feature_extractor.FE('mean', window=(1, 0))

    # definition of the physiological signals to be extracted
    eda_ = physio.EDA(f3)
    hr_ = physio.HR(f3)
    pupil_ = behavior.PUPIL(f3)

    # extraction of the desired data from the dataset
    d = datasets.FEAR(signals={hr_,pupil_,eda_}, subjects={str(i)})

    for s in d.signals:
        # preprocess ...
        if s.name == 'EDA':
            s.preprocess(show=show,new_fps=500)
            s.feature_ext.extract_feat(s,show=show)
        else:
            if s.name == 'HR':
                list_hr_test = s.raw[0]['data']
                s.preprocess(show=show, useneurokit=True)
                s.feature_ext.extract_feat(s,show=show)

            else:
                s.feature_ext.extract_feat_without_preprocess(s, show=show)

        #add feature extraction for eda before preprocessing

        # ... and extract features from each signal type


    for sig in d.signals:
        if sig.name == 'EDA':
            eda_data = sig.features
        if sig.name == 'HR':
            hr_data = sig.features
        if sig.name == 'PUPIL':
            pupil_data = sig.features

    TRIAL = 160

    hr = np.array(hr_data)
    hr = hr.reshape((TRIAL, int(hr.shape[0]/TRIAL*hr.shape[1])))

    pupil = np.array(pupil_data)
    pupil = pupil.reshape((TRIAL, int(pupil.shape[0]/TRIAL*pupil.shape[1])))

    eda = np.array(eda_data)
    eda = eda.reshape((TRIAL,int(eda.shape[0]/TRIAL*eda.shape[1])))

    N_pupil = pupil.shape[0]
    D_pupil = pupil.shape[1]

    N_hr = hr.shape[0]
    D_hr = hr.shape[1]

    N_eda = eda.shape[0]
    D_eda = eda.shape[1]
    K = 3

    with pm.Model() as sPPCA:
        # dati osservabili
        hr_data = pm.MutableData("hr_data", hr.T)
        pupil_data = pm.MutableData("pupil_data", pupil.T)
        eda_data = pm.MutableData("eda_data", eda.T)

        e_data = pm.ConstantData("e_data", e_labels.T)

        # matrici pesi
        Whr = pm.Normal('Whr', mu=at.zeros([D_hr, K]), sigma=2.0 * at.ones([D_hr, K]), shape=[D_hr, K])
        Wpupil = pm.Normal('Wpupil', mu=at.zeros([D_pupil, K]), sigma=2.0 * at.ones([D_pupil, K]), shape=[D_pupil, K])

        Weda = pm.Normal('Weda', mu=at.zeros([D_eda, K]), sigma=2.0 * at.ones([D_eda, K]), shape=[D_eda, K])

        # weight matrix for pain expectation.
        # check mu,sigma,shape
        We = pm.Normal('W_e', mu=at.zeros([D_e, K]), sigma=2.0 * at.ones([D_e, K]), shape=[D_e, K])

        # latent space
        c = pm.Normal('c', mu=at.zeros([N_hr, K]), sigma=at.ones([N_hr, K]), shape=[N_hr, K])

        # dati dell'hrv interpretati come una gaussiana
        mu_hr = pm.Normal('mu_hr', Whr.dot(c.T), at.ones([D_hr, N_hr]))  # hyperprior 1
        sigma_hr = pm.Exponential('sigma_hr', at.ones([D_hr, N_hr]))  # hyperprior 2
        x_hr = pm.Normal('x_hr', mu=mu_hr, sigma=sigma_hr, shape=[D_hr, N_hr], observed=hr_data)

        # dati della dilatazione pupille interpretati come una gaussiana
        mu_pupil = pm.Normal('mu_pupil', Wpupil.dot(c.T), at.ones([D_pupil, N_pupil]))  # hyperprior 1
        sigma_pupil = pm.Exponential('sigma_pupil', at.ones([D_pupil, N_pupil]))  # hyperprior 2
        x_pupil = pm.Normal('x_pupil', mu=mu_pupil, sigma=sigma_pupil, shape=[D_pupil, N_pupil], observed=pupil_data)

        # eda
        mu_eda = pm.Normal('mu_eda', Weda.dot(c.T), at.ones([D_eda, N_eda]))  # hyperprior 1
        sigma_eda = pm.Exponential('sigma_eda', at.ones([D_eda, N_eda]))  # hyperprior 2
        x_eda = pm.Normal('x_eda', mu=mu_eda, sigma=sigma_eda, shape=[D_eda, N_eda], observed=eda_data)

        # pain expectation. ciò che dovremmo inferire dato c
        # due strade: binary o multiclass (1-4)
        # p = probability of success?
        x_e = pm.Bernoulli('x_e', p=pm.math.sigmoid(We.dot(c.T)), shape=[D_e, N_e], observed=e_data)

        # x_hr = pm.Bernoulli('x_hr', p=pm.math.sigmoid(Whr.dot(c.T)), shape=[D_hr, N_hr], observed=hr_data)
        # x_eda = pm.Bernoulli('x_eda', p=pm.math.sigmoid(Weda.dot(c.T)), shape=[D_eda, N_eda], observed=eda_data)

    gv = pm.model_to_graphviz(sPPCA)

    with sPPCA:
        approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
        trace = approx.sample(500)

    #az.plot_trace(trace);
    with sPPCA:
        # update values of predictors:
        # pm.set_data({"pupil_data": pupil, "hr_data": hr, "eda_data": eda})
        # use the updated values and predict outcomes and probabilities:
        posterior_predictive = pm.sample_posterior_predictive(
            trace, random_seed=123)
    e_pred = posterior_predictive.posterior_predictive["x_e"]
    e_pred_mode = np.squeeze(stats.mode(e_pred[0], keepdims=False)[0])[:,np.newaxis]

    train_accuracy_exp = accuracy_score(e_labels, e_pred_mode)

    logging.basicConfig(level=logging.INFO, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("Subject number: " + str(i) + " Train Accuracy Pain Expectation: " + str(train_accuracy_exp) + " script: " +
                 os.path.basename(__file__))




