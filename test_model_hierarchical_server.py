import logging
import os

import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import aesara.tensor as at
import arviz as az
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import extract_correct_csv
from scipy import stats
import pandas as pd
import warnings

from deepemogp import feature_extractor
from deepemogp.signal import physio as physio
from deepemogp import datasets as datasets
from deepemogp.signal import behavior as behavior

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

scaler = StandardScaler()

prova_3_subj = extract_correct_csv.extract_only_valid_subject()

global_e_labels = []
global_subject = []
num_trials_to_remove = 48

for i in prova_3_subj:
    subj_ = extract_correct_csv.read_correct_subject_csv(i)
    csv_ = 'data/LookAtMe_0' + str(subj_) + '.csv'
    global_data = pd.read_csv(csv_, sep='\t')
    y = np.array(list([int(d > 2) for d in global_data['rating']]))
    e_labels = y[:, np.newaxis]  # rating > 2
    e_labels = e_labels[num_trials_to_remove:]
    global_e_labels = global_e_labels + e_labels.tolist()
    subject = np.array(list([s for s in global_data['subject']]))[:, np.newaxis]
    global_subject = global_subject + subject.tolist()
#e_labels = e_labels[num_trials_to_remove:]
'''N_e = e_labels.shape[0]
D_e = e_labels.shape[1]'''
global_e_labels = np.array(global_e_labels)
global_subject = np.array(global_subject)

N_e = global_e_labels.shape[0]
D_e = global_e_labels.shape[1]

N_sub = global_subject.shape[0]
D_sub = global_subject.shape[1]

global_subject_df = pd.DataFrame(global_subject, columns=['subject'])

subject_dict = dict(zip(global_subject_df.subject.unique(), range(len(prova_3_subj))))
subj_def = global_subject_df.replace(subject_dict).values

show = False
# definition of the feature extractors to be used later
f2 = feature_extractor.FE('wavelet', window=(2, 1))
f3 = feature_extractor.FE('mean', window=(1, 0))

# definition of the physiological signals to be extracted
eda_ = physio.EDA(f2)
hr_ = physio.HR(f2)
pupil_ = behavior.PUPIL(f3)

# extraction of the desired data from the dataset
elenco_subj = ([str(d) for d in prova_3_subj])
d = datasets.FEAR(signals={hr_, pupil_, eda_}, subjects=set(elenco_subj))

for s in d.signals:
    # preprocess ...
    if s.name == 'EDA':
        s.preprocess(show=show, new_fps=500)
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
    if sig.name=='EDA':
        eda_data = sig.features
    if sig.name=='HR':
        hr_data = sig.features
    if sig.name=='PUPIL':
        pupil_data = sig.features

NUM_TRIAL = 160
TRIAL = NUM_TRIAL*len(prova_3_subj)

def populate_array(x, name):
    return name[NUM_TRIAL*(x-1)+num_trials_to_remove:NUM_TRIAL*x]


hr_temp = np.array(hr_data)
hr_temp = hr_temp.reshape((TRIAL, int(hr_temp.shape[0]/TRIAL*hr_temp.shape[1])))
hr = np.concatenate([populate_array(x, hr_temp) for x in range(1, len(prova_3_subj)+1)])

pupil_temp = np.array(pupil_data)
pupil_temp = pupil_temp.reshape((TRIAL, int(pupil_temp.shape[0]/TRIAL*pupil_temp.shape[1])))
pupil = np.concatenate([populate_array(x, pupil_temp) for x in range(1, len(prova_3_subj)+1)])

eda_temp = np.array(eda_data)
eda_temp = eda_temp.reshape((TRIAL,int(eda_temp.shape[0]/TRIAL*eda_temp.shape[1])))
eda = np.concatenate([populate_array(x, eda_temp) for x in range(1, len(prova_3_subj)+1)])


N_pupil = pupil.shape[0]
D_pupil = pupil.shape[1]

N_hr = hr.shape[0]
D_hr = hr.shape[1]

N_eda = eda.shape[0]
D_eda = eda.shape[1]
K = 3

coords = {'subject': global_subject_df.subject.unique(), 'tot_trial':np.arange(N_hr)}

with pm.Model(coords=coords) as sPPCA:
    #dati osservabili
    hr_data = pm.MutableData("hr_data", hr)
    pupil_data = pm.MutableData("pupil_data", pupil)
    eda_data = pm.MutableData("eda_data", eda)

    e_data = pm.ConstantData("e_data", global_e_labels)

    #matrici pesi
    mu_hr = pm.Normal('mu_hr',at.zeros([D_hr, K]), 10) # hyperprior 1
    sigma_hr = pm.Exponential('sigma_hr', 2*at.ones([D_hr, K]))# hyperprior 2
    Whr = pm.Normal('Whr', mu=mu_hr, sigma=sigma_hr)

    mu_pupil = pm.Normal('mu_pupil',at.zeros([D_pupil, K]), 10) # hyperprior 1
    sigma_pupil = pm.Exponential('sigma_pupil', 2*at.ones([D_pupil, K]))# hyperprior 2
    Wpupil = pm.Normal('Wpupil', mu=mu_pupil, sigma=sigma_pupil)

    mu_eda = pm.Normal('mu_eda',at.zeros([D_eda, K]), 10) # hyperprior 1
    sigma_eda = pm.Exponential('sigma_eda', 2*at.ones([D_eda, K]))# hyperprior 2
    Weda = pm.Normal('Weda', mu=mu_eda, sigma=sigma_eda)

    #weight matrix for pain expectation.
    #check mu,sigma,shape
    mu_e = pm.Normal('mu_e',at.zeros([D_e, K]), 10) # hyperprior 1
    sigma_e = pm.Exponential('sigma_e', 2*at.ones([D_e, K]))# hyperprior 2
    We = pm.Normal('W_e', mu=mu_e, sigma=sigma_e, shape=[D_e, K])

    #latent space
    c = pm.Normal('c', mu=at.zeros([N_hr, K]), sigma=at.ones([N_hr,K]))
    #subject_idx = 2
    subject_idx = pm.MutableData("subject_idx", np.squeeze(subj_def))

    # dati dell'hrv interpretati come una gaussiana
    a_subjects_hr= pm.Normal("a_subjects_hr",mu=c.dot(Whr.T), sigma=at.ones([N_hr,D_hr]))
    theta_hr = a_subjects_hr[subject_idx]
    sigma2_hr = pm.Exponential("sigma2_hr",1.0)
    x_hr = pm.Normal('x_hr', mu=theta_hr, sigma=sigma2_hr, observed=hr_data)


    # dati della dilatazione pupille interpretati come una gaussiana
    a_subjects_pupil= pm.Normal("a_subjects_pupil",mu=c.dot(Wpupil.T), sigma=at.ones([N_pupil,D_pupil]))
    theta_pupil = a_subjects_pupil[subject_idx]
    sigma2_pupil = pm.Exponential("sigma2_pupil",1.0)
    x_pupil = pm.Normal('x_pupil', mu=theta_pupil, sigma=sigma2_pupil, observed=pupil_data)

    #eda
    a_subjects_eda = pm.Normal("a_subjects_eda",mu=c.dot(Weda.T), sigma=at.ones([N_eda,D_eda]))
    theta_eda = a_subjects_eda[subject_idx]
    sigma2_eda = pm.Exponential("sigma2_eda",1.0)
    x_eda = pm.Normal('x_eda', mu=theta_eda, sigma=sigma2_eda, observed=eda_data)

    # pain expectation. ciò che dovremmo inferire dato c
    # due strade: binary o multiclass (1-4)
    # p = probability of success?
    x_e = pm.Bernoulli('x_e' , p=pm.math.sigmoid(c.dot(We.T)), shape=[N_e,D_e], observed=e_data)


with sPPCA:
    approx = pm.fit(10000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
    trace = approx.sample(500)

with sPPCA:
    # update values of predictors:
    # pm.set_data({"hr_data":hr})
    # use the updated values and predict outcomes and probabilities:
    posterior_predictive = pm.sample_posterior_predictive(
        trace, var_names=['x_e'], random_seed=123)

e_pred = posterior_predictive.posterior_predictive["x_e"]

e_pred_mode = np.squeeze(stats.mode(e_pred[0], keepdims=False)[0])[:,np.newaxis]

train_accuracy_exp = accuracy_score(global_e_labels, e_pred_mode)
logging.basicConfig(level=logging.INFO, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info("Train Accuracy Pain Expectation using all valid subjects: " + str(train_accuracy_exp) + " script: " +
             os.path.basename(__file__) + "latent space dimension: " + str(K))
