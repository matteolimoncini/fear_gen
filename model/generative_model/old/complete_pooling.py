import logging
import os.path

import numpy as np
import pymc as pm
import aesara.tensor as at
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import scipy
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

prova_3_subj = extract_correct_csv.extract_only_valid_subject()
valid_k_list = list(range(1, 10))

k = 5
i = 2
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
    subject = subject[num_trials_to_remove:]
    global_subject = global_subject + subject.tolist()

# e_labels = e_labels[num_trials_to_remove:]
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

NUM_TRIAL = 160
TRIAL = NUM_TRIAL * len(prova_3_subj)


def populate_array(x, name):
    return name[NUM_TRIAL * (x - 1) + num_trials_to_remove:NUM_TRIAL * x]


hr_temp = np.concatenate([pd.read_csv('data/features/hr/' + str(x) + '.csv') for x in prova_3_subj])
hr = np.concatenate([populate_array(x, hr_temp) for x in range(1, len(prova_3_subj) + 1)])

pupil_temp = np.concatenate([pd.read_csv('data/features/pupil/' + str(x) + '.csv') for x in prova_3_subj])
pupil = np.concatenate([populate_array(x, pupil_temp) for x in range(1, len(prova_3_subj) + 1)])

eda_temp = np.concatenate([pd.read_csv('data/features/eda/' + str(x) + '.csv') for x in prova_3_subj])
eda = np.concatenate([populate_array(x, eda_temp) for x in range(1, len(prova_3_subj) + 1)])

N_pupil = pupil.shape[0]
D_pupil = pupil.shape[1]

N_hr = hr.shape[0]
D_hr = hr.shape[1]

N_eda = eda.shape[0]
D_eda = eda.shape[1]
K = k

# print(N_hr,D_hr)
# print(N_eda,D_eda)
# print(N_pupil,D_pupil)
# print(K)

with pm.Model() as sPPCA:
    # dati osservabili
    hr_data = pm.MutableData("hr_data", hr.T)
    pupil_data = pm.MutableData("pupil_data", pupil.T)
    eda_data = pm.MutableData("eda_data", eda.T)

    e_data = pm.ConstantData("e_data", global_e_labels.T)

    # matrici pesi
    Whr = pm.Normal('Whr', mu=at.zeros([D_hr, K]), sigma=2.0 * at.ones([D_hr, K]), shape=[D_hr, K])
    Wpupil = pm.Normal('Wpupil', mu=at.zeros([D_pupil, K]), sigma=2.0 * at.ones([D_pupil, K]),
                       shape=[D_pupil, K])

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
    x_pupil = pm.Normal('x_pupil', mu=mu_pupil, sigma=sigma_pupil, shape=[D_pupil, N_pupil],
                        observed=pupil_data)

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

# gv = pm.model_to_graphviz(sPPCA)
# gv.view('complete_pooling')

with sPPCA:
    approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
    trace = approx.sample(500)

# ax.set_yticks(np.arange(nvar));
# ax.set_yticklabels(max_rhat.coords["variable"].to_numpy()[::-1]);

with sPPCA:
    # update values of predictors:
    # pm.set_data({"pupil_data": pupil, "hr_data": hr, "eda_data": eda})
    # use the updated values and predict outcomes and probabilities:
    posterior_predictive = pm.sample_posterior_predictive(
        trace, random_seed=123)

e_pred = posterior_predictive.posterior_predictive["x_e"]
e_pred_mode = np.squeeze(stats.mode(e_pred[0], keepdims=False)[0])[:, np.newaxis]

train_accuracy_exp = accuracy_score(global_e_labels, e_pred_mode)

logging.basicConfig(level=logging.INFO, filename="logfile_label_complete_pooling", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info("Subj num: " + str(i) + " Train Accuracy Pain Expect: " + str(train_accuracy_exp) + " script: " +
             os.path.basename(__file__) + ", feat extract HR and EDA: wavelet" +
             ', feat extract PUPIL: mean, latent space dims: ' + str(K))
