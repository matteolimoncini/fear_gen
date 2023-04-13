import logging
import os.path
import numpy as np
import pymc as pm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import warnings
import extract_correct_csv

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

scaler = StandardScaler()

subjects = extract_correct_csv.extract_only_valid_subject()
valid_k_list = list(range(1, 10, 2))

logging.basicConfig(level=logging.INFO, filename="log/complete_pooled/pooled_label_split_subject", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
TEST_PERC = 0.25  # 1-TRAIN_PERC
subjects_test = subjects[-int(len(subjects) * TEST_PERC):]
subjects_train = subjects[:-int(len(subjects) * TEST_PERC)]

num_trials_to_remove = 48
TRIAL = (160 - num_trials_to_remove) * len(subjects)

global_e_labels_train = []
for i in subjects_train:
    subj_ = extract_correct_csv.read_correct_subject_csv(i)
    csv_ = 'data/LookAtMe_0' + str(subj_) + '.csv'
    global_data = pd.read_csv(csv_, sep='\t')
    y = np.array(list([int(d > 2) for d in global_data['rating']]))
    e_labels = y[:, np.newaxis]  # rating > 2
    e_labels = e_labels[num_trials_to_remove:]
    global_e_labels_train = global_e_labels_train + e_labels.tolist()

global_e_labels_test = []
for i in subjects_test:
    subj_ = extract_correct_csv.read_correct_subject_csv(i)
    csv_ = 'data/LookAtMe_0' + str(subj_) + '.csv'
    global_data = pd.read_csv(csv_, sep='\t')
    y = np.array(list([int(d > 2) for d in global_data['rating']]))
    e_labels = y[:, np.newaxis]  # rating > 2
    e_labels = e_labels[num_trials_to_remove:]
    global_e_labels_test = global_e_labels_test + e_labels.tolist()

global_e_labels_train = np.array(global_e_labels_train)
global_e_labels_test = np.array(global_e_labels_test)

N_e_train = global_e_labels_train.shape[0]
D_e_train = global_e_labels_train.shape[1]

NUM_TRIAL = 160
TRIAL = NUM_TRIAL * len(subjects_train)


def populate_array(x, name):
    return name[NUM_TRIAL * (x - 1) + num_trials_to_remove:NUM_TRIAL * x]


# train
hr_temp = np.concatenate([pd.read_csv('data/features/hr/' + str(x) + '.csv') for x in subjects_train])
hr_train = np.concatenate([populate_array(x, hr_temp) for x in range(1, len(subjects_train) + 1)])

pupil_temp = np.concatenate([pd.read_csv('data/features/pupil/' + str(x) + '.csv') for x in subjects_train])
pupil_train = np.concatenate([populate_array(x, pupil_temp) for x in range(1, len(subjects_train) + 1)])

eda_temp = np.concatenate([pd.read_csv('data/features/eda/' + str(x) + '.csv') for x in subjects_train])
eda_train = np.concatenate([populate_array(x, eda_temp) for x in range(1, len(subjects_train) + 1)])

# test
hr_temp = np.concatenate([pd.read_csv('data/features/hr/' + str(x) + '.csv') for x in subjects_test])
hr_test = np.concatenate([populate_array(x, hr_temp) for x in range(1, len(subjects_test) + 1)])

pupil_temp = np.concatenate([pd.read_csv('data/features/pupil/' + str(x) + '.csv') for x in subjects_train])
pupil_test = np.concatenate([populate_array(x, pupil_temp) for x in range(1, len(subjects_test) + 1)])

eda_temp = np.concatenate([pd.read_csv('data/features/eda/' + str(x) + '.csv') for x in subjects_test])
eda_test = np.concatenate([populate_array(x, eda_temp) for x in range(1, len(subjects_test) + 1)])

FINAL_TRIAL = NUM_TRIAL - num_trials_to_remove

N_train = FINAL_TRIAL * len(subjects_train)
N_test = FINAL_TRIAL * len(subjects_test)

D_features_physio = eda_train.shape[1]
D_e = global_e_labels_train.shape[1]
D_pupil = pupil_train.shape[1]

for k in valid_k_list:
    with pm.Model() as sPPCA:
        sPPCA.add_coord('physio_n', np.arange(N_train), mutable=True)
        sPPCA.add_coord('physio_d', np.arange(D_features_physio), mutable=False)
        sPPCA.add_coord('e_label_d', np.arange(D_e), mutable=True)
        sPPCA.add_coord('K', np.arange(k), mutable=True)
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
                           observed=global_e_labels_train.T)
    with sPPCA:
        approx = pm.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
        trace = approx.sample(500)
    name = 'complete_pooled_split_subjects/ADVI_100000_k' + str(k) + '_allsubjects_'
    trace.to_netcdf(name + 'trace.nc')

    with sPPCA:
        posterior_pred = pm.sample_posterior_predictive(
            trace, var_names=["x_e"], random_seed=123)
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

    test_accuracy_exp = accuracy_score(global_e_labels_test, e_pred_mode)
    logging.info("ADVI_100000_split_subjects Test Acc Pain Expect: " + str(test_accuracy_exp) + " script: " +
                 os.path.basename(__file__) + ", ft extr HR e EDA: wav" +
                 ', ft extr PUP: mean, lat space dims: ' + str(k))
