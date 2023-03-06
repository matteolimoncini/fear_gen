import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from sklearn import preprocessing

import extract_correct_csv

# # Rational agent
# 
# The idea is to train n models where each model is trained using n trials.

valid_sub = extract_correct_csv.extract_only_valid_subject()

# read dataset
df = pd.read_csv('data/newLookAtMe/newLookAtMe02.csv')
df_rational = df[['morphing level', 'shock']]
df_rational['shock'] = df_rational['shock'].astype(int)  # setting shock as int instead of boolean
df_rational['morphing level'] = [int(d == 6) for d in df_rational['morphing level']]  # if morphing level==6 -> 1

# uniform prior
prior_counts = torch.ones((2, 2))


# model
def model(data):
    prior = pyro.sample("prior", dist.Dirichlet(prior_counts))
    total_counts = int(data.sum())
    pyro.sample("likelihood", dist.Multinomial(total_counts, prior), obs=data)


data_np = df_rational.to_numpy()

HABITUATION_TRIALS = 16
ACQUISITION_TRIALS = 48
data_all = data_np[16:]
learning_data = data_np[HABITUATION_TRIALS:ACQUISITION_TRIALS]  # remove only habituation

data = torch.tensor(data_all)
N = data.shape[0]

counter = torch.zeros((N, 4))

for i in range(len(data_all)):
    dict_ = {'[0 0]': 0, '[0 1]': 0, '[1 0]': 0, '[1 1]': 0}
    tmp_data = data_all[:i + 1]

    # count occurencies
    for x in tmp_data:
        dict_[str(x)] += 1
    values = np.array(list(dict_.values()))
    counter[i] = torch.tensor(values)

# reshape in order to match the Dirichlet distribution
counter = counter.reshape((len(data_all), 2, 2))

nuts_kernel = NUTS(model)
num_samples, warmup_steps = (300, 200)

mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=True)
all_means = []

# sampling
for i in range(len(counter)):
    mcmc.run(counter[i])
    hmc_samples = {k: v.detach().cpu().numpy()
                   for k, v in mcmc.get_samples().items()}
    means = hmc_samples['prior'].mean(axis=0)
    stds = hmc_samples['prior'].std(axis=0)
    print('observation: ', data_all[i])
    print('probabilities: ', means)
    all_means.append(means)

all_means = np.array(all_means)
# all means -> value of the rational agent

# all 144 trials
X = np.arange(HABITUATION_TRIALS, HABITUATION_TRIALS + len(all_means))
y = all_means[:, 1, 1]

# only simulated training trials
X_training = np.arange(HABITUATION_TRIALS, ACQUISITION_TRIALS)
y_training = all_means[:32, 1, 1]

# points with observation [1 1]
list_ = []
for index, i in enumerate(data_all):
    if np.equal(i, np.array([1, 1])).all():
        list_.append(np.array([X[index], y[index]]))
x_points, y_points = np.array(list_)[:, 0], np.array(list_)[:, 1]

total_array_simulated = []

# split between cs+ e non-cs+ visual stimuli
array_csplus_simulated = []
array_csminus_simulated = []

for index, data in enumerate(data_all):
    if data[0] == 1:
        array_csplus_simulated.append([X[index], all_means[index, 1, 1]])
        total_array_simulated.append(all_means[index, 1, 1])
    else:
        array_csminus_simulated.append([X[index], all_means[index, 0, 1]])
        total_array_simulated.append(all_means[index, 0, 1])

array_csplus_simulated = np.array(array_csplus_simulated)
array_csminus_simulated = np.array(array_csminus_simulated)
total_array_simulated = np.array(total_array_simulated)

np.save('output/pyro/complete_rational/csplus.npy', array_csplus_simulated)
np.save('output/pyro/complete_rational/csminus.npy', array_csminus_simulated)
np.save('output/pyro/complete_rational/total.npy', total_array_simulated)

# # rational agent sliding window

# The idea is to have a rational agent with a limited memory over previous trials.
# k = param sliding window dimension


import extract_correct_csv

valid_sub = extract_correct_csv.extract_only_valid_subject()

# read dataset
df = pd.read_csv('data/newLookAtMe/newLookAtMe02.csv')
df_rational = df[['morphing level', 'shock']]
df_rational['shock'] = df_rational['shock'].astype(int)  # setting shock as int instead of boolean
df_rational['morphing level'] = [int(d == 6) for d in df_rational['morphing level']]  # if morphing level==6 -> 1

data_np = df_rational.to_numpy()


def counter_window(data, k=0):
    N = data.shape[0]
    counter = torch.zeros((N, 4))
    for i in range(len(data)):
        dict_ = {'[0 0]': 0, '[0 1]': 0, '[1 0]': 0, '[1 1]': 0}
        if k == 0 or k > i:
            tmp_data = data[:i + 1]
        else:
            tmp_data = data[i - k:i + 1]
            # print('im here')
        # count occurencies
        for x in tmp_data:
            dict_[str(x)] += 1
        values = np.array(list(dict_.values()))
        counter[i] = torch.tensor(values)
    return counter


list_k = [2, 5, 10, 25, 50, 100, 150]

for k_window in list_k:

    counter = counter_window(data_np, k_window)
    counter = counter.reshape((len(data_np), 2, 2))

    # uniform prior
    prior_counts = torch.ones((2, 2))


    # model
    def model(data):
        prior = pyro.sample("prior", dist.Dirichlet(prior_counts))
        total_counts = int(data.sum())
        pyro.sample("likelihood", dist.Multinomial(total_counts, prior), obs=data)


    nuts_kernel = NUTS(model)
    num_samples, warmup_steps = (300, 200)

    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=True)
    all_means = []

    # sampling
    for i in range(len(counter)):
        mcmc.run(counter[i])
        hmc_samples = {k: v.detach().cpu().numpy()
                       for k, v in mcmc.get_samples().items()}
        means = hmc_samples['prior'].mean(axis=0)
        stds = hmc_samples['prior'].std(axis=0)
        print('observation: ', data_np[i])
        print('probabilities: ', means)
        all_means.append(means)

    # all trials

    total_array_simulated = []

    # split between cs+ e non-cs+ visual stimuli
    array_csplus_simulated = []
    array_csminus_simulated = []

    for index, data in enumerate(data_np):
        if data[0] == 1:
            array_csplus_simulated.append([X[index], all_means[index, 1, 1]])
            total_array_simulated.append(all_means[index, 1, 1])
        else:
            array_csminus_simulated.append([X[index], all_means[index, 0, 1]])
            total_array_simulated.append(all_means[index, 0, 1])

    array_csplus_simulated = np.array(array_csplus_simulated)
    array_csminus_simulated = np.array(array_csminus_simulated)
    total_array_simulated = np.array(total_array_simulated)
    np.save('output/pyro/sliding_wind/k' + str(k_window) + '_csplus.npy', array_csplus_simulated)
    np.save('output/pyro/sliding_wind/k' + str(k_window) + '_csminus.npy', array_csminus_simulated)
    np.save('output/pyro/sliding_wind/k' + str(k_window) + '_total.npy', total_array_simulated)
