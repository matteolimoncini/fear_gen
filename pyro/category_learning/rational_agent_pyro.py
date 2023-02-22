import logging

import pandas as pd
import numpy as np
import torch
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
import pyro.distributions as dist

logging.basicConfig(format='%(message)s', level=logging.INFO)
# read dataset
df_ = pd.read_csv('fear_gen/data/newLookAtMe01.csv')
print(df_)
df_ = df_[['morphing level', 'shock']]
df_['shock'] = df_['shock'].astype(int)
df_['morphing level'] = [int(d==6) for d in df_['morphing level']]

# categorical/multinomial distribution
# LDA
prior_counts = torch.ones((2,2))
def model(data):
    prior = pyro.sample("prior", dist.Dirichlet(prior_counts))
    total_counts = int(data.sum())
    pyro.sample("posterior", dist.Multinomial(total_counts, prior), obs=data)


# data = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0],
# [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
data_np = df_.to_numpy()
data_np = data_np[16:]
data = torch.tensor(data_np)

N = data.shape[0]

counter = torch.zeros((N,4))
for i in range(len(data_np)):
    tmp_array = np.unique(data_np[:i+1], axis=0, return_counts=True)[1]
    for j in range(len(tmp_array)):
        if len(tmp_array) > 1:
            if j != 0:
                counter[i][j+1] = tmp_array[j]
            else:
                counter[i][0] = tmp_array[0]
        else:
            counter[i][j] = tmp_array[j]

counter = counter.reshape((len(data_np), 2, 2))

nuts_kernel = NUTS(model)
num_samples, warmup_steps = (1000, 200)
mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=True)
all_means = []
for i in range(len(counter)):
    mcmc.run(counter[i])
    hmc_samples = {k: v.detach().cpu().numpy()
                for k, v in mcmc.get_samples().items()}
    means = hmc_samples['prior'].mean(axis=0)
    stds = hmc_samples['prior'].std(axis=0)
    logging.info("observation: {}".format(data_np[i]))
    logging.info("probabilities: {}".format(means))
    all_means.append(means)


