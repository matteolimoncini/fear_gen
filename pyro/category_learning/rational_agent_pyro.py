import logging


import numpy as np
import torch
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
import pyro.distributions as dist

logging.basicConfig(format='%(message)s', level=logging.INFO)


# categorical/multinomial distribution
# LDA
prior_counts = torch.ones(4)


def model(data):
    prior = pyro.sample("prior", dist.Dirichlet(prior_counts))
    total_counts = int(data.sum())
    pyro.sample("posterior", dist.Multinomial(total_counts, prior), obs=data)


data = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
data_np = data.numpy()

counter = torch.zeros((len(data_np), 4))
for i in range(len(data_np)):
    tmp_array = np.unique(data_np[:i+1], axis=0, return_counts=True)[1]
    for j in range(len(tmp_array)):
        counter[i][j] = tmp_array[j]
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
    print('observation: ', data_np[i])
    print('probabilities: ', means)
    all_means.append(means)




