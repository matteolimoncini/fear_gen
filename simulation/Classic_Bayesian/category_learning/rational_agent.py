import pyro
import pyro.distributions as dist
import torch

# Define true probabilities
# true_p = torch.tensor([0.8, 0.2, 0.2, 0.8])

# Define prior
prior_counts = torch.ones(4)
prior = dist.Dirichlet(prior_counts)

# Initialize posterior
posterior_counts = prior_counts.clone()

# Define data
data = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [1, 1], [1, 1], [1, 1], [1, 1]])

# Online learning loop
for i in range(len(data)):
    # Infer relying on previous posterior distribution

    # Observe the data
    observation = data[i]
    # Update the posterior
    posterior_counts[2 * observation[0] + observation[1]] += 1
    # Get the posterior probabilities
    posterior_probs = posterior_counts / posterior_counts.sum()
    # Draw a sample from the posterior
    # sample = dist.Bernoulli(probs=posterior_probs).sample()
    # Print the sample and the true probability
    # print(f"Sample: {sample}, True probability: {true_p[2 * observation[0] + observation[1]]}")
    print("Observed: ", observation, "\nestimated_p=", posterior_probs)
