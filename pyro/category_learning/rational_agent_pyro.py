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


# Define a Pyro model
def model(data):
    # Define the Dirichlet prior
    with pyro.plate("latent_plate"):
        prior_counts = torch.ones(4)
        p_latent = pyro.sample("p_latent", dist.Dirichlet(prior_counts))
    # Loop over the observed data
    with pyro.plate("data_plate", len(data)):
        for i in range(len(data)):
            # Get the observed data
            observation = data[i]
            # Draw a sample from the Bernoulli likelihood using the latent variable
            pyro.sample(f"obs_{i}", dist.Bernoulli(probs=p_latent[2 * observation[0] + observation[1]]),
                        obs=observation)


# Define a Pyro guide
def guide(data):
    # Define the variational parameters for the Dirichlet posterior
    posterior_counts = prior_counts.clone()
    # Loop over the observed data
    with pyro.plate("data_plate", len(data)):
        for i in range(len(data)):
            # Get the observed data
            observation = data[i]
            # Update the posterior
            posterior_counts[2 * observation[0] + observation[1]] += 1
            # Get the posterior probabilities
            posterior_probs = posterior_counts / posterior_counts.sum()
            # Sample a value from the Dirichlet posterior
            pyro.sample(f"p_latent_{i}", dist.Dirichlet(posterior_probs))


# Run inference
pyro.clear_param_store()
svi = pyro.infer.SVI(model, guide, pyro.optim.Adam({"lr": 0.01}), pyro.infer.Trace_ELBO())
for i in range(1000):
    loss = svi.step(data)
    if i % 100 == 0:
        print(f"step {i}, loss = {loss}")

# Print the estimated posterior probabilities
posterior_probs = pyro.param("p_latent").detach()
print("Estimated posterior probabilities: ", posterior_probs)
