import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Define the model
def model(data):
    # Define the priors
    loc = pyro.param("loc", torch.tensor(0.))
    scale = pyro.param("scale", torch.tensor(1.))
    with pyro.plate("data", len(data)):
        # Sample from the Normal distribution
        pyro.sample("obs", dist.Normal(loc, scale), obs=data)

# Define the guide
def guide(data):
    # Define the variational parameters
    loc_q = pyro.param("loc_q", torch.tensor(0.))
    scale_q = pyro.param("scale_q", torch.tensor(1.))
    # Sample from the variational distribution
    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Normal(loc_q, scale_q))

# Generate some data
data = torch.tensor([1., 2., 3., 4., 5.])

# Setup the SVI object
svi = SVI(model=model, guide=guide, optim=Adam({"lr": 0.01}), loss=Trace_ELBO())

# Run the optimization loop
num_steps = 1000
for step in range(num_steps):
    loss = svi.step(data)
    if step % 100 == 0:
        print("step {}: loss = {}".format(step, loss))