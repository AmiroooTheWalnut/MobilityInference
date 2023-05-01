import torch
import pyro
import pyro.distributions as dist
from pyro.optim import Adagrad
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceTailAdaptive_ELBO, RenyiELBO, TraceGraph_ELBO, TraceTMC_ELBO, TraceMeanField_ELBO
import logging

def model():
    param1 = pyro.param("pp1", torch.tensor(12.5))
    param2 = pyro.param("pp2", torch.tensor(6.0))
    p1 = param1 * torch.tensor(2)-torch.tensor(4.0)
    p2 = param2 * torch.tensor(3)+torch.tensor(4.0)
    obs = torch.tensor(20.0)+torch.randn(10)*2
    with pyro.plate("data", 10):
        val=pyro.sample("var1",dist.Normal(param1,param2),obs=obs)
        # print(val)
def guide():
    param1 = pyro.param("pp1", torch.tensor(12.5))
    param2 = pyro.param("pp2", torch.tensor(6.0))
    p1 = param1 * torch.tensor(2) - torch.tensor(4.0)
    p2 = param2 * torch.tensor(3) + torch.tensor(4.0)
    pyro.sample("var1", dist.Normal(param1, param2))

adagrad_params = {"lr": 5, "maximize": False, "lr_decay":0.000001}
optimizer = Adagrad(adagrad_params)

# adam_params = {"lr": 0.001, "betas": (0.9, 0.999), "maximize": False}
# optimizer = Adam(adam_params)

Elbo = RenyiELBO
elbo = Elbo(alpha=0.5, num_particles=2)

svi = SVI(model, guide, optim=optimizer, loss=elbo)

n_steps = 10000

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

for step in range(n_steps):
    loss = svi.step()
    # print("param1 {}".format(param1))
    # print("param2 {}".format(param2))
    if step % 100 == 0:
        logging.info("{: >5d}\t{}".format(step, loss))

for name in pyro.get_param_store():
    value = pyro.param(name)
    print("{} = {}".format(name, value.detach().cpu().numpy()))