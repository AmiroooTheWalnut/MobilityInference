#   This model assumes that each pair of age-occupation group has needs and one alpha and one beta. The alpha and beta determines how frequently
# this group attends POIs. This means that individuals can't compensate their attendance because the alpha and beta are shared for a group.
# This model distinguished the type of visits and the POIs are distinguished. The home CBG is sampled for each individual and it affects
# the chance of visiting a POI.
# - Individual level samples (latent variable)
# - Age-Occupation level parameters
# - POI distinguished visits
# - CBG home sampled and probability of visit is applied

import pandas as pd
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal, AutoDiscreteParallel
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceTailAdaptive_ELBO, RenyiELBO, TraceGraph_ELBO, TraceTMC_ELBO, TraceMeanField_ELBO
import numpy as np
import logging
from torch.distributions import constraints
from pyro.optim import ASGD
from pyro.optim import Adagrad
from pyro.optim import RAdam
from pyro.optim import ExponentialLR
from pyro.optim import Rprop
from pyro.optim import AdamW
from pyro.optim import Adadelta
import gc

class AllData:
    def __init__(self,data):
        self.pOIs = data[0]
        self.needs = data[1]
        self.pOIShops = torch.from_numpy(data[4].iloc[:, 1:].values)
        self.pOISchools = torch.tensor(data[5].iloc[:, 1:].values)
        self.pOIReligion = torch.tensor(data[6].iloc[:, 1:].values)
        self.pOIShopProb = torch.tensor(data[7].values).flatten()
        self.pOISchoolProb = torch.tensor(data[8].values).flatten()
        self.pOIReligionProb = torch.tensor(data[9].values).flatten()
        self.ageCategories = 3
        self.occupationCategories = 5
        self.needCategories = 3
        self.ageProb = torch.zeros(3) # age0: 0-18, age1: 18-65, age2: 65+
        self.ageProb[0] = 0.25
        self.ageProb[1] = 0.58
        self.ageProb[2] = 0.17
        self.occupationProb = torch.zeros(3,5) # occupation0: student, occupation1: service, occupation2: driver, occupation3: education, occupation4: unemployed
        self.occupationProb[0][0] = 0.95
        self.occupationProb[0][1] = 0.05
        self.occupationProb[0][2] = 0
        self.occupationProb[0][3] = 0
        self.occupationProb[0][4] = 0

        self.occupationProb[1][0] = 0.173
        self.occupationProb[1][1] = 0.75
        self.occupationProb[1][2] = 0.012
        self.occupationProb[1][3] = 0.045
        self.occupationProb[1][4] = 0.02

        self.occupationProb[2][0] = 0
        self.occupationProb[2][1] = 0.1
        self.occupationProb[2][2] = 0
        self.occupationProb[2][3] = 0
        self.occupationProb[2][4] = 0.9

        self.P = len(self.pOIs)  # places
        self.N = data[2]  # people
        self.Nshop = pOIShops.shape[0]
        self.Nschool = pOISchools.shape[0]
        self.Nreligion = pOIReligion.shape[0]
        self.D = pOIShops.shape[1]  # days
        self.M = 1  # months
        self.NE = 3  # needs
        self.G = 15 # age/occupation groups
        self.needsTensor = torch.tensor(needs.values)
        self.isFirst=data[3]
        self.alpha_paramShop = torch.ones(self.G)
        self.alpha_paramSchool = torch.ones(self.G)
        self.alpha_paramReligion = torch.ones(self.G)
        self.beta_paramShop = torch.ones(self.G)
        self.beta_paramSchool = torch.ones(self.G)
        self.beta_paramReligion = torch.ones(self.G)

        self.cBGPopProbs = torch.tensor(data[10].values).flatten()
        self.cBGShopProb = torch.tensor(data[11].transpose().values)
        self.cBGSchoolProb = torch.tensor(data[12].transpose().values)
        self.cBGReligionProb = torch.tensor(data[13].transpose().values)


pyro.clear_param_store()

def model(data):
    # alpha_paramShop = torch.ones(data.G).cuda()
    # alpha_paramSchool = torch.ones(data.G).cuda()
    # alpha_paramReligion = torch.ones(data.G).cuda()
    # beta_paramShop = torch.ones(data.G).cuda()
    # beta_paramSchool = torch.ones(data.G).cuda()
    # beta_paramReligion = torch.ones(data.G).cuda()

    with pyro.plate("N", data.N) as n:
        cBGLocation = pyro.sample("cbg", dist.Categorical(data.cBGPopProbs))
        selAge = pyro.sample("age", dist.Categorical(data.ageProb))
        selOccupation = pyro.sample("occupation", dist.Categorical(data.occupationProb[selAge[n], :]))
        with pyro.plate("Nshop", data.Nshop) as nshop:
            shopVisits = pyro.sample("Tu_Shop", dist.BetaBinomial(torch.abs(data.alpha_paramShop[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramShop[selAge[n]][selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 0]))
            newShopVisits = torch.mul(shopVisits, data.cBGShopProb[:, cBGLocation])
            diff = shopVisits.sum() - newShopVisits.sum()
            shopVisits = torch.add(newShopVisits, torch.div(diff, newShopVisits.size(dim=0) * newShopVisits.size(dim=1)))
        with pyro.plate("Nschool", data.Nschool) as nschool:
            schoolVisits = pyro.sample("Tu_School", dist.BetaBinomial(torch.abs(data.alpha_paramSchool[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramSchool[selAge[n] * 5 + selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 1]))
            newSchoolVisits = torch.mul(schoolVisits, data.cBGSchoolProb[:, cBGLocation])
            diff = schoolVisits.sum() - newSchoolVisits.sum()
            schoolVisits = torch.add(newSchoolVisits, torch.div(diff, newSchoolVisits.size(dim=0) * newSchoolVisits.size(dim=1)))
        with pyro.plate("Nreligion", data.Nreligion) as nreligion:
            religionVisits = pyro.sample("Tu_Religion", dist.BetaBinomial(torch.abs(data.alpha_paramReligion[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramReligion[selAge[n] * 5 + selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 2]))
            newReligionVisits = torch.mul(religionVisits, data.cBGReligionProb[:, cBGLocation])
            diff = religionVisits.sum() - newReligionVisits.sum()
            religionVisits = torch.add(newReligionVisits, torch.div(diff, newReligionVisits.size(dim=0) * newReligionVisits.size(dim=1)))

    print(torch.mul(torch.abs(shopVisits.sum(1)), data.pOIShopProb).sum() - data.pOIShops.sum() + torch.mul(torch.abs(schoolVisits.sum(1)), data.pOISchoolProb).sum() - data.pOISchools.sum() + torch.mul(torch.abs(religionVisits.sum(1)), data.pOIReligionProb).sum() - data.pOIReligion.sum())

    if data.isFirst:
        # obsRaw = np.transpose(data.pOIs.iloc[:][1])
        # obs = torch.zeros(data.NE)
        # for i in range(data.NE):
        #     obs[i] = obsRaw.iloc[i]
        #     obs[i] = torch.div(obs[i], 100)
        print(torch.mul(torch.abs(shopVisits.sum(1)), data.pOIShopProb).sum() - data.pOIShops.sum() + torch.mul(torch.abs(schoolVisits.sum(1)), data.pOISchoolProb).sum() - data.pOISchools.sum() + torch.mul(torch.abs(religionVisits.sum(1)), data.pOIReligionProb).sum() - data.pOIReligion.sum())
        data.isFirst = False

    with pyro.plate("Nshop_prime", data.Nshop) as nshop:
        shopVisitsPOIs=shopVisits.sum(1)
        shopVisitsPOIsAdjusted=torch.mul(shopVisitsPOIs, data.pOIShopProb)
        expectedSum = shopVisitsPOIs.sum()
        newSum = shopVisitsPOIsAdjusted.sum()
        diff = expectedSum-newSum
        newSum = torch.add(shopVisitsPOIsAdjusted,torch.div(diff,shopVisitsPOIs.size(dim=0)))
        pyro.sample("S_Shop", dist.Poisson(newSum).to_event(1),obs=data.pOIShops)
        # del shopVisitsPOIs
        # del shopVisits
        # del shopVisitsPOIsAdjusted
        # del expectedSum
        # del newSum
        # gc.collect()
        # torch.cuda.empty_cache()
    with pyro.plate("Nschool_prime", data.Nschool) as nschool:
        schoolVisitsPOIs = schoolVisits.sum(1)
        schoolVisitsPOIsAdjusted = torch.mul(schoolVisitsPOIs, data.pOISchoolProb)
        expectedSum = schoolVisitsPOIs.sum()
        newSum = schoolVisitsPOIsAdjusted.sum()
        diff = expectedSum - newSum
        newSum = torch.add(schoolVisitsPOIsAdjusted, torch.div(diff, schoolVisitsPOIs.size(dim=0)))
        pyro.sample("S_School", dist.Poisson(newSum).to_event(1),obs=data.pOISchools)
        # del schoolVisitsPOIs
        # del schoolVisits
        # del schoolVisitsPOIsAdjusted
        # del expectedSum
        # del newSum
        # gc.collect()
        # torch.cuda.empty_cache()
    with pyro.plate("Nreligion_prime", data.Nreligion) as nreligion:
        religionVisitsPOIs = religionVisits.sum(1)
        religionVisitsPOIsAdjusted = torch.mul(religionVisitsPOIs, data.pOIReligionProb)
        expectedSum = religionVisitsPOIs.sum()
        newSum = religionVisitsPOIsAdjusted.sum()
        diff = expectedSum - newSum
        newSum = torch.add(religionVisitsPOIsAdjusted, torch.div(diff, religionVisitsPOIs.size(dim=0)))
        pyro.sample("S_Religion",dist.Poisson(newSum).to_event(1), obs=data.pOIReligion)
        # del religionVisitsPOIs
        # del religionVisits
        # del religionVisitsPOIsAdjusted
        # del expectedSum
        # del newSum
        # gc.collect()
        # torch.cuda.empty_cache()

    # obsRaw = np.transpose(data.pOIs.iloc[:][1])
    # obs = torch.zeros(data.NE)
    # for i in range(data.NE):
    #     obs[i] = obsRaw.iloc[i]
    #     obs[i] = torch.div(obs[i],100)

    # del shopVisits
    # del schoolVisits
    # del religionVisits
    # del shopVisitsPOIs
    # del schoolVisitsPOIs
    # del religionVisitsPOIs
    # torch.cuda.empty_cache()
   # return shopVisitsObs, schoolVisitsObs, religionVisitsObs

def guide(data):
    # maxParam = 100

    # register prior parameter value. It'll be updated in the guide function
    data.alpha_paramShop = pyro.param("alpha_paramShop_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive)
    data.beta_paramShop = pyro.param("beta_paramShop_G", torch.add(torch.ones(data.G), 12), constraint=constraints.positive)
    data.alpha_paramSchool = pyro.param("alpha_paramSchool_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive)
    data.beta_paramSchool = pyro.param("beta_paramSchool_G", torch.add(torch.ones(data.G), 12), constraint=constraints.positive)
    data.alpha_paramReligion = pyro.param("alpha_paramReligion_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive)
    data.beta_paramReligion = pyro.param("beta_paramReligion_G", torch.add(torch.ones(data.G), 12), constraint=constraints.positive)

    with pyro.plate("N", data.N) as n:
        cBGLocation = pyro.sample("cbg", dist.Categorical(data.cBGPopProbs))
        selAge = pyro.sample("age", dist.Categorical(data.ageProb))
        selOccupation = pyro.sample("occupation", dist.Categorical(data.occupationProb[selAge[n], :]))
        with pyro.plate("Nshop", data.Nshop) as nshop:
            shopVisits = pyro.sample("Tu_Shop", dist.BetaBinomial(torch.abs(data.alpha_paramShop[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramShop[selAge[n]][selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 0]))
            newShopVisits = torch.mul(shopVisits, data.cBGShopProb[:,cBGLocation])
            diff = shopVisits.sum() - newShopVisits.sum()
            shopVisits = torch.add(newShopVisits, torch.div(diff, newShopVisits.size(dim=0)*newShopVisits.size(dim=1)))
        with pyro.plate("Nschool", data.Nschool) as nschool:
            schoolVisits = pyro.sample("Tu_School", dist.BetaBinomial(torch.abs(data.alpha_paramSchool[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramSchool[selAge[n] * 5 + selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 1]))
            newSchoolVisits = torch.mul(schoolVisits, data.cBGSchoolProb[:,cBGLocation])
            diff = schoolVisits.sum() - newSchoolVisits.sum()
            schoolVisits = torch.add(newSchoolVisits, torch.div(diff, newSchoolVisits.size(dim=0)*newSchoolVisits.size(dim=1)))
        with pyro.plate("Nreligion", data.Nreligion) as nreligion:
            religionVisits = pyro.sample("Tu_Religion", dist.BetaBinomial(torch.abs(data.alpha_paramReligion[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramReligion[selAge[n] * 5 + selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 2]))
            newReligionVisits = torch.mul(religionVisits, data.cBGReligionProb[:,cBGLocation])
            diff = religionVisits.sum() - newReligionVisits.sum()
            religionVisits = torch.add(newReligionVisits, torch.div(diff, newReligionVisits.size(dim=0)*newReligionVisits.size(dim=1)))

    # del selAge
    # del selOccupation
    # gc.collect()
    # torch.cuda.empty_cache()

# torch.set_default_tensor_type(torch.cuda.FloatTensor)

cBGPopProbs=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_CBGPopProb.csv', header=None)

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

visits=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_FullSimple.csv', header=None)
pOIShops=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_shopLocVis.csv', header=None)
pOISchools=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_schoolLocVis.csv', header=None)
pOIReligion=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_religionLocVis.csv', header=None)
needs=pd.read_csv('Needs_data_numbers.csv', header=None)

cBGShopProb=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_sourceCBG_shopCBG_probability.csv', header=None)
cBGSchoolProb=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_sourceCBG_schoolCBG_probability.csv', header=None)
cBGReligionProb=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_sourceCBG_religionCBG_probability.csv', header=None)

population=84000

isFirst=True

pOIShopsProb=pd.DataFrame(np.zeros(pOIShops.shape[0]))
pOISchoolsProb=pd.DataFrame(np.zeros(pOISchools.shape[0]))
pOIReligionProb=pd.DataFrame(np.zeros(pOIReligion.shape[0]))
sShop=pOIShops.iloc[:,1].sum()
for i in range(pOIShops.shape[0]):
    pOIShopsProb.at[i,0]=pOIShops.iloc[i,1]/sShop
sSch=pOISchools.iloc[:,1].sum()
for i in range(pOISchools.shape[0]):
    pOISchoolsProb.at[i,0]=pOISchools.iloc[i,1]/sSch
sRel=pOIReligion.iloc[:,1].sum()
for i in range(pOIReligion.shape[0]):
    pOIReligionProb.at[i,0]=pOIReligion.iloc[i,1]/sRel

data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb,cBGPopProbs,cBGShopProb,cBGSchoolProb,cBGReligionProb]
allData=AllData(data)

# graph=pyro.render_model(model, model_args=(allData,), render_distributions=True, render_params=True)
# graph.view()

# setup the optimizer
adam_params = {"lr": 0.001, "betas": (0.9, 0.999), "maximize": False}
optimizer = Adam(adam_params)

# asgd_params = {"lr": 0.00001, "maximize": False}
# optimizer = ASGD(asgd_params)

# adagrad_params = {"lr": 0.0001, "maximize": False}
# optimizer = Adagrad(adagrad_params)

# radam_params = {"lr": 0.01, "betas": (0.6, 0.9)}
# optimizer = RAdam(radam_params)

# exponentialLR_params = {"gamma ": 0.01}
# optimizer = ExponentialLR(exponentialLR_params)

# rprop_params = {}
# optimizer = Rprop(rprop_params)

# adamW_params = {}
# optimizer = AdamW(adamW_params)

# adadelta_params = {}
# optimizer = Adadelta(adadelta_params)

# auto_guide = AutoDiscreteParallel(model)

# Elbo=Trace_ELBO
# elbo = Elbo(num_particles=5)

# Elbo = TraceTailAdaptive_ELBO
# elbo = Elbo(num_particles=5, vectorize_particles=True)

# Elbo = TraceGraph_ELBO
# elbo = Elbo(num_particles=5)

Elbo = RenyiELBO
elbo = Elbo(alpha=0.1,num_particles=4)

# Elbo = TraceMeanField_ELBO
# elbo = Elbo(num_particles=5)

# Elbo = TraceTMC_ELBO
# elbo = Elbo(num_particles=5)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=elbo)

# svi.num_chains=1

loss = elbo.loss(model, guide, allData)
logging.info("first loss train SantaFe = {}".format(loss))

n_steps=10000

# do gradient steps
for step in range(n_steps):
    loss=svi.step(allData)
    if step % 10 == 0:
        logging.info("{: >5d}\t{}".format(step, loss))
        #print('.', end='')
        # for name in pyro.get_param_store():
        #     value = pyro.param(name)
        #     print("{} = {}".format(name, value.detach().cpu().numpy()))
print("Final evalulation")
data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb,cBGPopProbs,cBGShopProb,cBGSchoolProb,cBGReligionProb]
allData=AllData(data)
loss = elbo.loss(model, guide, allData)
logging.info("final loss train SantaFe = {}".format(loss))

for name in pyro.get_param_store():
    value = pyro.param(name)
    print("{} = {}".format(name, value.detach().cpu().numpy()))

visits=pd.read_csv('USA_WI_Outagamie County_Appleton_FullSimple.csv', header=None)
population=75000
data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb,cBGPopProbs,cBGShopProb,cBGSchoolProb,cBGReligionProb]
allData=AllData(data)

loss = elbo.loss(model, guide, allData)
logging.info("final loss test Appleton = {}".format(loss))

visits=pd.read_csv('USA_WI_Brown County_Green Bay_FullSimple.csv', header=None)
population=107400
data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb,cBGPopProbs,cBGShopProb,cBGSchoolProb,cBGReligionProb]
allData=AllData(data)

loss = elbo.loss(model, guide, allData)
logging.info("final loss test Green bay = {}".format(loss))

visits=pd.read_csv('USA_NY_Richmond County_New York_FullSimple.csv', header=None)
population=8468000
data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb,cBGPopProbs,cBGShopProb,cBGSchoolProb,cBGReligionProb]
allData=AllData(data)

loss = elbo.loss(model, guide, allData)
logging.info("final loss test New york city = {}".format(loss))

visits=pd.read_csv('USA_WA_King County_Seattle_FullSimple.csv', header=None)
population=760000
data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb,cBGPopProbs,cBGShopProb,cBGSchoolProb,cBGReligionProb]
allData=AllData(data)

loss = elbo.loss(model, guide, allData)
logging.info("final loss test Seattle = {}".format(loss))