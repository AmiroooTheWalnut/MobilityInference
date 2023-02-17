#   This model assumes that each pair of age-occupation group has needs and one alpha and one beta. The alpha and beta determines how frequently
# this group attends POI types. This means that individuals can't compensate their attendance because the alpha and beta is shared for a group.
# This model only investigate the general type of visits and there is no POI involved.

import os
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal, AutoDiscreteParallel
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceTailAdaptive_ELBO, RenyiELBO, TraceGraph_ELBO, TraceTMC_ELBO, TraceMeanField_ELBO
import logging
from torch.distributions import constraints
from pyro.optim import ASGD
from pyro.optim import Adagrad
from pyro.optim import RAdam
from pyro.optim import ExponentialLR
from pyro.optim import Rprop
from pyro.optim import AdamW
from pyro.optim import Adadelta
from dataProcessing_CUDA import *

def model(data):
    # with pyro.plate("G", data.G) as g:
    #     alpha_paramShop = torch.ones(data.G)
    #     alpha_paramSchool = torch.ones(data.G)
    #     alpha_paramReligion = torch.ones(data.G)
    #     beta_paramShop = torch.ones(data.G)
    #     beta_paramSchool = torch.ones(data.G)
    #     beta_paramReligion = torch.ones(data.G)

    with pyro.plate("N", data.N) as n:
        selAge = pyro.sample("age", dist.Categorical(data.ageProb))
        selOccupation = pyro.sample("occupation", dist.Categorical(data.occupationProb[selAge[n], :]))
        shopVisits = pyro.sample("Tu_Shop", dist.BetaBinomial(torch.abs(data.alpha_paramShop[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramShop[selAge[n]][selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 0]))
        schoolVisits = pyro.sample("Tu_School", dist.BetaBinomial(torch.abs(data.alpha_paramSchool[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramSchool[selAge[n] * 5 + selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 1]))
        religionVisits = pyro.sample("Tu_Religion", dist.BetaBinomial(torch.abs(data.alpha_paramReligion[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramReligion[selAge[n] * 5 + selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 2]))

    shopVisitsObs = pyro.sample("S_Shop", dist.Poisson(torch.abs(shopVisits.sum(-1, True))).to_event(1), obs=data.pOIs[0, 1])
    schoolVisitsObs = pyro.sample("S_School", dist.Poisson(torch.abs(schoolVisits.sum(-1, True))).to_event(1), obs=data.pOIs[1, 1])
    religionVisitsObs = pyro.sample("S_Religion", dist.Poisson(torch.abs(religionVisits.sum(-1, True))).to_event(1), obs=data.pOIs[2, 1])

    # obsRaw = np.transpose(data.pOIs.iloc[:][1])
    # obs = torch.zeros(data.NE)
    # for i in range(data.NE):
    #     obs[i] = obsRaw.iloc[i]
    #     obs[i] = torch.div(obs[i],100)
    print(shopVisits.sum(-1, True) - data.pOIShops.sum() + schoolVisits.sum(-1, True) - data.pOISchools.sum() + religionVisits.sum(-1, True) - data.pOIReligion.sum())

    if data.isFirst == True:
        # obsRaw = np.transpose(data.pOIs.iloc[:][1])
        # obs = torch.zeros(data.NE)
        # for i in range(data.NE):
        #     obs[i] = obsRaw.iloc[i]
        #     obs[i] = torch.div(obs[i], 100)
        print(shopVisits.sum(-1, True) - data.pOIShops.sum() + schoolVisits.sum(-1, True) - data.pOISchools.sum() + religionVisits.sum(-1, True) - data.pOIReligion.sum())
        data.isFirst = False

    return shopVisitsObs, schoolVisitsObs, religionVisitsObs


def guide(data):
    # maxParam = 10

    # register prior parameter value. It'll be updated in the guide function
    # with pyro.plate("G", data.G) as g:
    data.alpha_paramShop = pyro.param("alpha_paramShop_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive).cuda()
    data.beta_paramShop = pyro.param("beta_paramShop_G", torch.add(torch.ones(data.G), 8), constraint=constraints.positive).cuda()
    data.alpha_paramSchool = pyro.param("alpha_paramSchool_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive).cuda()
    data.beta_paramSchool = pyro.param("beta_paramSchool_G", torch.add(torch.ones(data.G), 8), constraint=constraints.positive).cuda()
    data.alpha_paramReligion = pyro.param("alpha_paramReligion_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive).cuda()
    data.beta_paramReligion = pyro.param("beta_paramReligion_G", torch.add(torch.ones(data.G), 8), constraint=constraints.positive).cuda()

    with pyro.plate("N", data.N) as n:
        selAge = pyro.sample("age", dist.Categorical(data.ageProb))
        selOccupation = pyro.sample("occupation", dist.Categorical(data.occupationProb[selAge[n], :]))
        pyro.sample("Tu_Shop", dist.BetaBinomial(torch.abs(data.alpha_paramShop[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramShop[selAge[n]][selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 0]))
        pyro.sample("Tu_School", dist.BetaBinomial(torch.abs(data.alpha_paramSchool[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramSchool[selAge[n] * 5 + selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 1]))
        pyro.sample("Tu_Religion", dist.BetaBinomial(torch.abs(data.alpha_paramReligion[selAge[n] * 5 + selOccupation[n]]), torch.abs(data.beta_paramReligion[selAge[n] * 5 + selOccupation[n]]), data.needsTensor[selAge[n] * 5 + selOccupation[n]][:, 2]))

#def makeTestConfig(test):
#    jsonStr = json.dumps(test.__dict__)
#    print(jsonStr)

# testConfig= Config(0,0,[0],[1])
# file = open("tucson_test1.conf", "w")
# file.write(testConfig.toJSON())
# file.close()

pyro.clear_param_store()

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

isReadConfigFile=True # IF FALSE, MANUAL SELECTION IS ON
configFileName='tucson_test1.conf'
# I'LL ADD CONFIGURATION FILES TO AVOID RAPIDLY INPUTTING CITY AND MONTH
if isReadConfigFile==True:

    file = open('tucson_test1.conf', 'r')

    configStr=file.read()
    retConfig = json.loads(configStr, object_hook=customConfigDecoder)

cities = os.listdir('..\\TimedData')

for i in range(len(cities)):
    print("[{}] {}".format(i + 1, cities[i]))

if isReadConfigFile==False:
    selectedTrainCity = input("Select train city")
    selectedTrainCityIndex = int(selectedTrainCity) - 1
    selectedTestCity = input("Select test city")
    selectedTestCityIndex = int(selectedTestCity) - 1
else:
    selectedTrainCityIndex=retConfig.trainCityIndex
    print("City [{}] selected".format(selectedTrainCityIndex))
    selectedTestCityIndex = retConfig.testCityIndex
    print("City [{}] selected".format(selectedTestCityIndex))

times = os.listdir('..\\TimedData\\' + cities[selectedTrainCityIndex])
dates = set()
for i in range(len(times)):
    noExtension = times[i].split(".")
    parts = noExtension[0].split("_")
    dates.add(parts[len(parts) - 1 - 1] + "_" + parts[len(parts) - 1])

print("Available dates")
dates = sorted(dates)
for i in range(len(dates)):
    print("[{}] {}".format(i + 1, dates[i]))

if isReadConfigFile==False:
    selectedTrainRange = input("Select train range (, and - ranges max two digits i.e. 1,3-5,21: 1,3,4,5,21)")
    selectedTestRange = input("Select test range (, and - ranges max two digits i.e. 1,3-5,21: 1,3,4,5,21)")

    selectedTrainRangeIndices = processInputRanges(selectedTrainRange)
    selectedTestRangeIndices = processInputRanges(selectedTestRange)

    print(selectedTrainRangeIndices)
    print(selectedTestRangeIndices)
else:
    selectedTrainRangeIndices=retConfig.trainTimeRange
    selectedTestRangeIndices=retConfig.testTimeRange
    print("Selected train date range {}".format(selectedTrainRangeIndices))
    print("Selected test date range {}".format(selectedTestRangeIndices))


allData = loadData(cities[selectedTrainCityIndex],cities[selectedTestCityIndex], dates, selectedTrainRangeIndices,selectedTestRangeIndices)

# DataBundle.loadDataToGPU(allData.trainData.monthlyData[0])

graph = pyro.render_model(model, model_args=(allData.trainData.monthlyData[0],), render_distributions=True, render_params=True)
graph.view()

# setup the optimizer
adam_params = {"lr": 0.01, "betas": (0.9, 0.999), "maximize": False}
optimizer = Adam(adam_params)

# asgd_params = {"lr": 0.0001, "maximize": False}
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
elbo = Elbo(alpha=0.1, num_particles=5)

# Elbo = TraceMeanField_ELBO
# elbo = Elbo(num_particles=5)

# Elbo = TraceTMC_ELBO
# elbo = Elbo(num_particles=5)


# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=elbo)

# svi.num_chains=1

loss = elbo.loss(model, guide, allData.trainData.monthlyData[0])
logging.info("first loss train SantaFe = {}".format(loss))

n_steps = 10000

# do gradient steps
for step in range(n_steps):
    loss = svi.step(allData.trainData.monthlyData[0])
    if step % 10 == 0:
        logging.info("{: >5d}\t{}".format(step, loss))
        # print('.', end='')
        # for name in pyro.get_param_store():
        #     value = pyro.param(name)
        #     print("{} = {}".format(name, value.detach().cpu().numpy()))
print("Final evalulation")


# allDataTrain = loadData(cities[selectedTrainCityIndex], dates, times[selectedTrainRangeIndices])
allData.trainData.monthlyData[0].isFirst=True
loss = elbo.loss(model, guide, allData.trainData.monthlyData[0])
logging.info("final loss train Tucson = {}".format(loss))

for name in pyro.get_param_store():
    value = pyro.param(name)
    print("{} = {}".format(name, value.detach().cpu().numpy()))

# DataBundle.unloadDataToGPU(allData.trainData.monthlyData[0])
# DataBundle.loadDataToGPU(allData.testData.monthlyData[0])

# visits = pd.read_csv('USA_WI_Outagamie County_Appleton_FullSimple.csv', header=None)
# population = 75000
# data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb]
# allData = AllData(data)
#

for i in range(len(allData.testData.monthlyData)):
    loss = elbo.loss(model, guide, allData.testData.monthlyData[i])
    logging.info("final loss test Appleton = {}".format(loss))

#
# visits = pd.read_csv('USA_WI_Brown County_Green Bay_FullSimple.csv', header=None)
# population = 107400
# data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb]
# allData = AllData(data)
#
# loss = elbo.loss(model, guide, allData)
# logging.info("final loss test Green bay = {}".format(loss))
#
# visits = pd.read_csv('USA_NY_Richmond County_New York_FullSimple.csv', header=None)
# population = 8468000
# data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb]
# allData = AllData(data)
#
# loss = elbo.loss(model, guide, allData)
# logging.info("final loss test New york city = {}".format(loss))
#
# visits = pd.read_csv('USA_WA_King County_Seattle_FullSimple.csv', header=None)
# population = 760000
# data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb]
# allData = AllData(data)
#
# loss = elbo.loss(model, guide, allData)
# logging.info("final loss test Seattle = {}".format(loss))
