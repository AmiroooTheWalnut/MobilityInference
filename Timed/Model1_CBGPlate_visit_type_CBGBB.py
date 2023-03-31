#   This model assumes that each pair of age-occupation group has needs and one alpha and one beta. The alpha and beta determines how frequently
# this group attends POI types. This means that individuals can't compensate their attendance because the alpha and beta are shared for a group.
# This model only investigate the general type of visits and there is no POI involved.
# - Individual level samples (latent variable)
# - Age-Occupation level parameters

import os
import multiprocessing
from multiprocessing import Queue
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import torch
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
from pyro.optim import SGD
from dataProcessing import *
from kFoldCrossVal import *
from VisualizeAlphaBeta import *

class Test():

    def __init__(self):
        self.stepValue=0

    def run(self,numParticles,lr,elboType,alpha,i,retVals):
        globalError = np.zeros(1, dtype=np.int32)
        self.index=i

        def model(data):
            groupProbs = (torch.transpose(data.occupationProb, 0, 1) * data.ageProb).reshape([data.G, 1])
            needsSh = data.needsTensor[:, 0].reshape([data.G,1])
            needsSch = data.needsTensor[:, 1].reshape([data.G, 1])
            needsRel = data.needsTensor[:, 2].reshape([data.G, 1])
            BBNSh = (needsSh * groupProbs * data.populationNum.repeat([data.G, 1])).round()
            BBNSch = (needsSch * groupProbs * data.populationNum.repeat([data.G, 1])).round()
            BBNRel = (needsRel * groupProbs * data.populationNum.repeat([data.G, 1])).round()
            BBNSh[BBNSh == 0] = 1
            BBNSch[BBNSch == 0] = 1
            BBNRel[BBNRel == 0] = 1

            # \/\/\/ DEBUG EXPECTED VALUE
            # groupCBGPop=groupProbs * data.populationNum.repeat([data.G, 1])
            # firstGroupFirstCBGPop=groupCBGPop[0,0]
            # onePersonInCBG=pyro.sample("Tu_Shop_debugIP", dist.BetaBinomial(data.alpha_paramShop[0], data.beta_paramShop[0], data.needsTensor[0][0]))
            #
            # oneBigPersonInCBG = pyro.sample("Tu_Shop_debugIB", dist.BetaBinomial(data.alpha_paramShop[0], data.beta_paramShop[0], BBNSh[0][0]))
            # data.resultSamplesIB.append(oneBigPersonInCBG)
            # resultFromSample=onePersonInCBG*firstGroupFirstCBGPop
            # data.resultSamplesIP.append(resultFromSample)
            # resultFromExpectation=firstGroupFirstCBGPop*data.needsTensor[0][0]*data.alpha_paramShop[0]/(data.alpha_paramShop[0]+data.beta_paramShop[0])
            # # print("resultFromSample")
            # # print("resultFromSample "+str(resultFromSample.item()))
            # # if resultFromSample.item()>0:
            # #     print("!!!")
            # # print("resultFromExpectation")
            # # print("resultFromExpectation "+str(resultFromExpectation.item()))
            # data.resultFromSampleSumIP=data.resultFromSampleSumIP+resultFromSample
            # data.expectationDebugCounter=data.expectationDebugCounter+1
            # resultFromSampleAvgIP=data.resultFromSampleSumIP/data.expectationDebugCounter
            # data.resultFromSampleSumIB = data.resultFromSampleSumIB + oneBigPersonInCBG
            # data.resultFromAvgAllIP.append(resultFromSampleAvgIP.item())
            # data.resultFromEEAll.append(resultFromExpectation.item())
            # data.resultFromAvgAllIB.append(data.resultFromSampleSumIB.item()/data.expectationDebugCounter)
            # # print("resultFromSampleAvg " + str(resultFromSampleAvg.item()))
            # ^^^ DEBUG EXPECTED VALUE



            if self.stepValue > 3500 and self.stepValue < 3520:
                print('Stop here!')

            # \/\/\/ Big individual
            with pyro.plate("NCBG", data.NCBG) as ncbg:
            # with pyro.plate("G", data.G) as g:
                # selAge = pyro.sample("age", dist.Categorical(data.ageProb))
                # selOccupation = pyro.sample("occupation", dist.Categorical(data.occupationProb[selAge[n], :]))
                with pyro.plate("G", data.G) as g:
                # with pyro.plate("NCBG", data.NCBG) as ncbg:
                    # shopVisits = pyro.sample("Tu_Shop", dist.BetaBinomial(torch.abs(data.alpha_paramShop[g]), torch.abs(data.beta_paramShop[g]), needsSh*groupProbs[g]*data.populationNum.repeat([data.G,1])))
                    # schoolVisits = pyro.sample("Tu_School", dist.BetaBinomial(torch.abs(data.alpha_paramSchool[g]), torch.abs(data.beta_paramSchool[g]), data.needsTensor[g][:, 1]*data.populationNum[ncbg]*groupProbs[g]))
                    # religionVisits = pyro.sample("Tu_Religion", dist.BetaBinomial(torch.abs(data.alpha_paramReligion[g]), torch.abs(data.beta_paramReligion[g]), data.needsTensor[g][:, 2]*data.populationNum[ncbg]*groupProbs[g]))

                    shopVisits = pyro.sample("Tu_Shop", dist.BetaBinomial(torch.transpose(data.alpha_paramShop[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramShop[g].repeat([data.NCBG, 1]), 0, 1), BBNSh))
                    schoolVisits = pyro.sample("Tu_School", dist.BetaBinomial(torch.transpose(data.alpha_paramSchool[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramSchool[g].repeat([data.NCBG, 1]), 0, 1), BBNSch))
                    religionVisits = pyro.sample("Tu_Religion", dist.BetaBinomial(torch.transpose(data.alpha_paramReligion[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramReligion[g].repeat([data.NCBG, 1]), 0, 1), BBNRel))

                    # shopVisits=pyro.sample("Tu_Shop", dist.BetaBinomial(torch.transpose(data.alpha_paramShop[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramShop[g].repeat([data.NCBG, 1]), 0, 1), 1900))
                    # schoolVisits=pyro.sample("Tu_School", dist.BetaBinomial(torch.transpose(data.alpha_paramSchool[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramSchool[g].repeat([data.NCBG, 1]), 0, 1), 1900))
                    # religionVisits=pyro.sample("Tu_Religion", dist.BetaBinomial(torch.transpose(data.alpha_paramReligion[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramReligion[g].repeat([data.NCBG, 1]), 0, 1), 1900))

            sumValShop=shopVisits.sum()
            sumValSchool=schoolVisits.sum()
            sumValRel=religionVisits.sum()
            # ^^^ Big individual


            # shopVisits = ((shopVisits * data.occupationProb.flatten()).sum(-1, False)).mul(data.populationNum)
            # schoolVisits = ((schoolVisits * data.occupationProb.flatten()).sum(-1, False)).mul(data.populationNum)
            # religionVisits = ((religionVisits * data.occupationProb.flatten()).sum(-1, False)).mul(data.populationNum)
            # print(((shopVisits * (torch.transpose(data.occupationProb, 0, 1)).flatten()).sum(-1, False).mul(data.populationNum)).sum())
            # print(((shopVisits * (torch.transpose(data.occupationProb, 0, 1) * data.ageProb).flatten()).sum(-1, False).mul(data.populationNum)).sum())
            # sumVal = 0
            # for m in range(404):
            #     temp = ((data.populationNum[m] * (torch.transpose(data.occupationProb, 0, 1)).flatten()) * shopVisits[m][:]).sum()
            #     print(temp)
            #     sumVal = sumVal + temp
            # print(sumVal)
            # shopVisits = (shopVisits * (torch.transpose(data.occupationProb, 0, 1) * data.ageProb).flatten()).sum(-1, False).mul(data.populationNum)
            # schoolVisits = (schoolVisits * (torch.transpose(data.occupationProb, 0, 1) * data.ageProb).flatten()).sum(-1, False).mul(data.populationNum)
            # religionVisits = (religionVisits * (torch.transpose(data.occupationProb, 0, 1) * data.ageProb).flatten()).sum(-1, False).mul(data.populationNum)
            #
            # print(shopVisits.sum(-1, True))



            # \/\/\/ Individual multipiled to population
            # # with pyro.plate("NCBG", data.NCBG) as ncbg:
            # with pyro.plate("G", data.G) as g:
            #     # selAge = pyro.sample("age", dist.Categorical(data.ageProb))
            #     # selOccupation = pyro.sample("occupation", dist.Categorical(data.occupationProb[selAge[n], :]))
            #     # with pyro.plate("G", data.G) as g:
            #     with pyro.plate("NCBG", data.NCBG) as ncbg:
            #         # print(data.needsTensor[g][:, 0])
            #         # print(data.needsTensor[g][:, 1])
            #         # print(data.needsTensor[g][:, 2])
            #         shopVisits = pyro.sample("Tu_Shop", dist.BetaBinomial(torch.abs(data.alpha_paramShop), torch.abs(data.beta_paramShop), data.needsTensor[:, 0]))
            #         schoolVisits = pyro.sample("Tu_School", dist.BetaBinomial(torch.abs(data.alpha_paramSchool), torch.abs(data.beta_paramSchool), data.needsTensor[:, 1]))
            #         religionVisits = pyro.sample("Tu_Religion", dist.BetaBinomial(torch.abs(data.alpha_paramReligion), torch.abs(data.beta_paramReligion), data.needsTensor[:, 2]))
            #
            # # shopVisits = torch.zeros((data.G,data.NCBG))
            # # schoolVisits = torch.zeros((data.G, data.NCBG))
            # # religionVisits = torch.zeros((data.G, data.NCBG))
            # # for g in range(data.G):
            # #     for ncbg in range(data.NCBG):
            # #         shopVisits[g, ncbg] = pyro.sample("Tu_Shop" + str(g) + "_" + str(ncbg), dist.BetaBinomial(torch.abs(data.alpha_paramShop[g]), torch.abs(data.beta_paramShop[g]), data.needsTensor[g, 0]))
            # #         schoolVisits[g, ncbg] = pyro.sample("Tu_School" + str(g) + "_" + str(ncbg), dist.BetaBinomial(torch.abs(data.alpha_paramSchool[g]), torch.abs(data.beta_paramSchool[g]), data.needsTensor[g, 1]))
            # #         religionVisits[g, ncbg] = pyro.sample("Tu_Religion" + str(g) + "_" + str(ncbg), dist.BetaBinomial(torch.abs(data.alpha_paramReligion[g]), torch.abs(data.beta_paramReligion[g]), data.needsTensor[g, 2]))
            #
            # # shopVisits = ((shopVisits * data.occupationProb.flatten()).sum(-1, False)).mul(data.populationNum)
            # # schoolVisits = ((schoolVisits * data.occupationProb.flatten()).sum(-1, False)).mul(data.populationNum)
            # # religionVisits = ((religionVisits * data.occupationProb.flatten()).sum(-1, False)).mul(data.populationNum)
            # # print(((shopVisits * (torch.transpose(data.occupationProb, 0, 1)).flatten()).sum(-1, False).mul(data.populationNum)).sum())
            # # print(((shopVisits * (torch.transpose(data.occupationProb, 0, 1) * data.ageProb).flatten()).sum(-1, False).mul(data.populationNum)).sum())
            #
            # # frac = data.populationNum[0] / (data.populationNum[0] * groupProbs).sum()
            # groupCBGPop = groupProbs * data.populationNum.repeat([data.G, 1])
            # sumValShop = (shopVisits * torch.transpose(groupCBGPop,0,1)).sum()
            # sumValSchool = (schoolVisits * torch.transpose(groupCBGPop,0,1)).sum()
            # sumValRel = (religionVisits * torch.transpose(groupCBGPop,0,1)).sum()
            # ^^^ Individual multipiled to population

            # # \/\/\/ MANUAL CALCULATION. SLOW
            # sumValShop = 0
            # sumValSchool = 0
            # sumValRel = 0
            # for m in range(404):
            #     tempShop = (((data.populationNum[m]*frac) * (torch.transpose(data.occupationProb, 0, 1)).flatten()) * shopVisits[m][:]).sum()
            #     tempSchool = (((data.populationNum[m] * frac) * (torch.transpose(data.occupationProb, 0, 1)).flatten()) * schoolVisits[m][:]).sum()
            #     tempRel = (((data.populationNum[m] * frac) * (torch.transpose(data.occupationProb, 0, 1)).flatten()) * religionVisits[m][:]).sum()
            #     # print(temp)
            #     if not np.isnan(frac):
            #         sumValShop = sumValShop + tempShop
            #         sumValSchool = sumValSchool + tempSchool
            #         sumValRel = sumValRel + tempRel
            #         # print('!!!')
            #     # sumVal=sumVal+temp
            # # print(sumVal)
            # # ^^^ MANUAL CALCULATION. SLOW

            shopVisitsObs = pyro.sample("S_Shop", dist.Poisson(sumValShop), obs=torch.round(data.pOIs[0, 1]))
            schoolVisitsObs = pyro.sample("S_School", dist.Poisson(sumValSchool), obs=torch.round(data.pOIs[1, 1]))
            religionVisitsObs = pyro.sample("S_Religion", dist.Poisson(sumValRel), obs=torch.round(data.pOIs[2, 1]))

            # shopVisitsObs = pyro.sample("S_Shop", dist.Poisson(torch.sum(torch.transpose(torch.div(data.alpha_paramShop.repeat([data.NCBG, 1]), data.alpha_paramShop.repeat([data.NCBG, 1]) + data.beta_paramShop[g].repeat([data.NCBG, 1])), 0, 1) * BBNSh)), obs=torch.round(data.pOIs[0, 1]))
            # schoolVisitsObs = pyro.sample("S_School", dist.Poisson(torch.sum(torch.transpose(torch.div(data.alpha_paramSchool.repeat([data.NCBG, 1]), data.alpha_paramSchool.repeat([data.NCBG, 1]) + data.beta_paramSchool[g].repeat([data.NCBG, 1])), 0, 1) * BBNSch)), obs=torch.round(data.pOIs[1, 1]))
            # religionVisitsObs = pyro.sample("S_Religion", dist.Poisson(torch.sum(torch.transpose(torch.div(data.alpha_paramReligion.repeat([data.NCBG, 1]), data.alpha_paramReligion.repeat([data.NCBG, 1]) + data.beta_paramReligion[g].repeat([data.NCBG, 1])), 0, 1) * BBNRel)), obs=torch.round(data.pOIs[2, 1]))

            # obsRaw = np.transpose(data.pOIs.iloc[:][1])
            # obs = torch.zeros(data.NE)
            # for i in range(data.NE):
            #     obs[i] = obsRaw.iloc[i]
            #     obs[i] = torch.div(obs[i],100)
            for i in range(globalError.shape[0]):
                if globalError[i] == 0:
                    shopAnalytical=torch.sum(torch.transpose(torch.div(data.alpha_paramShop.repeat([data.NCBG, 1]), data.alpha_paramShop.repeat([data.NCBG, 1]) + data.beta_paramShop[g].repeat([data.NCBG, 1])), 0, 1) * BBNSh)
                    schoolAnalytical = torch.sum(torch.transpose(torch.div(data.alpha_paramSchool.repeat([data.NCBG, 1]), data.alpha_paramSchool.repeat([data.NCBG, 1]) + data.beta_paramSchool[g].repeat([data.NCBG, 1])), 0, 1) * BBNSch)
                    religionAnalytical = torch.sum(torch.transpose(torch.div(data.alpha_paramReligion.repeat([data.NCBG, 1]), data.alpha_paramReligion.repeat([data.NCBG, 1]) + data.beta_paramReligion[g].repeat([data.NCBG, 1])), 0, 1) * BBNRel)
                    globalError[i] = torch.absolute(shopAnalytical - data.pOIs[0, 1]) + torch.absolute(schoolAnalytical - data.pOIs[1, 1]) + torch.absolute(religionAnalytical - data.pOIs[2, 1])
                    # print("within errors {}".format(globalError[i]))
                    break

            if data.isFirst == True:
                # obsRaw = np.transpose(data.pOIs.iloc[:][1])
                # obs = torch.zeros(data.NE)
                # for i in range(data.NE):
                #     obs[i] = obsRaw.iloc[i]
                #     obs[i] = torch.div(obs[i], 100)
                print("within errors {}".format(globalError[0]))
                data.isFirst = False

            return shopVisitsObs, schoolVisitsObs, religionVisitsObs

        def guide(data):
            temp_alpha_paramShop = torch.add(torch.zeros(data.G), 0.2)
            temp_alpha_paramShop[6] = 0.5
            temp_alpha_paramShop[7] = 0.5
            temp_alpha_paramShop[8] = 0.4
            temp_alpha_paramShop[9] = 0.4
            temp_beta_paramShop = torch.add(torch.zeros(data.G), 13.4)
            temp_beta_paramShop[6] = 11
            temp_beta_paramShop[7] = 11
            temp_beta_paramShop[8] = 12
            temp_beta_paramShop[9] = 12
            temp_alpha_paramSchool = torch.add(torch.zeros(data.G), 0.2)
            temp_alpha_paramSchool[0] = 0.4
            temp_alpha_paramSchool[1] = 0.4
            temp_alpha_paramSchool[5] = 0.4
            temp_alpha_paramSchool[8] = 0.4
            temp_beta_paramSchool = torch.add(torch.zeros(data.G), 13.4)
            temp_beta_paramSchool[0] = 12
            temp_beta_paramSchool[1] = 12
            temp_beta_paramSchool[5] = 12
            temp_beta_paramSchool[8] = 12
            temp_alpha_paramRel = torch.add(torch.zeros(data.G), 0.2)
            temp_alpha_paramRel[11] = 0.4
            temp_alpha_paramRel[14] = 0.4
            temp_beta_paramRel = torch.add(torch.zeros(data.G), 13.4)
            temp_beta_paramRel[11] = 12
            temp_beta_paramRel[14] = 12
            data.alpha_paramShop = pyro.param("alpha_paramShop_G", temp_alpha_paramShop, constraint=constraints.positive)
            data.beta_paramShop = pyro.param("beta_paramShop_G", temp_beta_paramShop, constraint=constraints.positive)
            data.alpha_paramSchool = pyro.param("alpha_paramSchool_G", temp_alpha_paramSchool, constraint=constraints.positive)
            data.beta_paramSchool = pyro.param("beta_paramSchool_G", temp_beta_paramSchool, constraint=constraints.positive)
            data.alpha_paramReligion = pyro.param("alpha_paramReligion_G", temp_alpha_paramRel, constraint=constraints.positive)
            data.beta_paramReligion = pyro.param("beta_paramReligion_G", temp_beta_paramRel, constraint=constraints.positive)

            # register prior parameter value. It'll be updated in the guide function
            # data.alpha_paramShop = pyro.param("alpha_paramShop_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive)
            # data.beta_paramShop = pyro.param("beta_paramShop_G", torch.add(torch.ones(data.G), 13.4), constraint=constraints.positive)
            # data.alpha_paramSchool = pyro.param("alpha_paramSchool_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive)
            # data.beta_paramSchool = pyro.param("beta_paramSchool_G", torch.add(torch.ones(data.G), 13.4), constraint=constraints.positive)
            # data.alpha_paramReligion = pyro.param("alpha_paramReligion_G", torch.add(torch.zeros(data.G), 0.2), constraint=constraints.positive)
            # data.beta_paramReligion = pyro.param("beta_paramReligion_G", torch.add(torch.ones(data.G), 13.4), constraint=constraints.positive)

            groupProbs = (torch.transpose(data.occupationProb, 0, 1) * data.ageProb).reshape([data.G, 1])
            needsSh = data.needsTensor[:, 0].reshape([data.G, 1])
            needsSch = data.needsTensor[:, 1].reshape([data.G, 1])
            needsRel = data.needsTensor[:, 2].reshape([data.G, 1])
            BBNSh = (needsSh * groupProbs * data.populationNum.repeat([data.G, 1])).round()
            BBNSch = (needsSch * groupProbs * data.populationNum.repeat([data.G, 1])).round()
            BBNRel = (needsRel * groupProbs * data.populationNum.repeat([data.G, 1])).round()
            BBNSh[BBNSh == 0] = 1
            BBNSch[BBNSch == 0] = 1
            BBNRel[BBNRel == 0] = 1
            with pyro.plate("NCBG", data.NCBG) as ncbg:
                with pyro.plate("G", data.G) as g:
                    pyro.sample("Tu_Shop", dist.BetaBinomial(torch.transpose(data.alpha_paramShop[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramShop[g].repeat([data.NCBG, 1]), 0, 1), BBNSh))
                    pyro.sample("Tu_School", dist.BetaBinomial(torch.transpose(data.alpha_paramSchool[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramSchool[g].repeat([data.NCBG, 1]), 0, 1), BBNSch))
                    pyro.sample("Tu_Religion", dist.BetaBinomial(torch.transpose(data.alpha_paramReligion[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramReligion[g].repeat([data.NCBG, 1]), 0, 1), BBNRel))

                    # pyro.sample("Tu_Shop", dist.BetaBinomial(torch.transpose(data.alpha_paramShop[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramShop[g].repeat([data.NCBG, 1]), 0, 1), 1900))
                    # pyro.sample("Tu_School", dist.BetaBinomial(torch.transpose(data.alpha_paramSchool[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramSchool[g].repeat([data.NCBG, 1]), 0, 1), 1900))
                    # pyro.sample("Tu_Religion", dist.BetaBinomial(torch.transpose(data.alpha_paramReligion[g].repeat([data.NCBG, 1]), 0, 1), torch.transpose(data.beta_paramReligion[g].repeat([data.NCBG, 1]), 0, 1), 1900))

            # \/\/\/ Individual multipiled to population
            # # with pyro.plate("NCBG", data.NCBG) as ncbg:
            # with pyro.plate("G", data.G) as g:
            #     # selAge = pyro.sample("age", dist.Categorical(data.ageProb))
            #     # selOccupation = pyro.sample("occupation", dist.Categorical(data.occupationProb[selAge[n], :]))
            #     # with pyro.plate("G", data.G) as g:
            #     with pyro.plate("NCBG", data.NCBG) as ncbg:
            #         # print(data.needsTensor[g][:, 0])
            #         # print(data.needsTensor[g][:, 1])
            #         # print(data.needsTensor[g][:, 2])
            #         pyro.sample("Tu_Shop", dist.BetaBinomial(torch.abs(data.alpha_paramShop), torch.abs(data.beta_paramShop), data.needsTensor[:, 0]))
            #         pyro.sample("Tu_School", dist.BetaBinomial(torch.abs(data.alpha_paramSchool), torch.abs(data.beta_paramSchool), data.needsTensor[:, 1]))
            #         pyro.sample("Tu_Religion", dist.BetaBinomial(torch.abs(data.alpha_paramReligion), torch.abs(data.beta_paramReligion), data.needsTensor[:, 2]))
            #
            # # TOO SLOW
            # # shopVisits = torch.zeros((data.G, data.NCBG))
            # # schoolVisits = torch.zeros((data.G, data.NCBG))
            # # religionVisits = torch.zeros((data.G, data.NCBG))
            # # for g in range(data.G):
            # #     for ncbg in range(data.NCBG):
            # #         shopVisits[g, ncbg] = pyro.sample("Tu_Shop" + str(g) + "_" + str(ncbg), dist.BetaBinomial(torch.abs(data.alpha_paramShop[g]), torch.abs(data.beta_paramShop[g]), data.needsTensor[g, 0]))
            # #         schoolVisits[g, ncbg] = pyro.sample("Tu_School" + str(g) + "_" + str(ncbg), dist.BetaBinomial(torch.abs(data.alpha_paramSchool[g]), torch.abs(data.beta_paramSchool[g]), data.needsTensor[g, 1]))
            # #         religionVisits[g, ncbg] = pyro.sample("Tu_Religion" + str(g) + "_" + str(ncbg), dist.BetaBinomial(torch.abs(data.alpha_paramReligion[g]), torch.abs(data.beta_paramReligion[g]), data.needsTensor[g, 2]))
            # ^^^ Individual multipiled to population

        # def makeTestConfig(test):
        #    jsonStr = json.dumps(test.__dict__)
        #    print(jsonStr)

        # testConfig= Config(0,0,[0],[1])
        # file = open("tucson_test1.conf", "w")
        # file.write(testConfig.toJSON())
        # file.close()

        pyro.clear_param_store()

        logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

        isReadConfigFile = True  # IF FALSE, MANUAL SELECTION IS ON
        isKFoldCrossVal = True
        configFileName = 'tucson_test1.conf'
        # I'LL ADD CONFIGURATION FILES TO AVOID RAPIDLY INPUTTING CITY AND MONTH
        if isReadConfigFile == True:
            file = open('tucson_test1.conf', 'r')

            configStr = file.read()
            retConfig = json.loads(configStr, object_hook=customConfigDecoder)

        cities = os.listdir('..' + os.sep + 'TimedData')

        for i in range(len(cities)):
            print("[{}] {}".format(i + 1, cities[i]))

        if isReadConfigFile == False:
            selectedTrainCity = input("Select train city")
            selectedTrainCityIndex = int(selectedTrainCity) - 1
            selectedTestCity = input("Select test city")
            selectedTestCityIndex = int(selectedTestCity) - 1
        else:
            selectedTrainCityIndex = retConfig.trainCityIndex
            print("City [{}] selected".format(selectedTrainCityIndex))
            selectedTestCityIndex = retConfig.testCityIndex
            print("City [{}] selected".format(selectedTestCityIndex))

        times = os.listdir('..' + os.sep + 'TimedData' + os.sep + cities[selectedTrainCityIndex])
        dates = set()
        for i in range(len(times)):
            noExtension = times[i].split(".")
            parts = noExtension[0].split("_")
            dates.add(parts[len(parts) - 1 - 1] + "_" + parts[len(parts) - 1])

        print("Available dates")
        dates = sorted(dates)
        for i in range(len(dates)):
            print("[{}] {}".format(i + 1, dates[i]))

        if isReadConfigFile == False:
            selectedTrainRange = input("Select train range (, and - ranges max two digits i.e. 1,3-5,21: 1,3,4,5,21)")
            selectedTestRange = input("Select test range (, and - ranges max two digits i.e. 1,3-5,21: 1,3,4,5,21)")

            selectedTrainRangeIndices = processInputRanges(selectedTrainRange)
            selectedTestRangeIndices = processInputRanges(selectedTestRange)

            print(selectedTrainRangeIndices)
            print(selectedTestRangeIndices)
        else:
            selectedTrainRangeIndices = retConfig.trainTimeRange
            selectedTestRangeIndices = retConfig.testTimeRange
            print("Selected train date range {}".format(selectedTrainRangeIndices))
            print("Selected test date range {}".format(selectedTestRangeIndices))

        allData = loadData(cities[selectedTrainCityIndex], cities[selectedTestCityIndex], dates, selectedTrainRangeIndices, selectedTestRangeIndices)

        needsVerbose = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + 'Needs_data_numbers.csv')

        showAlphaBetaRange(None, allData.trainData.monthlyData[0].alpha_paramShop, allData.trainData.monthlyData[0].beta_paramShop, needsVerbose)

        # graph = pyro.render_model(model, model_args=(allData.trainData.monthlyData[0],), render_distributions=True, render_params=True)
        # graph.view()

        # setup the optimizer
        # adam_params = {"lr": lr, "betas": (0.9, 0.999), "maximize": False}
        # optimizer = Adam(adam_params)

        # asgd_params = {"lr": lr, "maximize": False}
        # optimizer = ASGD(asgd_params)

        adagrad_params = {"lr": lr, "maximize": False, "lr_decay":0.2}
        optimizer = Adagrad(adagrad_params)

        # radam_params = {"lr": lr, "betas": (0.6, 0.9)}
        # optimizer = RAdam(radam_params)

        # exponentialLR_params = {"gamma ": 0.01}
        # optimizer = ExponentialLR(exponentialLR_params)

        # rprop_params = {}
        # optimizer = Rprop(rprop_params)

        # adamW_params = {"lr": lr, "betas": (0.8, 0.9), "maximize": False, "weight_decay": 0.01}
        # optimizer = AdamW(adamW_params)

        # adadelta_params = {}
        # optimizer = Adadelta(adadelta_params)

        # sgd_params = {"lr": lr}
        # optimizer = SGD(sgd_params)

        # auto_guide = AutoDiscreteParallel(model)

        if elboType == 'Trace_ELBO':
             Elbo=Trace_ELBO
             elbo = Elbo(num_particles=numParticles)

        # Elbo = TraceTailAdaptive_ELBO
        # elbo = Elbo(num_particles=5, vectorize_particles=True)

        # Elbo = TraceGraph_ELBO
        # elbo = Elbo(num_particles=5)

        if elboType == 'RenyiELBO':
            Elbo = RenyiELBO
            elbo = Elbo(alpha=alpha, num_particles=numParticles)

        # Elbo = TraceMeanField_ELBO
        # elbo = Elbo(num_particles=5)

        # Elbo = TraceTMC_ELBO
        # elbo = Elbo(num_particles=5)

        # setup the inference algorithm
        svi = SVI(model, guide, optimizer, loss=elbo)
        globalError = np.zeros(numParticles, dtype=np.int32)

        # svi.num_chains=1

        if elboType == 'RenyiELBO':
            extraMessage = elboType + "_alpha" + str(alpha) + "_numParticle" + str(numParticles) + "_lr" + str(lr)
        else:
            extraMessage = elboType + "_numParticle" + str(numParticles) + "_lr" + str(lr)

        if retConfig.isKFoldCrossVal == 1:
            runCrossVal(svi, elbo, model, guide, allData.testData.monthlyData, globalError, dates, cities[selectedTestCityIndex])
        else:
            loss = elbo.loss(model, guide, allData.trainData.monthlyData[0])
            logging.info("first loss train SantaFe = {}".format(loss))

            n_steps = 3000
            error_tolerance = 1

            losses = []
            maxErrors = []

            plt.figure("loss fig online")

            # do gradient steps
            for step in range(n_steps):
                self.stepValue=step
                loss = svi.step(allData.trainData.monthlyData[0])
                maxError = np.max(np.absolute(globalError))
                losses.append(loss)
                maxErrors.append(maxError)

                plt.cla()
                plt.plot(maxErrors[-50:])
                plt.pause(0.01)

                # print("maxError {}".format(maxError))
                globalError = np.zeros(numParticles, dtype=np.int32)
                # svi.run()
                if step % 100 == 0:
                    logging.info("{: >5d}\t{}".format(step, loss))
                    print("maxError {}".format(maxError))
                    # print('.', end='')
                    # for name in pyro.get_param_store():
                    #     value = pyro.param(name)
                    #     print("{} = {}".format(name, value.detach().cpu().numpy()))
                if maxError <= error_tolerance:
                    break


            # FOR DEBUGGING
            # plt.plot(allData.trainData.monthlyData[0].resultFromAvgAllIP,label="numeric mean one indiv with small N multiplied to pop")
            # plt.plot(allData.trainData.monthlyData[0].resultFromAvgAllIB, label="numeric mean one indiv with large N")
            # plt.plot(allData.trainData.monthlyData[0].resultFromEEAll, label="expectation value")
            # plt.legend()
            # plt.title('Difference between numeric average and expectation for the case one person is made in CBG for a group and it is multiplied to the entire CBG population')
            # plt.show()
            #
            # plt.figure()
            # plt.plot(allData.trainData.monthlyData[0].resultSamplesIP)
            # plt.title('Samples from individual map to population')
            # plt.show()
            #
            # plt.figure()
            # plt.plot(allData.trainData.monthlyData[0].resultSamplesIB)
            # plt.title('Samples from big individual')
            # plt.show()
            # FOR DEBUGGING


            # print(os.path.dirname(__file__))
            now = datetime.now()
            dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
            plt.figure("loss fig")
            plt.plot(losses)
            plt.savefig(os.path.dirname(__file__) + os.sep + 'tests' + os.sep + 'loss_' + dt_string + '_' + cities[selectedTrainCityIndex] + '_' + extraMessage + '.png')
            loss_path=os.path.dirname(__file__) + os.sep + 'tests' + os.sep + 'loss_' + '_' + cities[selectedTrainCityIndex] + '_' + extraMessage
            plt.figure("error fig")
            plt.plot(maxErrors)
            plt.savefig(os.path.dirname(__file__) + os.sep + 'tests' + os.sep + 'error_' + dt_string + '_' + cities[selectedTrainCityIndex] + '_' + extraMessage + '.png')
            error_path=os.path.dirname(__file__) + os.sep + 'tests' + os.sep + 'error_' + '_' + cities[selectedTrainCityIndex] + '_' + extraMessage

            with open('tests' + os.sep + 'losses_M1CBG_{}_{}_{}.csv'.format(dt_string, cities[selectedTrainCityIndex], extraMessage), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(losses)):
                    writer.writerow([losses[i]])
            with open('tests' + os.sep + 'errors_M1CBG_{}_{}_{}.csv'.format(dt_string, cities[selectedTrainCityIndex], extraMessage), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(maxErrors)):
                    writer.writerow([maxErrors[i]])

            print("Final evalulation")

            # allDataTrain = loadData(cities[selectedTrainCityIndex], dates, times[selectedTrainRangeIndices])
            allData.trainData.monthlyData[0].isFirst = True
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
                globalError = np.zeros(numParticles, dtype=np.int32)
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
            # print('^^^'+loss_path)
            # print('^^^' + error_path)
            retVals.append([losses,maxErrors,loss_path,error_path])

if __name__ ==  '__main__':
    matplotlib.use("Qt5agg")
    numCPUs = 1
    numTests = 1
    manager = multiprocessing.Manager()
    retVals = manager.list()

    # retVals = []
    p_losses = []
    p_maxErrors = []
    tests = []
    processes = []
    pool = multiprocessing.Pool(processes=numCPUs)

    # for i in range(numTests):
    #     test = Test()
    #     tests.append(test)
    #     # p = multiprocessing.Process(target=test.run, args=(3, 0.001, "RenyiELBO", 0.2, i, retVals))
    #     test.run(3, 0.001, "RenyiELBO", 0.2, i, retVals)
    #     # p.start()
    #     # processes.append(p)
    #     print('RUN: '+str(i))

    for i in range(numTests):
        test = Test()
        tests.append(test)
        # print('!!!123')
        p = multiprocessing.Process(target=test.run,args=(5, 0.8, "RenyiELBO", 0.2,i,retVals))
        p.start()
        processes.append(p)
        # print('!!!')

    for i in range(numTests):
        print('!!!WAITING')
        processes[i].join()

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.figure("loss fig")
    for i in range(len(retVals)):
        # print('!123'+str(retVals[i][0]))
        # print('!1234'+retVals[i][2] + '_' + dt_string + '_summary' + '.png')
        plt.plot(retVals[i][0])
    plt.savefig(retVals[0][2] + '_' + dt_string + '_summary' + '.png')
    plt.figure("error fig")
    for i in range(len(retVals)):
        # print('!123'+str(retVals[i][1]))
        # print('!1234'+retVals[i][3] + '_' + dt_string + '_summary' + '.png')
        plt.plot(retVals[i][1])
    plt.savefig(retVals[0][3] + '_' + dt_string + '_summary' + '.png')

    print('!!!')
    # test1=Test()
    # #test.run(numParticles=1,lr=0.001,elboType="RenyiELBO",alpha=0.2)
    # p1 = multiprocessing.Process(target=test1.run(numParticles=1,lr=0.001,elboType="RenyiELBO",alpha=0.2))

