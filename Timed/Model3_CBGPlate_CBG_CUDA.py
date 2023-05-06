#   This model assumes that each pair of age-occupation group has needs and one alpha and one beta. The alpha and beta determines how frequently
# this group attends different POIs. This means that individuals can't compensate their attendance because the alpha and beta are shared for a group.
# This model distinguished the type of visits and the POIs are distinguished.
# - Individual level samples (latent variable)
# - Age-Occupation level parameters
# - POI distinguished visits

from line_profiler import LineProfiler
from line_profiler_pycharm import profile
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
        # globalError = np.zeros(1, dtype=np.int32)
        self.index=i

        @profile
        def model(data):
            # alphaShopReady = torch.transpose(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # betaShopReady = torch.transpose(data[0].beta_paramShop.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # alphaSchoolReady = torch.transpose(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # betaSchoolReady = torch.transpose(data[0].beta_paramSchool.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # alphaRelReady = torch.transpose(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # betaRelReady = torch.transpose(data[0].beta_paramReligion.repeat([data[0].NCBG, 1]), 0, 1).cuda()

            temp_alpha_paramShop = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramShop[6] = 0.5
            temp_alpha_paramShop[7] = 0.5
            temp_alpha_paramShop[8] = 0.4
            temp_alpha_paramShop[9] = 0.4
            temp_beta_paramShop = torch.add(torch.zeros(data[0].G), 1.4)
            temp_beta_paramShop[6] = 2.1
            temp_beta_paramShop[7] = 2.1
            temp_beta_paramShop[8] = 2.2
            temp_beta_paramShop[9] = 2.2
            temp_alpha_paramSchool = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramSchool[0] = 0.4
            temp_alpha_paramSchool[1] = 0.4
            temp_alpha_paramSchool[5] = 0.4
            temp_alpha_paramSchool[8] = 0.4
            temp_beta_paramSchool = torch.add(torch.zeros(data[0].G), 1.4)
            temp_beta_paramSchool[0] = 2.2
            temp_beta_paramSchool[1] = 2.2
            temp_beta_paramSchool[5] = 2.2
            temp_beta_paramSchool[8] = 2.2
            temp_alpha_paramRel = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramRel[11] = 0.4
            temp_alpha_paramRel[14] = 0.4
            temp_beta_paramRel = torch.add(torch.zeros(data[0].G), 1.4)
            temp_beta_paramRel[11] = 1
            temp_beta_paramRel[14] = 1
            data[0].alpha_paramShop = pyro.param("alpha_paramShop_G", temp_alpha_paramShop, constraint=constraints.positive).cuda()
            data[0].beta_paramShop = pyro.param("beta_paramShop_G", temp_beta_paramShop, constraint=constraints.positive).cuda()
            data[0].alpha_paramSchool = pyro.param("alpha_paramSchool_G", temp_alpha_paramSchool, constraint=constraints.positive).cuda()
            data[0].beta_paramSchool = pyro.param("beta_paramSchool_G", temp_beta_paramSchool, constraint=constraints.positive).cuda()
            data[0].alpha_paramReligion = pyro.param("alpha_paramReligion_G", temp_alpha_paramRel, constraint=constraints.positive).cuda()
            data[0].beta_paramReligion = pyro.param("beta_paramReligion_G", temp_beta_paramRel, constraint=constraints.positive).cuda()

            data[0].gapParamShop = pyro.param("gapShop_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamSchool = pyro.param("gapSchool_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamRel = pyro.param("gapRel_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamShopFrac = pyro.param("gapShopFrac_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamSchoolFrac = pyro.param("gapSchoolFrac_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamRelFrac = pyro.param("gapRelFrac_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].multiVisitVarShParam = pyro.param("multiVisitVarSh_Param", torch.ones(1),constraint=constraints.positive).cuda()
            data[0].multiVisitVarSchParam = pyro.param("multiVisitVarSch_Param", torch.ones(1),constraint=constraints.positive).cuda()
            data[0].multiVisitVarRelParam = pyro.param("multiVisitVarRel_Param", torch.ones(1),constraint=constraints.positive).cuda()

            with pyro.plate("NCBG", data[0].NCBG) as ncbg:
                with pyro.plate("G", data[0].G) as g:
                    newSumShopFinal = []
                    newSumSchoolFinal = []
                    newSumRelFinal = []
                    obsPOIShops = []
                    obsPOISchools = []
                    obsPOIRels = []
                    for i in pyro.plate("visitObs_plate", len(data)):
                        obsPOIShops.append(data[i].pOIShops.flatten())
                        obsPOISchools.append(data[i].pOISchools.flatten())
                        obsPOIRels.append(data[i].pOIReligion.flatten())
                        with pyro.plate("Nshop_{}".format(i), data[i].Nshop) as nshop:
                            shopVisits = pyro.sample("Tu_Shop_{}".format(i), dist.BetaBinomial(
                                torch.transpose(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), 0, 1),
                                torch.transpose(data[0].beta_paramShop.repeat([data[0].NCBG, 1]), 0, 1),
                                data[0].BBNSh)).cuda()

                            newShopVisits = torch.mul(shopVisits.sum(1), data[i].cBGShopProb[:, ncbg])
                            diff = shopVisits.sum() - newShopVisits.sum()
                            shopVisits = torch.add(newShopVisits, torch.div(diff, newShopVisits.size(
                                dim=0) * newShopVisits.size(dim=1)))

                            shopVisitsPOIs = shopVisits.sum(1).cuda()
                            shopVisitsPOIsAdjusted = torch.mul(shopVisitsPOIs, data[i].pOIShopProb)
                            expectedSum = shopVisitsPOIs.sum().cuda()
                            newSum = shopVisitsPOIsAdjusted.sum().cuda()
                            diff = expectedSum - newSum
                            newSumShop = torch.add(shopVisitsPOIsAdjusted,
                                                   torch.div(diff, shopVisitsPOIs.size(dim=0))).cuda()
                            newSumShopFinal.append(newSumShop)

                        with pyro.plate("Nschool_{}".format(i), data[i].Nschool) as nschool:
                            schoolVisits = pyro.sample("Tu_School_{}".format(i), dist.BetaBinomial(
                                torch.transpose(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), 0, 1),
                                torch.transpose(data[0].beta_paramSchool.repeat([data[0].NCBG, 1]), 0, 1),
                                data[0].BBNSch)).cuda()

                            newSchoolVisits = torch.mul(schoolVisits.sum(1), data[i].cBGSchoolProb[:, ncbg])
                            diff = schoolVisits.sum() - newSchoolVisits.sum()
                            schoolVisits = torch.add(newSchoolVisits, torch.div(diff, newSchoolVisits.size(
                                dim=0) * newSchoolVisits.size(dim=1)))

                            schoolVisitsPOIs = schoolVisits.sum(1)
                            schoolVisitsPOIsAdjusted = torch.mul(schoolVisitsPOIs, data[i].pOISchoolProb)
                            expectedSum = schoolVisitsPOIs.sum().cuda()
                            newSum = schoolVisitsPOIsAdjusted.sum().cuda()
                            diff = expectedSum - newSum
                            newSumSchool = torch.add(schoolVisitsPOIsAdjusted,
                                                     torch.div(diff, schoolVisitsPOIs.size(dim=0))).cuda()
                            newSumSchoolFinal.append(newSumSchool)

                        with pyro.plate("Nreligion_{}".format(i), data[i].Nreligion) as nreligion:
                            religionVisits = pyro.sample("Tu_Religion_{}".format(i), dist.BetaBinomial(
                                torch.transpose(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), 0, 1),
                                torch.transpose(data[0].beta_paramReligion.repeat([data[0].NCBG, 1]), 0, 1),
                                data[0].BBNRel)).cuda()

                            newReligionVisits = torch.mul(religionVisits.sum(1), data[i].cBGReligionProb[:, ncbg])
                            diff = religionVisits.sum() - newReligionVisits.sum()
                            religionVisits = torch.add(newReligionVisits, torch.div(diff, newReligionVisits.size(
                                dim=0) * newReligionVisits.size(dim=1)))

                            religionVisitsPOIs = religionVisits.sum(1)
                            religionVisitsPOIsAdjusted = torch.mul(religionVisitsPOIs, data[i].pOIReligionProb)
                            expectedSum = religionVisitsPOIs.sum().cuda()
                            newSum = religionVisitsPOIsAdjusted.sum().cuda()
                            diff = expectedSum - newSum
                            newSumRel = torch.add(religionVisitsPOIsAdjusted,
                                                  torch.div(diff, religionVisitsPOIs.size(dim=0))).cuda()
                            newSumRelFinal.append(newSumRel)
            if data[0].isFirst == True:
                np.savetxt('CBGShopVisitsM3.csv', shopVisits.sum(1).cpu().numpy(), delimiter=',')
                np.savetxt('CBGSchoolVisitsM3.csv', schoolVisits.sum(1).cpu().numpy(), delimiter=',')
                np.savetxt('CBGRelVisitsM3.csv', religionVisits.sum(1).cpu().numpy(), delimiter=',')

            with pyro.plate('observe_data'):
                shopVisitsObs = pyro.sample("S_Shop",dist.Poisson(torch.cat(newSumShopFinal) / data[0].gapParamShop).to_event(1),obs=torch.cat(obsPOIShops))
                schoolVisitsObs = pyro.sample("S_School", dist.Poisson(torch.cat(newSumSchoolFinal) / data[0].gapParamSchool).to_event(1), obs=torch.cat(obsPOISchools))
                religionVisitsObs = pyro.sample("S_Religion",dist.Poisson(torch.cat(newSumRelFinal) / data[0].gapParamRel).to_event(1), obs=torch.cat(obsPOIRels))

            temp = []
            for i in range(len(data)):
                temp.append(data[i].shopFrac)
            trainShopFracObs = torch.Tensor(temp).cuda()
            temp = []
            for i in range(len(data)):
                temp.append(data[i].schoolFrac)
            trainSchoolFracObs = torch.Tensor(temp).cuda()
            temp = []
            for i in range(len(data)):
                temp.append(data[i].relFrac)
            trainRelFracObs = torch.Tensor(temp).cuda()
            with pyro.plate('observed_fracs'):
                alphaShop_nonZero = torch.gather(data[0].alpha_paramShop, 0, data[0].nonZeroNeedsShopIndices)
                betaShop_nonZero = torch.gather(data[0].beta_paramShop, 0, data[0].nonZeroNeedsShopIndices)
                alphaSchool_nonZero = torch.gather(data[0].alpha_paramSchool, 0, data[0].nonZeroNeedsSchoolIndices)
                betaSchool_nonZero = torch.gather(data[0].beta_paramSchool, 0, data[0].nonZeroNeedsSchoolIndices)
                alphaRel_nonZero = torch.gather(data[0].alpha_paramReligion, 0, data[0].nonZeroNeedsRelIndices)
                betaRel_nonZero = torch.gather(data[0].beta_paramReligion, 0, data[0].nonZeroNeedsRelIndices)
                # shopMultiVisit = torch.maximum(torch.gather(data[0].oneAuxVal, 0, data[0].nonZeroNeedsShopIndices), (alphaShop_nonZero / (alphaShop_nonZero + betaShop_nonZero)) * (data[0].nonZeroNeedsShopIndices)).cuda()
                # schoolMultiVisit = torch.maximum(torch.gather(data[0].oneAuxVal, 0, data[0].nonZeroNeedsSchoolIndices), (alphaSchool_nonZero / (alphaSchool_nonZero + betaSchool_nonZero)) * (data[0].nonZeroNeedsSchoolIndices)).cuda()
                # relMultiVisit = torch.maximum(torch.gather(data[0].oneAuxVal, 0, data[0].nonZeroNeedsRelIndices), (alphaRel_nonZero / (alphaRel_nonZero + betaRel_nonZero)) * (data[0].nonZeroNeedsRelIndices)).cuda()
                shopMultiVisit = ((alphaShop_nonZero / (alphaShop_nonZero + betaShop_nonZero)) * (
                data[0].needsTensor[:, 0][data[0].nonZeroNeedsShopIndices])).cuda()
                schoolMultiVisit = ((alphaSchool_nonZero / (alphaSchool_nonZero + betaSchool_nonZero)) * (
                data[0].needsTensor[:, 1][data[0].nonZeroNeedsSchoolIndices])).cuda()
                relMultiVisit = ((alphaRel_nonZero / (alphaRel_nonZero + betaRel_nonZero)) * (
                data[0].needsTensor[:, 2][data[0].nonZeroNeedsRelIndices])).cuda()
                if data[0].isTrainedOnOneMonth == 1:
                    pyro.sample("M_Shop", dist.Poisson(shopMultiVisit.mean() / data[0].gapParamShopFrac),obs=trainShopFracObs)
                    pyro.sample("M_School", dist.Poisson(schoolMultiVisit.mean() / data[0].gapParamSchoolFrac),obs=trainSchoolFracObs)
                    pyro.sample("M_Religion", dist.Poisson(relMultiVisit.mean() / data[0].gapParamRelFrac),obs=trainRelFracObs)
                else:
                    pyro.sample("M_Shop", dist.Normal(shopMultiVisit.mean() / data[0].gapParamShopFrac,data[0].multiVisitVarShParam), obs=trainShopFracObs)
                    pyro.sample("M_School", dist.Normal(schoolMultiVisit.mean() / data[0].gapParamSchoolFrac,data[0].multiVisitVarSchParam), obs=trainSchoolFracObs)
                    pyro.sample("M_Religion", dist.Normal(relMultiVisit.mean() / data[0].gapParamRelFrac,data[0].multiVisitVarRelParam), obs=trainRelFracObs)


            # shopVisitsObs=0
            # schoolVisitsObs=0
            # religionVisitsObs=0

            # obsRaw = np.transpose(data.pOIs.iloc[:][1])
            # obs = torch.zeros(data.NE)
            # for i in range(data.NE):
            #     obs[i] = obsRaw.iloc[i]
            #     obs[i] = torch.div(obs[i],100)
            # print(torch.mul(torch.abs(shopVisits.sum(1).sum(1)), data[0].pOIShopProb).sum() - data[0].pOIShops.sum() + torch.mul(torch.abs(schoolVisits.sum(1).sum(1)), data[0].pOISchoolProb).sum() - data[0].pOISchools.sum() + torch.mul(torch.abs(religionVisits.sum(1).sum(1)), data[0].pOIReligionProb).sum() - data[0].pOIReligion.sum())

            for i in range(data[0].globalErrorFrac.shape[0]):
                if data[0].globalErrorFrac[i] == 0:
                    data[0].globalErrorFrac[i] = torch.absolute(
                        shopMultiVisit.mean() / data[0].gapParamShopFrac - trainShopFracObs.mean()) + torch.absolute(
                        schoolMultiVisit.mean() / data[0].gapParamSchoolFrac - trainSchoolFracObs.mean()) + torch.absolute(
                        relMultiVisit.mean() / data[0].gapParamRelFrac - trainRelFracObs.mean())
                    # print("within errors {}".format(globalError[i]))
                    break

            for i in range(data[0].globalError.shape[0]):
                if data[0].globalError[i] == 0:
                    # data[0].globalError[i] = torch.absolute(((newSumShopFinal.sum()/data[0].gapParamShop) - obsPOIShops.sum()) / (data[0].pOIs[0, 1])) + torch.absolute(((newSumSchool.sum()/data[0].gapParamSchool) - data[0].pOIs[1, 1]) / (data[0].pOIs[1, 1])) + torch.absolute(((newSumRel.sum()/data[0].gapParamRel) - data[0].pOIs[2, 1]) / (data[0].pOIs[2, 1]))
                    shopDiff = torch.absolute(((torch.cat(newSumShopFinal).sum() / data[0].gapParamShop) - torch.cat(obsPOIShops).sum()) / (torch.cat(obsPOIShops).sum()))
                    schoolDiff = torch.absolute(((torch.cat(newSumSchoolFinal).sum() / data[0].gapParamSchool) - torch.cat(obsPOISchools).sum()) / (torch.cat(obsPOISchools).sum()))
                    relDiff = torch.absolute(((torch.cat(newSumRelFinal).sum() / data[0].gapParamRel) - torch.cat(obsPOIRels).sum()) / (torch.cat(obsPOIRels).sum()))
                    data[0].globalError[i] = torch.absolute(shopDiff) + torch.absolute(schoolDiff) + torch.absolute(relDiff)
                    # print("within errors {}".format(globalError[i]))
                    break

            if data[0].isFirst == True:
                print("shopMultiVisit {} observed {}".format(shopMultiVisit.mean(), trainShopFracObs.mean()))
                print("schoolMultiVisit {} observed {}".format(schoolMultiVisit.mean(), trainSchoolFracObs.mean()))
                print("relMultiVisit {} observed {}".format(relMultiVisit.mean(), trainRelFracObs.mean()))
                print("gapParamShop {}".format(data[0].gapParamShop))
                print("gapParamSchool {}".format(data[0].gapParamSchool))
                print("gapParamRel {}".format(data[0].gapParamRel))
                data[0].isFirst = False

            # for name,param in list(locals().items()):
            #     if isinstance(param,torch.Tensor):
            #         print(name,param.device)

            return shopVisitsObs, schoolVisitsObs, religionVisitsObs

        @profile
        def guide(data):
            temp_alpha_paramShop = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramShop[6] = 0.5
            temp_alpha_paramShop[7] = 0.5
            temp_alpha_paramShop[8] = 0.4
            temp_alpha_paramShop[9] = 0.4
            temp_beta_paramShop = torch.add(torch.zeros(data[0].G), 1.4)
            temp_beta_paramShop[6] = 2.1
            temp_beta_paramShop[7] = 2.1
            temp_beta_paramShop[8] = 2.2
            temp_beta_paramShop[9] = 2.2
            temp_alpha_paramSchool = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramSchool[0] = 0.4
            temp_alpha_paramSchool[1] = 0.4
            temp_alpha_paramSchool[5] = 0.4
            temp_alpha_paramSchool[8] = 0.4
            temp_beta_paramSchool = torch.add(torch.zeros(data[0].G), 1.4)
            temp_beta_paramSchool[0] = 2.2
            temp_beta_paramSchool[1] = 2.2
            temp_beta_paramSchool[5] = 2.2
            temp_beta_paramSchool[8] = 2.2
            temp_alpha_paramRel = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramRel[11] = 0.4
            temp_alpha_paramRel[14] = 0.4
            temp_beta_paramRel = torch.add(torch.zeros(data[0].G), 1.4)
            temp_beta_paramRel[11] = 1
            temp_beta_paramRel[14] = 1
            data[0].alpha_paramShop = pyro.param("alpha_paramShop_G", temp_alpha_paramShop, constraint=constraints.positive).cuda()
            data[0].beta_paramShop = pyro.param("beta_paramShop_G", temp_beta_paramShop, constraint=constraints.positive).cuda()
            data[0].alpha_paramSchool = pyro.param("alpha_paramSchool_G", temp_alpha_paramSchool, constraint=constraints.positive).cuda()
            data[0].beta_paramSchool = pyro.param("beta_paramSchool_G", temp_beta_paramSchool, constraint=constraints.positive).cuda()
            data[0].alpha_paramReligion = pyro.param("alpha_paramReligion_G", temp_alpha_paramRel, constraint=constraints.positive).cuda()
            data[0].beta_paramReligion = pyro.param("beta_paramReligion_G", temp_beta_paramRel, constraint=constraints.positive).cuda()

            data[0].gapParamShop = pyro.param("gapShop_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamSchool = pyro.param("gapSchool_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamRel = pyro.param("gapRel_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamShopFrac = pyro.param("gapShopFrac_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamSchoolFrac = pyro.param("gapSchoolFrac_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].gapParamRelFrac = pyro.param("gapRelFrac_param", torch.ones(1).add(1),constraint=constraints.positive).cuda()
            data[0].multiVisitVarShParam = pyro.param("multiVisitVarSh_Param", torch.ones(1),constraint=constraints.positive).cuda()
            data[0].multiVisitVarSchParam = pyro.param("multiVisitVarSch_Param", torch.ones(1),constraint=constraints.positive).cuda()
            data[0].multiVisitVarRelParam = pyro.param("multiVisitVarRel_Param", torch.ones(1),constraint=constraints.positive).cuda()

            # groupProbs = (torch.transpose(data[0].occupationProb, 0, 1) * data[0].ageProb).reshape([data[0].G, 1])
            # needsSh = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0], data[0].occupationProb.shape[1]]).transpose(0, 1).reshape([data[0].G, 1])
            # needsSch = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0], data[0].occupationProb.shape[1]]).transpose(0, 1).reshape([data[0].G, 1])
            # needsRel = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0], data[0].occupationProb.shape[1]]).transpose(0, 1).reshape([data[0].G, 1])
            # data[0].BBNSh = (needsSh * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            # data[0].BBNSch = (needsSch * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            # data[0].BBNRel = (needsRel * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            # data[0].BBNSh[data[0].BBNSh == 0] = 1
            # data[0].BBNSch[data[0].BBNSch == 0] = 1
            # data[0].BBNRel[data[0].BBNRel == 0] = 1

            # alphaShopReady = torch.transpose(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # betaShopReady = torch.transpose(data[0].beta_paramShop.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # alphaSchoolReady = torch.transpose(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # betaSchoolReady = torch.transpose(data[0].beta_paramSchool.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # alphaRelReady = torch.transpose(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            # betaRelReady = torch.transpose(data[0].beta_paramReligion.repeat([data[0].NCBG, 1]), 0, 1).cuda()

            with pyro.plate("NCBG", data[0].NCBG) as ncbg:
                with pyro.plate("G", data[0].G) as g:
                    newSumShopFinal = []
                    newSumSchoolFinal = []
                    newSumRelFinal = []
                    obsPOIShops = []
                    obsPOISchools = []
                    obsPOIRels = []
                    for i in pyro.plate("visitObs_plate", len(data)):
                        obsPOIShops.append(data[i].pOIShops.flatten())
                        obsPOISchools.append(data[i].pOISchools.flatten())
                        obsPOIRels.append(data[i].pOIReligion.flatten())
                        with pyro.plate("Nshop_{}".format(i), data[i].Nshop) as nshop:
                            shopVisits = pyro.sample("Tu_Shop_{}".format(i), dist.BetaBinomial(
                                torch.transpose(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), 0, 1),
                                torch.transpose(data[0].beta_paramShop.repeat([data[0].NCBG, 1]), 0, 1),
                                data[0].BBNSh)).cuda()

                            newShopVisits = torch.mul(shopVisits.sum(1), data[i].cBGShopProb[:, ncbg])
                            diff = shopVisits.sum() - newShopVisits.sum()
                            shopVisits = torch.add(newShopVisits, torch.div(diff, newShopVisits.size(
                                dim=0) * newShopVisits.size(dim=1)))

                            shopVisitsPOIs = shopVisits.sum(1).cuda()
                            shopVisitsPOIsAdjusted = torch.mul(shopVisitsPOIs, data[i].pOIShopProb)
                            expectedSum = shopVisitsPOIs.sum().cuda()
                            newSum = shopVisitsPOIsAdjusted.sum().cuda()
                            diff = expectedSum - newSum
                            newSumShop = torch.add(shopVisitsPOIsAdjusted,
                                                   torch.div(diff, shopVisitsPOIs.size(dim=0))).cuda()
                            newSumShopFinal.append(newSumShop)

                        with pyro.plate("Nschool_{}".format(i), data[i].Nschool) as nschool:
                            schoolVisits = pyro.sample("Tu_School_{}".format(i), dist.BetaBinomial(
                                torch.transpose(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), 0, 1),
                                torch.transpose(data[0].beta_paramSchool.repeat([data[0].NCBG, 1]), 0, 1),
                                data[0].BBNSch)).cuda()

                            newSchoolVisits = torch.mul(schoolVisits.sum(1), data[i].cBGSchoolProb[:, ncbg])
                            diff = schoolVisits.sum() - newSchoolVisits.sum()
                            schoolVisits = torch.add(newSchoolVisits, torch.div(diff, newSchoolVisits.size(
                                dim=0) * newSchoolVisits.size(dim=1)))

                            schoolVisitsPOIs = schoolVisits.sum(1)
                            schoolVisitsPOIsAdjusted = torch.mul(schoolVisitsPOIs, data[i].pOISchoolProb)
                            expectedSum = schoolVisitsPOIs.sum().cuda()
                            newSum = schoolVisitsPOIsAdjusted.sum().cuda()
                            diff = expectedSum - newSum
                            newSumSchool = torch.add(schoolVisitsPOIsAdjusted,
                                                     torch.div(diff, schoolVisitsPOIs.size(dim=0))).cuda()
                            newSumSchoolFinal.append(newSumSchool)

                        with pyro.plate("Nreligion_{}".format(i), data[i].Nreligion) as nreligion:
                            religionVisits = pyro.sample("Tu_Religion_{}".format(i), dist.BetaBinomial(
                                torch.transpose(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), 0, 1),
                                torch.transpose(data[0].beta_paramReligion.repeat([data[0].NCBG, 1]), 0, 1),
                                data[0].BBNRel)).cuda()

                            newReligionVisits = torch.mul(religionVisits.sum(1), data[i].cBGReligionProb[:, ncbg])
                            diff = religionVisits.sum() - newReligionVisits.sum()
                            religionVisits = torch.add(newReligionVisits, torch.div(diff, newReligionVisits.size(
                                dim=0) * newReligionVisits.size(dim=1)))

                            religionVisitsPOIs = religionVisits.sum(1)
                            religionVisitsPOIsAdjusted = torch.mul(religionVisitsPOIs, data[i].pOIReligionProb)
                            expectedSum = religionVisitsPOIs.sum().cuda()
                            newSum = religionVisitsPOIsAdjusted.sum().cuda()
                            diff = expectedSum - newSum
                            newSumRel = torch.add(religionVisitsPOIsAdjusted,
                                                  torch.div(diff, religionVisitsPOIs.size(dim=0))).cuda()
                            newSumRelFinal.append(newSumRel)

            with pyro.plate('observe_data'):
                pyro.sample("S_Shop",dist.Poisson(torch.cat(newSumShopFinal) / data[0].gapParamShop).to_event(1))
                pyro.sample("S_School", dist.Poisson(torch.cat(newSumSchoolFinal) / data[0].gapParamSchool).to_event(1))
                pyro.sample("S_Religion",dist.Poisson(torch.cat(newSumRelFinal) / data[0].gapParamRel).to_event(1))

            with pyro.plate('observed_fracs'):
                alphaShop_nonZero = torch.gather(data[0].alpha_paramShop, 0, data[0].nonZeroNeedsShopIndices)
                betaShop_nonZero = torch.gather(data[0].beta_paramShop, 0, data[0].nonZeroNeedsShopIndices)
                alphaSchool_nonZero = torch.gather(data[0].alpha_paramSchool, 0, data[0].nonZeroNeedsSchoolIndices)
                betaSchool_nonZero = torch.gather(data[0].beta_paramSchool, 0, data[0].nonZeroNeedsSchoolIndices)
                alphaRel_nonZero = torch.gather(data[0].alpha_paramReligion, 0, data[0].nonZeroNeedsRelIndices)
                betaRel_nonZero = torch.gather(data[0].beta_paramReligion, 0, data[0].nonZeroNeedsRelIndices)
                # shopMultiVisit = torch.maximum(torch.gather(data[0].oneAuxVal, 0, data[0].nonZeroNeedsShopIndices), (alphaShop_nonZero / (alphaShop_nonZero + betaShop_nonZero)) * (data[0].nonZeroNeedsShopIndices)).cuda()
                # schoolMultiVisit = torch.maximum(torch.gather(data[0].oneAuxVal, 0, data[0].nonZeroNeedsSchoolIndices), (alphaSchool_nonZero / (alphaSchool_nonZero + betaSchool_nonZero)) * (data[0].nonZeroNeedsSchoolIndices)).cuda()
                # relMultiVisit = torch.maximum(torch.gather(data[0].oneAuxVal, 0, data[0].nonZeroNeedsRelIndices), (alphaRel_nonZero / (alphaRel_nonZero + betaRel_nonZero)) * (data[0].nonZeroNeedsRelIndices)).cuda()
                shopMultiVisit = ((alphaShop_nonZero / (alphaShop_nonZero + betaShop_nonZero)) * (
                    data[0].needsTensor[:, 0][data[0].nonZeroNeedsShopIndices])).cuda()
                schoolMultiVisit = ((alphaSchool_nonZero / (alphaSchool_nonZero + betaSchool_nonZero)) * (
                    data[0].needsTensor[:, 1][data[0].nonZeroNeedsSchoolIndices])).cuda()
                relMultiVisit = ((alphaRel_nonZero / (alphaRel_nonZero + betaRel_nonZero)) * (
                    data[0].needsTensor[:, 2][data[0].nonZeroNeedsRelIndices])).cuda()
                if data[0].isTrainedOnOneMonth == 1:
                    pyro.sample("M_Shop", dist.Poisson(shopMultiVisit.mean() / data[0].gapParamShopFrac))
                    pyro.sample("M_School", dist.Poisson(schoolMultiVisit.mean() / data[0].gapParamSchoolFrac))
                    pyro.sample("M_Religion", dist.Poisson(relMultiVisit.mean() / data[0].gapParamRelFrac))
                else:
                    pyro.sample("M_Shop", dist.Normal(shopMultiVisit.mean() / data[0].gapParamShopFrac,data[0].multiVisitVarShParam))
                    pyro.sample("M_School", dist.Normal(schoolMultiVisit.mean() / data[0].gapParamSchoolFrac,data[0].multiVisitVarSchParam))
                    pyro.sample("M_Religion", dist.Normal(relMultiVisit.mean() / data[0].gapParamRelFrac,data[0].multiVisitVarRelParam))
                # pyro.sample("M_Shop", dist.Normal(shopMultiVisit.mean(),data[0].multiVisitVarShParam), obs=trainShopFracObs)
                # pyro.sample("M_School", dist.Normal(schoolMultiVisit.mean(),data[0].multiVisitVarSchParam), obs=trainSchoolFracObs)
                # pyro.sample("M_Religion", dist.Normal(relMultiVisit.mean(),data[0].multiVisitVarRelParam), obs=trainRelFracObs)

            # for name,param in list(globals().items()):
            #     if isinstance(param,torch.Tensor):
            #         print(name,param.device)
            #
            # print("!!!")


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

        modelTypeIndex = 2
        allData = loadData(cities[selectedTrainCityIndex], cities[selectedTestCityIndex], dates, selectedTrainRangeIndices, selectedTestRangeIndices, modelTypeIndex)

        needsVerbose = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + 'Needs_data.csv')

        # graph = pyro.render_model(model, model_args=(allData.trainData.monthlyData[0],), render_distributions=True, render_params=True)
        # graph.view()

        # setup the optimizer
        # adam_params = {"lr": lr, "betas": (0.9, 0.999), "maximize": False}
        # optimizer = Adam(adam_params)

        # asgd_params = {"lr": lr, "maximize": False}
        # optimizer = ASGD(asgd_params)

        # adagrad_params = {"lr": lr, "maximize": False, "lr_decay": 0.01}
        # optimizer = Adagrad(adagrad_params)

        # radam_params = {"lr": lr, "betas": (0.6, 0.9)}
        # optimizer = RAdam(radam_params)

        # exponentialLR_params = {"gamma ": 0.01}
        # optimizer = ExponentialLR(exponentialLR_params)

        rprop_params = {"lr": lr, "step_sizes":(1e-06, 40), "maximize": False}
        optimizer = Rprop(rprop_params)

        # adamW_params = {"lr": lr, "betas": (0.8, 0.9), "maximize": False, "weight_decay": 0.01}
        # optimizer = AdamW(adamW_params)

        # adadelta_params = {}
        # optimizer = Adadelta(adadelta_params)

        # sgd_params = {"lr": lr}
        # optimizer = SGD(sgd_params)

        # auto_guide = AutoDiscreteParallel(model)

        if elboType == 'Trace_ELBO':
            Elbo = Trace_ELBO
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
        allData.trainData.monthlyData[0].globalError = np.zeros(numParticles, dtype=np.float32)
        allData.trainData.monthlyData[0].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)

        # svi.num_chains=1

        for i in range(len(allData.trainData.monthlyData)):
            DataBundle.loadDataToGPU(allData.trainData.monthlyData[i])

        for i in range(len(allData.testData.monthlyData)):
            DataBundle.loadDataToGPU(allData.testData.monthlyData[i])

        if elboType == 'RenyiELBO':
            extraMessage = "M2_"+elboType + "_alpha" + str(alpha) + "_numParticle" + str(numParticles) + "_lr" + str(lr)
        else:
            extraMessage = "M2_"+elboType + "_numParticle" + str(numParticles) + "_lr" + str(lr)

        if retConfig.isKFoldCrossVal == 1:
            allMonths = []
            for i in range(len(allData.trainData.monthlyData)):
                allData.trainData.monthlyData[i].globalError = np.zeros(numParticles, dtype=np.float32)
                allData.trainData.monthlyData[i].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)
                allMonths.append(allData.trainData.monthlyData[i])

            for i in range(len(allData.testData.monthlyData)):
                allData.testData.monthlyData[i].globalError = np.zeros(numParticles, dtype=np.float32)
                allData.testData.monthlyData[i].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)
                allMonths.append(allData.testData.monthlyData[i])
            runCrossVal(svi, elbo, model, guide, allMonths, numParticles, dates, cities[selectedTestCityIndex],"M3")
        else:
            loss = elbo.loss(model, guide, allData.trainData.monthlyData)
            logging.info("first loss train SantaFe = {}".format(loss))

            n_steps = 200
            error_tolerance = 1

            losses = []
            maxErrors = []
            maxErrorsFrac = []

            # do gradient steps
            for step in range(n_steps):
                self.stepValue = step
                loss = svi.step(allData.trainData.monthlyData)
                maxError = np.max(np.absolute(allData.trainData.monthlyData[0].globalError))
                maxErrorFrac = np.max(np.absolute(allData.trainData.monthlyData[0].globalErrorFrac))
                losses.append(loss)
                maxErrors.append(maxError)
                maxErrorsFrac.append(maxErrorFrac)

                plt.figure("error fig online")
                plt.cla()
                plt.plot(maxErrors[-50:])
                plt.pause(0.01)

                plt.figure("error frac fig online")
                plt.cla()
                plt.plot(maxErrorsFrac[-50:])
                plt.pause(0.01)

                # print("maxError {}".format(maxError))
                allData.trainData.monthlyData[0].globalError = np.zeros(numParticles, dtype=np.float32)
                allData.trainData.monthlyData[0].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)
                # svi.run()
                if step % 100 == 0:
                    logging.info("{: >5d}\t{}".format(step, loss))
                    print("maxError {}".format(maxError))
                    # print('.', end='')
                    # for name in pyro.get_param_store():
                    #     value = pyro.param(name)
                    #     print("{} = {}".format(name, value.detach().cpu().numpy()))
                # if maxError <= error_tolerance:
                #     break

            showAlphaBetaRange("CBG based simulation", allData.trainData.monthlyData[0], needsVerbose)

            # print(os.path.dirname(__file__))
            now = datetime.now()
            dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
            plt.figure("loss fig")
            plt.plot(losses)
            plt.savefig(os.path.dirname(__file__) + os.sep + 'tests' + os.sep + 'loss_' + dt_string + '_' + cities[selectedTrainCityIndex] + '_' + extraMessage + '.png')
            loss_path = os.path.dirname(__file__) + os.sep + 'tests' + os.sep + 'loss_' + '_' + cities[selectedTrainCityIndex] + '_' + extraMessage
            plt.figure("error fig")
            plt.plot(maxErrors)
            plt.savefig(os.path.dirname(__file__) + os.sep + 'tests' + os.sep + 'error_' + dt_string + '_' + cities[selectedTrainCityIndex] + '_' + extraMessage + '.png')
            error_path = os.path.dirname(__file__) + os.sep + 'tests' + os.sep + 'error_' + '_' + cities[selectedTrainCityIndex] + '_' + extraMessage

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
            loss = elbo.loss(model, guide, allData.trainData.monthlyData)
            logging.info("final loss train Tucson = {}".format(loss))

            for name in pyro.get_param_store():
                value = pyro.param(name)
                print("{} = {}".format(name, value.detach().cpu().numpy()))

            validate(allData.testData.monthlyData,elbo,model, guide, numParticles, cities[selectedTestCityIndex])

            retVals.append([losses, maxErrors, loss_path, error_path])

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
        p = multiprocessing.Process(target=test.run,args=(5, 0.7, "RenyiELBO", 0.5,i,retVals))
        p.start()
        processes.append(p)
        # print('!!!')

    for i in range(numTests):
        print('!!!WAITING')
        processes[i].join()

    if len(retVals) > 0:
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