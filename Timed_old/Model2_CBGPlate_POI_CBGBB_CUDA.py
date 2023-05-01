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
            alphaShopReady = torch.transpose(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            betaShopReady = torch.transpose(data[0].beta_paramShop.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            alphaSchoolReady = torch.transpose(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            betaSchoolReady = torch.transpose(data[0].beta_paramSchool.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            alphaRelReady = torch.transpose(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            betaRelReady = torch.transpose(data[0].beta_paramReligion.repeat([data[0].NCBG, 1]), 0, 1).cuda()

            with pyro.plate("NCBG", data[0].NCBG) as ncbg:
                with pyro.plate("G", data[0].G) as g:
                    with pyro.plate("Nshop", data[0].Nshop) as nshop:
                        shopVisits = pyro.sample("Tu_Shop", dist.BetaBinomial(alphaShopReady, betaShopReady, data[0].BBNSh)).cuda()
                        shopVisitsPOIs = shopVisits.sum([1,2]).cuda()
                        shopVisitsPOIsAdjusted = torch.mul(shopVisitsPOIs, data[0].pOIShopProb)
                        expectedSum = shopVisitsPOIs.sum().cuda()
                        newSum = shopVisitsPOIsAdjusted.sum().cuda()
                        diff = expectedSum - newSum
                        newSumShop = torch.add(shopVisitsPOIsAdjusted, torch.div(diff, shopVisitsPOIs.size(dim=0))).cuda()

                    with pyro.plate("Nschool", data[0].Nschool) as nschool:
                        schoolVisits = pyro.sample("Tu_School", dist.BetaBinomial(alphaSchoolReady, betaSchoolReady, data[0].BBNSch)).cuda()
                        schoolVisitsPOIs = schoolVisits.sum([1,2])
                        schoolVisitsPOIsAdjusted = torch.mul(schoolVisitsPOIs, data[0].pOISchoolProb)
                        expectedSum = schoolVisitsPOIs.sum().cuda()
                        newSum = schoolVisitsPOIsAdjusted.sum().cuda()
                        diff = expectedSum - newSum
                        newSumSchool = torch.add(schoolVisitsPOIsAdjusted, torch.div(diff, schoolVisitsPOIs.size(dim=0))).cuda()

                    with pyro.plate("Nreligion", data[0].Nreligion) as nreligion:
                        religionVisits = pyro.sample("Tu_Religion", dist.BetaBinomial(alphaRelReady, betaRelReady, data[0].BBNRel)).cuda()
                        religionVisitsPOIs = religionVisits.sum([1,2])
                        religionVisitsPOIsAdjusted = torch.mul(religionVisitsPOIs, data[0].pOIReligionProb)
                        expectedSum = religionVisitsPOIs.sum().cuda()
                        newSum = religionVisitsPOIsAdjusted.sum().cuda()
                        diff = expectedSum - newSum
                        newSumRel = torch.add(religionVisitsPOIsAdjusted, torch.div(diff, religionVisitsPOIs.size(dim=0))).cuda()


            shopVisitsObs = pyro.sample("S_Shop", dist.Poisson(newSumShop).to_event(1), obs=data[0].pOIShops.flatten())
            schoolVisitsObs = pyro.sample("S_School", dist.Poisson(newSumSchool).to_event(1), obs=data[0].pOISchools.flatten())
            religionVisitsObs = pyro.sample("S_Religion", dist.Poisson(newSumRel).to_event(1), obs=data[0].pOIReligion.flatten())

            # obsRaw = np.transpose(data.pOIs.iloc[:][1])
            # obs = torch.zeros(data.NE)
            # for i in range(data.NE):
            #     obs[i] = obsRaw.iloc[i]
            #     obs[i] = torch.div(obs[i],100)
            # print(torch.mul(torch.abs(shopVisits.sum(1).sum(1)), data[0].pOIShopProb).sum() - data[0].pOIShops.sum() + torch.mul(torch.abs(schoolVisits.sum(1).sum(1)), data[0].pOISchoolProb).sum() - data[0].pOISchools.sum() + torch.mul(torch.abs(religionVisits.sum(1).sum(1)), data[0].pOIReligionProb).sum() - data[0].pOIReligion.sum())

            for i in range(data[0].globalError.shape[0]):
                if data[0].globalError[i] == 0:
                    # shopAnalytical=torch.sum(torch.transpose(torch.div(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), data[0].alpha_paramShop.repeat([data[0].NCBG, 1]) + data[0].beta_paramShop[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNSh)
                    # schoolAnalytical = torch.sum(torch.transpose(torch.div(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]) + data[0].beta_paramSchool[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNSch)
                    # religionAnalytical = torch.sum(torch.transpose(torch.div(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]) + data[0].beta_paramReligion[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNRel)
                    data[0].globalError[i] = newSumShop.sum() - data[0].pOIShops.sum() + newSumSchool.sum() - data[0].pOISchools.sum() + newSumRel.sum() - data[0].pOIReligion.sum()
                    # print("within errors {}".format(globalError[i]))
                    break


            # if data[0].isFirst == True:
            #     print(torch.mul(torch.abs(shopVisits.sum(1).sum(1)), data[0].pOIShopProb).sum() - data[0].pOIShops.sum() + torch.mul(torch.abs(schoolVisits.sum(1).sum(1)), data[0].pOISchoolProb).sum() - data[0].pOISchools.sum() + torch.mul(torch.abs(religionVisits.sum(1).sum(1)), data[0].pOIReligionProb).sum() - data[0].pOIReligion.sum())
            #     data[0].isFirst = False

            return shopVisitsObs, schoolVisitsObs, religionVisitsObs

        @profile
        def guide(data):
            temp_alpha_paramShop = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramShop[6] = 0.5
            temp_alpha_paramShop[7] = 0.5
            temp_alpha_paramShop[8] = 0.4
            temp_alpha_paramShop[9] = 0.4
            temp_beta_paramShop = torch.add(torch.zeros(data[0].G), 13.4)
            temp_beta_paramShop[6] = 11
            temp_beta_paramShop[7] = 11
            temp_beta_paramShop[8] = 12
            temp_beta_paramShop[9] = 12
            temp_alpha_paramSchool = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramSchool[0] = 0.4
            temp_alpha_paramSchool[1] = 0.4
            temp_alpha_paramSchool[5] = 0.4
            temp_alpha_paramSchool[8] = 0.4
            temp_beta_paramSchool = torch.add(torch.zeros(data[0].G), 13.4)
            temp_beta_paramSchool[0] = 12
            temp_beta_paramSchool[1] = 12
            temp_beta_paramSchool[5] = 12
            temp_beta_paramSchool[8] = 12
            temp_alpha_paramRel = torch.add(torch.zeros(data[0].G), 0.2)
            temp_alpha_paramRel[11] = 0.4
            temp_alpha_paramRel[14] = 0.4
            temp_beta_paramRel = torch.add(torch.zeros(data[0].G), 13.4)
            temp_beta_paramRel[11] = 12
            temp_beta_paramRel[14] = 12
            data[0].alpha_paramShop = pyro.param("alpha_paramShop_G", temp_alpha_paramShop, constraint=constraints.positive)
            data[0].beta_paramShop = pyro.param("beta_paramShop_G", temp_beta_paramShop, constraint=constraints.positive)
            data[0].alpha_paramSchool = pyro.param("alpha_paramSchool_G", temp_alpha_paramSchool, constraint=constraints.positive)
            data[0].beta_paramSchool = pyro.param("beta_paramSchool_G", temp_beta_paramSchool, constraint=constraints.positive)
            data[0].alpha_paramReligion = pyro.param("alpha_paramReligion_G", temp_alpha_paramRel, constraint=constraints.positive)
            data[0].beta_paramReligion = pyro.param("beta_paramReligion_G", temp_beta_paramRel, constraint=constraints.positive)

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

            alphaShopReady = torch.transpose(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            betaShopReady = torch.transpose(data[0].beta_paramShop.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            alphaSchoolReady = torch.transpose(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            betaSchoolReady = torch.transpose(data[0].beta_paramSchool.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            alphaRelReady = torch.transpose(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), 0, 1).cuda()
            betaRelReady = torch.transpose(data[0].beta_paramReligion.repeat([data[0].NCBG, 1]), 0, 1).cuda()

            with pyro.plate("NCBG", data[0].NCBG) as ncbg:
                with pyro.plate("G", data[0].G) as g:
                    with pyro.plate("Nshop", data[0].Nshop) as nshop:
                        pyro.sample("Tu_Shop", dist.BetaBinomial(alphaShopReady, betaShopReady, data[0].BBNSh))
                        # shopVisitsPOIs = shopVisits.sum([1, 2])
                        # shopVisitsPOIsAdjusted = torch.mul(shopVisitsPOIs, data[0].pOIShopProb)
                        # expectedSum = shopVisitsPOIs.sum()
                        # newSum = shopVisitsPOIsAdjusted.sum()
                        # diff = expectedSum - newSum
                        # newSumShop = torch.add(shopVisitsPOIsAdjusted, torch.div(diff, shopVisitsPOIs.size(dim=0)))

                    with pyro.plate("Nschool", data[0].Nschool) as nschool:
                        pyro.sample("Tu_School", dist.BetaBinomial(alphaSchoolReady, betaSchoolReady, data[0].BBNSch))
                        # schoolVisitsPOIs = schoolVisits.sum([1, 2])
                        # schoolVisitsPOIsAdjusted = torch.mul(schoolVisitsPOIs, data[0].pOISchoolProb)
                        # expectedSum = schoolVisitsPOIs.sum()
                        # newSum = schoolVisitsPOIsAdjusted.sum()
                        # diff = expectedSum - newSum
                        # newSumSchool = torch.add(schoolVisitsPOIsAdjusted, torch.div(diff, schoolVisitsPOIs.size(dim=0)))

                    with pyro.plate("Nreligion", data[0].Nreligion) as nreligion:
                        pyro.sample("Tu_Religion", dist.BetaBinomial(alphaRelReady, betaRelReady, data[0].BBNRel))
                        # religionVisitsPOIs = religionVisits.sum([1, 2])
                        # religionVisitsPOIsAdjusted = torch.mul(religionVisitsPOIs, data[0].pOIReligionProb)
                        # expectedSum = religionVisitsPOIs.sum()
                        # newSum = religionVisitsPOIsAdjusted.sum()
                        # diff = expectedSum - newSum
                        # newSumRel = torch.add(religionVisitsPOIsAdjusted, torch.div(diff, religionVisitsPOIs.size(dim=0)))


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

        adagrad_params = {"lr": lr, "maximize": False, "lr_decay": 0.005}
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
        allData.trainData.monthlyData[0].globalError = np.zeros(numParticles, dtype=np.int32)

        # svi.num_chains=1

        for i in range(len(allData.trainData.monthlyData)):
            DataBundle.loadDataToGPU(allData.trainData.monthlyData[i])

        for i in range(len(allData.testData.monthlyData)):
            DataBundle.loadDataToGPU(allData.testData.monthlyData[i])

        if elboType == 'RenyiELBO':
            extraMessage = elboType + "_alpha" + str(alpha) + "_numParticle" + str(numParticles) + "_lr" + str(lr)
        else:
            extraMessage = elboType + "_numParticle" + str(numParticles) + "_lr" + str(lr)

        if retConfig.isKFoldCrossVal == 1:
            runCrossVal(svi, elbo, model, guide, allData.testData.monthlyData, numParticles, dates, cities[selectedTestCityIndex])
        else:
            loss = elbo.loss(model, guide, allData.trainData.monthlyData)
            logging.info("first loss train SantaFe = {}".format(loss))

            n_steps = 2000
            error_tolerance = 1

            losses = []
            maxErrors = []

            plt.figure("loss fig online")

            # do gradient steps
            for step in range(n_steps):
                self.stepValue = step
                loss = svi.step(allData.trainData.monthlyData)
                maxError = np.max(np.absolute(allData.trainData.monthlyData[0].globalError))
                losses.append(loss)
                maxErrors.append(maxError)

                plt.cla()
                plt.plot(maxErrors[-50:])
                plt.pause(0.01)

                # print("maxError {}".format(maxError))
                allData.trainData.monthlyData[0].globalError = np.zeros(numParticles, dtype=np.int32)
                # svi.run()
                if step % 10 == 0:
                    logging.info("{: >5d}\t{}".format(step, loss))
                    print("maxError {}".format(maxError))
                    # print('.', end='')
                    # for name in pyro.get_param_store():
                    #     value = pyro.param(name)
                    #     print("{} = {}".format(name, value.detach().cpu().numpy()))
                # if maxError <= error_tolerance:
                #     break

            showAlphaBetaRange("CBG based simulation", allData.trainData.monthlyData[0], allData.trainData.monthlyData[0].alpha_paramShop, allData.trainData.monthlyData[0].beta_paramShop, needsVerbose)

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

            # DataBundle.unloadDataToGPU(allData.trainData.monthlyData[0])
            # DataBundle.loadDataToGPU(allData.testData.monthlyData[0])

            # visits = pd.read_csv('USA_WI_Outagamie County_Appleton_FullSimple.csv', header=None)
            # population = 75000
            # data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb]
            # allData = AllData(data)
            #

            for i in range(len(allData.testData.monthlyData)):
                allData.testData.monthlyData[i].globalError = np.zeros(numParticles, dtype=np.int32)
                loss = elbo.loss(model, guide, [allData.testData.monthlyData[i]])
                logging.info("final loss test Appleton = {}".format(loss))
                logging.info("final error test Appleton = {}".format(allData.testData.monthlyData[i].globalError))

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



# logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)
#
# visits=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_FullSimple.csv', header=None)
# pOIShops=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_shopLocVis.csv', header=None)
# pOISchools=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_schoolLocVis.csv', header=None)
# pOIReligion=pd.read_csv('USA_NM_Santa Fe County_Santa Fe_religionLocVis.csv', header=None)
# needs=pd.read_csv('Needs_data_numbers.csv', header=None)
# population=84000
#
# isFirst=True
#
# pOIShopsProb=pd.DataFrame(np.zeros(pOIShops.shape[0]))
# pOISchoolsProb=pd.DataFrame(np.zeros(pOISchools.shape[0]))
# pOIReligionProb=pd.DataFrame(np.zeros(pOIReligion.shape[0]))
# sShop=pOIShops.iloc[:,1].sum()
# for i in range(pOIShops.shape[0]):
#     pOIShopsProb.at[i,0]=pOIShops.iloc[i,1]/sShop
# sSch=pOISchools.iloc[:,1].sum()
# for i in range(pOISchools.shape[0]):
#     pOISchoolsProb.at[i,0]=pOISchools.iloc[i,1]/sSch
# sRel=pOIReligion.iloc[:,1].sum()
# for i in range(pOIReligion.shape[0]):
#     pOIReligionProb.at[i,0]=pOIReligion.iloc[i,1]/sRel
#
# data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb]
# allData=AllData(data)
#
# graph=pyro.render_model(model, model_args=(allData,), render_distributions=True, render_params=True)
# graph.view()
#
# # setup the optimizer
# adam_params = {"lr": 0.01, "betas": (0.9, 0.999), "maximize": False}
# optimizer = Adam(adam_params)
#
# # asgd_params = {"lr": 0.00001, "maximize": False}
# # optimizer = ASGD(asgd_params)
#
# # adagrad_params = {"lr": 0.0001, "maximize": False}
# # optimizer = Adagrad(adagrad_params)
#
# # radam_params = {"lr": 0.01, "betas": (0.6, 0.9)}
# # optimizer = RAdam(radam_params)
#
# # exponentialLR_params = {"gamma ": 0.01}
# # optimizer = ExponentialLR(exponentialLR_params)
#
# # rprop_params = {}
# # optimizer = Rprop(rprop_params)
#
# # adamW_params = {}
# # optimizer = AdamW(adamW_params)
#
# # adadelta_params = {}
# # optimizer = Adadelta(adadelta_params)
#
# # auto_guide = AutoDiscreteParallel(model)
#
# # Elbo=Trace_ELBO
# # elbo = Elbo(num_particles=5)
#
# # Elbo = TraceTailAdaptive_ELBO
# # elbo = Elbo(num_particles=5, vectorize_particles=True)
#
# # Elbo = TraceGraph_ELBO
# # elbo = Elbo(num_particles=5)
#
# Elbo = RenyiELBO
# elbo = Elbo(alpha=0.1,num_particles=5)
#
# # Elbo = TraceMeanField_ELBO
# # elbo = Elbo(num_particles=5)
#
# # Elbo = TraceTMC_ELBO
# # elbo = Elbo(num_particles=5)
#
# # setup the inference algorithm
# svi = SVI(model, guide, optimizer, loss=elbo)
#
# # svi.num_chains=1
#
# loss = elbo.loss(model, guide, allData)
# logging.info("first loss train SantaFe = {}".format(loss))
#
# n_steps=10000
#
# # do gradient steps
# for step in range(n_steps):
#     loss=svi.step(allData)
#     if step % 10 == 0:
#         logging.info("{: >5d}\t{}".format(step, loss))
#         #print('.', end='')
#         # for name in pyro.get_param_store():
#         #     value = pyro.param(name)
#         #     print("{} = {}".format(name, value.detach().cpu().numpy()))
# print("Final evalulation")
# data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb]
# allData=AllData(data)
# loss = elbo.loss(model, guide, allData)
# logging.info("final loss train SantaFe = {}".format(loss))
#
# for name in pyro.get_param_store():
#     value = pyro.param(name)
#     print("{} = {}".format(name, value.detach().cpu().numpy()))
#
# visits=pd.read_csv('USA_WI_Outagamie County_Appleton_FullSimple.csv', header=None)
# population=75000
# data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb]
# allData=AllData(data)
#
# loss = elbo.loss(model, guide, allData)
# logging.info("final loss test Appleton = {}".format(loss))
#
# visits=pd.read_csv('USA_WI_Brown County_Green Bay_FullSimple.csv', header=None)
# population=107400
# data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb]
# allData=AllData(data)
#
# loss = elbo.loss(model, guide, allData)
# logging.info("final loss test Green bay = {}".format(loss))
#
# visits=pd.read_csv('USA_NY_Richmond County_New York_FullSimple.csv', header=None)
# population=8468000
# data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb]
# allData=AllData(data)
#
# loss = elbo.loss(model, guide, allData)
# logging.info("final loss test New york city = {}".format(loss))
#
# visits=pd.read_csv('USA_WA_King County_Seattle_FullSimple.csv', header=None)
# population=760000
# data=[visits,needs,population,isFirst,pOIShops,pOISchools,pOIReligion,pOIShopsProb,pOISchoolsProb,pOIReligionProb]
# allData=AllData(data)
#
# loss = elbo.loss(model, guide, allData)
# logging.info("final loss test Seattle = {}".format(loss))