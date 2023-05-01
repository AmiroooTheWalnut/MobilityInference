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
import torch
import pyro

class Test():

    def __init__(self):
        self.stepValue=0

    def run(self,numParticles,lr,elboType,alpha,i,retVals):
        # globalError = np.zeros(1, dtype=np.int32)
        self.index=i

        def model(data):
            groupProbs = (torch.transpose(data[0].occupationProb, 0, 1) * data[0].ageProb).reshape([data[0].G, 1])
            needsSh = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0],data[0].occupationProb.shape[1]]).transpose(0,1).reshape([data[0].G,1])
            needsSch = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0],data[0].occupationProb.shape[1]]).transpose(0,1).reshape([data[0].G,1])
            needsRel = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0],data[0].occupationProb.shape[1]]).transpose(0,1).reshape([data[0].G,1])
            data[0].BBNSh = (needsSh * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            data[0].BBNSch = (needsSch * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            data[0].BBNRel = (needsRel * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            data[0].BBNSh[data[0].BBNSh == 0] = 1
            data[0].BBNSch[data[0].BBNSch == 0] = 1
            data[0].BBNRel[data[0].BBNRel == 0] = 1


            data[0].gapVal=20

            # \/\/\/ Big individual
            with pyro.plate("NCBG", data[0].NCBG) as ncbg:
                with pyro.plate("G", data[0].G) as g:
                    shopVisits = pyro.sample("Tu_Shop", dist.BetaBinomial(torch.transpose(data[0].alpha_paramShop[g].repeat([data[0].NCBG, 1]), 0, 1), torch.transpose(data[0].beta_paramShop[g].repeat([data[0].NCBG, 1]), 0, 1), data[0].BBNSh))
                    schoolVisits = pyro.sample("Tu_School", dist.BetaBinomial(torch.transpose(data[0].alpha_paramSchool[g].repeat([data[0].NCBG, 1]), 0, 1), torch.transpose(data[0].beta_paramSchool[g].repeat([data[0].NCBG, 1]), 0, 1), data[0].BBNSch))
                    religionVisits = pyro.sample("Tu_Religion", dist.BetaBinomial(torch.transpose(data[0].alpha_paramReligion[g].repeat([data[0].NCBG, 1]), 0, 1), torch.transpose(data[0].beta_paramReligion[g].repeat([data[0].NCBG, 1]), 0, 1), data[0].BBNRel))

            sumValShop=shopVisits.sum()
            sumValSchool=schoolVisits.sum()
            sumValRel=religionVisits.sum()
            # ^^^ Big individual

            temp=[]
            for i in range(len(data)):
                temp.append(torch.round(data[i].pOIs[0, 1]))
            trainShopObs = torch.Tensor(temp)
            temp = []
            for i in range(len(data)):
                temp.append(torch.round(data[i].pOIs[1, 1]))
            trainSchoolObs = torch.Tensor(temp)
            temp = []
            for i in range(len(data)):
                temp.append(torch.round(data[i].pOIs[2, 1]))
            trainRelObs = torch.Tensor(temp)

            with pyro.plate('observe_data'):
                shopVisitsObs = pyro.sample("S_Shop", dist.Poisson(torch.sum(torch.transpose(torch.div(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), data[0].alpha_paramShop.repeat([data[0].NCBG, 1]) + data[0].beta_paramShop[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNSh)), obs=trainShopObs*data[0].gapVal)
                schoolVisitsObs = pyro.sample("S_School", dist.Poisson(torch.sum(torch.transpose(torch.div(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]) + data[0].beta_paramSchool[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNSch)), obs=trainSchoolObs*data[0].gapVal)
                religionVisitsObs = pyro.sample("S_Religion", dist.Poisson(torch.sum(torch.transpose(torch.div(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]) + data[0].beta_paramReligion[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNRel)), obs=trainRelObs*data[0].gapVal)

            for i in range(data[0].globalError.shape[0]):
                if data[0].globalError[i] == 0:
                    shopAnalytical=torch.sum(torch.transpose(torch.div(data[0].alpha_paramShop.repeat([data[0].NCBG, 1]), data[0].alpha_paramShop.repeat([data[0].NCBG, 1]) + data[0].beta_paramShop[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNSh)
                    schoolAnalytical = torch.sum(torch.transpose(torch.div(data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]), data[0].alpha_paramSchool.repeat([data[0].NCBG, 1]) + data[0].beta_paramSchool[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNSch)
                    religionAnalytical = torch.sum(torch.transpose(torch.div(data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]), data[0].alpha_paramReligion.repeat([data[0].NCBG, 1]) + data[0].beta_paramReligion[g].repeat([data[0].NCBG, 1])), 0, 1) * data[0].BBNRel)
                    data[0].globalError[i] = torch.absolute(shopAnalytical - data[0].pOIs[0, 1]*data[0].gapVal) + torch.absolute(schoolAnalytical - data[0].pOIs[1, 1]*data[0].gapVal) + torch.absolute(religionAnalytical - data[0].pOIs[2, 1]*data[0].gapVal)
                    # print("within errors {}".format(globalError[i]))
                    break

            return shopVisitsObs, schoolVisitsObs, religionVisitsObs

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

            groupProbs = (torch.transpose(data[0].occupationProb, 0, 1) * data[0].ageProb).reshape([data[0].G, 1])
            needsSh = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0], data[0].occupationProb.shape[1]]).transpose(0, 1).reshape([data[0].G, 1])
            needsSch = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0], data[0].occupationProb.shape[1]]).transpose(0, 1).reshape([data[0].G, 1])
            needsRel = data[0].needsTensor[:, 0].reshape([data[0].occupationProb.shape[0], data[0].occupationProb.shape[1]]).transpose(0, 1).reshape([data[0].G, 1])
            data[0].BBNSh = (needsSh * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            data[0].BBNSch = (needsSch * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            data[0].BBNRel = (needsRel * groupProbs * data[0].populationNum.repeat([data[0].G, 1])).round()
            data[0].BBNSh[data[0].BBNSh == 0] = 1
            data[0].BBNSch[data[0].BBNSch == 0] = 1
            data[0].BBNRel[data[0].BBNRel == 0] = 1
            with pyro.plate("NCBG", data[0].NCBG) as ncbg:
                with pyro.plate("G", data[0].G) as g:
                    pyro.sample("Tu_Shop", dist.BetaBinomial(torch.transpose(data[0].alpha_paramShop[g].repeat([data[0].NCBG, 1]), 0, 1), torch.transpose(data[0].beta_paramShop[g].repeat([data[0].NCBG, 1]), 0, 1), data[0].BBNSh))
                    pyro.sample("Tu_School", dist.BetaBinomial(torch.transpose(data[0].alpha_paramSchool[g].repeat([data[0].NCBG, 1]), 0, 1), torch.transpose(data[0].beta_paramSchool[g].repeat([data[0].NCBG, 1]), 0, 1), data[0].BBNSch))
                    pyro.sample("Tu_Religion", dist.BetaBinomial(torch.transpose(data[0].alpha_paramReligion[g].repeat([data[0].NCBG, 1]), 0, 1), torch.transpose(data[0].beta_paramReligion[g].repeat([data[0].NCBG, 1]), 0, 1), data[0].BBNRel))

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

        modelTypeIndex = 1
        allData = loadData(cities[selectedTrainCityIndex], cities[selectedTestCityIndex], dates, selectedTrainRangeIndices, selectedTestRangeIndices, modelTypeIndex)

        needsVerbose = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + 'Needs_data.csv')

        # graph = pyro.render_model(model, model_args=(allData.trainData.monthlyData[0],), render_distributions=True, render_params=True)
        # graph.view()

        # setup the optimizer
        # adam_params = {"lr": lr, "betas": (0.9, 0.999), "maximize": False}
        # optimizer = Adam(adam_params)

        # asgd_params = {"lr": lr, "maximize": False}
        # optimizer = ASGD(asgd_params)

        adagrad_params = {"lr": lr, "maximize": False, "lr_decay":0.000001}
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
        allData.trainData.monthlyData[0].globalError = np.zeros(numParticles, dtype=np.int32)

        # svi.num_chains=1

        if elboType == 'RenyiELBO':
            extraMessage = elboType + "_alpha" + str(alpha) + "_numParticle" + str(numParticles) + "_lr" + str(lr)
        else:
            extraMessage = elboType + "_numParticle" + str(numParticles) + "_lr" + str(lr)

        if retConfig.isKFoldCrossVal == 1:
            runCrossVal(svi, elbo, model, guide, allData.testData.monthlyData, numParticles, dates, cities[selectedTestCityIndex])
        else:
            loss = elbo.loss(model, guide, allData.trainData.monthlyData)
            logging.info("first loss train SantaFe = {}".format(loss))

            n_steps = 500
            error_tolerance = 1

            losses = []
            maxErrors = []

            plt.figure("loss fig online")

            # do gradient steps
            for step in range(n_steps):
                self.stepValue=step
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
                if step % 100 == 0:
                    logging.info("{: >5d}\t{}".format(step, loss))
                    print("maxError {}".format(maxError))
                    # print('.', end='')
                    # for name in pyro.get_param_store():
                    #     value = pyro.param(name)
                    #     print("{} = {}".format(name, value.detach().cpu().numpy()))
                # if maxError <= error_tolerance:
                #     break

            showAlphaBetaRange("CBG based simulation", allData.trainData.monthlyData[0], allData.trainData.monthlyData[0].alpha_paramShop, allData.trainData.monthlyData[0].beta_paramShop, needsVerbose)


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
            loss = elbo.loss(model, guide, allData.trainData.monthlyData)
            logging.info("final loss train Tucson = {}".format(loss))

            for name in pyro.get_param_store():
                value = pyro.param(name)
                print("{} = {}".format(name, value.detach().cpu().numpy()))

            for i in range(len(allData.testData.monthlyData)):
                allData.testData.monthlyData[i].globalError = np.zeros(numParticles, dtype=np.int32)
                loss = elbo.loss(model, guide, [allData.testData.monthlyData[i]])
                logging.info("final loss test Appleton = {}".format(loss))
                logging.info("final error test Appleton = {}".format(allData.testData.monthlyData[i].globalError))

            retVals.append([losses,maxErrors,loss_path,error_path])

if __name__ ==  '__main__':
    print("pyro version:", pyro.__version__)
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
        p = multiprocessing.Process(target=test.run,args=(5, 4, "RenyiELBO", 0.2,i,retVals))
        p.start()
        processes.append(p)

    for i in range(numTests):
        print('!!!WAITING')
        processes[i].join()

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.figure("loss fig")
    for i in range(len(retVals)):
        plt.plot(retVals[i][0])
    plt.savefig(retVals[0][2] + '_' + dt_string + '_summary' + '.png')
    plt.figure("error fig")
    for i in range(len(retVals)):
        plt.plot(retVals[i][1])
    plt.savefig(retVals[0][3] + '_' + dt_string + '_summary' + '.png')

    print('!!!')

