import logging
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceTailAdaptive_ELBO, RenyiELBO, TraceGraph_ELBO, TraceTMC_ELBO, TraceMeanField_ELBO
import sys
import pyro
import csv
import numpy as np

class FoldOutput:
    def __init__(self,trainIndex,trainError,trainLoss,testErrors,testLosses):
        self.trainIndex=trainIndex
        self.trainError=trainError
        self.trainLoss = trainLoss
        self.testErrors=testErrors
        self.testLosses=testLosses

def runCrossVal(svi,elbo,model,guide,tests,numParticles,dates,cityName):
    allFolds=[]
    dataIndices=[]
    for i in range(len(tests)):
        dataIndices.append(i)
    for i in range(len(tests)):
        test=[tests[i]]
        trains_temp=tests.copy()
        del trains_temp[i]
        dataIndices_temp=dataIndices.copy()
        del dataIndices_temp[i]
        foldOutput=runOneFold(svi,elbo,model,guide,trains_temp,i,test,dataIndices_temp,numParticles)
        allFolds.append(foldOutput)
    saveKFoldCrossVal(allFolds,dates,cityName)

def runOneFold(svi,elbo,model,guide,train,trainIndex,tests,dataIndices,numParticles):
    loss = elbo.loss(model, guide, train)
    logging.info("first loss train SantaFe = {}".format(loss))

    n_steps = 500
    error_tolerance = 10

    # do gradient steps
    for step in range(n_steps):
        loss = svi.step(train)
        # globalError = np.zeros(numParticles, dtype=np.int32)
        if step % 10 == 0:
            logging.info("{: >5d}\t{}".format(step, loss))
            # print('.', end='')
            # for name in pyro.get_param_store():
            #     value = pyro.param(name)
            #     print("{} = {}".format(name, value.detach().cpu().numpy()))
        if loss <= error_tolerance:
            print("Error tolerance {} is obtained".format(error_tolerance))
            break
    print("Final evalulation")

    # allDataTrain = loadData(cities[selectedTrainCityIndex], dates, times[selectedTrainRangeIndices])
    train[0].isFirst = True
    train[0].globalError = np.zeros(numParticles, dtype=np.int32)
    trainLoss = elbo.loss(model, guide, train)
    logging.info("final loss = {}".format(trainLoss))
    trainError=train[0].globalError[0]

    for name in pyro.get_param_store():
        value = pyro.param(name)
        print("{} = {}".format(name, value.detach().cpu().numpy()))

    testLosses=[]
    testErrors=[]
    for i in range(len(tests)):
        tests[0].globalError = np.zeros(numParticles, dtype=np.int32)
        loss = elbo.loss(model, guide, [tests[i]])
        logging.info("loss fold {} = {}".format(dataIndices[i],loss))
        testLosses.append(loss)
        testErrors.append(tests[0].globalError[0])


    foldOutput = FoldOutput(trainIndex,trainError,trainLoss,testErrors,testLosses)

    return foldOutput

def saveKFoldCrossVal(allFolds,dates,cityName):
    with open('KFCV_{}.csv'.format(cityName), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # numColumns = len(allFolds[0].testErrors) + 2
        for i in range(len(allFolds)):
            header = []
            header.append("Fold {} {}".format(i, dates[i]))
            errorRow = []
            errorRow.append("Error")
            lossRow = []
            lossRow.append("Loss")
            header.append("Train")
            header.append("Test")
            errorRow.append(allFolds[i].trainError.item())
            lossRow.append(allFolds[i].trainLoss)
            errorRow.append(allFolds[i].testErrors[0].item())
            lossRow.append(allFolds[i].testLosses[0])
            # counter = 0
            # for j in range(numColumns-1):
            #     if allFolds[i].trainIndex == j:
            #         header.append("Train")
            #         errorRow.append(allFolds[i].trainError.item())
            #         lossRow.append(allFolds[i].trainLoss)
            #     else:
            #         if len(allFolds[i].testErrors)>counter:
            #             header.append("Test")
            #             errorRow.append(allFolds[i].testErrors[counter].item())
            #             lossRow.append(allFolds[i].testLosses[counter])
            #             counter = counter + 1
            writer.writerow(header)
            writer.writerow(errorRow)
            writer.writerow(lossRow)