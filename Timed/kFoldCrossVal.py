import logging
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceTailAdaptive_ELBO, RenyiELBO, TraceGraph_ELBO, TraceTMC_ELBO, TraceMeanField_ELBO
import sys
import pyro
import csv
import numpy as np
from Validator import *
import matplotlib
import matplotlib.pyplot as plt

class FoldOutput:
    def __init__(self,trainIndex,trainLoss,trainError,trainErrorFrac,testLosses,testErrors,testErrorsFrac):
        self.trainIndex=trainIndex
        self.trainLoss = trainLoss
        self.trainError=trainError
        self.trainErrorFrac = trainErrorFrac
        self.testLosses = testLosses
        self.testErrors=testErrors
        self.testErrorsFrac = testErrorsFrac

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
        foldOutput=runOneFold(svi,elbo,model,guide,trains_temp,i,test,dataIndices_temp,cityName,numParticles)
        allFolds.append(foldOutput)
    saveKFoldCrossVal(allFolds,dates,cityName)

def runOneFold(svi,elbo,model,guide,train,trainIndex,tests,dataIndices,trainCity,numParticles):
    train[0].globalError = np.zeros(numParticles, dtype=np.float32)
    train[0].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)
    loss = elbo.loss(model, guide, train)
    logging.info("first loss train {} = {}".format(trainCity,loss))

    n_steps = 1000
    error_tolerance = 1

    train[0].globalError = np.zeros(numParticles, dtype=np.float32)
    train[0].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)

    losses = []
    maxErrors = []
    maxErrorsFrac = []

    # do gradient steps
    for step in range(n_steps):
        loss = svi.step(train)

        maxError = np.max(np.absolute(train[0].globalError))
        maxErrorFrac = np.max(np.absolute(train[0].globalErrorFrac))
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

        train[0].globalError = np.zeros(numParticles, dtype=np.float32)
        train[0].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)
        if step % 100 == 0:
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
    train[0].globalError = np.zeros(numParticles, dtype=np.float32)
    train[0].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)
    trainLoss = elbo.loss(model, guide, train)
    logging.info("final loss = {}".format(trainLoss))
    trainError = train[0].globalError[0]
    trainErrorFrac = train[0].globalErrorFrac[0]

    for name in pyro.get_param_store():
        value = pyro.param(name)
        print("{} = {}".format(name, value.detach().cpu().numpy()))

    # testLosses=[]
    # testErrors=[]
    # for i in range(len(tests)):
    #     tests[0].globalError = np.zeros(numParticles, dtype=np.int32)
    #     loss = elbo.loss(model, guide, [tests[i]])
    #     logging.info("loss fold {} = {}".format(dataIndices[i],loss))
    #     testLosses.append(loss)
    #     testErrors.append(tests[0].globalError[0])
    losses,errors,errorsFrac=validate(tests, elbo, model, guide, numParticles, trainCity)


    foldOutput = FoldOutput(trainIndex,trainLoss,trainError,trainErrorFrac,losses,errors,errorsFrac)

    return foldOutput

def saveKFoldCrossVal(allFolds,dates,cityName):
    with open('KFCV_{}.csv'.format(cityName), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # numColumns = len(allFolds[0].testErrors) + 2
        header = []
        header.append("")
        header.append("loss")
        header.append("")
        # header.append("")
        header.append("error")
        header.append("")
        # header.append("")
        header.append("errorFrac")
        header.append("")
        # header.append("")
        writer.writerow(header)
        header2 = []
        header2.append("")
        header2.append("train")
        header2.append("test")
        # header2.append("test var")
        header2.append("train")
        header2.append("test")
        # header2.append("test var")
        header2.append("train")
        header2.append("test")
        # header2.append("test var")
        writer.writerow(header2)
        for i in range(len(allFolds)):
            row = []
            row.append("F{}".format(i))
            row.append(allFolds[i].trainLoss)
            row.append(np.mean(allFolds[i].testLosses))
            # row.append(np.var(allFolds[i].testLosses))
            row.append(allFolds[i].trainError)
            row.append(np.mean(allFolds[i].testErrors))
            # row.append(np.var(allFolds[i].testErrors))
            row.append(allFolds[i].trainErrorFrac)
            row.append(np.mean(allFolds[i].testErrorsFrac))
            # row.append(np.var(allFolds[i].testErrorsFrac))

            writer.writerow(row)