import logging
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceTailAdaptive_ELBO, RenyiELBO, TraceGraph_ELBO, TraceTMC_ELBO, TraceMeanField_ELBO
import sys
import pyro
import csv
import numpy as np
from Validator import *
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import datetime

class FoldOutput:
    def __init__(self,trainIndex,trainLoss,trainError,trainErrorFrac,testLosses,testErrors,testErrorsFrac,params):
        self.trainIndex = trainIndex
        self.trainLoss = trainLoss
        self.trainError = trainError
        self.trainErrorFrac = trainErrorFrac
        self.testLosses = testLosses
        self.testErrors = testErrors
        self.testErrorsFrac = testErrorsFrac

        self.alpha_paramShop = params[0]
        self.beta_paramShop = params[1]

        self.alpha_paramSchool = params[2]
        self.beta_paramSchool = params[3]

        self.alpha_paramReligion = params[4]
        self.beta_paramReligion = params[5]

        self.gapParamShop = params[6]
        self.gapParamSchool = params[7]
        self.gapParamRel = params[8]


        self.multiVisitVarShParam = params[9]
        self.multiVisitVarSchParam = params[10]
        self.multiVisitVarRelParam = params[11]


        # self.alpha_paramShop = alpha_paramShop
        # self.beta_paramShop = beta_paramShop
        #
        # self.alpha_paramSchool = alpha_paramSchool
        # self.beta_paramSchool = beta_paramSchool
        #
        # self.alpha_paramReligion = alpha_paramReligion
        # self.beta_paramReligion = beta_paramReligion
        #
        # self.gapParamShop = gapParamShop
        # self.gapParamSchool = gapParamSchool
        # self.gapParamRel = gapParamRel
        #
        # self.gapParamShopFrac = gapParamShopFrac
        # self.gapParamSchoolFrac = gapParamSchoolFrac
        # self.gapParamRelFrac = gapParamRelFrac
        #
        # self.multiVisitVarShParam = multiVisitVarShParam
        # self.multiVisitVarSchParam = multiVisitVarSchParam
        # self.multiVisitVarRelParam = multiVisitVarRelParam


def runCrossVal(svi,elbo,model,guide,tests,numParticles,dates,cityName,extraMessage):
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
    saveKFoldCrossVal(allFolds,dates,cityName,extraMessage)

def runOneFold(svi,elbo,model,guide,train,trainIndex,tests,dataIndices,trainCity,numParticles):
    train[0].globalError = np.zeros(numParticles, dtype=np.float32)
    train[0].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)
    loss = elbo.loss(model, guide, train)
    logging.info("first loss train {} = {}".format(trainCity,loss))

    n_steps = 100
    error_tolerance = 1

    train[0].globalError = np.zeros(numParticles, dtype=np.float32)
    train[0].globalErrorFrac = np.zeros(numParticles, dtype=np.float32)

    losses = []
    maxErrors = []
    maxErrorsFrac = []

    now = datetime.datetime.now()
    seed=now.year+now.month+now.day+now.hour+now.minute+now.second
    print("seed: {}".format(seed))
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

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

    losses,errors,errorsFrac=validate(train[0],tests, elbo, model, guide, numParticles, trainCity)

    params = []

    params.append(train[0].alpha_paramShop)
    params.append(train[0].beta_paramShop)

    params.append(train[0].alpha_paramSchool)
    params.append(train[0].beta_paramSchool)

    params.append(train[0].alpha_paramReligion)
    params.append(train[0].beta_paramReligion)

    params.append(train[0].gapParamShop)
    params.append(train[0].gapParamSchool)
    params.append(train[0].gapParamRel)

    # params.append(train[0].gapParamShopFrac)
    # params.append(train[0].gapParamSchoolFrac)
    # params.append(train[0].gapParamRelFrac)

    params.append(train[0].multiVisitVarShParam)
    params.append(train[0].multiVisitVarSchParam)
    params.append(train[0].multiVisitVarRelParam)

    foldOutput = FoldOutput(trainIndex,trainLoss,trainError,trainErrorFrac,losses,errors,errorsFrac,params)

    return foldOutput

def writeOneParam(writer,nFolds,name, vals):
    vals = np.zeros((nFolds, len(vals)))
    for i in range(nFolds):
        rowAlphaShop = []
        rowAlphaShop.append(name)
        for j in range(len(vals)):
            rowAlphaShop.append(vals[j].item())
            vals[i, j] = vals[j].item()
        writer.writerow(rowAlphaShop)
    varVals = vals.std(axis=0)
    varRow = []
    varRow.append("STD {}".format(name))
    for j in range(len(vals)):
        varRow.append(varVals[j])
    writer.writerow(varRow)

def saveKFoldCrossVal(allFolds,dates,cityName,extraMessage):
    index=0
    for i in range(10):
        if os.path.isfile('KFCV_{}_{}_{}.csv'.format(extraMessage,cityName,index))==False:
            break
        else:
            index=index+1
    with open('KFCV_{}_{}_{}.csv'.format(extraMessage,cityName,index), 'w', encoding='UTF8', newline='') as f:
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

    with open('KFCV_{}_{}_{}_params.csv'.format(extraMessage, cityName, index), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # writeOneParam(writer,len(allFolds),"AlphaShop",)

        vals = np.zeros((len(allFolds),len(allFolds[0].alpha_paramShop)))
        for i in range(len(allFolds)):
            rowAlphaShop = []
            rowAlphaShop.append("AlphaShop")
            for j in range(len(allFolds[i].alpha_paramShop)):
                rowAlphaShop.append(allFolds[i].alpha_paramShop[j].item())
                vals[i,j]=allFolds[i].alpha_paramShop[j].item()
            writer.writerow(rowAlphaShop)
        varVals=vals.std(axis=0)
        varRow=[]
        varRow.append("STD AlphaShop")
        for j in range(len(allFolds[0].alpha_paramShop)):
            varRow.append(varVals[j])
        writer.writerow(varRow)

        vals = np.zeros((len(allFolds), len(allFolds[0].alpha_paramSchool)))
        for i in range(len(allFolds)):
            rowAlphaSchool = []
            rowAlphaSchool.append("AlphaSchool")
            for j in range(len(allFolds[i].alpha_paramSchool)):
                rowAlphaSchool.append(allFolds[i].alpha_paramSchool[j].item())
                vals[i, j] = allFolds[i].alpha_paramSchool[j].item()
            writer.writerow(rowAlphaSchool)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD AlphaSchool")
        for j in range(len(allFolds[0].alpha_paramSchool)):
            varRow.append(varVals[j])
        writer.writerow(varRow)

        vals = np.zeros((len(allFolds), len(allFolds[0].alpha_paramReligion)))
        for i in range(len(allFolds)):
            rowAlphaRel = []
            rowAlphaRel.append("AlphaRel")
            for j in range(len(allFolds[i].alpha_paramReligion)):
                rowAlphaRel.append(allFolds[i].alpha_paramReligion[j].item())
                vals[i, j] = allFolds[i].alpha_paramReligion[j].item()
            writer.writerow(rowAlphaRel)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD AlphaRel")
        for j in range(len(allFolds[0].alpha_paramReligion)):
            varRow.append(varVals[j])
        writer.writerow(varRow)




        vals = np.zeros((len(allFolds), len(allFolds[0].beta_paramShop)))
        for i in range(len(allFolds)):
            rowBetaShop = []
            rowBetaShop.append("BetaShop")
            for j in range(len(allFolds[i].beta_paramShop)):
                rowBetaShop.append(allFolds[i].beta_paramShop[j].item())
                vals[i, j] = allFolds[i].beta_paramShop[j].item()
            writer.writerow(rowBetaShop)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD BetaShop")
        for j in range(len(allFolds[0].beta_paramShop)):
            varRow.append(varVals[j])
        writer.writerow(varRow)

        vals = np.zeros((len(allFolds), len(allFolds[0].beta_paramSchool)))
        for i in range(len(allFolds)):
            rowBetaSchool = []
            rowBetaSchool.append("BetaSchool")
            for j in range(len(allFolds[i].beta_paramSchool)):
                rowBetaSchool.append(allFolds[i].beta_paramSchool[j].item())
                vals[i, j] = allFolds[i].beta_paramSchool[j].item()
            writer.writerow(rowBetaSchool)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD BetaSchool")
        for j in range(len(allFolds[0].beta_paramSchool)):
            varRow.append(varVals[j])
        writer.writerow(varRow)

        vals = np.zeros((len(allFolds), len(allFolds[0].beta_paramReligion)))
        for i in range(len(allFolds)):
            rowBetaRel = []
            rowBetaRel.append("BetaRel")
            for j in range(len(allFolds[i].beta_paramReligion)):
                rowBetaRel.append(allFolds[i].beta_paramReligion[j].item())
                vals[i, j] = allFolds[i].beta_paramReligion[j].item()
            writer.writerow(rowBetaRel)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD BetaRel")
        for j in range(len(allFolds[0].beta_paramReligion)):
            varRow.append(varVals[j])
        writer.writerow(varRow)




        vals = np.zeros((len(allFolds), len(allFolds[0].gapParamShop)))
        for i in range(len(allFolds)):
            rowGapParamShop = []
            rowGapParamShop.append("GapParamShop")
            for j in range(len(allFolds[i].gapParamShop)):
                rowGapParamShop.append(allFolds[i].gapParamShop[j].item())
                vals[i, j] = allFolds[i].gapParamShop[j].item()
            writer.writerow(rowGapParamShop)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD GapParamShop")
        for j in range(len(allFolds[0].gapParamShop)):
            varRow.append(varVals[j])
        writer.writerow(varRow)

        vals = np.zeros((len(allFolds), len(allFolds[0].gapParamSchool)))
        for i in range(len(allFolds)):
            rowGapParamSchool = []
            rowGapParamSchool.append("GapParamSchool")
            for j in range(len(allFolds[i].gapParamSchool)):
                rowGapParamSchool.append(allFolds[i].gapParamSchool[j].item())
                vals[i, j] = allFolds[i].gapParamSchool[j].item()
            writer.writerow(rowGapParamSchool)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD GapParamSchool")
        for j in range(len(allFolds[0].gapParamSchool)):
            varRow.append(varVals[j])
        writer.writerow(varRow)

        vals = np.zeros((len(allFolds), len(allFolds[0].gapParamRel)))
        for i in range(len(allFolds)):
            rowGapParamRel = []
            rowGapParamRel.append("GapParamRel")
            for j in range(len(allFolds[i].gapParamRel)):
                rowGapParamRel.append(allFolds[i].gapParamRel[j].item())
                vals[i, j] = allFolds[i].gapParamRel[j].item()
            writer.writerow(rowGapParamRel)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD GapParamRel")
        for j in range(len(allFolds[0].gapParamRel)):
            varRow.append(varVals[j])
        writer.writerow(varRow)








        vals = np.zeros((len(allFolds), len(allFolds[0].multiVisitVarShParam)))
        for i in range(len(allFolds)):
            rowMultiVisitVarShParam = []
            rowMultiVisitVarShParam.append("multiVisitVarShParam")
            for j in range(len(allFolds[i].multiVisitVarShParam)):
                rowMultiVisitVarShParam.append(allFolds[i].multiVisitVarShParam[j].item())
                vals[i, j] = allFolds[i].multiVisitVarShParam[j].item()
            writer.writerow(rowMultiVisitVarShParam)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD multiVisitVarShParam")
        for j in range(len(allFolds[0].multiVisitVarShParam)):
            varRow.append(varVals[j])
        writer.writerow(varRow)

        vals = np.zeros((len(allFolds), len(allFolds[0].multiVisitVarSchParam)))
        for i in range(len(allFolds)):
            rowMultiVisitVarSchParam = []
            rowMultiVisitVarSchParam.append("multiVisitVarSchParam")
            for j in range(len(allFolds[i].multiVisitVarSchParam)):
                rowMultiVisitVarSchParam.append(allFolds[i].multiVisitVarSchParam[j].item())
                vals[i, j] = allFolds[i].multiVisitVarSchParam[j].item()
            writer.writerow(rowMultiVisitVarSchParam)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD multiVisitVarSchParam")
        for j in range(len(allFolds[0].multiVisitVarSchParam)):
            varRow.append(varVals[j])
        writer.writerow(varRow)

        vals = np.zeros((len(allFolds), len(allFolds[0].multiVisitVarRelParam)))
        for i in range(len(allFolds)):
            rowMultiVisitVarRelParam = []
            rowMultiVisitVarRelParam.append("multiVisitVarRelParam")
            for j in range(len(allFolds[i].multiVisitVarRelParam)):
                rowMultiVisitVarRelParam.append(allFolds[i].multiVisitVarRelParam[j].item())
                vals[i, j] = allFolds[i].multiVisitVarRelParam[j].item()
            writer.writerow(rowMultiVisitVarRelParam)
        varVals = vals.std(axis=0)
        varRow = []
        varRow.append("STD multiVisitVarRelParam")
        for j in range(len(allFolds[0].multiVisitVarRelParam)):
            varRow.append(varVals[j])
        writer.writerow(varRow)
