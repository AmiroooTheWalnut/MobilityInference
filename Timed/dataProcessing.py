import pandas as pd
import torch
import json
import numpy as np
import os
from collections import namedtuple

class Config:
    def __init__(self, trainCityIndex,testCityIndex,trainTimeRange,testTimeRange):
        self.trainCityIndex=trainCityIndex
        self.testCityIndex=testCityIndex
        self.trainTimeRange=trainTimeRange
        self.testTimeRange=testTimeRange

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

class AllData:
    def __init__(self, trainData, testData):
        self.trainData=trainData
        self.testData=testData
class DataBundle:
    def __init__(self, city, dates):
        self.city = city
        self.dates = dates
        self.monthlyData = []

    @staticmethod
    def loadDataToGPU(data):
        # data.pOIs.cuda()
        # data.pOIShops.cuda()
        # data.pOISchools.cuda()
        # data.pOIReligion.cuda()
        # data.pOIShopProb.cuda()
        # data.pOISchoolProb.cuda()
        # data.pOIReligionProb.cuda()
        # data.needsTensor.cuda()
        # data.alpha_paramShop.cuda()
        # data.alpha_paramSchool.cuda()
        # data.alpha_paramReligion.cuda()
        # data.beta_paramShop.cuda()
        # data.beta_paramSchool.cuda()
        # data.beta_paramReligion.cuda()

        data.pOIs=data.pOIs.cuda()
        data.needsTensor=data.needsTensor.cuda()
        data.pOIShops=data.pOIShops.cuda()
        data.pOISchools=data.pOISchools.cuda()
        data.pOIReligion=data.pOIReligion.cuda()

        data.shopFrac = data.shopFrac.cuda()
        data.schoolFrac = data.schoolFrac.cuda()
        data.relFrac = data.relFrac.cuda()

        data.pOIShopProb=data.pOIShopProb.cuda()
        data.pOISchoolProb=data.pOISchoolProb.cuda()
        data.pOIReligionProb=data.pOIReligionProb.cuda()
        # data.ageCategories.cuda()
        # data.occupationCategories.cuda()
        # data.needCategories.cuda()
        data.ageProb=data.ageProb.cuda()  # age0: 0-18, age1: 18-65, age2: 65+
        data.needsTensor=data.needsTensor.cuda()
        # data.ageProb.cuda()
        data.occupationProb=data.occupationProb.cuda() # occupation0: student, occupation1: service, occupation2: driver, occupation3: education, occupation4: unemployed

        # data.P.cuda() # places
        # data.N.cuda()  # people
        # data.Nshop.cuda()
        # data.Nschool.cuda()
        # data.Nreligion.cuda()
        # data.D.cuda()  # days
        # data.M.cuda()  # months
        # data.NE.cuda()  # needs
        # data.G.cuda()  # age/occupation groups
        # self.needsTensor = torch.tensor(self.needs.values).div(4).ceil()
        # self.isFirst = data[3]
        data.alpha_paramShop=data.alpha_paramShop.cuda()
        data.alpha_paramSchool=data.alpha_paramSchool.cuda()
        data.alpha_paramReligion=data.alpha_paramReligion.cuda()
        data.beta_paramShop=data.beta_paramShop.cuda()
        data.beta_paramSchool=data.beta_paramSchool.cuda()
        data.beta_paramReligion=data.beta_paramReligion.cuda()
        data.populationCBG=data.populationCBG.cuda()
        # data.NCBG.cuda()
        data.populationNum=data.populationNum.cuda()

        # self.BBNSh=0
        # self.BBNSch=0
        # self.BBNRel=0

        data.gapVal=data.gapVal.cuda()

        # data.globalError.cuda()

        # RELATED TO CBG MODELING
        data.groupProbs=data.groupProbs.cuda()
        data.needsSh=data.needsSh.cuda()
        data.needsSch=data.needsSch.cuda()
        data.needsRel=data.needsRel.cuda()
        data.BBNSh=data.BBNSh.cuda()
        data.BBNSch=data.BBNSch.cuda()
        data.BBNRel=data.BBNRel.cuda()

        # DEBUG
        # self.expectationDebugCounter = 0
        # self.resultFromSampleSumIP = 0
        # self.resultFromSampleSumIB = 0
        # self.resultFromAvgAllIP = []
        # self.resultFromAvgAllIB = []
        # self.resultFromEEAll = []
        # self.resultSamplesIP = []
        # self.resultSamplesIB = []

    @staticmethod
    def unloadDataToGPU(data):
        del data.pOIs
        del data.pOIShops
        del data.pOISchools
        del data.pOIReligion
        del data.pOIShopProb
        del data.pOISchoolProb
        del data.pOIReligionProb
        del data.needsTensor
        del data.alpha_paramShop
        del data.alpha_paramSchool
        del data.alpha_paramReligion
        del data.beta_paramShop
        del data.beta_paramSchool
        del data.beta_paramReligion

class MonthData:
    def __init__(self, data):
        self.isTrainedOnOneMonth = 0

        self.pOIs = torch.tensor(data[0].values)
        self.needs = data[1]

        self.isModel2 = 0

        self.pOIShops = torch.tensor(data[4].iloc[:, 1:].values)
        self.pOISchools = torch.tensor(data[5].iloc[:, 1:].values)
        self.pOIReligion = torch.tensor(data[6].iloc[:, 1:].values)
        self.pOIShopProb = torch.tensor(data[7].values).flatten()
        self.pOISchoolProb = torch.tensor(data[8].values).flatten()
        self.pOIReligionProb = torch.tensor(data[9].values).flatten()

        self.shopFrac = torch.tensor(data[12][0].values)
        self.schoolFrac = torch.tensor(data[12][1].values)
        self.relFrac = torch.tensor(data[12][2].values)

        self.cBGShopProb = torch.tensor(data[13].transpose().values).cuda()
        self.cBGSchoolProb = torch.tensor(data[14].transpose().values).cuda()
        self.cBGReligionProb = torch.tensor(data[15].transpose().values).cuda()

        self.ageCategories = 3
        self.occupationCategories = 5
        self.needCategories = 3
        self.ageProb = torch.zeros(3)  # age0: 0-18, age1: 18-65, age2: 65+
        self.ageProb[0] = 0.25
        self.ageProb[1] = 0.58
        self.ageProb[2] = 0.17
        self.occupationProb = torch.zeros(3, 5)  # occupation0: student, occupation1: service, occupation2: driver, occupation3: education, occupation4: unemployed
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
        self.N = data[2][0][0]  # people
        self.Nshop = data[4].shape[0]
        self.Nschool = data[5].shape[0]
        self.Nreligion = data[6].shape[0]
        self.D = data[4].shape[1]  # days
        self.M = 1  # months
        self.NE = 3  # needs
        self.G = 15  # age/occupation groups

        self.oneAuxVal = torch.ones(self.G).cuda()

        #self.needsTensor = torch.tensor(self.needs.values).div(4).ceil()
        self.needsTensor = torch.tensor(self.needs.values)

        self.nonZeroNeedsShopIndices = self.needsTensor[:, 0].nonzero().flatten().cuda()
        self.nonZeroNeedsSchoolIndices = self.needsTensor[:, 1].nonzero().flatten().cuda()
        self.nonZeroNeedsRelIndices = self.needsTensor[:, 2].nonzero().flatten().cuda()

        self.isFirst = data[3]
        self.alpha_paramShop = torch.ones(self.G)
        self.alpha_paramSchool = torch.ones(self.G)
        self.alpha_paramReligion = torch.ones(self.G)
        self.beta_paramShop = torch.ones(self.G)
        self.beta_paramSchool = torch.ones(self.G)
        self.beta_paramReligion = torch.ones(self.G)
        self.multiVisitVarShParam = torch.ones(1)
        self.multiVisitVarSchParam = torch.ones(1)
        self.multiVisitVarRelParam = torch.ones(1)
        # self.obsVarShParam = torch.ones(1)
        # self.obsVarSchParam = torch.ones(1)
        # self.obsVarRelParam = torch.ones(1)
        self.populationCBG = torch.tensor(data[10].values)
        self.NCBG = data[10].shape[0]
        self.populationNum = (self.populationCBG*self.N).flatten()

        # self.BBNSh=0
        # self.BBNSch=0
        # self.BBNRel=0

        self.globalError = np.zeros(1, dtype=np.float32)
        self.globalErrorFrac = np.zeros(1, dtype=np.float32)

        # RELATED TO CBG MODELING
        if data[11] == 1:
            self.groupProbs = (torch.transpose(self.occupationProb, 0, 1) * self.ageProb).reshape([self.G, 1])
            self.needsSh = self.needsTensor[:, 0].reshape([self.occupationProb.shape[0], self.occupationProb.shape[1]]).transpose(0, 1).reshape([self.G, 1])
            self.needsSch = self.needsTensor[:, 1].reshape([self.occupationProb.shape[0], self.occupationProb.shape[1]]).transpose(0, 1).reshape([self.G, 1])
            self.needsRel = self.needsTensor[:, 2].reshape([self.occupationProb.shape[0], self.occupationProb.shape[1]]).transpose(0, 1).reshape([self.G, 1])
        elif data[11] == 2:
            self.groupProbs = (torch.transpose(self.occupationProb, 0, 1) * self.ageProb).reshape([self.G, 1])
            self.needsSh = self.needsTensor[:, 0].reshape([self.occupationProb.shape[0], self.occupationProb.shape[1]]).transpose(0, 1).reshape([self.G, 1])
            self.needsSch = self.needsTensor[:, 1].reshape([self.occupationProb.shape[0], self.occupationProb.shape[1]]).transpose(0, 1).reshape([self.G, 1])
            self.needsRel = self.needsTensor[:, 2].reshape([self.occupationProb.shape[0], self.occupationProb.shape[1]]).transpose(0, 1).reshape([self.G, 1])
            self.needsSh = self.needsSh / self.Nshop
            self.needsSch = self.needsSch / self.Nschool
            self.needsRel = self.needsRel / self.Nreligion
        self.BBNSh = (self.needsSh * self.groupProbs * self.populationNum.repeat([self.G, 1])).round()
        self.BBNSch = (self.needsSch * self.groupProbs * self.populationNum.repeat([self.G, 1])).round()
        self.BBNRel = (self.needsRel * self.groupProbs * self.populationNum.repeat([self.G, 1])).round()
        self.BBNSh[self.BBNSh == 0] = 1
        self.BBNSch[self.BBNSch == 0] = 1
        self.BBNRel[self.BBNRel == 0] = 1

        # RELATED TO GAP
        self.gap_param_up = torch.ones(1)
        self.gap_param_down = torch.ones(1)
        self.gapVal = torch.tensor(1)
        self.gapParam = torch.ones(1)

        # DEBUG
        self.expectationDebugCounter = 0
        self.resultFromSampleSumIP = 0
        self.resultFromSampleSumIB = 0
        self.resultFromAvgAllIP = []
        self.resultFromAvgAllIB = []
        self.resultFromEEAll = []
        self.resultSamplesIP = []
        self.resultSamplesIB = []

class RunConfig:
    def __init__(self, trainCityIndex, trainMonthIndices, testCityIndex, testMonthIndices):
        self.trainCityIndex=trainCityIndex
        self.trainMonthIndices=trainMonthIndices
        self.testCityIndex=testCityIndex
        self.testMonthIndices=testMonthIndices

def loadData(cityTrain, cityTest, dates, monthsTrain, monthsTest, modelTypeIndex):
    trainBundle = DataBundle(cityTrain, monthsTrain)
    for i in range(len(monthsTrain)):
        visits = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'FullSimple_' + dates[monthsTrain[i]] + '.csv', header=None)
        pOIShops = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'shopLocVis_' + dates[monthsTrain[i]] + '.csv', header=None)
        pOISchools = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'schoolLocVis_' + dates[monthsTrain[i]] + '.csv', header=None)
        pOIReligion = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'religionLocVis_' + dates[monthsTrain[i]] + '.csv', header=None)

        shopFrac = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'MVShop_' + dates[monthsTrain[i]] + '.csv', header=None)
        schoolFrac = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'MVSchool_' + dates[monthsTrain[i]] + '.csv', header=None)
        religionFrac = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'MVReligion_' + dates[monthsTrain[i]] + '.csv', header=None)

        cBGShopProb = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'sourceCBG_shopCBG_probability_' + dates[monthsTrain[i]] + '.csv', header=None)
        cBGSchoolProb = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'sourceCBG_schoolCBG_probability_' + dates[monthsTrain[i]] + '.csv', header=None)
        cBGReligionProb = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'sourceCBG_religionCBG_probability_' + dates[monthsTrain[i]] + '.csv', header=None)

        population = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + cityTrain + '_population.csv', header=None)
        populationCBG = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + cityTrain + '_CBGPopProb.csv', header=None)
        needs = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + 'Needs_data_numbers.csv', header=None)

        isFirst = True

        pOIShopsProb = pd.DataFrame(np.zeros(pOIShops.shape[0]))
        pOISchoolsProb = pd.DataFrame(np.zeros(pOISchools.shape[0]))
        pOIReligionProb = pd.DataFrame(np.zeros(pOIReligion.shape[0]))
        sShop = pOIShops.iloc[:, 1].sum()
        for j in range(pOIShops.shape[0]):
            pOIShopsProb.at[j, 0] = pOIShops.iloc[j, 1] / sShop
        sSch = pOISchools.iloc[:, 1].sum()
        for j in range(pOISchools.shape[0]):
            pOISchoolsProb.at[j, 0] = pOISchools.iloc[j, 1] / sSch
        sRel = pOIReligion.iloc[:, 1].sum()
        for j in range(pOIReligion.shape[0]):
            pOIReligionProb.at[j, 0] = pOIReligion.iloc[j, 1] / sRel

        fracs = [shopFrac,schoolFrac,religionFrac]

        data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb, populationCBG, modelTypeIndex, fracs, cBGShopProb, cBGSchoolProb, cBGReligionProb]

        monthData = MonthData(data)
        trainBundle.monthlyData.append(monthData)

    testBundle = DataBundle(cityTest, monthsTest)
    for i in range(len(monthsTest)):
        visits = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTest + os.sep + 'FullSimple_' + dates[monthsTest[i]] + '.csv', header=None)
        pOIShops = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTest + os.sep + 'shopLocVis_' + dates[monthsTest[i]] + '.csv', header=None)
        pOISchools = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTest + os.sep + 'schoolLocVis_' + dates[monthsTest[i]] + '.csv', header=None)
        pOIReligion = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTest + os.sep + 'religionLocVis_' + dates[monthsTest[i]] + '.csv', header=None)

        shopFrac = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'MVShop_' + dates[monthsTest[i]] + '.csv', header=None)
        schoolFrac = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'MVSchool_' + dates[monthsTest[i]] + '.csv', header=None)
        religionFrac = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'MVReligion_' + dates[monthsTest[i]] + '.csv', header=None)

        cBGShopProb = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'sourceCBG_shopCBG_probability_' + dates[monthsTest[i]] + '.csv', header=None)
        cBGSchoolProb = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'sourceCBG_schoolCBG_probability_' + dates[monthsTest[i]] + '.csv', header=None)
        cBGReligionProb = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'sourceCBG_religionCBG_probability_' + dates[monthsTest[i]] + '.csv', header=None)

        population = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + cityTest + '_population.csv', header=None)
        populationCBG = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + cityTrain + '_CBGPopProb.csv', header=None)
        needs = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + 'Needs_data_numbers.csv', header=None)

        isFirst = True

        pOIShopsProb = pd.DataFrame(np.zeros(pOIShops.shape[0]))
        pOISchoolsProb = pd.DataFrame(np.zeros(pOISchools.shape[0]))
        pOIReligionProb = pd.DataFrame(np.zeros(pOIReligion.shape[0]))
        sShop = pOIShops.iloc[:, 1].sum()
        for j in range(pOIShops.shape[0]):
            pOIShopsProb.at[j, 0] = pOIShops.iloc[j, 1] / sShop
        sSch = pOISchools.iloc[:, 1].sum()
        for j in range(pOISchools.shape[0]):
            pOISchoolsProb.at[j, 0] = pOISchools.iloc[j, 1] / sSch
        sRel = pOIReligion.iloc[:, 1].sum()
        for j in range(pOIReligion.shape[0]):
            pOIReligionProb.at[j, 0] = pOIReligion.iloc[j, 1] / sRel

        fracs = [shopFrac, schoolFrac, religionFrac]

        data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb, populationCBG, modelTypeIndex, fracs, cBGShopProb, cBGSchoolProb, cBGReligionProb]
        monthData = MonthData(data)
        testBundle.monthlyData.append(monthData)

    allData = AllData(trainBundle, testBundle)

    return allData

def processInputRanges(input):
    output = []
    skip = 0
    for i in range(len(input)):
        if skip:
            skip -= 1
            continue
        if i == 0:
            if input[i].isnumeric() == False:
                print("Wrong input, first letter is not a number. CODE STOPPED!")
            if i + 1 < len(input):
                if input[i + 1].isnumeric() == True:
                    output.append(int(input[i] + input[i + 1])-1)
                    skip = 1
                else:
                    output.append(int(input[i])-1)
            else:
                output.append(int(input[i])-1)
        else:
            if input[i] == '-':
                rightSide = 0
                if i+2<len(input):
                    if input[i + 2].isnumeric() == True:
                        rightSide = int(input[i + 1] + input[i + 2])-1
                        skip = 2
                    else:
                        rightSide = int(input[i + 1])-1
                        skip = 1
                else:
                    rightSide = int(input[i + 1])-1
                    skip = 1
                leftSide = output[len(output) - 1]
                for m in range(leftSide + 1, rightSide + 1):
                    output.append(m)
            elif input[i].isnumeric():
                if i + 1 < len(input):
                    if input[i + 1].isnumeric() == True:
                        output.append(int(input[i] + input[i + 1])-1)
                        skip = 1
                    else:
                        output.append(int(input[i])-1)
                else:
                    output.append(int(input[i])-1)

    return output

def customConfigDecoder(config):
    return namedtuple('X', config.keys())(*config.values())