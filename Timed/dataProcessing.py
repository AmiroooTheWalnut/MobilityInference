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
        data.pOIs.cuda()
        data.pOIShops.cuda()
        data.pOISchools.cuda()
        data.pOIReligion.cuda()
        data.pOIShopProb.cuda()
        data.pOISchoolProb.cuda()
        data.pOIReligionProb.cuda()
        data.needsTensor.cuda()
        data.alpha_paramShop.cuda()
        data.alpha_paramSchool.cuda()
        data.alpha_paramReligion.cuda()
        data.beta_paramShop.cuda()
        data.beta_paramSchool.cuda()
        data.beta_paramReligion.cuda()

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
        self.pOIs = torch.tensor(data[0].values)
        self.needs = data[1]
        self.pOIShops = torch.tensor(data[4].iloc[:, 1:].values)
        self.pOISchools = torch.tensor(data[5].iloc[:, 1:].values)
        self.pOIReligion = torch.tensor(data[6].iloc[:, 1:].values)
        self.pOIShopProb = torch.tensor(data[7].values).flatten()
        self.pOISchoolProb = torch.tensor(data[8].values).flatten()
        self.pOIReligionProb = torch.tensor(data[9].values).flatten()
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
        #self.needsTensor = torch.tensor(self.needs.values).div(4).ceil()
        self.needsTensor = torch.tensor(self.needs.values)
        self.isFirst = data[3]
        self.alpha_paramShop = torch.ones(self.G)
        self.alpha_paramSchool = torch.ones(self.G)
        self.alpha_paramReligion = torch.ones(self.G)
        self.beta_paramShop = torch.ones(self.G)
        self.beta_paramSchool = torch.ones(self.G)
        self.beta_paramReligion = torch.ones(self.G)
        self.populationCBG = torch.tensor(data[10].values)
        self.NCBG = data[10].shape[0]
        self.populationNum = (self.populationCBG*self.N).flatten()

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

def loadData(cityTrain, cityTest, dates, monthsTrain, monthsTest):
    trainBundle = DataBundle(cityTrain, monthsTrain)
    for i in range(len(monthsTrain)):
        visits = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'FullSimple_' + dates[monthsTrain[i]] + '.csv', header=None)
        pOIShops = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'shopLocVis_' + dates[monthsTrain[i]] + '.csv', header=None)
        pOISchools = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'schoolLocVis_' + dates[monthsTrain[i]] + '.csv', header=None)
        pOIReligion = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTrain + os.sep + 'religionLocVis_' + dates[monthsTrain[i]] + '.csv', header=None)

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

        data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb, populationCBG]
        monthData = MonthData(data)
        trainBundle.monthlyData.append(monthData)

    testBundle = DataBundle(cityTest, monthsTest)
    for i in range(len(monthsTrain)):
        visits = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTest + os.sep + 'FullSimple_' + dates[monthsTest[i]] + '.csv', header=None)
        pOIShops = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTest + os.sep + 'shopLocVis_' + dates[monthsTest[i]] + '.csv', header=None)
        pOISchools = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTest + os.sep + 'schoolLocVis_' + dates[monthsTest[i]] + '.csv', header=None)
        pOIReligion = pd.read_csv('..' + os.sep + 'TimedData' + os.sep + cityTest + os.sep + 'religionLocVis_' + dates[monthsTest[i]] + '.csv', header=None)

        population = pd.read_csv('..' + os.sep + 'FixedData' + os.sep + cityTest + '_population.csv', header=None)
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

        data = [visits, needs, population, isFirst, pOIShops, pOISchools, pOIReligion, pOIShopsProb, pOISchoolsProb, pOIReligionProb, populationCBG]
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