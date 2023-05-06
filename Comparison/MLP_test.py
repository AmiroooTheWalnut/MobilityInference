from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import numpy as np
data = pd.read_csv('TabularData_Tucson.csv')
data = data.to_numpy()

for i in range(len(data)):
    XTrain = np.copy(data)
    XTrain = XTrain[:, 0:4]
    XTrain = np.delete(XTrain, i, 0)
    YTrain = np.copy(data)
    YTrain = YTrain[:, 4:7]
    YTrain = np.delete(YTrain, i, 0)

    XTest = data[i, 0:4].reshape((1,4))
    YTest = data[i, 4:7].reshape((1,3))

    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(XTrain, YTrain)
    pred = clf.predict(XTest)
    error = (abs(pred[0, 0] - YTest[0, 0]) / YTest[0, 0]) + (abs(pred[0, 1] - YTest[0, 1]) / YTest[0, 1]) + (abs(pred[0, 2] - YTest[0, 2]) / YTest[0, 2])
    print("error tree: {}".format(error))

for i in range(len(data)):
    XTrain = np.copy(data)
    XTrain = XTrain[:, 0:4]
    XTrain = np.delete(XTrain, i, 0)
    YTrain = np.copy(data)
    YTrain = YTrain[:, 4:7]
    YTrain = np.delete(YTrain, i, 0)

    XTest = data[i, 0:4].reshape((1,4))
    YTest = data[i, 4:7].reshape((1,3))

    regr = MLPRegressor(random_state=1, max_iter=50000).fit(XTrain, YTrain)
    pred = regr.predict(XTest)
    error = (abs(pred[0, 0] - YTest[0, 0]) / YTest[0, 0]) + (abs(pred[0, 1] - YTest[0, 1]) / YTest[0, 1]) + (abs(pred[0, 2] - YTest[0, 2]) / YTest[0, 2])
    print("error MLP: {}".format(error))

