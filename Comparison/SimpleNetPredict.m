clc
clear
data=readmatrix('TabularData_Tucson.csv');
trainX=data(1:6,1:4);
trainY=data(1:6,5:7);
testX=data(7,1:4);
testY=data(7,5:7);