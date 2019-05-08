import numpy as np
def LoadDataSet():
    dataMat = []; labelMat = [] ;
    fr = open('text.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid (inX):
    return 1.0/(1+exit(-inX))

def gradAscent(dataMatin,classLabels):
    dataMatrix = np.mat(dataMatin)
    labelMat = np.mat(classLabels).transpose
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights= np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights

