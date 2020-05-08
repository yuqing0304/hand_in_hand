#!/bin/usr/env python3
# *-- coding:utf-8 --*
# 李柱 201921198685
'''
    第八次作业（集成学习）
'''
import numpy as np

def loadData():
    datMat = np.matrix([[0., 1., 3.],
                     [0., 3., 1.],
                     [1., 2., 2.],
                     [1., 1., 3.],
                     [1., 2., 3.],
                     [0., 1., 2.],
                     [1., 1., 2.],
                     [1., 1., 1.],
                     [1., 3., 1.],
                     [0., 2., 1.]])
    classLabels = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    return datMat, classLabels


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t'))  # get number of fields
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  # init error sum, to +infinity
    for i in range(n):  # loop over all dimensions
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max();

        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)

                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan
                errArr = np.mat(np.ones((m, 1)))

                errArr[predictedVals == labelMat] = 0

                weightedError = D.T * errArr  # calc total error multiplied by D

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


if __name__ == '__main__':
    datMat, classLabels = loadData()
    d = np.mat(np.ones((10, 1)) / 10)
    bestStump, minError, bestClasEst = buildStump(datMat, classLabels, d)
    print("BestStump: {},\n minError: {},\n BestClasEst: \n{}".format(bestStump, minError, bestClasEst))