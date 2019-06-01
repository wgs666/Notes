'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # print(stringArr)
    datArr = [list(map(float,line)) for line in stringArr]
    # python3 needs list()
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    print(meanVals)
    print("----")
    meanRemoved = dataMat - meanVals #remove mean, then the mean of each dimension will be 0
    print(meanRemoved)
    print("----")
    covMat = cov(meanRemoved, rowvar=0)
    print(covMat)
    print("----")
    eigVals,eigVects = linalg.eig(mat(covMat))
    print("eigVals:")
    print(eigVals)
    print("----")
    print("eigVects:")
    print(eigVects)
    print("----")
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest, return the index
    # here is a demo for argsort()
    # x = array([1,4,3,-1,6,9])
    # y = x.argsort() 
    # print(y)  
    # y is [3 0 2 1 4 5], because x[3]==-1 is the smallest, x[5]==9 is the largest

    print("eigValInd:")
    print(eigValInd)
    print("----")
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    print("eigValInd cut off:")
    print(eigValInd)
    print("----")
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    print("redEigVects:")
    print(redEigVects)
    print("----")
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    print("lowDDataMat:")
    print(lowDDataMat)
    print("----")
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print("reconMat:")
    print(reconMat)
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

dataMat = loadDataSet('testSet0.txt')
# dataMat = loadDataSet('testSet.txt')
lowDMat, reconMat = pca(dataMat, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],  dataMat[:,1].flatten().A[0], marker='x', s=90)
ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
plt.show()
