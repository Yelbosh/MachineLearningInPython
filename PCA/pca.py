# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 19:42:23 2016
@author: Yelbosh
code of PCA Algrithom
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#根据要求的方差百分比，求出所需要的特征值的个数n
def percent2n(eigVals,percent):
    sortArray = np.sort(eigVals) #升序
    sortArray = sortArray[::-1]  #逆转，即降序
    arraySum = sum(sortArray)
    tmp = 0
    num = 0
    for i in sortArray:
        tmp += i
        num += 1
        if tmp >= arraySum*percent:
            return num
#零均值化
def zeroMean(dataMat):      
    meanVal = np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData,meanVal

#pca算法主题部分
def pca(dataMat,percent=0.95):
    newData,meanVal = zeroMean(dataMat)
    covMat = np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n = percent2n(eigVals,percent)                #要达到percent的方差百分比，需要前n个特征向量
    eigValIndice = np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect = eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect             #低维特征空间的数据
    reconMat = (lowDDataMat*n_eigVect.T) + meanVal    #重构数据 .T 转制矩阵,是将降维的数据重新还原
    return lowDDataMat,reconMat

def loaddata(datafile):
    return np.array(pd.read_csv(datafile,sep="\t",header=-1)).astype(np.float)

def plotBestFit(data1, data2):    
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)
    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        axis_x2.append(dataArr2[i,0]) 
        axis_y2.append(dataArr2[i,1])                 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()  
    
def main():    
    datafile = "data.txt"
    XMat = loaddata(datafile)
    k = 0.95
    return pca(XMat, k)
    
if __name__ == "__main__":
    finalData, reconMat = main()
    plotBestFit(finalData, reconMat)

