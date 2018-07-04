import csv
from collections import defaultdict
import pandas as pd  
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.datasets import load_iris
import math
import random  
import threading
import time
import sys
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

trainSampleNum = 50
testSampleNum = 20
train_dir = "simple_data/train.txt"
test_dir = "simple_data/test.txt"

# trainSampleNum = 1719692
# testSampleNum = 429923
# train_dir = "data/train.txt"
# test_dir = "data/test.txt"

featureNum = 201
threads = []

def writeCSV(predictLable, fileName = "pre.csv"):
    print(">> writing to csv...")
    names = range(testSampleNum) 
    dataframe = pd.DataFrame({'id':names,'label':predictLable})
    dataframe.to_csv(fileName,index=False,sep=',', encoding = "utf-8")

def getData(dataDir = "train.txt", isTrain = True):
    print(">> start loading data %s..." %(dataDir))
    '''
    readin dataset according to dataDir
    if train dataset is needed
    return augmented trainingData[m, n+1], the last colomn is label(1 or 0)
    if test dataset is needed
    return augmented testData[m, n], without label

    '''
    if isTrain:
        data = np.zeros((trainSampleNum, featureNum + 1))
    else:
        data = np.zeros((testSampleNum, featureNum))

    f = open(dataDir)  
    lines = f.readline() 
    sampleCount = 0
    while lines:  
        line = lines.split(' ')
        for index in range(len(line)):
            if index == 0:
                if isTrain:
                    data[sampleCount, -1] = int(line[0])
                continue
            colon = line[index].index(':')
            data[sampleCount, int(line[index][:colon]) - 1] = float(line[index][colon+1:])
        lines = f.readline()  
        sampleCount += 1
        sys.stdout.write('\r>> %d ' % (sampleCount))
    f.close()  
    print(">> finish load data %s..." %(dataDir))
    
    return data

if __name__ == '__main__':
    time_start=time.time()

    data = getData(train_dir, True)
    label = data[:,-1]
    data = data[:, :-1]   
    loadTrainingDataTime = float(time.time() - time_start)
    print('>> load training data time %.2fs.' %(loadTrainingDataTime))

    clf = RandomForestClassifier(n_estimators=10)
    print(">> fitting...")
    clf.fit(data, label)

    fitTime = float(time.time() - time_start)
    print('>> fit time %.2fs.' %(fitTime))


    data = getData(test_dir, False)
    #data = min_max_scaler.fit_transform(data)  
    loadTestDataTime = float(time.time() - fitTime - time_start)
    print('>> predict time %.2fs.' %(loadTestDataTime))

    print(">> predicting...")
    pre = clf.predict(data)
    pre.astype(int)
    predictTime = float(time.time() - fitTime - time_start - loadTestDataTime)
    print('>> predict time %.2fs.' %(predictTime))

    writeCSV(pre, "predForest.csv")