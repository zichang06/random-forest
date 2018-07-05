import csv
from collections import defaultdict
import pandas as pd  
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import math
import random  
import threading
import time
import sys

trainSampleNum = 50
testSampleNum = 20
train_dir = "simple_data/train.txt"
test_dir = "simple_data/test.txt"
treeNum = 100

# trainSampleNum = 1719692
# testSampleNum = 429923
# train_dir = "data/train.txt"
# test_dir = "data/test.txt"
# treeNum = 100

preDir = "myForest_100_mul.csv"
featureNum = 201
threads = []

def writeCSV(predictLable, fileName = "pre.csv"):
    print(">> writing to csv %s..." %fileName)
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

class myForest:
    '''
    '''
    def __init__(self, n_bootstrapSamples=10):
        self.n_bootstrapSamples = n_bootstrapSamples
        self.list_tree = []  # 随机森林
        self.list_featureIndex= [] 

    
    def generateBootstrapSamples(self, data):
        '''
        构造bootstrap样本，对已得一般的样本随机选取70%的特征
        '''
        k = int(0.7 * featureNum)
        featureIndex = random.sample(range(len(data[0])-1), k)
        sampledData = np.zeros((len(data), k))
        for i in range(k):
            sampledData[:, i] = [x[featureIndex[i]] for x in data]

        return sampledData, featureIndex

    def getBootstrapSamples(self, data, featureIndex):
        '''
        根据该树的featureIndex,构建样本
        '''
        sampledData = np.zeros((len(data), len(featureIndex)))
        for i in range(len(featureIndex)):
            sampledData[:, i] = [x[featureIndex[i]] for x in data]

        return sampledData

    def generateHalfSamples(self, data, label):
        '''
        随机选取一半的样本  
        '''
        rowRange = np.arange(trainSampleNum)
        np.random.shuffle(rowRange)
        halfSampleNum = int(trainSampleNum / 2)
        halfData = [data[i] for i in rowRange[0:halfSampleNum]]
        halfLabel = [label[i] for i in rowRange[0:halfSampleNum]]
        return halfData, halfLabel
     
    def buildTree(self, data, label, index): 
        '''
        构建一棵树  
        '''
        print("split data for the %dst tree..." %(index + 1)) 
        halfData, sampleLabel = self.generateHalfSamples(data, label)
        sampledData, featureIndex = self.generateBootstrapSamples(halfData)
        print("finish spliting, building the %dst tree..." %(index + 1)) 
        currentTree = tree.DecisionTreeClassifier(max_depth = 10)
        currentTree.fit(sampledData, sampleLabel)

        self.list_tree.append(currentTree)
        self.list_featureIndex.append(featureIndex)

        del halfData, sampleLabel, sampledData, featureIndex


    def fitWithMultiThread(self, data, label):
        '''
        多线程构造随机森林
        '''
        for i in range(self.n_bootstrapSamples):
            t = threading.Thread(target=self.buildTree,args=(data, label, i))
            threads.append(t)
        
        for t in threads:
            t.setDaemon(True)
            t.start()   
        
        for t in threads:
            t.join()
        
        threads.clear()
        print("finish building forest tree, main thread go on...")

    def fit(self, data, label):
        '''
        构造随机森林,不是多线程
        '''
        for i in range(self.n_bootstrapSamples):
            self.buildTree(data, label, i)
        
    def treePredict(self, data, results, i):
        print("split data for the %dst tree..." %(i + 1)) 
        sampledData = self.getBootstrapSamples(data, self.list_featureIndex[i])
        print("finish spliting, predicting for the %dst tree..." %(i + 1)) 
        tmp = self.list_tree[i].predict(sampledData)
        results[:, i] = tmp
        del sampledData

    def predictWithMultiThread(self, data):
        '''
        多线程利用随机森林对给定观测数据进行分类
        '''
        print("predicting...") 
        results = np.zeros((len(data), self.n_bootstrapSamples))

        for i in range(len(self.list_tree)):
            t = threading.Thread(target=self.treePredict,args=(data, results, i))
            threads.append(t)

        for t in threads:
            t.setDaemon(True)
            t.start()   
        
        for t in threads:
            t.join()
        
        threshold = np.full(len(data), 0.5)
        tmp = np.mean(results, axis=1) # 计算每一行的均值
        finalResult = np.greater(tmp, threshold)
        return finalResult

    def predict(self, data):
        '''
        利用随机森林对给定观测数据进行分类
        '''
        print("predicting...") 
        results = np.zeros((len(data), self.n_bootstrapSamples))
        
        for i in range(len(self.list_tree)):
            self.treePredict( data, results, i)
        
        threshold = np.full(len(data), 0.5)
        tmp = np.mean(results, axis=1) # 计算每一行的均值
        finalResult = np.greater(tmp, threshold)
        return finalResult

if __name__ == '__main__':
    time_start=time.time()
    print('>> my Forest with %d trees.' %(treeNum))

    data = getData(train_dir, True)
    label = data[:,-1]
    data = data[:, :-1]   
    loadTrainingDataTime = float(time.time() - time_start)
    print('>> load training data time %.2fs.' %(loadTrainingDataTime))

    #clf = RandomForestClassifier(10)
    clf = myForest(treeNum)
    print(">> fitting...")
    clf.fitWithMultiThread(data, label)

    fitTime = float(time.time() - time_start)
    print('>> fit time %.2fs.' %(fitTime))


    data = getData(test_dir, False)
    loadTestDataTime = float(time.time() - fitTime - time_start)
    print('>> load test data time %.2fs.' %(loadTestDataTime))

    print(">> predicting...")
    pre = clf.predictWithMultiThread(data)
    pre = pre.astype(int)
    predictTime = float(time.time() - fitTime - time_start - loadTestDataTime)
    print('>> predict time %.2fs.' %(predictTime))

    writeCSV(pre, preDir)