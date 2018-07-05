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

from myTree import myTree

trainSampleNum = 50
testSampleNum = 20
train_dir = "simple_data/train.txt"
test_dir = "simple_data/test.txt"
treeNum = 1
maxDepth = 5

# trainSampleNum = 1719692
# testSampleNum = 429923
# train_dir = "data/train.txt"
# test_dir = "data/test.txt"
# treeNum = 20
# maxDepth = 5

preDir = "test.csv"
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
    print("\n>> finish load data %s..." %(dataDir))
    
    return data

class myForest:
    def __init__(self, n_bootstrapSamples=10):
        self.n_bootstrapSamples = n_bootstrapSamples
        self.list_tree = []  
        self.list_featureIndex= [] 

    
    def generateBootstrapSamples(self):
        '''
        构造bootstrap样本，随机选取50%的样本，并随机选取70%的特征
        返回两个list，rows和columns
        '''
        k = int(0.7 * featureNum)
        columns = random.sample(range(len(data[0])-1), k)

        rowRange = np.arange(trainSampleNum)
        np.random.shuffle(rowRange)
        halfSampleNum = int(trainSampleNum / 2)
        rows = rowRange[:halfSampleNum]

        return rows, columns
     
    def buildTree(self, data, label, index): 
        '''
        构建一棵树  
        '''
        # 返回两个list
        rows, columns = self.generateBootstrapSamples()
        print("building the %dst tree..." %(index + 1)) 
        currentTree = myTree(maxLevel = 10)
        currentTree.fit(data = self.data, label = self.label, rows = rows, columns = columns)

        self.list_tree.append(currentTree)
        self.list_featureIndex.append(columns)

    def fit(self, data, label):
        '''
        构造随机森林,不是多线程
        '''
        self.data = data
        self.label = label

        for i in range(self.n_bootstrapSamples):
            self.buildTree(self.data, label, i)
        
    def fitWithMultiThread(self, data, label):
        '''
        多线程构造随机森林
        '''
        self.data = data
        self.label = label

        for i in range(self.n_bootstrapSamples):
            t = threading.Thread(target=self.buildTree,args=(label, i))
            threads.append(t)
        
        for t in threads:
            t.setDaemon(True)
            t.start()   
        
        for t in threads:
            t.join()
        
        threads.clear()
        print("finish building forest tree, main thread go on...")

    def treePredict(self, results, i):
        '''
        一棵树预测
        '''
        print("predicting for the %dst tree..." %(i + 1)) 
        tmp = self.list_tree[i].predict(self.data)
        results[:, i] = tmp

    def predict(self, data):
        '''
        利用随机森林对给定观测数据进行分类
        '''
        print("predicting...") 
        self.data = data
        results = np.zeros((len(self.data), self.n_bootstrapSamples))
        
        for i in range(len(self.list_tree)):
            self.treePredict(results, i)
        
        threshold = np.full(len(self.data), 0.5)
        tmp = np.mean(results, axis=1) # 计算每一行的均值
        finalResult = np.greater(tmp, threshold)
        return finalResult

    def predictWithMultiThread(self, data):
        '''
        多线程利用随机森林对给定观测数据进行分类
        '''
        self.data = data
        print("predicting...") 
        results = np.zeros((len(self.data), self.n_bootstrapSamples))

        for i in range(len(self.list_tree)):
            t = threading.Thread(target=self.treePredict,args=(results, i))
            threads.append(t)

        for t in threads:
            t.setDaemon(True)
            t.start()   
        
        for t in threads:
            t.join()
        
        threshold = np.full(len(self.data), 0.5)
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
    clf = myForest(treeNum, maxDepth)
    print(">> fitting...")
    clf.fit(data, label)

    fitTime = float(time.time() - time_start)
    print('>> fit time %.2fs.' %(fitTime))


    data = getData(test_dir, False)
    loadTestDataTime = float(time.time() - fitTime - time_start)
    print('>> load test data time %.2fs.' %(loadTestDataTime))

    print(">> predicting...")
    pre = clf.predict(data)
    pre = pre.astype(int)
    predictTime = float(time.time() - fitTime - time_start - loadTestDataTime)
    print('>> predict time %.2fs.' %(predictTime))

    writeCSV(pre, preDir)