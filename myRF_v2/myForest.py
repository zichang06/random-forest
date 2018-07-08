import csv
from collections import defaultdict
import pandas as pd  
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
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

trainSampleNum = 1719692
testSampleNum = 429923
train_dir = "data/train.txt"
test_dir = "data/test.txt"
treeNum = 1
maxDepth = 5

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

def sortFeatures(data, label):
    '''
    '''
    sortedFeatures = []
    for j in range(len(data[0])):
        results = {}
        for i in range(len(data)):
            r = data[i][j]
            if r not in results:
                results[r] = {1: [], 0:[]}
            if label[i] == 1:
                results[r][1].append(i)
            if label[i] == 0:
                results[r][0].append(i)
        results = OrderedDict(sorted(results.items()))
        sortedFeatures.append(results)
    print(">>finish sorting Features.")
    return sortedFeatures

class myForest:
    def __init__(self, n_bootstrapSamples=10, maxDepth = 10):
        self.n_bootstrapSamples = n_bootstrapSamples
        self.list_tree = []  
        self.list_featureIndex= [] 
        self.maxDepth = maxDepth

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
     
    def buildTree(self, index): 
        '''
        构建一棵树  
        '''
        # 返回两个list
        rows, columns = self.generateBootstrapSamples()
        print("building the %dst tree..." %(index + 1)) 
        currentTree = myTree(maxLevel = self.maxDepth)
        currentTree.fit(data = self.data, label = self.label, sortedFeatures = self.sortedFeatures, rows = rows, columns = columns)

        self.list_tree.append(currentTree)
        self.list_featureIndex.append(columns)
        print("finish building the %dst tree..." %(index + 1)) 

    def fit(self, data, label, sortedFeatures):
        '''
        构造随机森林,不是多线程
        '''
        self.data = data
        self.label = label
        self.sortedFeatures = sortedFeatures

        for i in range(self.n_bootstrapSamples):
            self.buildTree(i)
        
    def fitWithMultiThread(self, data, label, sortedFeatures):
        '''
        多线程构造随机森林
        '''
        self.data = data
        self.label = label
        self.sortedFeatures = sortedFeatures

        for i in range(self.n_bootstrapSamples):
            t = threading.Thread(target=self.buildTree,args=(i,))
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
        print("finish predicting for the %dst tree..." %(i + 1)) 

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
    print('>> my Forest with %d trees(depth: %d)' %(treeNum, maxDepth))

    data = getData(train_dir, True)
    label = data[:,-1]
    data = data[:, :-1]   
    sortedFeatures = sortFeatures(data, label)
    loadTrainingDataTime = float(time.time() - time_start)
    print('>> load training data time %.2fs.' %(loadTrainingDataTime))

    #clf = RandomForestClassifier(10)
    clf = myForest(treeNum, maxDepth)
    print(">> fitting...")
    #clf.fit(data, label, sortedFeatures)
    clf.fitWithMultiThread(data, label, sortedFeatures)

    fitTime = float(time.time() - time_start)
    print('>> fit time %.2fs.' %(fitTime))


    data = getData(test_dir, False)
    loadTestDataTime = float(time.time() - fitTime - time_start)
    print('>> load test data time %.2fs.' %(loadTestDataTime))

    print(">> predicting...")
    #pre = clf.predict(data)
    pre = clf.predictWithMultiThread(data)
    pre = pre.astype(int)
    predictTime = float(time.time() - fitTime - time_start - loadTestDataTime)
    print('>> predict time %.2fs.' %(predictTime))

    writeCSV(pre, preDir)