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

#!!! 样本不均衡，导致容易误判成负样本，增加对数变换y = log2(x + 1)
# 或许解决的更好方法，是实现计算出所有正负样本之比，在结点投票时，把弱势样本乘以一个比例

trainSampleNum = 50
testSampleNum = 20
train_dir = "simple_data/train.txt"
test_dir = "simple_data/test.txt"
treeNum = 100
maxDepth = 5

trainSampleNum = 1719692
testSampleNum = 429923
train_dir = "data/train.txt"
test_dir = "data/test.txt"
treeNum = 1
maxDepth = 5

preDir = "test.csv"
maxSplitNum = 10
featureNum = 201
threads = []

def writeCSV(predictLable, fileName = "pre.csv"):
    print(">> writing to csv %s..." %fileName)
    names = range(testSampleNum) 
    dataframe = pd.DataFrame({'id':names,'label':predictLable})
    dataframe.to_csv(fileName,index=False,sep=',', encoding = "utf-8")

def loadTrainingData(dataDir = "train.txt"):
    print(">> start loading data %s..." %(dataDir))
    '''
    readin dataset according to dataDir
    if train dataset is needed
    return augmented trainingData[m, n+1], the last colomn is label(1 or 0)
    if test dataset is needed
    return augmented testData[m, n], without label

    '''
    label = []
    features = [{0: {1: [], 0:[]}} for n in range(featureNum)]
    f = open(dataDir)  
    lines = f.readline() 
    sampleCount = 0
    while lines:  
        line = lines.split(' ')
        for index in range(len(line)):
            if index == 0:
                label.append(int(line[0]))
                continue
            colon = line[index].index(':')
            r = float(line[index][colon+1:])  # 该样本的该特征值
            i = int(line[index][:colon]) - 1  # 哪个特征
            if r not in features[i]:
                features[i][r] = {1: [], 0:[]}
            if label[sampleCount] == 1:
                features[i][r][1].append(sampleCount)
            if label[sampleCount] == 0:
                features[i][r][0].append(sampleCount)
        lines = f.readline()  
        sampleCount += 1
        sys.stdout.write('\r>> %d ' % (sampleCount))

    for item in features:
        item.pop(0)
    f.close()  
    print("\n>> finish load data %s..." %(dataDir))
    
    return features, label

def sortFeatures(features, splitNum):
    '''
    找出特征的若干分割点，并从小到大排序
    '''
    sortedFeatures = []
    featureCount = 0
    # !!!选取分裂结点时，需要判断一下
    # sortedFeatures列表的下标表示哪一个特征，如果第i个特征不存在数据，那么len(sortedFeatures[i]) == 1,只有第一个
    for feature in features: # 遍历每个特征 O(m)  这个嵌套循环大概走2000次
        sortedFeatures.append({0:{1: [], 0:[]}})  # 遍历每个
        if len(feature.keys()) == 0:
            featureCount+=1
            continue
        splitNum = min(splitNum, len(feature.keys()))
        maxValue = max(feature.keys())
        minValue = min(feature.keys())
        adder = (maxValue - minValue) / splitNum 
        for i in range(splitNum):  # 遍历每个分裂点(k'≈10)
            splitPoint = float(minValue + (i + 0.5) * adder)
            sortedFeatures[featureCount][splitPoint] = {1: [], 0:[]}
        featureCount+=1
        sys.stdout.write('\r>> finish spilting %dst feature ' % (featureCount))
    print("\nbegin to arrange split points for features...")
    del featureCount
        

    for featureCount in range(len(sortedFeatures)):
        splitValues = list(sortedFeatures[featureCount])
        for oriPoint in features[featureCount]:

            diff = [abs(oriPoint - splitValue) for splitValue in splitValues]
            nearestSplitValue = splitValues[diff.index(min(diff))]
            sortedFeatures[featureCount][nearestSplitValue][1] += features[featureCount][oriPoint][1]
            sortedFeatures[featureCount][nearestSplitValue][0] += features[featureCount][oriPoint][0]
            sys.stdout.write('\r>> finish arranging %dst feature ' % (featureCount+1))

    print("\n>>finish sorting Features.")
    return sortedFeatures

def loadTestData(dataDir = "test.txt"):
    print(">> start loading data %s..." %(dataDir))
    '''
    readin dataset according to dataDir
    if test dataset is needed
    return augmented testData[m, n], without label
    可能要对test的缺失值进行补零

    '''
    data = np.zeros((testSampleNum, featureNum))
    f = open(dataDir)  
    lines = f.readline() 
    sampleCount = 0
    while lines:  
        line = lines.split(' ')
        for index in range(len(line)):
            if index == 0:
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
        columns = random.sample(range(featureNum), k)

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
        currentTree.fit(label = self.label, sortedFeatures = self.sortedFeatures, rows = rows, columns = columns, treeID = index)

        self.list_tree.append(currentTree)
        self.list_featureIndex.append(columns)
        print("finish building the %dst tree..." %(index + 1)) 

    def fit(self, sortedFeatures, label):
        '''
        构造随机森林,不是多线程
        '''
        self.label = label
        self.sortedFeatures = sortedFeatures

        for i in range(self.n_bootstrapSamples):
            self.buildTree(i)
        
    def fitWithMultiThread(self, sortedFeatures, label):
        '''
        多线程构造随机森林
        '''
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
        
        #threshold = np.full(len(self.data), 0.5)
        tmp = np.mean(results, axis=1) # 计算每一行的均值
        #!!! 样本不均衡，导致容易误判成负样本，增加对数变换y = log2(x + 1)
        finalResult = [math.log(x + 1, 2) for x in tmp]
        #finalResult = np.greater(tmp, threshold)
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
        
        #threshold = np.full(len(self.data), 0.5)
        tmp = np.mean(results, axis=1) # 计算每一行的均值
        #!!! 样本不均衡，导致容易误判成负样本，增加对数变换y = log2(x + 1)
        finalResult = [math.log(x + 1, 2) for x in tmp]
        #finalResult = np.greater(tmp, threshold)
        return finalResult

if __name__ == '__main__':
    t0 = time.time()
    print('>> my Forest with %d trees(depth: %d)' %(treeNum, maxDepth))

    features, label = loadTrainingData(train_dir)
    t1 = time.time()
    print('>> load training data time %.2fs.' %(float(t1 - t0)))
    sortedFeatures = sortFeatures(features, maxSplitNum)
    del features
    t2 = time.time()
    print('>> load training data time %.2fs.' %(float(t2 - t1)))

    #clf = RandomForestClassifier(10)
    clf = myForest(treeNum, maxDepth)
    print(">> fitting...")
    clf.fit(sortedFeatures, label)
    #clf.fitWithMultiThread(sortedFeatures, label)
    t3 = time.time()
    print('>> fit time %.2fs.' %(float(t3 - t2)))


    data = loadTestData(test_dir)
    t4 = time.time()
    print('>> load test data time %.2fs.' %(float(t4 - t3)))

    print(">> predicting...")
    pre = clf.predict(data)
    #pre = clf.predictWithMultiThread(data)
    #pre = pre.astype(int)
    t5 = time.time()
    print('>> predict time %.2fs.' %(float(t5 - t4)))

    writeCSV(pre, preDir)