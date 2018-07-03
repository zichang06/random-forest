import csv
from collections import defaultdict
import pandas as pd  
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.datasets import load_iris
import math
import random  
import decision_tree
import thread
import time
from decision_tree import *

trainSampleNum = 50
testSampleNum = 50
train_dir = "simple_data/train.txt"
test_dir = "simple_data/test.txt"

# trainSampleNum = 1719692
# testSampleNum = 429923
# train_dir = "data/train.txt"
# test_dir = "data/test.txt"

featureNum = 201

def writePreToCSV(predictLable, predDir = "predictLable.csv"):
    print("writing to csv...")
    head = ["label"]
    y_pred = pd.DataFrame (predictLable , columns = head)
    y_pred.to_csv (predDir , encoding = "utf-8")

def makeFeatureNameTree(featureNum):
    def __init__(self, n_bootstrapSamples=20):
        self.n_bootstrapSamples = n_bootstrapSamples
        self.list_tree = []
        self.list_random_k = []

    featureName = {}
    for index in range(featureNum):
        szCol = 'Column %d' % index
        szY = 'F%d' % index
        featureName[szCol] = str(szY)

    return featureName

def getDataTree(dataDir = "train.txt", isTrain = True):
    print("start loading data %s..." %(dataDir))
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
    f.close()  
    print("finish load data %s..." %(dataDir))
    
    return data

class RandomForestsClassifier:
    def __init__(self, n_bootstrapSamples=20):
        self.n_bootstrapSamples = n_bootstrapSamples
        self.list_tree = []  # 随机森林
        self.list_random_k = []

    def generateBootstrapSamples(self, data):#构造bootstrap样本
            k = int(0.7 * featureNum)
            samples = []
            random_k = random.sample(range(len(data[0])-1), k)
            random_k.append(len(data[0])-1)
            for i in range(len(data)):
                data1 = data[np.random.randint(len(data))]
                samples.append([data1[i] for i in random_k])
            return samples,random_k
        
    def generateHalfSamples(self, data):#构造bootstrap样本
        rowRange = np.arange(trainSampleNum)
        np.random.shuffle(rowRange)
        sampleNumATree = int(trainSampleNum / 2)
        #从鸢尾花数据集(容量为150)按照随机均匀抽样的原则选取70%的数据作为训练数据
        training_data = [data[i] for i in rowRange[0:sampleNumATree]]
        return training_data

    def fit(self, data):#构造随机森林
        for i in range(self.n_bootstrapSamples):
            halfData = self.generateHalfSamples(data)
            samples,random_k = self.generateBootstrapSamples(halfData)

            currentTree = buildDecisionTree(samples, evaluationFunction=gini)
            prune(currentTree, 0.4)
            self.list_tree.append(currentTree)
            self.list_random_k.append(random_k)

            self.printTree(currentTree)

    def predict_tree(self, observation, tree,random_k):#利用决策树进行分类
        if tree.results != None:
            return tree.getLabel()
        else:
            v = observation[random_k[tree.col]]
            branch = None
            if isinstance(v,int) or isinstance(v,float):
                if v >= tree.value: 
                    branch = tree.trueBranch
                else: 
                    branch = tree.falseBranch
            else:
                if v == tree.value: 
                    branch = tree.trueBranch
                else: 
                    branch = tree.falseBranch
            return self.predict_tree(observation,branch,random_k)

    def predict_randomForests(self, observation):#利用随机森林对给定观测数据进行分类
        results = {}
        for i in range(len(self.list_tree)):
            currentResult = self.predict_tree(observation, self.list_tree[i],self.list_random_k[i])
            if currentResult not in results:
                results[currentResult] = 0
            results[currentResult] = results[currentResult] + 1
        max_counts = 0
        for key in results.keys():
            if results[key] > max_counts:
                finalResult = key
                max_counts = results[key]
        return finalResult

    def printTree(self, tree,indent='    '):#以文本形式显示决策树
        if tree.results != None:
            print(str(tree.results))
        else:
            print(str(tree.col)+':>='+str(tree.value)+'?  ')
            print(indent+'T->    ', end=""),
            self.printTree(tree.trueBranch,indent+indent)
            print(indent+'F->    ', end=""),
            self.printTree(tree.falseBranch,indent+indent)

if __name__ == '__main__':
    featureName = makeFeatureNameTree(featureNum)
    data = getDataTree(train_dir, True)
    min_max_scaler = preprocessing.MinMaxScaler()  
    data = min_max_scaler.fit_transform(data)    

    classifier = RandomForestsClassifier(n_bootstrapSamples=10)#初始化随机森林
    #classifier.generateBootstrapSamples(training_data)
    classifier.fit(data)#利用训练数据进行拟合

    testData = getDataTree(train_dir, False)
    testData = min_max_scaler.fit_transform(testData)
    finalResults = []
    for row in testData:
        finalResult = classifier.predict_randomForests(row)#对检验数据集进行分类
        finalResults.append(finalResult)

    writePreToCSV(finalResults, predDir = "pred.csv")