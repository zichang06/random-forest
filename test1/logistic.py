import pandas as pd   
import numpy as np
import sklearn
from sklearn import preprocessing  
import csv
import random
import sys
import math

# trainSampleNum = 1719692
# testSampleNum = 429923
# train_dir = "train.txt"
# test_dir = "test.txt"
# alpha = 0.0001 
# lambdaR = 1
# batchSize = 1719692 
# epochMaxNum = 100000

trainSampleNum = 50
testSampleNum = 20
train_dir = "simple/train.txt"
test_dir = "simple/test.txt"
alpha = 0.0001 
lambdaR = 1
batchSize = 10 
epochMaxNum = 1000000


featureNum = 201
epsilon = 0.000002

def getData(dataDir = "train.txt", isTrain = True):
    print("start loading data %s..." %(dataDir))
    '''
    readin dataset according to dataDir
    if train dataset is needed
    return augmented features[m, n+1] and label[m]
    if test dataset is needed
    return augmented features[m, n+1]

    '''
    if isTrain:
        features = np.zeros((trainSampleNum, featureNum + 1))
        for i in range(trainSampleNum):
            features[i][0] = 1
        if isTrain:
            label = np.zeros((trainSampleNum))
    else:
        features = np.zeros((testSampleNum, featureNum + 1))
        for i in range(testSampleNum):
            features[i][0] = 1
        if isTrain:
            label = np.zeros((testSampleNum))

    f = open(dataDir)  
    lines = f.readline() 
    sampleCount = 0
    while lines:  
        line = lines.split(' ')
        for index in range(len(line)):
            if index == 0:
                if isTrain:
                    label[sampleCount] = int(line[0])
                continue
            colon = line[index].index(':')
            features[sampleCount, int(line[index][:colon])] = float(line[index][colon+1:])
        lines = f.readline()  
        sampleCount += 1
    f.close()  
    print("finish load data %s..." %(dataDir))
    
    if isTrain:
        return features, label 
    return features

class logisticRegression:
    def __init__(self, featureNum = 201):
        self.dim = featureNum+1
        self.theta = np.random.random(size = self.dim)

    def predict(self, x):
        z = np.dot(x, self.theta)
        yP = self.sigmoid(z)
        return yP

    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def lossFunction(self, hypothesis, y):
        '''
        hypothesis: calculated hypothesis by model
        y: real value
        return the mean of one-batch loss
        '''
        hypothesis = np.clip(hypothesis, 10e-8, 1.0-10e-8)
        entropys = - y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)
        loss = np.mean(entropys)
        return loss

    def trainWithMiniBatch(self, features, label, alpha, lambdaR, batchSize, epochMax):
        '''
        every time load in a minibatch to train
        '''
        lastLossPerEpoch = 0
        finalEpoch = epochMax
        for epoch in range(epochMax):
            currentLoss = self.oneStepGradientDescentRegulazed(x = features, 
                                                            y = label, 
                                                            alpha = alpha, 
                                                            lambdaR = lambdaR)
            diff = abs(lastLossPerEpoch - currentLoss)   
            lastLossPerEpoch = currentLoss                                                  
            sys.stdout.write('\r>>epoch %d, current loss = %f, diff = %f.' % (epoch, currentLoss, diff))         
            if epoch % 100 == 0:
                print('\n>>epoch %d, total loss this epoch = %f.\n' % (epoch, currentLoss))                                         
            if diff < epsilon:
                finalEpoch = epoch
                break
        return finalEpoch

    def oneStepGradientDescentRegulazed(self, x, y, alpha, lambdaR):
        '''
        for a minibatch dataset 
        do gradient descent for a step 
        return the loss for this step
        '''
        z = np.dot(x, self.theta)
        hypothesis = self.sigmoid(z)
        error = hypothesis - y
        loss = self.lossFunction(hypothesis, y)
    
        x_transpose =  x.T
        item2 = np.dot(x_transpose, error) * alpha / featureNum

        scalar = 1 - alpha * lambdaR / featureNum
        tmp = self.theta[0]
        self.theta *= scalar
        self.theta[0] = tmp

        self.theta -= item2
        return loss
        

def writeCSV(predictLable, predDir = "predictLable.csv"):
    head = ["label"]
    y_pred = pd.DataFrame (predictLable , columns = head)
    y_pred.to_csv (predDir , encoding = "utf-8")

def dataToSigmoid(dataDir):
    data = pd.read_csv(dataDir) 
    score = data.iloc[:, 1]
    return score


if __name__ == '__main__':
    # trainFeatures, trainLabel  = getData(dataDir = train_dir, isTrain = True)
    # min_max_scaler = preprocessing.MinMaxScaler()  
    # trainFeatures = min_max_scaler.fit_transform(trainFeatures)  

    lr = logisticRegression()
    # epoch = lr.trainWithMiniBatch(features = trainFeatures, 
    #                       label = trainLabel, 
    #                       alpha = alpha, 
    #                       lambdaR = lambdaR, 
    #                       batchSize = batchSize, 
    #                       epochMax = epochMaxNum)
    # print("finish training, with epoch = %d." %(epoch))

    # testFeatures  = getData(dataDir = test_dir, isTrain = False)
    # testFeatures = min_max_scaler.fit_transform(testFeatures)

    # print("start predict...")
    # predictLable = lr.predict(testFeatures)

    # print("write the result to csv and save...")
    # writeCSV(predictLable)
    _score = dataToSigmoid('predictLable.csv')
    score = np.zeros(429923)
    score[:] = _score[:429923]
    score = lr.sigmoid(score)
    writeCSV(score, predDir = "scores.csv")
    print('finish')