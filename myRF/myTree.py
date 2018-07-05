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
# 构造函数、fit、predict的cols和rows的处理

class node:
    def __init__(self, col=-1, value=None, results=None, trueBranch=None, falseBranch=None):
        self.col = col  # 分裂的是哪个特征
        self.value = value  # 分裂的这个特征值为何值
        self.results = results  # 本叶子结点的数据
        self.trueBranch = trueBranch  # 指向左子树
        self.falseBranch = falseBranch  # 指向右子树
        
    def getLabel(self):
        '''
        选取出本结点的主要类别
        如果非叶子结点，则返回None
        '''
        if self.results == None:
            return None
        else:
            max_counts = 0
            for key in self.results.keys():
                if self.results[key] > max_counts:
                    label = key
                    max_counts = self.results[key]
        return label

# Important part
class myTree:
    def __init__(self, data=None, label = None, maxLevel=10, rows = None, columns = None):
        self.data = data  #本树的data
        self.label = label  # 本树的label
        self.maxLevel = maxLevel  # 该树最大深度为多少
        self.rows = rows  # 构建本树所运用的样本下标，一维数组
        self.columns = columns  # 本树所分得特征对应的下标，一维数组 
        #data[sample[i]][columns[j]] 表示本棵树的第i个样本，第j个特征
        self.root = None  # 本树的根节点，即一个node

    def uniqueCounts(self, rows):
        '''
        计算该结点每个类别各有多少样本
        param rows: 该结点拥有data的哪些samples
        return: result是dict，label为Key, 对应的值为该结点下某label有多少个样本
        '''
        results = {}
        for i in rows:  # 遍历每一个样本
            r = self.label[i]
            if r not in results:
                results[r] = 0
            results[r] = results[r]+1
        return results

    def giniEstimate(self, rows):
        '''
        计算该结点的基尼系数
        param rows: 该结点拥有data的哪些samples
        参考：https://www.cnblogs.com/pinard/p/6053344.html
        运用的是该参考博客的第一个公式
        '''
        if len(rows)==0: return 0
        total = len(rows)
        counts = self.uniqueCounts(rows)
        gini = 0
        for target in counts:
            gini = gini + pow(counts[target],2)
        gini = 1 - gini / pow(total,2)
        return gini
    
    def divideSet(self, rows, column, value):
        '''
        划分子集，这里默认特征值都是连续值
        param rows: 该结点拥有data的哪些samples
        param column: 哪一个特征，这里传入的是真实的特征下标
        '''
        splitFunction = lambda row: self.data[row][column] >= value 
        rows1 = [row for row in rows if splitFunction(row)]
        rows2 = [row for row in rows if not splitFunction(row)]
        return (rows1,rows2)

    def buildTree(self, rows = [], level = 0):
        '''
        构造CART决策树
        param rows: 该结点拥有data的哪些samples
        param columns: 对应的特征下标
        '''
        if len(rows) == 0:
            return node()
        if level > self.maxLevel:
            return node(results=self.uniqueCounts(rows))

        currentGini = self.giniEstimate(rows)
        bestGain = 0
        bestCriteria = None
        bestSets = None

        colCount = len(self.columns)   # 特征数
        colRange = np.arange(colCount)  # colRange数组存储特征下标。对某一行的样本来说，
                                        # sample[colRange[j]],就是取该样本第j个特征，colRange数组中的下标
        np.random.shuffle(colRange)  # 乱序特征，存储的是下标, 随机选取特征总数开方的样本进行分裂
        for col in colRange[0:int(math.ceil(math.sqrt(colCount)))]:  # col是colRange数组中的特征下标，即哪一个特征
            colValues = {}  # dict，对于该特征，所有样本共多少种取值
            for row in rows:  # row 是某样本的下标
                colValues[self.data[row][self.columns[col]]] = 1  # self.columns[col]]就是真实的特征下标
            for value in colValues.keys():  # 寻找分裂点
                (rows1,rows2) = self.divideSet(rows,self.columns[col],value)
                gain = currentGini - (len(rows1)*self.giniEstimate(rows1) + len(rows2)*self.giniEstimate(rows2)) / len(rows)
                if gain > bestGain and len(rows1) > 0 and len(rows2) > 0:
                    bestGain = gain
                    bestCriteria = (self.columns[col],value)
                    bestSets = (rows1,rows2)
        if bestGain > 0:
            trueBranch = self.buildTree(bestSets[0], level = level+1)
            falseBranch = self.buildTree(bestSets[1], level = level+1)
            return node(col=bestCriteria[0],value=bestCriteria[1],trueBranch=trueBranch,falseBranch=falseBranch)
        else:
            return node(results=self.uniqueCounts(rows))

    def fit(self, data, label, rows, columns):
        '''
        初始化，并构造一棵树
        '''
        self.data = data
        self.label = label
        self.rows = rows
        self.columns = columns
        
        self.root = self.buildTree(self.rows, level = 0)
    
    def printTree(self, node, indent='    '):  
        '''
        以文本形式显示决策树
        调用：clf.printTree(clf.tree)
        '''
        if node.results != None:
            print(str(node.results))
        else:
            print('F'+ str(node.col)+' >='+str(node.value)+'?  ')
            print(indent+'T->    ', end=""),
            self.printTree(node.trueBranch, indent + '    ')
            print(indent+'F->    ', end=""),
            self.printTree(node.falseBranch, indent + '    ')

    def predict_one_sample(self, node, row):
        '''
        利用决策树进行分类
        param node: 该
        param row: 该行数据
        '''
        if node.results != None:
            return node.getLabel()
        else:
            v = row[node.col]
            branch = None
            if v >= node.value: 
                branch = node.trueBranch
            else: 
                branch = node.falseBranch
            return self.predict_one_sample(branch, row)

    def predict(self, data):
        '''
        利用决策树进行分类
        '''
        self.data = data
        self.rows = range(len(data))

        finalResults = []
        for row in self.data:
            finalResult = self.predict_one_sample(self.root, row)
            finalResults.append(finalResult)

        return finalResults