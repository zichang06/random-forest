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
# count sys.stdout.write,这两个关键词是用来检测进度的

lock=threading.Lock()

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

    def uniqueCounts(self, Trows, Frows):
        '''
        计算该结点每个类别各有多少样本
        param rows: 该结点拥有data的哪些samples
        return: result是dict，label为Key, 对应的值为该结点下某label有多少个样本
        '''
        results = {1:len(Trows), 0:len(Frows)}
        return results

    def giniEstimate(self, Trows, Frows):
        '''
        计算该结点的基尼系数
        param rows: 该结点拥有data的哪些samples
        参考：https://www.cnblogs.com/pinard/p/6053344.html
        运用的是该参考博客的第一个公式
        '''
        total = len(Trows) + len(Frows)
        if total == 0: return 0
        gini = pow(len(Trows), 2) + pow(len(Frows), 2)
        gini = 1 - gini / pow(total, 2)
        return gini
    
    def divideSet(self, Trows, Frows, column, value):
        '''
        划分子集，这里默认特征值都是连续值
        大的放在右子树1 trueBranch
        小的放在左子树2 frueBranch
        param rows: 该结点拥有data的哪些samples
        param column: 哪一个特征，这里传入的是真实的特征下标
        '''
        Trows1 = []
        Frows1 = []
        Trows2 = [] 
        Frows2 = []

        # lambda???
        for item in self.sortedFeatures[column].keys():
            if  item >= value:  #!!!不确定这样是否对，item是社么样
                Trows1 += self.sortedFeatures[column][item][1]
                Frows1 += self.sortedFeatures[column][item][0]
            else:
                Trows2 += self.sortedFeatures[column][item][1]
                Frows2 += self.sortedFeatures[column][item][0]

        Trows1 = list(set(Trows1).intersection(set(Trows)))
        Frows1 = list(set(Frows1).intersection(set(Frows)))
        Trows2 = list(set(Trows2).intersection(set(Trows)))
        Frows2 = list(set(Frows2).intersection(set(Frows)))

        subTrows = Trows1 + Trows2
        subFrows = Frows1 + Frows2

        # splitFunction = lambda row: self.data[row][column] >= value 
        # rows1 = [row for row in rows if splitFunction(row)]
        # rows2 = [row for row in rows if not splitFunction(row)]
        return (Trows1, Frows1, Trows2, Frows2, subTrows, subFrows)

    def buildTree(self, Trows = [], Frows = [], level = 0):
        '''
        构造CART决策树
        param rows: 该结点拥有data的哪些samples
        param columns: 对应的特征下标
        '''
        sys.stdout.write("--> now in depth %d. of %s st tree \n" %(level, self.treeID))
        if len(Trows) + len(Frows) == 0:
            return node()
        if level > self.maxLevel:
            return node(results=self.uniqueCounts(Trows, Frows))

        bestGain = 0
        bestCriteria = None
        bestSets = None

        colCount = len(self.columns)   # 特征数
        colRange = np.arange(colCount)  # colRange数组存储特征下标。对某一行的样本来说，
                                        # sample[colRange[j]],就是取该样本第j个特征，colRange数组中的下标
        np.random.shuffle(colRange)  # 乱序特征，存储的是下标, 随机选取特征总数开方的样本进行分裂
        countCol = 0
        total = int(math.ceil(colCount / 4)) 
        #total = int(math.ceil(math.sqrt(colCount)))  # 如果样本太少，可能找到的可行分裂特征会很少甚至没有
        for col in colRange[0:total]:  # self.columns[col]是原来真正的哪一个特征
            countCol+=1
            #sys.stdout.write("--> -- the %dst feature in total of %d. \n" %(countCol, total))
            colValues = self.sortedFeatures[self.columns[col]]  # dict，对于该特征，所有样本共多少种取值
            count = 0
            for value in colValues.keys():  # 寻找分裂点
                count += 1
                #sys.stdout.write("--> find the %d st split point of %d in total. \n" %(count, len(colValues.keys())))
                if count == 1 or count == len(colValues):
                    continue  # 取第一个值和最后一个值做分裂点毫无意义
                #!!!这里有个细节，对某个特征而言，分裂前后的rows不再相同,比较基尼系数，前面的应该用子集
                (Trows1, Frows1, Trows2, Frows2, subTrows, subFrows) = self.divideSet(Trows, Frows, self.columns[col], value)
                currentGini = self.giniEstimate(subTrows, subFrows)
                tmp =  (len(Trows1) + len(Frows1)) * self.giniEstimate(Trows1, Frows1) + (len(Trows2) + len(Frows2)) * self.giniEstimate(Trows2, Frows2)
                if len(subTrows) + len(subFrows) == 0:  #!!!存在分母为0的情况，这时该分裂点无参考价值
                    continue
                gain = currentGini - tmp / (len(subTrows) + len(subFrows))
                if gain > bestGain and len(Trows1) + len(Frows1) > 0 and len(Trows2) + len(Frows2) > 0:
                    bestGain = gain
                    bestCriteria = (self.columns[col], value)
                    bestSets = (Trows1, Frows1, Trows2, Frows2)
        if bestGain > 0:
            trueBranch = self.buildTree(bestSets[0], bestSets[1], level = level+1)
            falseBranch = self.buildTree(bestSets[2], bestSets[3], level = level+1)
            return node(col=bestCriteria[0],value=bestCriteria[1], trueBranch=trueBranch, falseBranch=falseBranch)
        else:
            return node(results=self.uniqueCounts(Trows, Frows))

    def fit(self, label, rows, columns, sortedFeatures, treeID = 0):
        '''
        初始化，并构造一棵树
        '''
        self.label = label
        self.rows = rows
        self.columns = columns
        self.sortedFeatures = sortedFeatures
        self.treeID = treeID

        Trows = []
        Frows = []
        for i in rows:  # 遍历每一个样本  lambda???
            if self.label[i] == 1:
                Trows.append(i)
            if self.label[i] == 0:
                Frows.append(i)

        self.root = self.buildTree(Trows, Frows, level = 0)
        #self.printTree(self.root)
    
    def printTree(self, node, indent='    '):  
        '''
        以文本形式显示决策树
        调用：clf.printTree(clf.tree)
        '''
        #lock.acquire()
        if node.results != None:
            print(str(node.results))
        else:
            print('F'+ str(node.col)+' >='+str(node.value)+'?  ')
            print(indent+'T->  ', end=""),
            self.printTree(node.trueBranch, indent + '    ')
            print(indent+'F->  ', end=""),
            self.printTree(node.falseBranch, indent + '    ')
        #lock.release()

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