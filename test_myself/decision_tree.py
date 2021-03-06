import csv
from collections import defaultdict
import pandas as pd  
import numpy as np
import sklearn
from sklearn import preprocessing  

'''
我现在用训练集作为测试集，效果很差，可能和打乱/预测函数有关
'''

trainSampleNum = 50
testSampleNum = 20
train_dir = "simple_data/train.txt"
test_dir = "simple_data/test.txt"

# trainSampleNum = 1719692
# testSampleNum = 429923
# train_dir = "data/train.txt"
# test_dir = "data/test.txt"

maxLevel = 2
featureNum = 201


def makeFeatureName(featureNum):
    featureName = {}
    for index in range(featureNum):
        szCol = 'Column %d' % index
        szY = 'F%d' % index
        featureName[szCol] = str(szY)

    return featureName

def getData(dataDir = "train.txt", isTrain = True):
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

# Important part
class Tree:
    def __init__(self, value=None, trueBranch=None, falseBranch=None, results=None, col=-1, summary=None, data=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.summary = summary
        self.data = data

    def getLabel(self):
        if self.results == None:
            return None
        else:
            max_counts = 0
            for key in self.results.keys():
                if self.results[key] > max_counts:
                    label = key
                    max_counts = self.results[key]
        return label


def calculateDiffCount(datas):
    # 将输入的数据汇总(input dataSet)
    # return results Set{type1:type1Count,type2:type2Count ... typeN:typeNCount}

    results = {}
    for data in datas:
        # data[-1] means dataType
        if data[-1] not in results:
            results[data[-1]] = 1
        else:
            results[data[-1]] += 1
    return results


def gini(rows):
    # 计算gini值(Calculate GINI)

    length = len(rows)
    results = calculateDiffCount(rows)
    imp = 0.0
    for i in results:
        imp += results[i] / length * results[i] / length
    return 1 - imp


def splitDatas(rows, value, column):
    # 根据条件分离数据集(splitDatas by value,column)
    # return 2 part(list1,list2)

    list1 = []
    list2 = []
    if (isinstance(value, int) or isinstance(value, float)):  # for int and float type
        for row in rows:
            if (row[column] >= value):
                list1.append(row)
            else:
                list2.append(row)
    else:  # for String type
        for row in rows:
            if row[column] == value:
                list1.append(row)
            else:
                list2.append(row)

    return (list1, list2)


def buildDecisionTree(rows, evaluationFunction=gini, level=0):
    '''
    #print("start building tree...")
    # 递归建立决策树,当gain = 0 时停止递归
    # bulid decision tree by recursive function
    # stop recursive function when gain = 0
    # return tree
    '''

    # 如果层数大于规定的最大层数，则终止
    if level >= maxLevel:
        return Tree(results=calculateDiffCount(rows),  data=rows)

    currentGain = evaluationFunction(rows)
    column_length = len(rows[0])
    rows_length = len(rows)
    best_gain = 0.0
    best_value = None
    best_set = None

    # choose the best gain
    for col in range(column_length - 1):
        col_value_set = set([x[col] for x in rows])
        for value in col_value_set:
            list1, list2 = splitDatas(rows, value, col)
            p = len(list1) / rows_length
            gain = currentGain - p * evaluationFunction(list1) - (1 - p) * evaluationFunction(list2)
            if gain > best_gain:
                best_gain = gain
                best_value = (col, value)
                best_set = (list1, list2)

    dcY = {'impurity': '%.3f' % currentGain, 'samples': '%d' % rows_length}

    # stop or not stop
    if best_gain > 0:
        trueBranch = buildDecisionTree(best_set[0], evaluationFunction, level+1)
        falseBranch = buildDecisionTree(best_set[1], evaluationFunction, level+1)
        return Tree(col=best_value[0], value=best_value[1], trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY)
    else:
        return Tree(results=calculateDiffCount(rows), summary=dcY, data=rows)


## 参考博客：https://blog.csdn.net/herosofearth/article/details/52425952
def prune(tree, miniGain, evaluationFunction=gini):
    #print("start pruning...")
    # 剪枝, when gain < mini Gain，合并(merge the trueBranch and the falseBranch)

    if tree.trueBranch.results == None: prune(tree.trueBranch, miniGain, evaluationFunction)
    if tree.falseBranch.results == None: prune(tree.falseBranch, miniGain, evaluationFunction)

    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        len1 = len(tree.trueBranch.data)
        len2 = len(tree.falseBranch.data)
        len3 = len(tree.trueBranch.data + tree.falseBranch.data)
        p = float(len1) / (len1 + len2)
        gain = evaluationFunction(tree.trueBranch.data + tree.falseBranch.data) - p * evaluationFunction(
            tree.trueBranch.data) - (1 - p) * evaluationFunction(tree.falseBranch.data)
        if (gain < miniGain):
            tree.data = tree.trueBranch.data + tree.falseBranch.data
            tree.results = calculateDiffCount(tree.data)
            tree.trueBranch = None
            tree.falseBranch = None


def classify(row, tree):
    if tree.results != None:
        return tree.getLabel()
    else:
        branch = None
        v = row[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if v == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        return classify(row, branch)

def predict(data, tree):
    print("start predicting...")
    pred = np.zeros(testSampleNum)
    for index in range(testSampleNum):
        result = classify(data[index], tree)
        pred[index] = result
    return pred


#下面是辅助代码画出树
#Unimportant part
#plot tree and load data
def plot(decisionTree):
    """Plots the obtained decision tree. """

    def toString(decisionTree, indent=''):
        if decisionTree.results != None:  # leaf node
            return str(decisionTree.results)
        else:
            szCol = 'Column %s' % decisionTree.col
            if szCol in featureName:
                szCol = featureName[szCol]
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = '%s >= %.2f?' % (szCol, decisionTree.value)
            else:
                decision = '%s == %.2f' % (szCol, decisionTree.value)
            trueBranch = indent + 'yes -> ' + toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no  -> ' + toString(decisionTree.falseBranch, indent + '\t\t')
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))

def writeCSV(predictLable, predDir = "predictLable.csv"):
    print("writing to csv...")
    head = ["label"]
    y_pred = pd.DataFrame (predictLable , columns = head)
    y_pred.to_csv (predDir , encoding = "utf-8")

if __name__ == '__main__':
    featureName = makeFeatureName(featureNum)
    data = getData(train_dir, True)
    min_max_scaler = preprocessing.MinMaxScaler()  
    data = min_max_scaler.fit_transform(data) 

    decisionTree = buildDecisionTree(data, evaluationFunction=gini)
    #plot(decisionTree)
    prune(decisionTree, 0.4) # notify, when a branch is pruned (one time in this example)
    plot(decisionTree)

    testData = getData(test_dir, False)
    testData = min_max_scaler.fit_transform(testData)
    predict = predict(testData, decisionTree)

    writeCSV(predict, predDir = "pred.csv")
    
    
