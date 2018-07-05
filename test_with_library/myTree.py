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
    def __init__(self, data=None, maxLevel=10):
        self.data = data #本树的data
        self.maxLevel = maxLevel  # 该树最大深度为多少

    def buildTree(self, data, label):
        '''
        构造CART决策树
        '''
        if len(data) == 0:
            return node()
        currentGini = self.giniEstimate(samples)
        bestGain = 0
        bestCriteria = None
        bestSets = None
        colCount = len(samples[0]) - 1
        colRange = range(0,colCount)
        np.random.shuffle(colRange)
        for col in colRange[0:int(math.ceil(math.sqrt(colCount)))]:
            colValues = {}
            for row in samples:
                colValues[row[col]] = 1
            for value in colValues.keys():
                (set1,set2) = self.divideSet(samples,col,value)
                gain = currentGini - (len(set1)*self.giniEstimate(set1) + len(set2)*self.giniEstimate(set2)) / len(samples)
                if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                    bestGain = gain
                    bestCriteria = (col,value)
                    bestSets = (set1,set2)
        if bestGain > 0:
            trueBranch = self.buildTree(bestSets[0])
            falseBranch = self.buildTree(bestSets[1])
            return node(col=bestCriteria[0],value=bestCriteria[1],trueBranch=trueBranch,falseBranch=falseBranch)
        else:
            return node(results=self.uniqueCounts(samples))


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