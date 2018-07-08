import csv
from collections import defaultdict
import pandas as pd  
import numpy as np
import sklearn
from sklearn import preprocessing
import math
import random  
import threading
import time
import xgboost as xgb

trainSampleNum = 1719692
testSampleNum = 429923
train_dir = "data/train.txt"
test_dir = "data/test.txt"
treeNum = 20


# read in data
dtrain = xgb.DMatrix(train_dir)
dtest = xgb.DMatrix(test_dir)
# specify parameters via map
param = {'max_depth':6, 'eta':0.3, 'silent':0, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
