# _*_coding:utf-8 _*_
"""
    Created by Dye_Char on 2017.07.22
    Use LSTM to help distinguish icing state
"""
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
import keras.backend as K
from sklearn.preprocessing import StandardScaler

# --------------------------some basic settings and definitions----------------------------------
# record time
STime = time.time()
# fix random seed for reproducibility
seed = 10
np.random.seed(seed)
# cols to drop (which are useless)
cols_drop = []


# define metric to evaluate offline
def score(y_true, y_pred):
    difference = y_pred - y_true
    FN = np.sum(difference == 1)
    FP = np.sum(difference == -1)
    N_normal = np.sum(y_true == 0)
    N_fault = np.sum(y_true == 1)
    return 100 - 50 * FN / N_normal - 50 * FP / N_fault


# --------------------------Load data and add some features----------------------------------
# load data: DataFrame
dataSet_15 = pd.read_csv('../DataSets/train/15_data_clean.csv')
dataSet_21 = pd.read_csv('../DataSets/train/21_data_clean.csv')
dataSet_08 = pd.read_csv('../DataSets/test/08_data.csv')

trainSet = dataSet_15
validSet = dataSet_21
testSet = dataSet_08

# add some features

# standardize
Scaler = StandardScaler()
