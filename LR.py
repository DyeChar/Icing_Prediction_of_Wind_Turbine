# _*_ coding:utf-8 _*_
"""
    Created by Dye_Char on 2017.07.22
    Used LR to predict icing state
"""
from __future__ import division
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# --------------------------some basic settings and definitions----------------------------------
# record time
STime = time.time()
# fix random seed for reproducibility
seed = 10
np.random.seed(seed)
# cols to drop (which are useless)
cols_drop = ['time',  # 'wind_direction',
             'wind_direction_mean',#'yaw_position','yaw_speed',
             'pitch1_speed','pitch2_speed','pitch3_speed',
             'acc_x','acc_y','int_tmp',
             # 'pitch1_ng5_tmp', 'pitch2_ng5_tmp', 'pitch3_ng5_tmp',
             'pitch1_ng5_DC', 'pitch2_ng5_DC', 'pitch3_ng5_DC',
             'group']


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

# prepare train, valid, and test dataSet
# drop some useless features eg: 'time', 'group'

# is_ice_index = trainSet['ice_state']==1
# is_ice = trainSet[is_ice_index]
# for i in range(0,13):
#    trainSet =pd.concat([trainSet,is_ice],axis=0)





# add some features

# -------transfer angles to sin, cos, tan
for col in ['yaw_position','pitch1_angle','pitch2_angle','pitch3_angle']:
    trainSet[col + '_sin'] = np.sin(trainSet[col])
    trainSet[col + '_cos'] = np.cos(trainSet[col])

    validSet[col + '_sin'] = np.sin(validSet[col])
    validSet[col + '_cos'] = np.cos(validSet[col])

    testSet[col + '_sin'] = np.sin(testSet[col])
    testSet[col + '_cos'] = np.cos(testSet[col])
'''
failInfo_15 = pd.read_csv('../DataSets/Origin/train/15/15_failureInfo.csv')
normInfo_15 = pd.read_csv('../DataSets/Origin/train/15/15_normalInfo.csv')

failInfo_21 = pd.read_csv('../DataSets/Origin/train/21/21_failureInfo.csv')
normInfo_21 = pd.read_csv('../DataSets/Origin/train/21/21_normalInfo.csv')

dataSet_15['environment_tmp_mean'] = np.nan
dataSet_21['environment_tmp_mean'] = np.nan
pd.options.mode.chained_assignment = None   # 关闭SettingWithCopyWarning警告


for row_index in range(0, failInfo_21.shape[0]):     # 处理故障信息, 将故障时间段的风机状态设置威1
    sTime_fail = failInfo_21.startTime[row_index]
    eTime_fail = failInfo_21.endTime[row_index]
    sTimeSpan = validSet.time >= sTime_fail
    eTimeSpan = validSet.time <= eTime_fail
    #dataSet_15.loc[sTimeSpan & eTimeSpan,'ice_state'] =1
    temp = validSet.loc[sTimeSpan & eTimeSpan,'environment_tmp'].values
    envir_tmp_temp = np.zeros(temp.shape[0])
    for i in range(temp.shape[0]):
        if i<50:
            envir_tmp_temp[i] = temp[i]
        else:
            envir_tmp_temp[i] = np.sum(temp[i-50:i])/50
    validSet.loc[sTimeSpan & eTimeSpan, 'environment_tmp_mean'] = envir_tmp_temp

for row_index in range(0, normInfo_21.shape[0]):  # 处理正常运行信息, 将正常运行时间段的风机状态设置威0
    sTime_fail = normInfo_21.startTime[row_index]
    eTime_fail = normInfo_21.endTime[row_index]
    sTimeSpan = validSet.time >= sTime_fail
    eTimeSpan = validSet.time <= eTime_fail
    #dataSet_15.loc[sTimeSpan & eTimeSpan,'ice_state'] = 0
    temp = validSet.loc[sTimeSpan & eTimeSpan,'environment_tmp'].values
    envir_tmp_temp = np.zeros(temp.shape[0])
    for i in range(temp.shape[0]):
        if i<50:
            envir_tmp_temp[i] = temp[i]
        else:
            envir_tmp_temp[i] = np.sum(temp[i-50:i])/50
    validSet.loc[sTimeSpan & eTimeSpan, 'environment_tmp_mean'] = envir_tmp_temp

for row_index in range(0, failInfo_15.shape[0]):     # 处理故障信息, 将故障时间段的风机状态设置威1
    sTime_fail = failInfo_15.startTime[row_index]
    eTime_fail = failInfo_15.endTime[row_index]
    sTimeSpan = trainSet.time >= sTime_fail
    eTimeSpan = trainSet.time <= eTime_fail
    #dataSet_15.loc[sTimeSpan & eTimeSpan,'ice_state'] =1
    temp = trainSet.loc[sTimeSpan & eTimeSpan,'environment_tmp'].values
    envir_tmp_temp = np.zeros(temp.shape[0])
    for i in range(temp.shape[0]):
        if i<50:
            envir_tmp_temp[i] = temp[i]
        else:
            envir_tmp_temp[i] = np.sum(temp[i-50:i])/50
    trainSet.loc[sTimeSpan & eTimeSpan, 'environment_tmp_mean'] = envir_tmp_temp

for row_index in range(0, normInfo_15.shape[0]):  # 处理正常运行信息, 将正常运行时间段的风机状态设置威0
    sTime_fail = normInfo_15.startTime[row_index]
    eTime_fail = normInfo_15.endTime[row_index]
    sTimeSpan = trainSet.time >= sTime_fail
    eTimeSpan = trainSet.time <= eTime_fail
    #dataSet_15.loc[sTimeSpan & eTimeSpan,'ice_state'] = 0
    temp = trainSet.loc[sTimeSpan & eTimeSpan,'environment_tmp'].values
    envir_tmp_temp = np.zeros(temp.shape[0])
    for i in range(temp.shape[0]):
        if i<50:
            envir_tmp_temp[i] = temp[i]
        else:
            envir_tmp_temp[i] = np.sum(temp[i-50:i])/50
    trainSet.loc[sTimeSpan & eTimeSpan, 'environment_tmp_mean'] = envir_tmp_temp

'''



# standardize


# scaler = MinMaxScaler()
# trainSet_X = scaler.fit_transform(trainSet_X)
# validSet_X = scaler.transform(validSet_X)
# testSet_X = scaler.transform(testSet_X)

trainSet = trainSet.dropna()
print 'shape of train is {}'.format(trainSet.shape)
validSet = validSet.dropna()
print 'shape of valid is {}'.format(validSet.shape)
Y_train = trainSet['ice_state']
Y_valid = validSet['ice_state']

trainSet_X = trainSet.drop(cols_drop + ['ice_state'], axis=1)
trainSet_X = trainSet_X - trainSet_X.mean()

validSet_X = validSet.drop(cols_drop + ['ice_state'], axis=1)
validSet_X = validSet_X - validSet_X.mean()

testSet_X = testSet.drop(cols_drop, axis=1)
testSet_X = testSet_X - testSet_X.mean()






columns = trainSet_X.columns
X_train = trainSet_X.values
X_valid = validSet_X.values
X_test = testSet_X.values

Y_train = Y_train.values
Y_valid = Y_valid.values

# ---------------------------------initial LR model------------------------------------------
clf_lr = LogisticRegressionCV(n_jobs=-1, verbose=-1,class_weight='balanced')
clf_lr.fit(X_train, Y_train)
#f_importance = pd.DataFrame()
# f_importance['feature_name'] = columns
#f_importance['feature_importance'] =np.array(clf_lr.coef_)
print columns
print clf_lr.coef_
# ---------------------------------evaluate and predict---------------------------------------
res_val = clf_lr.predict(X_valid)
res_test = clf_lr.predict(X_test)

power_larger_valid = validSet_X['power'] >= 2
res_val = np.array(res_val)  # list to array
res_val[power_larger_valid] = 0  # 添加一条规则，功率大于2时不会发生故障

power_larger_test = testSet_X['power'] >= 2
res_test = np.array(res_test)  # list to array
res_test[power_larger_test] = 0  # 添加一条规则，功率大于2时不会发生故障

# --------------create df and convert it to csv
if False:
    output_val = pd.DataFrame({'y_lr_eval': res_val})
    output_val.to_csv('../Results/lr-baseline_2122.csv', index=False)
    output_test = pd.DataFrame({'y_lr_test': res_test})
    output_test.to_csv('../Results/lr-baseline_0822.csv', index=False)
#print('\nEclapsed {}s'.format(time.time() - start_time))

print ('Score of ValidData is {}'.format(score(Y_valid, res_val)))


# get the confusion matrix

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

confusion_matrix = confusion_matrix(Y_valid, res_val)
print(confusion_matrix)
'''
plt.matshow(confusion_matrix)
plt.title('confusion_matrix')
plt.colorbar()
plt.ylabel('Actual class')
plt.xlabel('Predict class')
plt.show()
'''
