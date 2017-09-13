# _*_coding:utf:8 _*_
"""
    Created by Dye_Char on 2017.07.22
    Used to extract some important features
"""

import pandas as pd

dataSet_15 = pd.read_csv('../DataSets/train/15_data_clean.csv')
dataSet_21 = pd.read_csv('../DataSets/train/21_data_clean.csv')
dataSet_08 = pd.read_csv('../DataSets/test/08_data.csv')

train_failInfo = pd.read_csv('../DataSets/Origin/train/15/15_failureInfo.csv')
train_normInfo = pd.read_csv('../DataSets/Origin/train/15/15_normalInfo.csv')
# print train_data.head()
train_data['ice_state'] = np.nan
# print train_data.head()
pd.options.mode.chained_assignment = None   # 关闭SettingWithCopyWarning警告
for row_index in range(0, train_failInfo.shape[0]):     # 处理故障信息, 将故障时间段的风机状态设置威1
    sTime_fail = train_failInfo.startTime[row_index]
    eTime_fail = train_failInfo.endTime[row_index]
    sTimeSpan = train_data.time >= sTime_fail
    eTimeSpan = train_data.time <= eTime_fail
    train_data.loc[sTimeSpan & eTimeSpan,'ice_state'] =1

for row_index in range(0, train_normInfo.shape[0]):  # 处理正常运行信息, 将正常运行时间段的风机状态设置威0
    sTime_fail = train_normInfo.startTime[row_index]
    eTime_fail = train_normInfo.endTime[row_index]
    sTimeSpan = train_data.time >= sTime_fail
    eTimeSpan = train_data.time <= eTime_fail
    train_data.loc[sTimeSpan & eTimeSpan,'ice_state'] = 0









