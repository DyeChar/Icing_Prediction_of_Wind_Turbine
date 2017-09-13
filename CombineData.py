# _*_ coding:utf-8_*_
"""
    created by Dye_Char @2017.07.03 21:40
    Used to combine data
    1. glue all data
    2.去除风机正常时间和结冰时间均不覆盖的无效数据（V2017.07.04）

"""
import time
import numpy as np
import pandas as pd

# ################################# 15 ###############################################
# 把故障和正常运行数据合并、在表中添加一列ice_state，表征结冰与否，正常为0,结冰为1
#   preparing data
start_time = time.clock()

train_data = pd.read_csv('../DataSets/Origin/train/15/15_data.csv')
train_failInfo = pd.read_csv('../DataSets/Origin/train/15/15_failureInfo.csv')
train_normInfo = pd.read_csv('../DataSets/Origin/train/15/15_normalInfo.csv')
# print train_data.head()
pd.options.mode.chained_assignment = None  # 关闭SettingWithCopyWarning警告
for row_index in range(0, train_failInfo.shape[0]):  # 处理故障信息, 将故障时间段的风机状态设置威1
    sTime_fail = train_failInfo.startTime[row_index]
    eTime_fail = train_failInfo.endTime[row_index]
    sTimeSpan = train_data.time >= sTime_fail
    eTimeSpan = train_data.time <= eTime_fail
    train_data.loc[sTimeSpan & eTimeSpan, 'ice_state'] = 1

for row_index in range(0, train_normInfo.shape[0]):  # 处理正常运行信息, 将正常运行时间段的风机状态设置威0
    sTime_fail = train_normInfo.startTime[row_index]
    eTime_fail = train_normInfo.endTime[row_index]
    sTimeSpan = train_data.time >= sTime_fail
    eTimeSpan = train_data.time <= eTime_fail
    train_data.loc[sTimeSpan & eTimeSpan, 'ice_state'] = 0

# print train_data.head()
print 'there is {} useless data in trainSet_15!'.format(train_data.shape[0] - train_data.dropna().shape[0])
# train_data.to_csv('input/train/15/15_data_all.csv',index=False)     # 保存成新的文件

# save to csv
train_data.dropna().to_csv('../DataSets/train/15_data_clean.csv', index=False)  # 去除无效数据并保存成新的文件

end_time = time.clock()
print ('Time confused: {} senconds'.format(end_time - start_time))

# ################################# 21 ###############################################
# 把故障和正常运行数据合并、在表中添加一列ice_state，表征结冰与否，正常为0,结冰为1
#   preparing data
start_time = time.clock()

train_data = pd.read_csv('../DataSets/Origin/train/21/21_data.csv')
train_failInfo = pd.read_csv('../DataSets/Origin/train/21/21_failureInfo.csv')
train_normInfo = pd.read_csv('../DataSets/Origin/train/21/21_normalInfo.csv')
# print train_data.head()
train_data['ice_state'] = np.nan
# print train_data.head()
pd.options.mode.chained_assignment = None  # 关闭SettingWithCopyWarning警告
for row_index in range(0, train_failInfo.shape[0]):  # 处理故障信息, 将故障时间段的风机状态设置威1
    sTime_fail = train_failInfo.startTime[row_index]
    eTime_fail = train_failInfo.endTime[row_index]
    sTimeSpan = train_data.time >= sTime_fail
    eTimeSpan = train_data.time <= eTime_fail
    train_data.loc[sTimeSpan & eTimeSpan, 'ice_state'] = 1

for row_index in range(0, train_normInfo.shape[0]):  # 处理正常运行信息, 将正常运行时间段的风机状态设置威0
    sTime_fail = train_normInfo.startTime[row_index]
    eTime_fail = train_normInfo.endTime[row_index]
    sTimeSpan = train_data.time >= sTime_fail
    eTimeSpan = train_data.time <= eTime_fail
    train_data.loc[sTimeSpan & eTimeSpan, 'ice_state'] = 0

# print train_data.head()
print 'there is {} useless data in trainSet_21!'.format(train_data.shape[0] - train_data.dropna().shape[0])
# train_data.to_csv('input/train/15/15_data_all.csv',index=False)     # 保存成新的文件

# save to csv
train_data.dropna().to_csv('../DataSets/train/21_data_clean.csv', index=False)  # 去除无效数据并保存成新的文件

end_time = time.clock()
print ('Time confused: {} senconds'.format(end_time - start_time))
