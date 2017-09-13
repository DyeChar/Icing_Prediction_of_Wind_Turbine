# _*_ coding:utf-8_*_
"""
    created by Dye_Char @2017.07.03 21:40
    Used to visualize data distribution

"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
from datetime import datetime
from imblearn.over_sampling import SMOTE

# #########################################################################################
# 绘制序列图，直观表达时间序列中结冰和不结冰的时间段，从中可以发现经常会有停机（可能是故障/人为删除数据）
if False:
    fail_data = pd.read_csv('../DataSets/Origin/train/21/21_failureInfo.csv')
    norm_data = pd.read_csv('../DataSets/Origin/train/21/21_normalInfo.csv')

    fig = pylab.figure(figsize=(50,5))
    fail_data = fail_data.applymap(str)
    norm_data = norm_data.applymap(str)
    for i in range(fail_data.shape[0]):
        fail = plt.hlines(1, pylab.date2num(datetime.strptime(fail_data.iloc[i, 0], '%Y-%m-%d %H:%M:%S')),
                          pylab.date2num(datetime.strptime(fail_data.iloc[i, 1], '%Y-%m-%d %H:%M:%S')), colors='r', lw=200)
    for i in range(norm_data.shape[0]):
        norm = plt.hlines(1, pylab.date2num(datetime.strptime(norm_data.iloc[i, 0], '%Y-%m-%d %H:%M:%S')),
                          pylab.date2num(datetime.strptime(norm_data.iloc[i, 1], '%Y-%m-%d %H:%M:%S')), colors='g', lw=200)

    # plt.hlines(range(10), fig_data['c'],fig_data['d'], colors='b',lw=50)
    plt.margins(0, 1)
    ax = fig.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=matplotlib.dates.MO))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%b-%-d'))

    plt.title('Status of Icing  (Red:Ice, Green:NoIce) ')
    plt.show()


# #########################################################################################
# 绘制数据分布图, 观察数据分布特点
if False:
    # 发现温度有一段时空白
    dataSet_15 = pd.read_csv('../DataSets/train/15_data_clean.csv')
    dataSet_21 = pd.read_csv('../DataSets/train/21_data_clean.csv')
    dataSet_08 = pd.read_csv('../DataSets/test/08_data.csv')


    col = ['pitch1_speed']
    mean15 = np.mean(dataSet_15[col].values)
    mean21 = np.mean(dataSet_21[col].values)
    mean08 = np.mean(dataSet_08[col].values)

    print mean15
    print mean21
    print mean08

    fig = plt.figure()
    fig.suptitle(str(col))

    ax1 = plt.subplot(3,2,1)
    plt.hist(dataSet_15[col].values, bins=100,label='15')
    plt.legend()

    ax2 = plt.subplot(3,2,2)
    plt.hist(dataSet_15[col].values-mean15, bins=100,label='15-mean')
    plt.legend()

    ax3 = plt.subplot(3,2,3,sharex=ax1)
    plt.hist(dataSet_21[col].values, bins=100,label='21')
    plt.legend()

    ax4 = plt.subplot(3,2,4,sharex=ax2)
    plt.hist(dataSet_21[col].values-mean21, bins=100,label='21-mean')
    plt.legend()

    ax5 = plt.subplot(3,2,5,sharex=ax1)
    plt.hist(dataSet_08[col].values, bins=100,label='08')
    plt.legend()

    ax6 = plt.subplot(3,2,6,sharex=ax2)
    plt.hist(dataSet_08[col].values-mean08, bins=100,label='08-mean')
    plt.legend()

    plt.show()

# #########################################################################################
# 分别绘制结冰与不结冰的数据的分布
if True:
    dataSet_15 = pd.read_csv('../DataSets/train/15_data_clean.csv')
    dataSet_21 = pd.read_csv('../DataSets/train/21_data_clean.csv')
    dataSet_08 = pd.read_csv('../DataSets/test/08_data.csv')

    col = ['generator_speed']
    is_ice_index_15 = dataSet_15['ice_state']==1
    is_ice_15 = dataSet_15[is_ice_index_15]
    no_ice_15 = dataSet_15[~is_ice_index_15]

    fig = plt.figure()
    fig.suptitle('Scatter of generator_speed and power')

   # plt.scatter(dataSet_08['generator_speed'].values, dataSet_08['power'].values)

    plt.scatter(no_ice_15['generator_speed'].values, no_ice_15['power'].values, c='g')

    plt.xlabel('generator_speed')
    plt.ylabel('power')
    plt.legend(['normal','ice'],loc='best')
    plt.show()


