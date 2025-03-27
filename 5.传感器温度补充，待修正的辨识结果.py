# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:54:54 2024

@author: 13121

Created on Thu Nov 16 16:02:15 2023
三个任务
1）data里拼入传感器的温度
2）片段的各个传感器最高温、最低温平均温度,pack的重新算下，覆盖掉
3）读取，SOH,R0的表格，参考之前代码整理成大表，分析cell成组关系,以及cell id与传感器探针温度的对应关系
@author: 13121
"""

import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def unfold(a):
    b=(a.split('_'))
    b=pd.to_numeric(b,'coerce')
    b=list(b)
    if len(b)==34:
        # 将列表转换为NumPy数组
        b_array = np.array(b)
        # 将字符串数组转换为浮点数数组
        b_array = b_array.astype(float)
        # 所有数都减去40
        b_array -= 40
        # 将数组中的0替换为np.nan
        b_array[b_array == -40] = np.nan
        b=list(b_array)
    else:
        b=list(np.full(34, np.nan))
    return b

def cell_temp(length,data_c):
    b=pd.DataFrame()
    for i in tqdm(range(int(len(data_c)/length+1))):
      try:
        chakan=data_c.iloc[i*length:i*length+length]
        a1=list(chakan['单体电池温度值'].apply(lambda x:unfold(x)))
        a1=pd.DataFrame(a1)
        b=pd.concat([b,a1])
      except:
        return i
    return b

names=['LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
       'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
       'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
       'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']


chuanganqinumber=34
# name='LB378Y4W3JA179379'
for name in names:
    #提取需要的编号
    tbl04=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'tbl04.csv', index_col=0)
    
    for listname in os.listdir('G:/数据/吉利原始数据/'+name+'/'):
        if '.csv' in (listname):
           raw_data=pd.read_csv('G:/数据/吉利原始数据/'+name+'/'+listname)
           sensors_temp=raw_data[['数据采集时间','单体电池温度值']]
           #去掉重复行
           sensors_temp = sensors_temp.drop_duplicates()
           del raw_data
           data=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'data.csv', index_col=0)
           data = pd.merge(data, sensors_temp, on='数据采集时间', how='left')
           del sensors_temp
           data['len_温度列表'] = data['单体电池温度值'].apply(lambda x: len(str(x)))
           sensor_temp=cell_temp(500000,data)
           new_col = []
           for k in range(1,chuanganqinumber+1):
            new_col.append('传感器温度'+str(k))
           sensor_temp.columns = new_col
           sensor_temp=sensor_temp.reset_index(drop=True)
           data = pd.concat([data,sensor_temp],axis = 1)
           del new_col
           del sensor_temp
            
           # 将计算结果赋给DataFrame中的对应列
           data['最高温度值'] = data.loc[:, '传感器温度1':'传感器温度34'].max(axis=1)
           data['最低温度值'] = data.loc[:, '传感器温度1':'传感器温度34'].min(axis=1)
           data['Tem'] = data.loc[:, '传感器温度1':'传感器温度34'].mean(axis=1)
           data.to_csv('G:/DBD4数据/DIHAO,参数提取片段与数据,温度补充/'+name+'data.csv',index=False)
    
    for i in tbl04['片段编号']:
        chakan=data[data['number']==i]
        tbl04.loc[tbl04['片段编号']==i,'temp_max']=chakan['最高温度值'].mean()
        tbl04.loc[tbl04['片段编号']==i,'temp_min']=chakan['最低温度值'].mean()
        tbl04.loc[tbl04['片段编号']==i,'temp_mean']=chakan['Tem'].mean()
        for k in range(1,chuanganqinumber+1):
            tbl04.loc[tbl04['片段编号']==i,'传感器温度'+str(k)+'_max']=chakan['传感器温度'+str(k)].max()
            tbl04.loc[tbl04['片段编号']==i,'传感器温度'+str(k)+'_min']=chakan['传感器温度'+str(k)].min()        
            tbl04.loc[tbl04['片段编号']==i,'传感器温度'+str(k)+'_mean']=chakan['传感器温度'+str(k)].mean()
    tbl04['平均温差'] = tbl04['temp_max']-tbl04['temp_min']
    
    tbl04.to_csv('G:/DBD4数据/DIHAO,参数提取片段与数据,温度补充/'+name+'tbl04.csv',index=False)

# chakan=data[data['number']==22894] 
#最后一步的内阻拼接，观察与验证，SOH的拼接与保存

from scipy import stats

def lvbo(r0_df):
    # 假设 r0_df 包含你的数据，其中 r0_df[1] 是时间序列，r0_df[5] 是滤波对象
    time_series = r0_df['1'].values
    filter_data = r0_df['3'].values
    
    # 窗口大小和步长
    window_span = 10000
    step_size = 2000
    
    # 存储滤波后的数据
    filtered_time_series = []
    filtered_filter_data = []
    
    # 使用滑动窗口来筛选异常值
    start_time = time_series[0]
    end_time = start_time + window_span
    
    while end_time <= time_series[-1]:
        # 确定窗口的起始和结束索引
        start_index = np.where(time_series >= start_time)[0][0]
        end_index = np.where(time_series <= end_time)[0][-1]
    
        window_time_series = time_series[start_index:end_index+1]
        window_filter_data = filter_data[start_index:end_index+1]
    
        # 使用Z-score方法计算窗口内数据的偏离程度
        z_scores = np.abs(stats.zscore(window_filter_data))
    
        # 使用阈值来过滤异常值
        threshold = 1.2  # 根据需要调整阈值
        filtered_window_time_series = window_time_series[z_scores < threshold]
        filtered_window_filter_data = window_filter_data[z_scores < threshold]
    
        # 将窗口内筛选后的数据添加到结果列表中
        filtered_time_series.extend(filtered_window_time_series)
        filtered_filter_data.extend(filtered_window_filter_data)
    
        # 更新窗口的起始和结束时间
        start_time += step_size
        end_time = start_time + window_span
    
    # 将筛选后的数据转换为数组
    filtered_time_series = np.array(filtered_time_series)
    filtered_filter_data = np.array(filtered_filter_data)
    
    data = {'Time_Series': filtered_time_series, 'Filtered_Filter_Data': filtered_filter_data}
    df = pd.DataFrame(data)
    
    # 去除重复行
    df = df.drop_duplicates()
    # plt.scatter(r0_df[1],r0_df[3])
    # plt.scatter(df['Time_Series'],df['Filtered_Filter_Data'],marker='o')
    # 只保留里程
    return df['Time_Series']


merge_data=pd.DataFrame()
for name in tqdm(names):  
    r0_df = pd.read_csv(f'C:/Users/13121/Desktop/Story 1/Vehicles/{name}r0_df.csv')
    soh_df = pd.read_csv(f'C:/Users/13121/Desktop/Story 1/Vehicles/{name}soh_df.csv')
    tbl04 = pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据,温度补充/'+name+'tbl04.csv')
    #这三行决定了是否滤波
    baoliulicheng =  lvbo(r0_df)
    r0_df = r0_df[r0_df['1'].isin(baoliulicheng)]
    soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]
    
    r0_df = r0_df.rename(columns={r0_df.columns[0]: '片段编号'})
    new_column_names = [f'estimated_r0_cell{i}' for i in range(1, 96)]
    r0_df = r0_df.rename(columns=dict(zip(r0_df.columns[3:98], new_column_names)))
    selected_columns = ['片段编号'] + [f'estimated_r0_cell{i}' for i in range(1, 96)]
    r0_df = r0_df.loc[:, selected_columns]
    
    soh_df = soh_df.rename(columns={soh_df.columns[0]: '片段编号'})
    new_column_names = [f'estimated_soh_cell{i}' for i in range(1, 96)]
    soh_df = soh_df.rename(columns=dict(zip(soh_df.columns[3:98], new_column_names)))
    selected_columns = ['片段编号'] + [f'estimated_soh_cell{i}' for i in range(1, 96)]
    soh_df = soh_df.loc[:, selected_columns]
    
    tbl04 = tbl04[tbl04['片段编号'].isin(r0_df['片段编号'])] 
    a= tbl04.loc[:, '传感器温度1_max':'传感器温度34_mean']
    tbl04 = tbl04[['片段编号', 'start_time', 'end_time', '状态', 'start_mile',
           'end_mile', '平均车速', '平均电流', '帧数', 'start_soc', 'end_soc', '容量(A·h)',
           'temp_max', 'temp_min', 'temp_mean', '平均温差', '平均压差', 'vin']]
    tbl04 = pd.concat([tbl04, a], axis=1)
    
    
    merged_df = pd.merge(r0_df, soh_df, on='片段编号', how='outer')
    merged_df = pd.merge(merged_df, tbl04, on='片段编号', how='outer')
    
    merge_data = pd.concat([merge_data, merged_df], axis=0)
merge_data['start_mile']=merge_data['start_mile']*0.1
merge_data['end_mile']=merge_data['end_mile']*0.1
merge_data['平均电流']=merge_data['平均电流']*0.1
merge_data['容量(A·h)']=merge_data['容量(A·h)']*0.1
    

def identify_group(number):
    if number <=10:
        return int((number-1)/5+1)
    elif 11 <= number <= 70:
        return int((number-11)/6+3)
    elif 71 <= number <= 95:
        return int((number-71)/5+13)


# # 输入数字
# result = identify_group(52)
# print(f"该数字属于：{result}")

#补充merge里的数据：
for i in tqdm(range(1, 96)):
    a=identify_group(i)
    b = 2*a-1
    c = 2*a
    merge_data['cell'+str(i) + 'max_temp a']= merge_data['传感器温度'+str(b)+'_max']
    merge_data['cell'+str(i) + 'min_temp a']= merge_data['传感器温度'+str(b)+'_min']
    merge_data['cell'+str(i) + 'mean_temp a']= merge_data['传感器温度'+str(b)+'_mean']
    merge_data['cell'+str(i) + 'max_temp b']= merge_data['传感器温度'+str(c)+'_max']
    merge_data['cell'+str(i) + 'min_temp b']= merge_data['传感器温度'+str(c)+'_min']
    merge_data['cell'+str(i) + 'mean_temp b']= merge_data['传感器温度'+str(c)+'_mean']

merge_data.to_csv('C:/Users/13121/Desktop/Story 1/results need to be fixed_filted.csv',index=None)

# a=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/results need to be fixed_filted.csv')



