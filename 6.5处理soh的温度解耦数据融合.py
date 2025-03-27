# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:49:15 2024

@author: 13121

整体上：
先不考虑温度影响，只考虑趋势，做趋势的预测
重点突出一件事（用尽可能少的数据预测尽可能高的精度，两个变量，一个是精度，一个是预测的时机，模型对比时用得到）
模型对比（效果不好就取极端情况（数据少、涉及隐私等等）
创新性：可以增加机理分析（找下拐点出现的时刻与内阻、温度（编号，暂时先不考虑）的相关性）

把数据处理好：实现任意抽调任意车辆的指定电池做训练集和验证集
跑通transformer模型（无grid search)
改装TTA模型
模型对比，讲故事（训练集限制的情况下怎么样，对拐点出现的适应性怎么样）

1)循环对19辆车执行：
先处理cap:读取一舟处理后的数据，处理成等间隔的样子，'里程'，'fixed','k','kk'(默认已知)
再处理内阻：'里程'，'r0'，'svr'，'fixed'
2）任意指定训练集、验证集
3）transformer效果观察（自己选择表达方式、预测起始点，出图观察）
4）tta效果观察

"""
import pandas as pd
import numpy as np
from scipy import stats
import pickle
from tqdm import tqdm



#%%数据准备
Vehicle_raw = {}
Vehicle_train = {}

pickle_file_path = 'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/一舟结果读取，与出图更新/cap_compare.pickle'

# 打开 pickle 文件并读取数据
with open(pickle_file_path, 'rb') as file:
    # 使用pickle.load()方法加载数据
    data_soh = pickle.load(file)

for name in tqdm(['LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
             'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
             'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
             'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']):

    soh_df = pd.read_csv(f'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/{name}soh_df.csv')
    filted_soh_df=soh_df.copy()
    chakan=data_soh[name]
    
    # 第一步：读取filted_r0_df['1']的最大值与最小值
    min_mileage = filted_soh_df['1'].min()
    max_mileage = filted_soh_df['1'].max()
    
    min_val = int(np.floor(soh_df['1'].min() / 500) * 500)  # 向下取整到 500 的倍数
    max_val = int(np.floor(soh_df['1'].max() / 500) * 500)  # 向下取整到 500 的倍数 
    # 生成 500 间隔的数列
    mileage_series = np.arange(min_val, max_val + 1, 500)
    # 创建 DataFrame
    intrval_soh_df = pd.DataFrame({'Mileage': mileage_series})
    
    # 第二步：处理每个电池单体
    for battery_number in range(1, 96):  # 电池编号从1到95
        # 获取电池状态轨迹
        battery_status = chakan[battery_number]
    
        # 确定里程对应的索引范围
        min_index = int(min_mileage / 5000)
        max_index = int(max_mileage / 5000)
    
        # 截取有效里程范围的电池状态
        effective_status = battery_status[min_index:max_index + 1]
        # 将 effective_status 转换为一维数组
        effective_status = effective_status.ravel()
        # 里程值的计算（每5000公里记录一次）
        mileage_values = np.arange(min_index * 5000, (max_index + 1) * 5000, 5000)
        # 10次多项式拟合
        coefficients = np.polyfit(mileage_values, effective_status, 10)
        # 创建一个多项式函数
        poly_func = np.poly1d(coefficients)
        
        # 计算一阶导数
        poly_func1 = np.polyder(poly_func, 1)
        # 计算二阶导数
        poly_func2 = np.polyder(poly_func, 2)
    
        # 在filted_r0_df上添加新列
        column_name = f'一舟{battery_number}Fixed'
        filted_soh_df[column_name] = filted_soh_df['1'].apply(poly_func)
        intrval_soh_df[column_name] = intrval_soh_df['Mileage'].apply(poly_func)
        




         


    
        
    #出出图用的数据
    # 第一列是 min_mileage 到 max_mileage 的均匀分布的 5000 行
    num_rows = 5000
    mileage_range = np.linspace(min_mileage, max_mileage, num_rows)
    new_df = pd.DataFrame(mileage_range, columns=['Mileage'])
    
    # 对于每个电池单体，应用多项式函数并添加为新列
    for battery_number in range(1, 96):  # 电池编号从1到95
        # 获取对应的多项式函数
        battery_status = chakan[battery_number]
        min_index = int(min_mileage / 5000)
        max_index = int(max_mileage / 5000)
        effective_status = battery_status[min_index:max_index + 1]
        effective_status = effective_status.ravel()
        mileage_values = np.arange(min_index * 5000, (max_index + 1) * 5000, 5000)
        coefficients = np.polyfit(mileage_values, effective_status, 10)
        poly_func = np.poly1d(coefficients)
    
        # 应用多项式函数
        column_name = f'一舟{battery_number}Fixed'
        new_df[column_name] = poly_func(new_df['Mileage'])  





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
    return df['Time_Series']

#%%数据准备
pickle_file_path = 'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/一舟结果读取，与出图更新/cap_compare.pickle'
# pickle_file_path = 'C:/Users/13121/Desktop/cap_cell_compare.pickle'


# 打开 pickle 文件并读取数据
with open(pickle_file_path, 'rb') as file:
    # 使用pickle.load()方法加载数据
    data_soh = pickle.load(file)

for name in tqdm(['LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
             'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
             'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
             'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']):


    r0_df = pd.read_csv(f'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/{name}r0_df.csv')
    soh_df = pd.read_csv(f'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/{name}soh_df.csv')


    # baoliulicheng =  lvbo(r0_df)
    # filted_r0_df = r0_df[r0_df['1'].isin(baoliulicheng)]
    # filted_soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]
    filted_soh_df=soh_df.copy()
    
    chakan=data_soh[name]
    
    # 第一步：读取filted_r0_df['1']的最大值与最小值
    min_mileage = filted_soh_df['1'].min()
    max_mileage = filted_soh_df['1'].max()
    
    # 第二步：处理每个电池单体
    for battery_number in range(1, 96):  # 电池编号从1到95
        # 获取电池状态轨迹
        battery_status = chakan[battery_number]
    
        # 确定里程对应的索引范围
        min_index = int(min_mileage / 5000)
        max_index = int(max_mileage / 5000)
    
        # 截取有效里程范围的电池状态
        effective_status = battery_status[min_index:max_index + 1]
        # 将 effective_status 转换为一维数组
        effective_status = effective_status.ravel()
        # 里程值的计算（每5000公里记录一次）
        mileage_values = np.arange(min_index * 5000, (max_index + 1) * 5000, 5000)
        # 10次多项式拟合
        coefficients = np.polyfit(mileage_values, effective_status, 10)
    
        # 创建一个多项式函数
        poly_func = np.poly1d(coefficients)
    
        # 在filted_r0_df上添加新列
        column_name = f'一舟{battery_number}Fixed'
        filted_soh_df[column_name] = filted_soh_df['1'].apply(poly_func)
        
    #出出图用的数据
    # 第一列是 min_mileage 到 max_mileage 的均匀分布的 5000 行
    num_rows = 5000
    mileage_range = np.linspace(min_mileage, max_mileage, num_rows)
    new_df = pd.DataFrame(mileage_range, columns=['Mileage'])
    
    # 对于每个电池单体，应用多项式函数并添加为新列
    for battery_number in range(1, 96):  # 电池编号从1到95
        # 获取对应的多项式函数
        battery_status = chakan[battery_number]
        min_index = int(min_mileage / 5000)
        max_index = int(max_mileage / 5000)
        effective_status = battery_status[min_index:max_index + 1]
        effective_status = effective_status.ravel()
        mileage_values = np.arange(min_index * 5000, (max_index + 1) * 5000, 5000)
        coefficients = np.polyfit(mileage_values, effective_status, 10)
        poly_func = np.poly1d(coefficients)
    
        # 应用多项式函数
        column_name = f'一舟{battery_number}Fixed'
        new_df[column_name] = poly_func(new_df['Mileage'])      
        
    
    filted_soh_df.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/NMC-PV cap 温度修正结果/'+name+'filtered_soh.csv',index=False)
    new_df.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/NMC-PV cap 温度修正结果/'+name+'soh_for_line.csv',index=False)
