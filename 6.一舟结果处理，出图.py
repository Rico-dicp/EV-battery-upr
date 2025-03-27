# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:46:17 2024

@author: 13121

Created on Thu Dec 21 11:55:10 2023

@author: 13121

目标：
1）出图展示考虑了温度的修正结果
2）准备好数据用于后续的指标计算

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times new roman'] 
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'

from scipy import stats

import pickle



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

#%%数据准备
name='LB378Y4W5JA179156'

r0_df = pd.read_csv(f'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/{name}r0_df.csv')
soh_df = pd.read_csv(f'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/{name}soh_df.csv')


baoliulicheng =  lvbo(r0_df)
filted_r0_df = r0_df[r0_df['1'].isin(baoliulicheng)]
filted_soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]


#数据融合
pickle_file_path = 'C:/Users/13121/Desktop/DBD 4 process/12.06/2.Temperature Fix/r0_compare.pickle'
# pickle_file_path = 'C:/Users/13121/Desktop/cap_cell_compare.pickle'


# 打开 pickle 文件并读取数据
with open(pickle_file_path, 'rb') as file:
    # 使用pickle.load()方法加载数据
    data_r0 = pickle.load(file)

chakan=data_r0[name]

# 第一步：读取filted_r0_df['1']的最大值与最小值
min_mileage = filted_r0_df['1'].min()
max_mileage = filted_r0_df['1'].max()

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
    filted_r0_df[column_name] = filted_r0_df['1'].apply(poly_func)
    
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

filted_r0_df.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/一舟结果读取，与出图更新/filtered_r0.csv')
new_df.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/一舟结果读取，与出图更新/r0_for_line.csv')

#################################################################################################################
pickle_file_path = 'C:/Users/13121/Desktop/DBD 4 process/12.06/2.Temperature Fix/cap_compare.pickle'
# pickle_file_path = 'C:/Users/13121/Desktop/cap_cell_compare.pickle'


# 打开 pickle 文件并读取数据
with open(pickle_file_path, 'rb') as file:
    # 使用pickle.load()方法加载数据
    data_soh = pickle.load(file)

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
    

filted_soh_df.to_csv('C:/Users/13121/Desktop/Story 1/一舟结果读取，与出图更新/filtered_soh.csv')
new_df.to_csv('C:/Users/13121/Desktop/Story 1/一舟结果读取，与出图更新/soh_for_line.csv')
#%%数据读取：

filted_r0_df = pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/一舟结果读取，与出图更新/filtered_r0.csv')
filted_soh_df = pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/一舟结果读取，与出图更新/filtered_soh.csv')
svr_r0_df = pd.read_csv(f'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/{name}svr_r0_df.csv')
svr_soh_df = pd.read_csv(f'E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/{name}svr_soh_df.csv')

#%%


# Setting the font parameters globally to ensure no text is bold, without changing the font size.
plt.rc('font', weight='normal', size=26)
plt.rc('axes', titleweight='normal', labelweight='normal', labelsize=26)
plt.rc('xtick', labelsize=26)
plt.rc('ytick', labelsize=26)
plt.rc('legend', fontsize=20)

# Recreate the plot with professional colors and no bold font styles, keeping the font sizes as specified before.
plt.figure(figsize=(16, 8))

# Plotting "Resistance" scatter plot
plt.scatter(filted_r0_df['1']/1000, filted_r0_df['3']*1000, color='deeppink', label='Resistance', marker='o')

# Plotting "SVR" line plot
plt.plot((svr_r0_df['mileage']/1000).values, (svr_r0_df['flitered1']*1000).values, marker='o', linestyle='-', color='darkblue', label='SVR')

# Plotting "Temperature Fixed Resistance" line plot
plt.plot((filted_r0_df['1']/1000).values, (filted_r0_df['一舟1Fixed']).values, marker='o', linestyle='-',  color='orange', label='Temperature Fixed Resistance')

# Setting chart title and axis labels with no bold font
plt.xlabel('Mileage (10³ km)')
plt.ylabel('R0 (mΩ)')
# 添加图例并增大图例字体大小
plt.legend(fontsize=20, loc='upper left')
# Displaying the plot with updated aesthetic colors and no bold text
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'Cell Internal Resistance Identification and Filtering Processing.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'Cell Internal Resistance Identification and Filtering Processing.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'Cell Internal Resistance Identification and Filtering Processing.pdf', format='pdf', bbox_inches='tight')
plt.close()

#%%

###经过观察，我认为SOH最好复刻r0的流程，或者基于R0的滤波结果展开计算
#先出一个电池soh的图
plt.figure(figsize=(16, 8))
# plt.rcParams['font.weight'] = 'bold'
# 设置字体为Times new Roman
plt.rcParams['font.family'] = 'Times new Roman'

# 绘制散点图
# plt.scatter(soh_df[1]/1000, soh_df[3], color='lightseagreen', label='Outlier', marker='o')
plt.scatter(filted_soh_df['1']/1000, filted_soh_df['3'],  color='deeppink', label='SOH', marker='o')
# plt.scatter(wendu_filted_soh_df[1]/1000, wendu_filted_soh_df[3]*1000,  color='gold', label='23-25\u00b0C', marker='o')
# 绘制折线图
plt.plot(svr_soh_df['mileage']/1000, svr_soh_df['flitered1'], marker='o', linestyle='-', color='darkblue', label='SVR',linewidth=2.5)
plt.plot(filted_soh_df['1']/1000, filted_soh_df['一舟1Fixed'], marker='o', linestyle='-',  color='orange', label='Temperature Fixed Resistance SOH')
# 设置图表标题和坐标轴标签
plt.xlabel('Mileage (10³ km)', fontsize=26)
plt.ylabel('SOH', fontsize=26)
# 设置y轴的范围
plt.ylim(0.82, 0.98)

# 增大坐标轴刻度数字的字体大小
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# 添加图例并增大图例字体大小
plt.legend(fontsize=20, loc='upper right')
# 显示图表
plt.show()
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'cell_soh.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'cell_soh.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'cell_soh.pdf', format='pdf', bbox_inches='tight')
plt.close()



#%%10.08针对SOH与R0的多个单体图，每个线不一样的颜色，出图

line_soh_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/一舟结果读取，与出图更新/soh_for_line.csv')
# 创建一个颜色映射，将95个线的颜色映射到一个色谱上
colors = plt.get_cmap('coolwarm', 95)

# 创建一个figure
plt.figure(figsize=(16, 8))
# plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Times new Roman'

subset = line_soh_df.iloc[-1, 2:98]
sorted_values = subset.sort_values()
sorted_indices = np.argsort(sorted_values)


# 循环绘制95条线，每条线的颜色根据色谱映射
for m in range(1, 96):
    m1 = line_soh_df['一舟' + str(m) +'Fixed'].iloc[-1]
    x_position = 95-np.where(sorted_indices == np.where(sorted_values == m1)[0][0])[0][0]
    color = colors(x_position/95)  # 根据映射获取颜色
    plt.plot(line_soh_df['Mileage'] / 1000, line_soh_df['一舟' + str(m) +'Fixed'], linestyle='-', color=color,linewidth=2.5)

# 设置图表标题和坐标轴标签
plt.xlabel('Mileage (10³ km)', fontsize=26)
plt.ylabel('SOH', fontsize=26)

# 增大坐标轴刻度数字的字体大小
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# 创建一个色谱条
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=1, vmax=95))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Index', fontsize=26)
# 增大色谱条上刻度字体的大小
cbar.ax.tick_params(labelsize=26)
cbar.set_label('Index', fontsize=26)

# 显示图表
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'soh_all.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'soh_all.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'soh_all.pdf', format='pdf', bbox_inches='tight')
plt.close()


#%%R0图
line_r0_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/一舟结果读取，与出图更新/r0_for_line.csv')
# 创建一个颜色映射，将95个线的颜色映射到一个色谱上
colors = plt.get_cmap('coolwarm', 95)

# 创建一个figure
plt.figure(figsize=(16, 8))
# plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Times new Roman'

subset = line_r0_df.iloc[-1, 2:98]
sorted_values = subset.sort_values()
sorted_indices = np.argsort(sorted_values)


# 循环绘制95条线，每条线的颜色根据色谱映射
for m in range(1, 96):
    m1 = line_r0_df['一舟' + str(m) +'Fixed'].iloc[-1]
    x_position = np.where(sorted_indices == np.where(sorted_values == m1)[0][0])[0][0]
    color = colors(x_position/95)  # 根据映射获取颜色
    plt.plot(line_r0_df['Mileage'] / 1000, line_r0_df['一舟' + str(m) +'Fixed'], linestyle='-', color=color,linewidth=2.5)

# 设置图表标题和坐标轴标签
plt.xlabel('Mileage (10³ km)', fontsize=26)
plt.ylabel('R0 (mΩ)', fontsize=26)

# 增大坐标轴刻度数字的字体大小
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.2, 0.6)
# 创建一个色谱条
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=1, vmax=95))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Index', fontsize=26)
# 增大色谱条上刻度字体的大小
cbar.ax.tick_params(labelsize=26)
cbar.set_label('Index', fontsize=26)
# 显示图表
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'r0_all.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'r0_all.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 2/'+name+'r0_all.pdf', format='pdf', bbox_inches='tight')
plt.close()

