# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:52:37 2024

@author: 13121
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial'] # 修改字体为宋体
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'
import time
from datetime import datetime
from tqdm import tqdm
import os
import pyswarms as ps
import math
from scipy import stats
from sklearn.svm import SVR

data=pd.read_csv('C:/Users/13121/Desktop/DBD4 trying/粒子群优化/'+'data.csv', index_col=0)
tbl04=pd.read_csv('C:/Users/13121/Desktop/DBD4 trying/粒子群优化/'+'tbl04.csv', index_col=0)


#拟合函数（lamda默认为0，即无正则项）
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def r(data_identification,m):
    i=data_identification[data_identification['总电流'].diff(1)>9].index.values[-1]
    chakan=data_identification.loc[i-1:i]
    r=-(chakan['单体电压'+str(m)].diff(1)/chakan['总电流'].diff(1)).mean()*0.001
    i=data_identification[data_identification['总电流'].diff(1)<-10].index.values[0]
    chakan=data_identification.loc[i-1:i]
    r1=-(chakan['单体电压'+str(m)].diff(1)/chakan['总电流'].diff(1)).mean()*0.001

    jihua=data_identification['单体电压'+str(m)].iloc[-1]*0.001-data_identification[data_identification['状态']==10]['单体电压'+str(m)].iloc[-1]*0.001-r*data_identification[data_identification['充电状态']==1]['总电流'].iloc[-1]
    return r,r1,jihua


def r1(data_identification,m):
    i=data_identification[data_identification['总电流'].diff(1)>9].index.values[-1]
    chakan=data_identification.loc[i-1:i]
    dert_u=chakan['单体电压'+str(m)].iloc[0]-data_identification['单体电压'+str(m)].iloc[-1]
    # dert_u=chakan['单体电压'+str(m)].iloc[0]-chakan['单体电压'+str(m)].iloc[-1]
    dert_i=chakan['总电流'].diff(1).mean()
    # dert_i=-data_identification[data_identification['充电状态']==1]['总电流'].mean()
    r=dert_u/dert_i*0.001
    dert_t=(data_identification['time'].iloc[-1]-chakan['time'].iloc[0]).total_seconds() / 3600
    return r,dert_t,dert_u,dert_i


#尝试统一用1792次，cell1的OCV-SOC曲线，计算，观察演化的规律与4次的总体分布情况
def OCV_SOC(m,i,data,xia,shang):
    add=list(range(i-1,i+3))
    data['time']=pd.to_datetime(data['time'],format = '%Y-%m-%d %H:%M:%S')
    data_identification=data[data['number'].isin(add)]

    r0=r(data_identification,m)[0]/2+r(data_identification,m)[1]/2
    jihua=r(data_identification,m)[2]
    data_identification['jihua']=jihua/len(data_identification)
    data_identification['jihua']=data_identification['jihua'].cumsum()
    data_identification.loc[data_identification['状态']==10,'r'+str(m)]=r0
    data_identification.loc[data_identification['状态']==10,'uocv'+str(m)]=r0*data_identification['总电流']+data_identification['单体电压'+str(m)]*0.001+data_identification['jihua']
    data_ocv=data_identification[data_identification['状态']==10]
    data_ocv['x']=range(len(data_ocv))
    M=4
    N=30
    x=np.array(range(len(data_ocv)))
    
    x1=np.array(range(-600,len(data_ocv)+400))
    x2=x1[0:600]
    x3=x1[-401:-1]
    
    x_n =np.linspace(min(x),max(x)/8,N).astype(int)
    t_n = data_ocv[data_ocv['x'].isin(x_n)]['uocv'+str(m)]
    clf = Pipeline([('poly', PolynomialFeatures(degree=M)),
         ('linear', LinearRegression(fit_intercept=False))])
    clf.fit(x_n[:, np.newaxis], t_n)
    p = clf.predict(x2[:, np.newaxis]) 
    
    x_nn =np.linspace(7*max(x)/8,max(x),N).astype(int)
    t_nn = data_ocv[data_ocv['x'].isin(x_nn)]['uocv'+str(m)]
    clf = Pipeline([('poly', PolynomialFeatures(degree=1)),
         ('linear', LinearRegression(fit_intercept=False))])
    clf.fit(x_nn[:, np.newaxis], t_nn)
    pp = clf.predict(x3[:, np.newaxis]) 
    #提取上下阈值对应的时间差，用于容量计算,获得SOH，进而计算SOC，到此获得四个要素：SOC，UOCV,内阻，SOH，出图SOC，开路电压随时间变化曲线，出图SOC-OCV图，标明电池编号与车辆，时间节点
    #首先将补全内容与原内容拼接成dataframe,用于SOH、SOC计算    
    ocv=data_ocv[['time','x','uocv'+str(m),'单体电压'+str(m),'总电流','单帧容量变化(A·h)']]
    ocv_a=pd.DataFrame({'uocv'+str(m):p[0:600],'单帧容量变化(A·h)':-10*ocv['总电流'].iloc[2]/3600})
    ocv_b=pd.DataFrame({'uocv'+str(m):pp[-401:-1],'单帧容量变化(A·h)':-10*ocv['总电流'].iloc[-2]/3600})
    ocv=pd.concat([ocv_a,ocv,ocv_b],axis=0)
    ocv=ocv[(ocv['uocv'+str(m)]>=xia)&(ocv['uocv'+str(m)]<=shang)]
    ocv['SOC']=ocv['单帧容量变化(A·h)'].cumsum()/(ocv['单帧容量变化(A·h)'].sum())*100
    
    soh1=ocv['单帧容量变化(A·h)'].sum()/150
    ocv['r']=(ocv['uocv'+str(m)]-ocv['单体电压'+str(m)]*0.001)/ocv['总电流']
    # 图像绘制部分
    t = data_ocv['uocv' + str(m)]
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    light_green = (0.7, 0.9, 0.7)  # 较浅的绿色
    dark_blue = (0.0, 0.0, 0.8)  # 较暗的蓝色
    dark_orange = (0.6, 0.3, 0.0)  # 较暗的橙色

    
    plt.plot(x, t, color=light_green, linewidth=3, label='Original curves')
    plt.plot(x2, p, color=dark_blue, linewidth=3, label='Extended curves')
    plt.plot(x3, pp, color=dark_orange, linewidth=3, label='Extended curves')
    plt.scatter(x_n, t_n, color='', marker='o', edgecolors=dark_blue, s=100, linewidth=3, label='Points used for simulation')
    plt.scatter(x_nn, t_nn, color='', marker='o', edgecolors=dark_orange, s=100, linewidth=3, label='Points used for simulation')
    
    xia = 2.8
    shang = 4.25
    plt.axhline(xia, color='black', ls='--', lw=3)
    plt.axhline(shang, color='black', ls='--', lw=3)
    
    # 使用plt.ylim()来设置Y轴范围
    plt.ylim(2.75, 4.4)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(u'Time Series (10s)', fontsize=16)
    plt.ylabel(u'Voltage (V)', fontsize=16)
    plt.legend(loc='lower right', bbox_to_anchor=(0.30, 0.58), bbox_transform=ax1.transAxes, ncol=1, prop={'size': 16})
    plt.text(366, 2.8, '2.8V, SOC=0', ha='center', va='bottom', fontsize=16)
    plt.text(366, 4.25, '4.25V, SOC=100', ha='center', va='bottom', fontsize=16)
    
    # 显示图形
    plt.show()
    plt.savefig('C:/Users/13121/Desktop/Story 1/Additional Figures/'+'OCV.png', format='png', bbox_inches='tight')
    plt.savefig('C:/Users/13121/Desktop/Story 1/Additional Figures/'+'OCV.svg', format='svg', bbox_inches='tight')
    plt.savefig('C:/Users/13121/Desktop/Story 1/Additional Figures/'+'OCV.pdf', format='pdf', bbox_inches='tight')
    # plt.close()


    return ocv,soh1,r(data_identification,m)[0],r(data_identification,m)[1]

ocv_soc_curve,soh,r01,r02=OCV_SOC(1,1792,data,2.8,4.25)
r0=ocv_soc_curve['r'].mean()


#%%出SOC计算的图

name = 'LB378Y4W8JA179782'
indicators = pd.read_csv('C:/Users/13121/Desktop/Story 1/Indicators/'+name+'α_df'+'.csv')
data = pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'data.csv', index_col=0)
r0_df=pd.read_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'r0_df'+'.csv')
soh_df=pd.read_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'soh_df'+'.csv')
soc_df=pd.read_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'soc_df'+'.csv')
rp_df=pd.read_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'rp_df'+'.csv')
cp_df=pd.read_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'cp_df'+'.csv')

jiezhiSOC_df=pd.read_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'jiezhiSOC_df'+'.csv')

from scipy.optimize import curve_fit

# 定义十次多项式函数
def polynomial_func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6 + a7*x**7 + a8*x**8 + a9*x**9 + a10*x**10

# 从DataFrame中获取x和y数据
x_data = np.array(ocv_soc_curve['SOC'])
y_data = np.array(ocv_soc_curve['uocv1'])

# 进行十次多项式拟合
initial_guess = [1] * 11  # 初始猜测参数
fit_params, _ = curve_fit(polynomial_func, x_data, y_data, p0=initial_guess)


plt.figure(figsize=(16, 8))
plt.plot(x_data, y_data*1000, color='g', linewidth=3, label='SOC-OCV curve')
# n = 17
# chakan = data[data['number']==n]
# soc = chakan[['time','总电压','总电流','SOC','Tem','单帧容量变化(A·h)']]
# soh1 = chakan['单帧容量变化(A·h)'].sum()
# for m in range(1, 96):
#     qishisoc=soc_df[soc_df['0']==n][str(m+2)].iloc[0]
#     soh=soh_df[soh_df['0']==n][str(m+2)].iloc[0]*1.5
#     # 找到标杆行的索引
#     benchmark_row_index = soc[soc['总电压'] >= 375].index[0]
#     # 计算['C']列值的和，并添加负号
#     soc['soc'+str(m)] = soc.apply(lambda row: qishisoc-soc['单帧容量变化(A·h)'].loc[row.name + 1:benchmark_row_index].sum()/soh if row.name < benchmark_row_index else qishisoc+soc['单帧容量变化(A·h)'].loc[benchmark_row_index:row.name + 1].sum()/soh, axis=1)


# colors = plt.get_cmap('coolwarm', 95)
# for m in range(1, 96):
#     color = colors(m/95)  # 根据映射获取颜色
# # 绘制折线图
#     plt.plot(soc['soc'+str(m)],chakan['单体电压'+str(m)], linestyle='-', color=color ,linewidth=2.5)
# # 在图上添加文本标签显示 soh1 的值
# plt.text(10, 4100, f'chargr_capacity1 = {soh1:.2f} (A·h)', fontsize=14, fontweight='bold', color='black')

n = 71
chakan = data[data['number']==n]
soc = chakan[['time','总电压','总电流','SOC','Tem','单帧容量变化(A·h)']]
soh1 = chakan['单帧容量变化(A·h)'].sum()
for m in range(1, 96):
    qishisoc=soc_df[soc_df['0']==n][str(m+2)].iloc[0]
    soh=soh_df[soh_df['0']==n][str(m+2)].iloc[0]*1.5
    # 找到标杆行的索引
    benchmark_row_index = soc[soc['总电压'] >= 375].index[0]
    # 计算['C']列值的和，并添加负号
    soc['soc'+str(m)] = soc.apply(lambda row: qishisoc-soc['单帧容量变化(A·h)'].loc[row.name + 1:benchmark_row_index].sum()/soh if row.name < benchmark_row_index else qishisoc+soc['单帧容量变化(A·h)'].loc[benchmark_row_index:row.name + 1].sum()/soh, axis=1)

colors = plt.get_cmap('coolwarm', 95)
for m in range(1, 96):
    color = colors(m/95)  # 根据映射获取颜色
# 绘制折线图
    plt.plot(soc['soc'+str(m)],chakan['单体电压'+str(m)], linestyle='-', color=color ,linewidth=2.5)
# 在图上添加文本标签显示 soh1 的值
plt.text(10, 4100, f'chargr_capacity1 = {soh1:.2f} (A·h)', fontsize=14, fontweight='bold', color='black')
 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel(u'SOC', fontsize=16)
plt.ylabel(u'Voltage (mV)', fontsize=16)
plt.legend( prop={'size': 12})
# 创建一个色谱条
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=1, vmax=95))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Index', fontsize=16, fontweight='bold')
# 增大色谱条上刻度字体的大小
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Index', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 显示图形
plt.show()
plt.savefig('C:/Users/13121/Desktop/Story 1/Additional Figures/'+'SOC-OCV_2.png', format='png', bbox_inches='tight')
plt.savefig('C:/Users/13121/Desktop/Story 1/Additional Figures/'+'SOC-OCV_2.svg', format='svg', bbox_inches='tight')
plt.savefig('C:/Users/13121/Desktop/Story 1/Additional Figures/'+'SOC-OCV_2.pdf', format='pdf', bbox_inches='tight')



#%%补充图

VS=[chr(ord('a') + i) for i in range(19)]

alpha3_svr_values = indicator_list['fixed_α3'].unique().tolist()
# 创建指定大小的图表
plt.figure(figsize=(12, 6))  # 设置图形大小
plt.bar(VS, alpha3_svr_values, color='royalblue')  # 绘制柱状图
plt.xlabel('Vehicles', fontdict=font)  # 设置横坐标标签
plt.ylabel('γ', fontdict=font)  # 设置纵坐标标签
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(0, 1)
# 显示柱状图
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()



#%%α5补充图

#出图
VS=[chr(ord('a') + i) for i in range(19)]

alpha5_mean_values = []
for name in names:
    alpha5_mean_values.append(indicator_list[indicator_list['vin']==name]['svrα5'].mean())

# 创建指定大小的图表
plt.figure(figsize=(12, 6))  # 设置图形大小
plt.bar(VS, alpha5_mean_values, color='royalblue')  # 绘制柱状图
plt.xlabel('Vehicles', fontdict=font)  # 设置横坐标标签
plt.ylabel('Average η', fontdict=font)  # 设置纵坐标标签
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(0, 1)
# 显示柱状图
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_average_α5.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_average_α5.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_average_α5.pdf', format='pdf', bbox_inches='tight')
plt.close()



# 创建指定大小的图表
plt.figure(figsize=(12, 6))

# 定义标记点的大小
marker_size = 4

# 定义20种不同的颜色
colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d']
# 定义标签列表
labels = ['Vehicle a', 'Vehicle b', 'Vehicle c', 'Vehicle d', 'Vehicle e', 'Vehicle f', 'Vehicle g', 'Vehicle h', 'Vehicle i', 'Vehicle j', 'Vehicle k', 'Vehicle l', 'Vehicle m', 'Vehicle n', 'Vehicle o', 'Vehicle p', 'Vehicle q', 'Vehicle r', 'Vehicle s', 'Vehicle t']

# 遍历并绘制折线
for i, name in enumerate(names):
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    color = colors[i % len(colors)]  # 循环使用颜色
    label = labels[i % len(labels)]  # 循环使用标签
    # if name=='LB378Y4W9JA174980':
    #     plt.scatter(data['mileage'] / 1000, data['α5'], s=marker_size*5, color=color)
    plt.plot(data['mileage'] / 1000, data['svrα5'], linestyle='-', markersize=marker_size, color=color, label=label, linewidth=2)

# 设置标签和其字体属性
plt.xlabel('Mileage ($10^3$ km)', fontdict=font)
plt.ylabel('η', fontdict=font)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

# 设置图例的位置和字体大小
plt.legend(fontsize=12.5, loc='upper left', bbox_to_anchor=(1, 1))

# 显示图表
plt.tight_layout()  # 确保所有元素适合图表
plt.show()
plt.ylim(0.75, 1)

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α5.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α5.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α5.pdf', format='pdf', bbox_inches='tight')
plt.close()


plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α3.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α3.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α3.pdf', format='pdf', bbox_inches='tight')
plt.close()

################总的
plt.rc('font', family='Times new Roman')
font_size = 16
font = {'size': font_size, 'family': 'Times new Roman'}


indicator_list=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'final_α'+'.csv')

    
alpha_values = indicator_list['α'].unique().tolist()

# 创建指定大小的图表
plt.figure(figsize=(16, 8))  # 设置图形大小
# 计算α的平均值
average_utilization = np.mean(alpha_values)

# 添加平均值虚线
plt.axhline(average_utilization, color='red', linestyle='--', linewidth=3, label=f'Average Utilization Rate: {average_utilization:.4f}')



plt.bar(VS, alpha_values, color='royalblue')  # 绘制柱状图
plt.xlabel('Vehicles', fontdict=font)  # 设置横坐标标签
plt.ylabel('θ', fontdict=font)

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(0, 1)
plt.legend(fontsize=font_size)  # 显示图例
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α.pdf', format='pdf', bbox_inches='tight')
plt.close()   



#%% energy 指标出图：NMC

names=['LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
             'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
             'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
             'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']
indicator_list=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy滤波结果，出图用/New_indicators'+'.csv')

indicator_list=indicator_list[indicator_list['energy_α1_svr']>0]
#事先去除有空值的行即可
plt.figure(figsize=(9, 6))

# 获取一个由19种颜色组成的色条
colors = plt.get_cmap('coolwarm', 19)
# 存储每个name的关键fixed_α1值
critical_values = []

for name in names:
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    # 找到mileage接近200000的点
    closest_row = data.iloc[(data['mileage'] - 200000).abs().argsort()[:1]]
    critical_values.append((name, closest_row['energy_α1_svr'].values[0]))

# 根据fixed_α1排序并映射颜色
critical_values.sort(key=lambda x: x[1])
name_color_map = {name: colors(i / (len(critical_values) - 1)) for i, (name, _) in enumerate(critical_values)}

# 绘制每个name的折线图
for name in names:
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    color = name_color_map[name]  # 使用映射后的颜色
    plt.plot(data['mileage'] / 1000, data['energy_α1_svr'], linestyle='-', markersize=4, color=color,  linewidth=2.0)

# 设置标签和字体属性
plt.xlabel('Mileage (10³ km)',fontsize = 26)
plt.ylabel('α_energy',fontsize = 26)
# 增大坐标轴刻度数字的字体大小
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tight_layout()
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α1_energy.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α1_energy.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α1_energy.pdf', format='pdf', bbox_inches='tight')
plt.close()


################################################


plt.figure(figsize=(9, 6))

# 获取一个由19种颜色组成的色条
colors = plt.get_cmap('coolwarm', 19)
# 存储每个name的关键fixed_α1值
critical_values = []

for name in names:
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    # 找到mileage接近200000的点
    closest_row = data.iloc[(data['mileage'] - 200000).abs().argsort()[:1]]
    critical_values.append((name, closest_row['energy_α2_svr'].values[0]))

# 根据fixed_α1排序并映射颜色
critical_values.sort(key=lambda x: x[1])
name_color_map = {name: colors((19-i) / (len(critical_values) - 1)) for i, (name, _) in enumerate(critical_values)}

# 绘制每个name的折线图
for name in names:
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    color = name_color_map[name]  # 使用映射后的颜色
    plt.plot(data['mileage'] / 1000, data['energy_α2_svr'], linestyle='-', markersize=4, color=color,  linewidth=2.0)

# 设置标签和字体属性
plt.xlabel('Mileage (10³ km)',fontsize = 26)
plt.ylabel('β_energy',fontsize = 26)
# 增大坐标轴刻度数字的字体大小
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tight_layout()
plt.show()


plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α2_energy.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α2_energy.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α2_energy.pdf', format='pdf', bbox_inches='tight')
plt.close()