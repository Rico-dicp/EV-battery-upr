# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:09:31 2024
批量循环计算可以计算的所有车辆
@author: 13121
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times new Roman'] 
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'
from tqdm import tqdm
import pyswarms as ps
from scipy import stats
from sklearn.svm import SVR

#%%系列函数定义

fit_params = [2.79349202e+00,  1.32456734e-01, -1.06408510e-02,  4.87568943e-04,
       -1.21132612e-05,  1.21430353e-07,  1.19820875e-09, -4.82818206e-11,
        5.64142933e-13, -3.06234456e-15,  6.54961010e-18]


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

# 定义十次多项式函数
def polynomial_func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6 + a7*x**7 + a8*x**8 + a9*x**9 + a10*x**10

# 定义可调用的拟合函数
def fitted_function(x,fit_params):
    return polynomial_func(x, *fit_params)


#粒子群优化
def system_function(particles, data_identification):
    a, b, c, Rp, Cp, Vp0 = particles[:, 0], particles[:, 1], particles[:, 2],particles[:, 3], particles[:, 4], particles[:, 5]
    cap = data_identification['cap'][:, np.newaxis]
    total_current = data_identification['总电流'][:, np.newaxis]
    y = fitted_function(a + cap / (b * 150)*100, fit_params) - (-total_current*Rp-Vp0)*np.exp(-np.divide(3600*cap, -Rp* Cp* total_current))-total_current*(c+Rp)
    return y


def objective_function(particles, n_particles, target_data, data_identification):
    predicted_data = system_function(particles, data_identification)
    target_data = np.tile(target_data, (n_particles, 1)).T
    mse = np.mean((target_data - predicted_data)**2, axis=0)  # 计算每个粒子的均方误差
    return mse

#参数辨识取375到385，每10个取一个
def identify_parameters(data, i, m, n_particles, cishu):
#先计算两个内阻出来，输出，并取均值然后乘以系数0.8到1.0作为参考范围
    i=int(i)
    add=list(range(i-1,i+3))
    data_identification=data[data['number'].isin(add)]
    r0,dert_t,dert_u,dert_i=r1(data_identification,m)
    
    data_identification = data[data['number'].isin([i])][2:]
    data_identification = data_identification[(data_identification['总电压']>=375)&(data_identification['总电压']<=385)]
    # data_identification = data_identification[(data_identification['总电压']>=375)]
    data_identification['cap'] = data_identification['单帧容量变化(A·h)'].cumsum()
    data_identification['t'] = np.array(range(len(data_identification)))[:, np.newaxis]*10
    data_identification=data_identification[::10]

    target_data = data_identification['单体电压'+str(m)].values * 0.001
    
    options = {'c1': 0.1, 'c2': 0.1, 'w': 0.9}
    bounds = ([0, 0.7, r0*0.21, r0*0.06, 60000, 0.001], [95, 1, 0.23*r0, r0*0.08, 70000, 0.0015])  # 参数取值范围   
    
    # 生成符合边界范围的初始粒子分布
    initial_particles = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_particles, 6))
    initial_particles = np.clip(initial_particles, bounds[0], bounds[1])
    
    # 使用固定的初始粒子分布创建优化器对象
    threshold=1e-9
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=6, options=options, bounds=bounds, init_pos=initial_particles,ftol=threshold,ftol_iter=10)
    cost, best_pos = optimizer.optimize(objective_function,iters=cishu, n_particles=n_particles, target_data=target_data, data_identification=data_identification,verbose=0)
    return best_pos

def lvbo(r0_df):
    # 假设 r0_df 包含你的数据，其中 r0_df[1] 是时间序列，r0_df[5] 是滤波对象
    time_series = r0_df[1].values
    filter_data = r0_df[5].values
    
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

def identify(tbltbl):
    n_particles=1000
    
    soc_df =[]
    soh_df =[]
    r0_df =[]
    rp_df=[]
    cp_df=[]
    Vp0_df=[]
    jiezhiSOC_df=[]
    
    for xunhuan in tqdm(tbltbl['片段编号']):
      try: 
        data_identification = data[data['number'].isin([xunhuan])][2:]
        licheng=data_identification['里程'].iloc[0]
        tem=(data_identification['最高温度值']/2+data_identification['最低温度值']/2).mean()
        soc=[xunhuan,licheng,tem]
        soh=[xunhuan,licheng,tem]
        resistance=[xunhuan,licheng,tem]  
        rp=[xunhuan,licheng,tem]
        cp=[xunhuan,licheng,tem]
        Vp0=[xunhuan,licheng,tem]
        jiezhiSOC=[xunhuan,licheng,tem]
        for m in range(1,96):
            i=int(xunhuan)
            add=list(range(i-1,i+3))
            data_identification=data[data['number'].isin(add)]
            identified_parameters = identify_parameters(data, xunhuan, m, n_particles, 60)
            jiezhi_SOC=identified_parameters[0]+data_identification[data_identification['总电压']>=375]['单帧容量变化(A·h)'].sum()/(identified_parameters[1]*150)*100
            soc.append(identified_parameters[0])
            soh.append(identified_parameters[1])
            resistance.extend([identified_parameters[2]])  
            rp.append(identified_parameters[3])
            cp.append(identified_parameters[4])
            Vp0.append(identified_parameters[5])
            jiezhiSOC.append(jiezhi_SOC)
        soc_df.append(soc)
        soh_df.append(soh)
        r0_df.append(resistance)
        rp_df.append(rp)
        cp_df.append(cp)
        Vp0_df.append(Vp0)
        jiezhiSOC_df.append(jiezhiSOC) 
    
      except:
        continue    
    
    soc_df = pd.DataFrame(soc_df)
    soh_df = pd.DataFrame(soh_df)
    r0_df = pd.DataFrame(r0_df)
    rp_df=pd.DataFrame(rp_df)
    cp_df=pd.DataFrame(cp_df)
    Vp0_df=pd.DataFrame(Vp0_df)
    jiezhiSOC_df=pd.DataFrame(jiezhiSOC_df)
    
    return soc_df,soh_df,r0_df,rp_df,cp_df,Vp0_df,jiezhiSOC_df


#用SVR获得平滑曲线：我需要的是输入参数（最优化的C和gamma，缩放倍数，以及去噪后的数据),输出滤波后的曲线，横坐标为里程，纵坐标为容量与内阻
def Smooth_curve(a, best_params, filted_df, n, n1, n2):
    X = filted_df[1].values.reshape(-1, 1)
    X_pred = np.arange(int(X.min()), int(X.max()),100).reshape(-1, 1)
    X = X / n1
    X_pred = X_pred / n1
    
    X1 = a[1].values.reshape(-1, 1)
    X1 = X1 / n1

    # 创建一个示例 DataFrame，包含一个名为 'index_column' 的索引
    data = {'index_column': range(len(X_pred))}
    df = pd.DataFrame(data)
    # 将 X_pred 添加为 'mileage' 列
    df['mileage'] = X_pred * n1
    for m in range(3, n):

        y = filted_df[m].values * n2
        best_svr = SVR(kernel='rbf', degree=3, C=best_params['C'], gamma=best_params['gamma'], epsilon=0.11, verbose=1)
        best_svr.fit(X, y)
        y_pred = best_svr.predict(X_pred)
        y_pred = y_pred / n2
        df['flitered' + str(m - 2)] = y_pred
        a['svr' + str(m - 2)] = best_svr.predict(X1) / n2
    df = df.iloc[:, 1:]
    return df, a


#%%数据读取
names=['LB378Y4W3JA179379','LB378Y4W9JA174980','LB378Y4W5JA179156','LB378Y4W4JA179259','LB378Y4W9JA176518',
       'LB378Y4W9JA176454','LB378Y4W6JA179408','LB378Y4W7JA178669','LB378Y4W4JA176278','LB378Y4W9JA176454',
       'LB378Y4W1JA175086','LB378Y4W1JA177517','LB378Y4W1JA179350','LB378Y4W6JA179408','LB378Y4W6JA179408',
       'LB378Y4W0JA174527','LB378Y4WXJA176169','LB378Y4W8JA179782','LB378Y4W4JA176118','LB378Y4W5JA179917',
       'LB378Y4W8JA175280','LB378Y4WXJA176348','LB378Y4W4JA179309','LB378Y4W1JA177534','LB378Y4W7JA177862',
       'LB378Y4W4JA177348','LB378Y4WXJA173319','LB378Y4W8JA177207','LB378Y4W0JA174706','LB378Y4W7JA175268',
       'LB378Y4W7JA179725','LB378Y4W1JA182166','LB378Y4W9JA173988','LB378Y4WXJA175104','LB378Y4W4JA176278']

for name in names:

    data=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'data.csv', index_col=0)
    tbl04=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'tbl04.csv', index_col=0)
    data['time']=pd.to_datetime(data['time'],format = '%Y-%m-%d %H:%M:%S')
    tbltbl=tbl04.copy()
    
    tbltbl=tbltbl[tbltbl['start_soc']<=25]
    soc_df,soh_df,r0_df,rp_df,cp_df,Vp0_df,jiezhiSOC_df=identify(tbltbl)
    
    
    baoliulicheng =  lvbo(r0_df)
    filted_r0_df = r0_df[r0_df[1].isin(baoliulicheng)]
    filted_soh_df = soh_df[soh_df[1].isin(baoliulicheng)]
    
    
    wendu_filted_r0_df=filted_r0_df[(filted_r0_df[2]>=23)&(filted_r0_df[2]<=25)]
    wendu_filted_soh_df=filted_soh_df[(filted_soh_df[2]>=23)&(filted_soh_df[2]<=25)]
    
    
    svr_r0_df, r0_df = Smooth_curve(r0_df, {'C': 10, 'gamma': 0.035}, filted_r0_df, 98, 10000, 10000)
    wendu_svr_r0_df, wendu_filted_r0_df = Smooth_curve(wendu_filted_r0_df,{'C': 0.5, 'gamma': 0.035}, wendu_filted_r0_df, 98, 10000, 10000)
    
    
    svr_soh_df, soh_df = Smooth_curve(soh_df, {'C': 10, 'gamma': 0.035}, filted_soh_df, 98, 10000, 100)
    
    
    
    #数据存储：原始辨识的结果，其中soh与r0有滤波结果，限制温度的r0滤波结果，soh与r0的全值滤波（出图用），r0的限制温度滤波（出图用）
    r0_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'r0_df'+'.csv',index=False)
    soh_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'soh_df'+'.csv',index=False)
    soc_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'soc_df'+'.csv',index=False)
    rp_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'rp_df'+'.csv',index=False)
    cp_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'cp_df'+'.csv',index=False)
    Vp0_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'Vp0_df'+'.csv',index=False)
    jiezhiSOC_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'jiezhiSOC_df'+'.csv',index=False)
    svr_r0_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'svr_r0_df'+'.csv',index=False)
    svr_soh_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'svr_soh_df'+'.csv',index=False)
    wendu_svr_r0_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'wendu_svr_r0_df'+'.csv',index=False)
    wendu_filted_r0_df.to_csv('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'wendu_filted_r0_df'+'.csv',index=False)

    
    #出图
    #先出一个电池r0的图
    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    # 设置字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # 绘制散点图
    plt.scatter(r0_df[1]/1000, r0_df[3]*1000, color='lightseagreen', label='Outlier', marker='o')
    plt.scatter(filted_r0_df[1]/1000, filted_r0_df[3]*1000,  color='deeppink', label='Outliers removed', marker='o')
    plt.scatter(wendu_filted_r0_df[1]/1000, wendu_filted_r0_df[3]*1000,  color='gold', label='23-25\u00b0C', marker='o')
    # 绘制折线图
    plt.plot(svr_r0_df['mileage']/1000, svr_r0_df['flitered1']*1000, marker='o', linestyle='-', color='darkblue', label='SVR')
    plt.plot(wendu_svr_r0_df['mileage']/1000, wendu_svr_r0_df['flitered1']*1000, marker='o', linestyle='-', color='orange', label='23-25\u00b0C SVR')
    # 设置图表标题和坐标轴标签
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('R0 (mΩ)', fontsize=16, fontweight='bold')
    
    # 增大坐标轴刻度数字的字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 添加图例并增大图例字体大小
    plt.legend(fontsize=14)
    # 显示图表
    plt.show()
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'Cell Internal Resistance Identification and Filtering Processing.png', format='png', bbox_inches='tight')
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'Cell Internal Resistance Identification and Filtering Processing.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    
    
    #多个/所有电池r0的svr图，不固定温度
    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    # 设置字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    
    for m in range(1, 96):
    # 绘制折线图
        plt.plot(svr_r0_df['mileage']/1000, svr_r0_df['flitered'+str(m)]*1000, linestyle='-', color='darkblue')
    # 设置图表标题和坐标轴标签
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('R0 (mΩ)', fontsize=16, fontweight='bold')
    
    # 增大坐标轴刻度数字的字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 添加图例并增大图例字体大小
    # 显示图表
    plt.show()
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'r0_svr.png', format='png', bbox_inches='tight')
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'r0_svr.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    #多个/所有电池r0的svr图，固定温度,这个调不好了，我觉得可能是传感器空间分布的问题
    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    # 设置字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    
    for m in range(1, 96):
    # 绘制折线图
        plt.plot(wendu_svr_r0_df['mileage']/1000, wendu_svr_r0_df['flitered'+str(m)]*1000, linestyle='-', color='red')
    # 设置图表标题和坐标轴标签
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('R0 (mΩ)', fontsize=16, fontweight='bold')
    
    # 增大坐标轴刻度数字的字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 添加图例并增大图例字体大小
    # 显示图表
    plt.show()
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'r0_svr_ct.png', format='png', bbox_inches='tight')
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'r0_svr_ct.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    
    ###经过观察，我认为SOH最好复刻r0的流程，或者基于R0的滤波结果展开计算
    #先出一个电池soh的图
    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    # 设置字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # 绘制散点图
    # plt.scatter(soh_df[1]/1000, soh_df[3], color='lightseagreen', label='Outlier', marker='o')
    plt.scatter(filted_soh_df[1]/1000, filted_soh_df[3],  color='deeppink', label='SOH', marker='o')
    # plt.scatter(wendu_filted_soh_df[1]/1000, wendu_filted_soh_df[3]*1000,  color='gold', label='23-25\u00b0C', marker='o')
    # 绘制折线图
    plt.plot(svr_soh_df['mileage']/1000, svr_soh_df['flitered1'], marker='o', linestyle='-', color='darkblue', label='SVR')
    
    # 设置图表标题和坐标轴标签
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('SOH', fontsize=16, fontweight='bold')
    
    # 增大坐标轴刻度数字的字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 添加图例并增大图例字体大小
    plt.legend(fontsize=14)
    # 显示图表
    plt.show()
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'cell_soh.png', format='png', bbox_inches='tight')
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'cell_soh.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    #多个/所有电池soh的图
    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    # 设置字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    
    for m in range(1, 96):
    # 绘制折线图
        plt.plot(svr_soh_df['mileage']/1000, svr_soh_df['flitered'+str(m)], linestyle='-', color='red')
    # 设置图表标题和坐标轴标签
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('SOH', fontsize=16, fontweight='bold')
    
    # 增大坐标轴刻度数字的字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 添加图例并增大图例字体大小
    # 显示图表
    plt.show()
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'soh.png', format='png', bbox_inches='tight')
    plt.savefig('C:/Users/13121/Desktop/Story 1/Vehicles/'+name+'soh.svg', format='svg', bbox_inches='tight')
    plt.close()
