# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:17:48 2024

@author: 13121

Created on Tue Sep 26 16:53:40 2023
@author: 13121
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times new roman'] # 修改字体为宋体
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'
from tqdm import tqdm
import os
from scipy import stats
from sklearn.svm import SVR

#%%

fit_params = [2.79349202e+00,  1.32456734e-01, -1.06408510e-02,  4.87568943e-04,
       -1.21132612e-05,  1.21430353e-07,  1.19820875e-09, -4.82818206e-11,
        5.64142933e-13, -3.06234456e-15,  6.54961010e-18]

# 定义十次多项式函数
def polynomial_func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6 + a7*x**7 + a8*x**8 + a9*x**9 + a10*x**10

# 定义可调用的拟合函数
def fitted_function(x,fit_params):
    return polynomial_func(x, *fit_params)


def EFC(soh, j, threshold,k):
    # 计算与阈值 threshold 的差值
    diff = soh.iloc[:, j] - threshold  
    # 找到正向和负向差值最近的行的索引
    try: 
        positive_idx = diff[diff >= 0].idxmin()
        negative_idx = diff[diff < 0].idxmax()  
        # 获取 j 列和 K 列的两个值
        positive_j_values = soh.iloc[positive_idx, j]
        positive_k_values = soh.loc[positive_idx, k]
        negative_j_values = soh.iloc[negative_idx, j]
        negative_k_values = soh.loc[negative_idx, k]  
        # 给定两个点的坐标
        x_values = [positive_j_values, negative_j_values]
        y_values = [positive_k_values, negative_k_values]
    
        # 要插值的目标值
        target_x = threshold
        # 进行一次线性拟合
        coefficients = np.polyfit(x_values, y_values, 1)
    
        # 提取拟合的斜率和截距
        slope = coefficients[0]
        intercept = coefficients[1]
    
        # 计算目标 x 对应的 efc 值
        efc = slope * target_x + intercept
    except:
        idx = (soh.iloc[:, j] - threshold).abs().idxmin()
        efc = soh.loc[idx, k]
    
    return efc


#构建函数：针对某个车，给定编号i,遍历所有单体形成soc表,针对每个单体，定位到k=0.9，获取其当前状态的电压，电流，到截止电压的电压差，读取其rp,cp,r0,代入公式进行计算电流最大值，而后取最小的最大电流值，计算所有单体的最大功率，然后计算指标

# name=unique_values[0]
def FW(name,tj,k,dert_t,soh_list,r0_list,rp_list,cp_list,soc_list,x):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    data=x
    data=data[data['number'].isin(tj['0'])]
    chakan1=tj
    alpha5_list=[]
    b1=[]
    b2=[]
    for i in (chakan1['0']):
      try:
        I_max_list=[]
        data_identification=data[data['number']==i]
        for m in range(1,96):
            r0=r0_list[r0_list['0']==i].iloc[:, m+97].values[0]
            rp=rp_list[rp_list['0']==i].iloc[:, m+2].values[0]
            cp=cp_list[cp_list['0']==i].iloc[:, m+2].values[0]
            I_soc=soc_list[soc_list['0']==i].iloc[:, m+2].values[0]
            soh=soh_list[soh_list['0']==i].iloc[:, m+97].values[0]
            # data_identification['soc'+str(m)]=I_soc+data_identification[data_identification['总电压']>=362]['单帧容量变化(A·h)'].cumsum()/(soh*150)*100
            data_identification.loc[:, 'soc'+str(m)] = I_soc + data_identification[data_identification['总电压']>=375]['单帧容量变化(A·h)'].cumsum() / (soh * 150) * 100
            yaosuzhen=data_identification[data_identification['soc'+str(m)]>=k*100].iloc[0]
            It=-yaosuzhen['总电流']
            vt=yaosuzhen['单体电压'+str(m)]*0.001-r0*It-fitted_function(k*100,fit_params)
            # vt=yaosuzhen['单体电压'+str(m)]*0.001-r0*It-4.123334296456136
            fenzi=4.25-yaosuzhen['单体电压'+str(m)]*0.001+r0*It+vt*(1-np.exp(-(dert_t/(rp*cp))))
            fenmu=r0+rp*(1-np.exp(-(dert_t/(rp*cp))))
            I_max=fenzi/fenmu    
            I_max_list.append(I_max)
        min_I_max=min(I_max_list)
        p_max_list=[]
        min_p_max_list=[]
        for m in range(1,96):
            r0=r0_list[(r0_list['0']==i)].iloc[:, m+97].values[0]
            I_max=I_max_list[m-1]
            It=-yaosuzhen['总电流']
            vt=yaosuzhen['单体电压'+str(m)]*0.001
            min_p_max=min_I_max*(vt+r0*(min_I_max-It))
            min_p_max_list.append(min_p_max)
            p_max=I_max*(vt+r0*(I_max-It))
            p_max_list.append(p_max)
        # fenzi=min(p_max_list)
        fenzi=np.array(min_p_max_list).sum()
        fenmu=np.array(p_max_list).sum()
        b1.append(fenzi)
        b2.append(fenmu)
        alpha5_list.append(fenzi/fenmu)
      except:
        alpha5_list.append(np.nan)       
    return alpha5_list,I_max_list,b1,b2
     
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

#用SVR获得平滑曲线：我需要的是输入参数（最优化的C和gamma，缩放倍数，以及去噪后的数据),输出滤波后的曲线，横坐标为里程，纵坐标为α4与α5
def Smooth_curve(a, best_params, filted_df, n, n1, n2, indicator):
    X = filted_df['1'].values.reshape(-1, 1)
    X_pred = np.arange(int(X.min()), int(X.max()),100).reshape(-1, 1)
    X = X / n1
    X_pred = X_pred / n1
    
    X1 = a['1'].values.reshape(-1, 1)
    X1 = X1 / n1

    # 创建一个示例 DataFrame，包含一个名为 'index_column' 的索引
    data = {'index_column': range(len(X_pred))}
    df = pd.DataFrame(data)
    # 将 X_pred 添加为 'mileage' 列
    df['mileage'] = X_pred * n1

    y = filted_df[indicator].values * n2
    best_svr = SVR(kernel='rbf', degree=3, C=best_params['C'], gamma=best_params['gamma'], epsilon=0.11, verbose=1)
    best_svr.fit(X, y)
    y_pred = best_svr.predict(X_pred)
    y_pred = y_pred / n2
    df['flitered' +indicator] = y_pred
    a['svr' + indicator] = best_svr.predict(X1) / n2
    df = df.iloc[:, 1:]
    return df, a

#%%
#定义一个函数：
def indicator_cal(name):
    r0_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'r0_df'+'.csv')
    soh_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/NMC-PV cap 温度修正结果/'+name+'filtered_soh'+'.csv')

   
    soc_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'soc_df'+'.csv')
    rp_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'rp_df'+'.csv')
    cp_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'cp_df'+'.csv')

    jiezhiSOC_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'jiezhiSOC_df'+'.csv')

   
    baoliulicheng =  lvbo(r0_df)
    filted_soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]
    #soh需要做个数据融合,用温度修正数据替换掉svr数据
    # filted_soh_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/NMC-PV cap 温度修正结果/'+name+'filtered_soh.csv')
       
    
    tbl01=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/片段数据/'+name+'_tbl.csv',index_col=False)
    tbl01=tbl01[tbl01['状态']==10]
    start_cycle= tbl01['start_mile'].iloc[0]*(tbl01['end_soc']-tbl01['start_soc']).cumsum().iloc[-1]/(tbl01['start_mile'].iloc[-1]-tbl01['start_mile'].iloc[0])/100
    tbl01['cycle_based on_soc']=(tbl01['end_soc']-tbl01['start_soc']).cumsum()/100+start_cycle
    start_cycle= tbl01['start_mile'].iloc[0]*(tbl01['容量(A·h)']).cumsum().iloc[-1]/(tbl01['start_mile'].iloc[-1]-tbl01['start_mile'].iloc[0])/1500
    tbl01['cycle_based on_cap']=(tbl01['容量(A·h)']).cumsum()/1500+start_cycle  
    tbl01=tbl01[tbl01['片段编号'].isin(soh_df['0'])]
    soh_df['cycle_based on_soc']=tbl01['cycle_based on_soc'].values
    soh_df['cycle_based on_cap']=tbl01['cycle_based on_cap'].values
    
    
    #α1
    soh_df['α1'] = soh_df.iloc[:, 3:98].std(axis=1)
    soh_df['α1_svr'] = soh_df.iloc[:, 98:193].std(axis=1)
    soh_df['fixed_α1'] = soh_df.iloc[:, 193:288].std(axis=1)
    #α2    
    soh_df['α2'] = soh_df.iloc[:, 3:98].min(axis=1)/soh_df.iloc[:, 3:98].mean(axis=1)
    soh_df['α2_svr'] = soh_df.iloc[:, 98:193].min(axis=1)/soh_df.iloc[:, 98:193].mean(axis=1)
    soh_df['fixed_α2'] = soh_df.iloc[:, 193:288].min(axis=1)/soh_df.iloc[:, 193:288].mean(axis=1)
    
    #α3是这样，0.92到不了的占比不多，直接取最大值再取均值影响不大，重点在于最小值：所以直接算就可以了
    EFC_list=[]
    for j in range(98,193):
        EFC_list.append(EFC(soh_df, j, 0.92, 'cycle_based on_cap'))
    soh_df['α3_svr'] = min(EFC_list)/(sum(EFC_list) / len(EFC_list))
    
 
    #α4
    four_list=[]
    for i in jiezhiSOC_df['0']:
        biaogan=jiezhiSOC_df[jiezhiSOC_df['0']==i].iloc[:, 3:98].max(axis=1).values[0]
        soccha=biaogan/100-jiezhiSOC_df[jiezhiSOC_df['0']==i].iloc[:, 3:98]/100
        sohzhi=soh_df[soh_df['0']==i].iloc[:, 98:193]
        soccha.columns = [None] * len(soccha.columns)
        sohzhi.columns = [None] * len(sohzhi.columns)
        fenmu=(soccha*sohzhi).sum(axis=1).values[0]
        fenzi=sohzhi.sum(axis=1).values[0]
        four_list.append(1-fenmu/fenzi)
    
    soh_df['α4']=four_list
    filted_soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]
    α4_soh_df, soh_df = Smooth_curve(soh_df, {'C': 10, 'gamma': 0.035}, filted_soh_df, 98, 10000, 1000,'α4')
    
    #α5
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    x=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'data.csv', index_col=0)
    alpha5_list,I_max_list,min_p_max_list,p_max_list=FW(name,soh_df,0.85,10,soh_df,r0_df,rp_df,cp_df,soc_df,x)       
    soh_df['α5']=alpha5_list
    
    soh_df = soh_df.dropna()
    filted_soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]
    α5_soh_df, soh_df = Smooth_curve(soh_df, {'C': 10, 'gamma': 0.035}, filted_soh_df, 98, 10000, 100,'α5')
    
    soh_df['vin']=name
    b = soh_df.iloc[:, :3].join(soh_df.iloc[:, -12:-1])
    b = soh_df.iloc[:, -1:].join(b)
    b = b.rename(columns={'0': '片段编号', '1': 'mileage', '2': 'temperature'})

    return b
#%%开始计算 
   
names=['LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
        'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
        'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
        'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']


import warnings

# 禁用 SettingWithCopyWarning
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)


for name in tqdm(names):    
    
  # if name+'α1_svr.png' not in os.listdir('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Indicators/'):
     
    a = indicator_cal(name)
    a.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'α_df'+'.csv',index=False)
    
    # plt.scatter(a['mileage'],a['α1_svr'],marker='o')
    plt.scatter(a['mileage'],a['fixed_α1'],marker='o')
    plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'α1_svr.png', format='png', bbox_inches='tight')
    plt.close()
    
    # plt.scatter(a['mileage'],a['α2_svr'],marker='o')
    plt.scatter(a['mileage'],a['fixed_α2'],marker='o')
    plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'α2_svr.png', format='png', bbox_inches='tight')
    plt.close()
    
    plt.scatter(a['mileage'],a['α3_svr'],marker='o')
    plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'α3_svr.png', format='png', bbox_inches='tight')
    plt.close()
    
    plt.scatter(a['mileage'],a['svrα4'],marker='o')
    plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'svrα4.png', format='png', bbox_inches='tight')
    plt.close()
    
    plt.scatter(a['mileage'],a['svrα5'],marker='o')
    plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'svrα5.png', format='png', bbox_inches='tight')
    plt.close()   



#%%全部读取，数据处理，拼接，简单出图

indicator_list=pd.DataFrame()

for name in tqdm(names):    
    a=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'α_df'+'.csv')
    indicator_list = pd.concat([indicator_list, a], axis=0)
    

indicator_list_orgin = indicator_list.copy()  


# #每个车只保留最高点
# for vin in names:
#     # 筛选出特定'vin'值的行
#     vin_rows = indicator_list[indicator_list['vin'] == vin]
    
#     # 找到'α1_svr'最大的行
#     max_row = vin_rows.loc[vin_rows['α1_svr'].idxmax()]
    
#     # 提取'mileage'值和'vin'值
#     mileage_value = max_row['mileage']
    
#     # 将结果添加到结果DataFrame中
#     indicator_list.loc[indicator_list['vin'] == vin,'threshold']=mileage_value-5000
#     # if vin == 'LB378Y4W9JA176518':
#     #    indicator_list.loc[indicator_list['vin'] == vin,'threshold']=5000000

# indicator_list=indicator_list[indicator_list['mileage']<=indicator_list['threshold']]




#%%new_α3的计算
def new_EFC(soh, j, threshold,k):
    # 计算与阈值 threshold 的差值
    diff = soh.iloc[:, j] - threshold  
    # 找到正向和负向差值最近的行的索引
    try: 
        positive_idx = diff[diff >= 0].idxmin()
        negative_idx = diff[diff < 0].idxmax()  
        # 获取 j 列和 K 列的两个值
        positive_j_values = soh.iloc[positive_idx, j]
        positive_k_values = soh.loc[positive_idx, k]
        negative_j_values = soh.iloc[negative_idx, j]
        negative_k_values = soh.loc[negative_idx, k]  
        # 给定两个点的坐标
        x_values = [positive_j_values, negative_j_values]
        y_values = [positive_k_values, negative_k_values]
    
        # 要插值的目标值
        target_x = threshold
        # 进行一次线性拟合
        coefficients = np.polyfit(x_values, y_values, 1)
    
        # 提取拟合的斜率和截距
        slope = coefficients[0]
        intercept = coefficients[1]
    
        # 计算目标 x 对应的 efc 值
        efc = slope * target_x + intercept
    except:
        idx = (soh.iloc[:, j] - threshold).abs().idxmin()
        efc = soh.loc[idx, k]
    
    return efc

#α3是这样，最快到0.85的比上均值到0.85的

# little=[]
for name in tqdm(names):    
    soh_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/NMC-PV cap 温度修正结果/'+name+'filtered_soh'+'.csv')
    # mean_value = soh_df.iloc[:, 98:193].mean(axis=1)
    mean_value = soh_df.iloc[:, 193:288].mean(axis=1)

    soh_df['soh_mean'] = mean_value
    # little.append(soh_df['soh_mean'].min())
    
    tbl01=pd.read_csv('H:/DBD4 Record 备份/DBD4 trying/片段数据/'+name+'_tbl.csv',index_col=False)
    tbl01=tbl01[tbl01['状态']==10]
    start_cycle= tbl01['start_mile'].iloc[0]*(tbl01['end_soc']-tbl01['start_soc']).cumsum().iloc[-1]/(tbl01['start_mile'].iloc[-1]-tbl01['start_mile'].iloc[0])/100
    tbl01['cycle_based on_soc']=(tbl01['end_soc']-tbl01['start_soc']).cumsum()/100+start_cycle
    start_cycle= tbl01['start_mile'].iloc[0]*(tbl01['容量(A·h)']).cumsum().iloc[-1]/(tbl01['start_mile'].iloc[-1]-tbl01['start_mile'].iloc[0])/1500
    tbl01['cycle_based on_cap']=(tbl01['容量(A·h)']).cumsum()/1500+start_cycle  
    tbl01=tbl01[tbl01['片段编号'].isin(soh_df['0'])]
    soh_df['cycle_based on_soc']=tbl01['cycle_based on_soc'].values
    soh_df['cycle_based on_cap']=tbl01['cycle_based on_cap'].values
    
    EFC_list=[]
    for j in range(193,289):

        efc_cell=new_EFC(soh_df, j, 0.85, 'cycle_based on_cap')
        EFC_list.append(efc_cell)
        
    if name == 'LB378Y4W9JA176518':
        EFC_list=[]
        for j in range(193,289):
            if j not in range(227, 234):
                efc_cell=new_EFC(soh_df, j, 0.85, 'cycle_based on_cap')
                EFC_list.append(efc_cell)
    
    #这两辆车温度修正有些问题，用SVR结果
    if (name=='LB378Y4W4JA177348')|(name=='LB378Y4W8JA175280'):
        mean_value = soh_df.iloc[:, 98:193].mean(axis=1)
        soh_df['soh_mean'] = mean_value
        EFC_list=[]
        for j in range(98,194):
                efc_cell=new_EFC(soh_df, j, 0.85, 'cycle_based on_cap')
                EFC_list.append(efc_cell)
        
    
    
    indicator_list.loc[indicator_list['vin']==name, 'new_α3_fixed'] = min(EFC_list)/EFC_list[-1]
    indicator_list.loc[indicator_list['vin']==name, 'mile_treshold'] = soh_df[soh_df['cycle_based on_cap']>=min(EFC_list)]['1'].iloc[0]
    # # 对 EFC_list 进行排序
    # sorted_efc_list = sorted(EFC_list)
    
    # # 默认使用最小的值
    # selected_efc = sorted_efc_list[0]
    
    # # 如果 min(EFC_list)/EFC_list[-1] 小于 0.65，改为使用第二小的值
    # if sorted_efc_list[0] / EFC_list[-1] < 0.65:
    #     if len(sorted_efc_list) > 1:
    #         selected_efc = sorted_efc_list[1]
    #     else:
    #         # 如果列表中只有一个元素，仍然使用它
    #         selected_efc = sorted_efc_list[0]
    
    # # 更新指标列表的 'new_α3_fixed' 列
    # indicator_list.loc[indicator_list['vin'] == name, 'new_α3_fixed'] = selected_efc / EFC_list[-1]
    
    # # 更新 'mile_treshold'，查找第一次满足 cycle_based on_cap 大于等于 selected_efc 的值
    # mile_threshold_value = soh_df[soh_df['cycle_based on_cap'] >= selected_efc]['1'].iloc[0]
    # indicator_list.loc[indicator_list['vin'] == name, 'mile_treshold'] = mile_threshold_value

# 'LB378Y4W0JA174527','LB378Y4W9JA173988','LB378Y4W8JA175280',三辆车，其中前两辆车温度修正效果不佳，后一辆整体很差，把前两个指标结果替换为SVR结果
indicator_list = indicator_list.reset_index(drop=True)
indicator_list.loc[indicator_list['vin']=='LB378Y4W0JA174527','fixed_α1']=indicator_list['α1_svr']
indicator_list.loc[indicator_list['vin']=='LB378Y4W0JA174527','fixed_α2']=indicator_list['α2_svr']
indicator_list.loc[indicator_list['vin']=='LB378Y4W9JA173988','fixed_α1']=indicator_list['α1_svr']
indicator_list.loc[indicator_list['vin']=='LB378Y4W9JA173988','fixed_α2']=indicator_list['α2_svr']
indicator_list.loc[indicator_list['vin']=='LB378Y4W8JA175280','fixed_α1']=indicator_list['α1_svr']
indicator_list.loc[indicator_list['vin']=='LB378Y4W8JA175280','fixed_α2']=indicator_list['α2_svr']




indicator_list.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'new_α3'+'.csv',index=False)

font=14
font_size=14
plt.rc('font', family='Arial')
font_size = 16
font = {'size': font_size, 'family': 'Times new Roman'}
VS=[chr(ord('a') + i) for i in range(19)]

alpha3_svr_values = indicator_list['new_α3_fixed'].unique().tolist()
# 创建指定大小的图表
plt.figure(figsize=(12, 6))  # 设置图形大小
plt.bar(VS, alpha3_svr_values, color='royalblue')  # 绘制柱状图
plt.xlabel('Vehicles', fontdict=font)  # 设置横坐标标签
plt.ylabel('α$_{\mathrm{EFC}}$', fontdict=font)  # 设置纵坐标标签
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(0, 1)
# 显示柱状图
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()


plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'α3.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'α3.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'α3.pdf', format='pdf', bbox_inches='tight')
plt.close()

#%% 最终的α_Overall capacity utilization
# Set font properties
# 设置字体属性
plt.rc('font', family='Arial')
font_size = 16
font = {'size': font_size, 'family': 'Arial'}
plt.rcParams['font.weight'] = 'bold'

indicator_list=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'new_α3'+'.csv')
indicator_list=indicator_list[indicator_list['mileage']<=indicator_list['mile_treshold']]
for name in tqdm(names):  
    chakan=indicator_list[indicator_list['vin']==name]
    indicator_list.loc[indicator_list['vin']==name, 'α'] = chakan['fixed_α2'].mean()*chakan['new_α3_fixed'].mean()

indicator_list.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'final_α'+'.csv',index=False)    
alpha_values = indicator_list['α'].unique().tolist()

# 创建指定大小的图表
plt.figure(figsize=(16, 8))  # 设置图形大小
# 计算α的平均值
average_utilization = np.mean(alpha_values)

# 添加平均值虚线
plt.axhline(average_utilization, color='red', linestyle='--', linewidth=3, label=f'Average Utilization Rate: {average_utilization:.3f}')



plt.bar(VS, alpha_values, color='royalblue')  # 绘制柱状图
plt.xlabel('Vehicles', fontdict=font)  # 设置横坐标标签
plt.ylabel('α$_{\mathrm{overall\ utilization\ rate}}$', fontdict=font)

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(0, 1)
plt.legend(fontsize=font_size)  # 显示图例
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'α.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'α.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'α.pdf', format='pdf', bbox_inches='tight')
plt.close()   
