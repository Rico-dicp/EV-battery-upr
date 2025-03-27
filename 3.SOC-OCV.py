# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:56:24 2024

@author: 13121
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial'] # 修改字体为宋体
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'


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