# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:32:06 2024

@author: 13121

数据瘦身，节省空间，方便后续的计算
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun'] # 修改字体为宋体
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'
from tqdm import tqdm
import os

table=pd.DataFrame()
for name in os.listdir('G:/DBD4数据/片段数据/'):
    tbl=pd.read_csv("G:/DBD4数据/片段数据/"+name)
    # tbl['vin']=name
    table=pd.concat([table,tbl],axis=0)

table01=table[table['状态']==10]
table01['start_time']= pd.to_datetime(table01['start_time'],format = '%Y-%m-%d %H:%M:%S')
table01['end_time']= pd.to_datetime(table01['end_time'],format = '%Y-%m-%d %H:%M:%S')
table01['start_mile']= table01['start_mile']*0.1
table01['end_mile']= table01['end_mile']*0.1
table01['时间跨度']=table01['end_time']-table01['start_time']
table01['时间跨度']=table01['时间跨度'].dt.seconds
table01['每帧时间']=(table01['时间跨度']+10)/table01['帧数']
table01=table01[(table01['每帧时间']>=10)&(table01['每帧时间']<=10.03)]
table01['充入soc']=table01['end_soc']-table01['start_soc']  
table01=table01[table01['充入soc']>20]
table01['计算容量(A·h)']=100*table01['容量(A·h)']/table01['充入soc']
table01['平均电流']=table01['平均电流']*-0.1
table01['计算容量(A·h)']=table01['计算容量(A·h)']*0.1 

table_ana=table01[(table01['平均电流']<18)&(table01['平均电流']>15)&(table01['end_soc']>=99)&(table01['start_soc']<=25)]


vin=[]
start=[]
end=[]
length=[]
cap1=[]
cap2=[]
for i in set(table_ana['vin']):
    vin.append(i)
    chakan=table_ana[table_ana['vin']==i]
    length.append(len(chakan))
    start.append(chakan['start_mile'].iloc[0])
    end.append(chakan['start_mile'].iloc[-1])
    cap1.append(chakan['计算容量(A·h)'].iloc[0])
    cap2.append(chakan['计算容量(A·h)'].iloc[-1])    
table_zong=pd.DataFrame({'vin':vin,'片段数':length,'起始里程':start,'结束里程':end,'起始SOH':cap1,'结束SOH':cap2})

table_zong=table_zong[table_zong['起始里程']<=100000]
table_zong=table_zong[table_zong['片段数']>100]
table_zong['衰退速率']=(table_zong['起始SOH']-table_zong['结束SOH'])/(table_zong['结束里程']-table_zong['起始里程'])
table_zong=table_zong[table_zong['vin']!='LB378Y4W6JA176637']


#先找出合适的全部车辆，然后把合适的片段输出成tbl04
names=table_zong['vin']
# names=['LB378Y4W3JA179379']
for name1 in tqdm(names):
    name=name1[0:17]
    table=pd.read_csv('H:/DBD4数据/片段数据/'+name+'_tbl.csv')
    
    table01=table[table['状态']==10]
    table01['start_time']= pd.to_datetime(table01['start_time'],format = '%Y-%m-%d %H:%M:%S')
    table01['end_time']= pd.to_datetime(table01['end_time'],format = '%Y-%m-%d %H:%M:%S')
    table01['start_mile']= table01['start_mile']*0.1
    table01['end_mile']= table01['end_mile']*0.1
    table01['时间跨度']=table01['end_time']-table01['start_time']
    table01['时间跨度']=table01['时间跨度'].dt.seconds
    table01['每帧时间']=(table01['时间跨度']+10)/table01['帧数']
    table01=table01[(table01['每帧时间']>=10)&(table01['每帧时间']<=10.03)]
    table01['充入soc']=table01['end_soc']-table01['start_soc']  
    table01=table01[table01['充入soc']>20]
    table01['计算容量(A·h)']=100*table01['容量(A·h)']/table01['充入soc']
    table01['平均电流']=table01['平均电流']*-0.1
    table01['计算容量(A·h)']=table01['计算容量(A·h)']*0.1 
    tbl01=table01
    chakan=tbl01[(tbl01['平均电流']<18)&(tbl01['平均电流']>15)&(tbl01['end_soc']>=99)&(tbl01['start_soc']<=60)&(tbl01['帧数']>=100)]
    chakan=chakan[['vin','片段编号','状态','start_soc','end_soc','start_mile','容量(A·h)']]
    tbl=pd.read_csv('H:/DBD4数据/片段数据/'+name+'_tbl.csv')
    tbl=tbl[(tbl['片段编号'].isin(chakan['片段编号']))|(tbl['状态']==30)|(tbl['状态']==40)]
    tbl['状态差1']=tbl['状态'].diff(1)
    tbl['状态差2']=tbl['状态'].diff(2)
    tbl['状态差3']=tbl['状态'].diff(-1)
    tbl['状态差4']=tbl['状态'].diff(-2)
    tbl04=tbl[(tbl['状态']==10)&(tbl['状态差2']==-20)&(tbl['状态差1']==-30)&(tbl['状态差3']==-30)&(tbl['状态差4']==-30)]
    bianhao=pd.DataFrame({'起始':tbl04['片段编号'].values-2,'截止':tbl04['片段编号'].values+2})
    bianhaojihe=[]
    for i in tqdm(range(len(bianhao))):
        add=list(range(int(bianhao['起始'].iloc[i]),int(bianhao['截止'].iloc[i]+1)))
        bianhaojihe=bianhaojihe+add
    data=pd.read_csv('G:/DBD4数据/处理后数据/'+name+'_chulihou.csv')
    data=data[data['number'].isin(bianhaojihe)]
    data['里程']=data['里程']*0.1
    data['单帧容量变化(A·h)']=data['单帧容量变化(A·h)']*0.1
    data['总电压']=data['总电压']*0.1
    data['总电流']=data['总电流']*0.1
    data['Tem']=data['最高温度值']/2+data['最低温度值']/2
    data['time']=pd.to_datetime(data['time'],format = '%Y-%m-%d %H:%M:%S')

    del tbl,chakan,bianhao,bianhaojihe,add
    
    data.to_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'data.csv')
    tbl04.to_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'tbl04.csv')