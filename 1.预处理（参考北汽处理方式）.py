# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:05:31 2024

@author: 13121

E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/暂时不用的程序，只做参考/New vehicles.py
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def unfold(a):
    b=(a.split('_'))
    b=pd.to_numeric(b,'coerce')
    # b=b[b>0]
    b=list(b)
    return b

def cell_v(length,data_c):
    b=pd.DataFrame()
    for i in tqdm(range(int(len(data_c)/length+1))):
      try:
        chakan=data_c.iloc[i*length:i*length+length]
        a1=list(chakan['单体电压'].apply(lambda x:unfold(x)))
        a1=pd.DataFrame(a1)
        b=pd.concat([b,a1])
      except:
        return i
    return b

#片段编号提取原数据 
def data_extraction(dataC,td):
    a=dataC[dataC['number']==td]
    return a  

#片段划分相关
def add_line(table_list,td,start_time,end_time,state,start_mile,end_mile,v,I,number,start_soc,end_soc,cap):
            line = [td,start_time,end_time,state,start_mile,end_mile,v,I,number,start_soc,end_soc,cap]
            table_list.append(line)
            return table_list

def state_sta(d):
    table_list = []
    for i in tqdm(set(d['number'])):
        a=d[d['number']==i]
        td=a['number'].iloc[0]      
        start_time=a['time'].iloc[0]
        end_time=a['time'].iloc[len(a)-1]
        state=a['状态'].iloc[0]
        start_mile=a['里程'].iloc[0]
        end_mile=a['里程'].iloc[len(a)-1]
        v=a['车速'].mean() 
        I=a['总电流'].mean() 
        number=len(a)
        start_soc=a['SOC'].iloc[0]
        end_soc=a['SOC'].iloc[-1]
        cap=a['单帧容量变化(A·h)'].sum()
        table_list = add_line(table_list,td,start_time,end_time,state,start_mile,end_mile,v,I,number,start_soc,end_soc,cap)
    table_df = pd.DataFrame(table_list,columns = ["片段编号","start_time","end_time","状态","start_mile","end_mile","平均车速","平均电流","帧数","start_soc","end_soc","容量(A·h)"])
    return  table_df  

#温度：平均最高温，平均最低温，平均温度,平均温度差,平均压差
def tem(data,tbl):
     for i in range(len(tbl)):
        t=data_extraction(data,tbl['片段编号'].iloc[i])
        t1=t['最高温度值'].mean()
        t2=t['最低温度值'].mean()
        t3=(t1+t2)/2
        t4=t1-t2
        t5=(t['电池单体电压最高值']-t['电池单体电压最低值']).mean()
        tbl.loc[(tbl['片段编号']==tbl['片段编号'].iloc[i]),'temp_max']=t1
        tbl.loc[(tbl['片段编号']==tbl['片段编号'].iloc[i]),'temp_min']=t2
        tbl.loc[(tbl['片段编号']==tbl['片段编号'].iloc[i]),'temp_mean']=t3
        tbl.loc[(tbl['片段编号']==tbl['片段编号'].iloc[i]),'平均温差']=t4
        tbl.loc[(tbl['片段编号']==tbl['片段编号'].iloc[i]),'平均压差']=t5  
     return tbl

def yuchuli(path,dantinumber,shijianjiange):
    data=pd.read_csv(path)
    data=data[['vin', '数据采集时间', '车辆状态', '充电状态', '车速', '里程', '总电压', '总电流', 'SOC',
       '最高电压电池单体代号', '电池单体电压最高值', '最低电压电池单体代号', '电池单体电压最低值', '最高温度探针号',
       '最高温度值', '最低温度探针号', '最低温度值', '单体电池总数', '单体电池包总数', '单体电池电压值', '单体电池温度值','经度', '纬度']]
    data.dropna(how='any', subset=['数据采集时间', '车辆状态', '充电状态','里程', '总电压', '总电流', 'SOC'],axis=0, inplace=True)
    data=data[data['SOC']>0]
    data=data[data['总电压']>0]
    data['len_单体电压列表'] = data['单体电池电压值'].apply(lambda x: len(str(x)))
    data = data[(data['len_单体电压列表']==((dantinumber*5)+1))|(data['len_单体电压列表']==((dantinumber*5)+11))]
    data.drop(columns = ['len_单体电压列表'],inplace = True)
    data['单体电压'] = data['单体电池电压值'].apply(lambda x: str(x)[2:])
    data.drop(columns = ['单体电池电压值'],inplace = True)
    data.drop_duplicates('数据采集时间',inplace = True)#去除时间重复帧
    data.dropna(how='any', subset=['数据采集时间', '车辆状态', '充电状态','里程', '总电压', '总电流', 'SOC'],axis=0, inplace=True)
    data.sort_values(by ='数据采集时间',inplace = True)
    data=data.reset_index(drop=True)
    pd.set_option('mode.chained_assignment', None)
    cell_voltage=cell_v(500000,data)
#这一块原来的91没改掉，会不会数据出问题？
    cell_voltage=cell_voltage.iloc[:, :dantinumber]
    new_col = []
    for k in range(1,dantinumber+1):
        new_col.append('单体电压'+str(k))
    cell_voltage.columns = new_col
    cell_voltage=cell_voltage.reset_index(drop=True)
    data = pd.concat([data,cell_voltage],axis = 1)
    del cell_voltage
    data.loc[data['充电状态']==1,'状态']=10
    data.loc[(data['车速']==0)&(data['车辆状态']==2)&(abs(data['总电流'])<3),'状态']=40
    data.loc[data['车辆状态']==1,'状态']=30
    data=data[data['状态']>0]
    data['数据采集时间'] = data['数据采集时间'].astype('str')
    data['time'] = pd.to_datetime(data['数据采集时间'],format = '%Y%m%d%H%M%S')
    data['dert_T']=data['time'].diff(1).dt.seconds    
    data['dert_状态']=data['状态'].diff() 
    data_number=data[(data['dert_T']>300)|(data['dert_状态']!=0)]
    data_number=data_number[data_number['状态']>0]
    data_number['number']=np.linspace(1,len(data_number),len(data_number))
    for i in tqdm(data_number['number']):
         times=data_number[data_number['number']==i]['time'].iloc[0]
         data.loc[data['time']>=times,'number']=i
    data.drop(columns = ['dert_T'],inplace = True)
    data.drop(columns = ['dert_状态'],inplace = True)    
    data.sort_values(by ='数据采集时间',inplace = True)
    data.reset_index(inplace = True,drop = True)
    data['time'] = pd.to_datetime(data['数据采集时间'],format = '%Y%m%d%H%M%S')
    data['dert_T']=data['time'].diff(1).dt.seconds   
    data.loc[data['dert_T']<40,'单帧容量变化(A·h)']=-data['总电流']*data['dert_T']/3600
    data.loc[data['单帧容量变化(A·h)'].isnull(),'单帧容量变化(A·h)']=-data['总电流']*shijianjiange/3600
    return data

#%%
troubles=[]
for m in tqdm(os.listdir('G:/数据/吉利原始数据/')):
    if m+'_chulihou.csv' not in os.listdir('G:/DBD4数据/处理后数据/'):
        for n in os.listdir('G:/数据/吉利原始数据/'+m+'/'):
            if '.csv' in n:   
              try:
                path='H:/数据/吉利原始数据/'+m+'/'+n
                data=yuchuli(path,95,10)
                data.to_csv('G:/DBD4数据/处理后数据/'+m+'_chulihou.csv',index=False)
                tbl=state_sta(data) 
                tbl=tem(data,tbl) 
                tbl.to_csv('G:/DBD4数据/片段数据/'+m+'_tbl.csv',index=False)
              except:
                troubles.append(m)
                continue
            
#读取前20行，观察

chakan=pd.read_csv('G:/DBD4数据/处理后数据/LB378Y4W0JA172387_chulihou.csv',nrows=10)
