# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:05:31 2024

@author: Rico

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle


def unfold(a):
    b=(a.split('_'))
    b=pd.to_numeric(b,'coerce')
    b=list(b)
    return b

def cell_v(length,data_c):
    b=pd.DataFrame()
    for i in tqdm(range(int(len(data_c)/length+1))):
      try:
        chakan=data_c.iloc[i*length:i*length+length]
        a1=list(chakan['CellVoltages'].apply(lambda x:unfold(x)))
        a1=pd.DataFrame(a1)
        b=pd.concat([b,a1])
      except:
        return i
    return b

#Extract original data using segment number
def data_extraction(dataC,td):
    a=dataC[dataC['number']==td]
    return a  

#Segment division
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
        state=a['State'].iloc[0]
        start_mile=a['Mileage'].iloc[0]
        end_mile=a['Mileage'].iloc[len(a)-1]
        v=a['Speed'].mean() 
        I=a['TotalCurrent'].mean() 
        number=len(a)
        start_soc=a['SOC'].iloc[0]
        end_soc=a['SOC'].iloc[-1]
        cap=a['frame_cap_diff'].sum()
        table_list = add_line(table_list,td,start_time,end_time,state,start_mile,end_mile,v,I,number,start_soc,end_soc,cap)
    table_df = pd.DataFrame(table_list,columns = ["segment_id","start_time","end_time","State","start_mile","end_mile","Ave_Speed","Ave_Current","frame_numbers","start_soc","end_soc","Cap(A·h)"])
    return  table_df  

def tem(data,tbl):
     for i in range(len(tbl)):
        t=data_extraction(data,tbl['segment_id'].iloc[i])
        t1=t['MaxTemp'].mean()
        t2=t['MinTemp'].mean()
        t3=(t1+t2)/2
        t4=t1-t2
        t5=(t['MaxCellVoltage']-t['MinVoltageCellID']).mean()
        tbl.loc[(tbl['segment_id']==tbl['segment_id'].iloc[i]),'temp_max']=t1
        tbl.loc[(tbl['segment_id']==tbl['segment_id'].iloc[i]),'temp_min']=t2
        tbl.loc[(tbl['segment_id']==tbl['segment_id'].iloc[i]),'temp_mean']=t3
        tbl.loc[(tbl['segment_id']==tbl['segment_id'].iloc[i]),'avg_temp_diff']=t4
        tbl.loc[(tbl['segment_id']==tbl['segment_id'].iloc[i]),'avg_voltage_diff']=t5  
     return tbl

def yuchuli(path,dantinumber,shijianjiange):
    # data=pd.read_csv(path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    data=data[['vin', 'Timestamp', 'VehicleStatus', 'ChargingStatus', 'Speed', 'Mileage', 'TotalVoltage', 'TotalCurrent', 'SOC',
       'MaxVoltageCellID', 'MaxCellVoltage', 'MinVoltageCellID', 'MinCellVoltage', 'MaxTempProbeID',
       'MaxTemp', 'MinTempProbeID', 'MinTemp', 'CellVoltages', 'CellTemperatures']]
    data.dropna(how='any', subset=['Timestamp', 'VehicleStatus', 'ChargingStatus','Mileage', 'TotalVoltage', 'TotalVoltage', 'SOC'],axis=0, inplace=True)
    data=data[data['SOC']>0]
    data=data[data['TotalVoltage']>0]
    data['len_CellVoltages_list'] = data['CellVoltages'].apply(lambda x: len(str(x)))
    data = data[(data['len_CellVoltages_list']==((dantinumber*5)+1))|(data['len_CellVoltages_list']==((dantinumber*5)+11))]
    data.drop(columns = ['len_CellVoltages_list'],inplace = True)
    data['CellVoltages'] = data['CellVoltages'].apply(lambda x: str(x)[2:])
    data.drop(columns = ['CellVoltages'],inplace = True)
    data.drop_duplicates('Timestamp',inplace = True)
    data.dropna(how='any', subset=['Timestamp', 'VehicleStatus', 'ChargingStatus','Mileage', 'TotalVoltage', 'TotalCurrent', 'SOC'],axis=0, inplace=True)
    data.sort_values(by ='Timestamp',inplace = True)
    data=data.reset_index(drop=True)
    pd.set_option('mode.chained_assignment', None)
    cell_voltage=cell_v(500000,data)

    cell_voltage=cell_voltage.iloc[:, :dantinumber]
    new_col = []
    for k in range(1,dantinumber+1):
        new_col.append('Cell_V'+str(k))
    cell_voltage.columns = new_col
    cell_voltage=cell_voltage.reset_index(drop=True)
    data = pd.concat([data,cell_voltage],axis = 1)
    del cell_voltage
    data.loc[data['ChargingStatus']==1,'State']=10
    data.loc[(data['Speed']==0)&(data['VehicleStatus']==2)&(abs(data['TotalVoltage'])<3),'State']=40
    data.loc[data['VehicleStatus']==1,'State']=30
    data=data[data['State']>0]
    data['Timestamp'] = data['Timestamp'].astype('str')
    data['time'] = pd.to_datetime(data['Timestamp'],format = '%Y%m%d%H%M%S')
    data['dert_T']=data['time'].diff(1).dt.seconds    
    data['dert_State']=data['State'].diff() 
    data_number=data[(data['dert_T']>300)|(data['dert_State']!=0)]
    data_number=data_number[data_number['State']>0]
    data_number['number']=np.linspace(1,len(data_number),len(data_number))
    for i in tqdm(data_number['number']):
         times=data_number[data_number['number']==i]['time'].iloc[0]
         data.loc[data['time']>=times,'number']=i
    data.drop(columns = ['dert_T'],inplace = True)
    data.drop(columns = ['dert_State'],inplace = True)    
    data.sort_values(by ='Timestamp',inplace = True)
    data.reset_index(inplace = True,drop = True)
    data['time'] = pd.to_datetime(data['Timestamp'],format = '%Y%m%d%H%M%S')
    data['dert_T']=data['time'].diff(1).dt.seconds   
    data.loc[data['dert_T']<40,'frame_cap_diff']=-data['TotalCurrent']*data['dert_T']/3600
    data.loc[data['frame_cap_diff'].isnull(),'frame_cap_diff']=-data['TotalCurrent']*shijianjiange/3600
    return data

#%%
troubles = []
for m in tqdm(os.listdir('XXX/Data_NMC_81/')):
    try:
        path = f'XXX/Data_NMC_81/{m}'
        data = yuchuli(path, 95, 10)
        data.to_csv(f'Processed data storage file/{m}_chulihou.csv', index=False)

        tbl = state_sta(data)
        tbl = tem(data, tbl)
        tbl.to_csv(f'Processed data storage file_segments/{m}_tbl.csv', index=False)

    except Exception as e:
        troubles.append((m, str(e)))
        continue

