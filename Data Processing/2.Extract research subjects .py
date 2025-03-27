# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:32:06 2024

@author: Rico

Extracting data to reduce storage usage and enable efficient downstream computation
"""

import pandas as pd
from tqdm import tqdm
import os

table=pd.DataFrame()
for name in os.listdir('Processed data storage file_segments/'):
    tbl=pd.read_csv("Processed data storage file_segments/"+name)
    # tbl['vin']=name
    table=pd.concat([table,tbl],axis=0)

table01=table[table['State']==10]
table01['start_time']= pd.to_datetime(table01['start_time'],format = '%Y-%m-%d %H:%M:%S')
table01['end_time']= pd.to_datetime(table01['end_time'],format = '%Y-%m-%d %H:%M:%S')
table01['start_mile']= table01['start_mile']*0.1
table01['end_mile']= table01['end_mile']*0.1
table01['time_span']=table01['end_time']-table01['start_time']
table01['time_span']=table01['time_span'].dt.seconds
table01['frame_time']=(table01['time_span']+10)/table01['frame_numbers']
table01=table01[(table01['frame_time']>=10)&(table01['frame_time']<=10.03)]
table01['charged_soc']=table01['end_soc']-table01['start_soc']  
table01=table01[table01['charged_soc']>20]
table01['Cal_Cap(A·h)']=100*table01['Cap(A·h)']/table01['charged_soc']
table01['Ave_Current']=table01['Ave_Current']*-0.1
table01['Cal_Cap(A·h)']=table01['Cal_Cap(A·h)']*0.1 

table_ana=table01[(table01['Ave_Current']<18)&(table01['Ave_Current']>15)&(table01['end_soc']>=99)&(table01['start_soc']<=25)]


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
    cap1.append(chakan['Cal_Cap(A·h)'].iloc[0])
    cap2.append(chakan['Cal_Cap(A·h)'].iloc[-1])    
table_zong=pd.DataFrame({'vin':vin,'fragment_numbers':length,'start_mileage':start,'end_mileage':end,'start_SOH':cap1,'end_SOH':cap2})

table_zong=table_zong[table_zong['start_mileage']<=100000]
table_zong=table_zong[table_zong['fragment_numbers']>100]
table_zong['degradation_rate']=(table_zong['start_SOH']-table_zong['end_SOH'])/(table_zong['end_mileage']-table_zong['start_mileage'])


#先找出合适的全部车辆，然后把合适的片段输出成tbl04
names=table_zong['vin']
for name in tqdm(names):
    table=pd.read_csv('Processed data storage file_segments/'+name+'_tbl.csv')
    
    table01=table[table['State']==10]
    table01['start_time']= pd.to_datetime(table01['start_time'],format = '%Y-%m-%d %H:%M:%S')
    table01['end_time']= pd.to_datetime(table01['end_time'],format = '%Y-%m-%d %H:%M:%S')
    table01['start_mile']= table01['start_mile']*0.1
    table01['end_mile']= table01['end_mile']*0.1
    table01['time_span']=table01['end_time']-table01['start_time']
    table01['time_span']=table01['time_span'].dt.seconds
    table01['frame_time']=(table01['time_span']+10)/table01['frame_numbers']
    table01=table01[(table01['frame_time']>=10)&(table01['frame_time']<=10.03)]
    table01['charged_soc']=table01['end_soc']-table01['start_soc']  
    table01=table01[table01['charged_soc']>20]
    table01['Cal_Cap(A·h)']=100*table01['Cap(A·h)']/table01['charged_soc']
    table01['Ave_Current']=table01['Ave_Current']*-0.1
    table01['Cal_Cap(A·h)']=table01['Cal_Cap(A·h)']*0.1 
    tbl01=table01
    chakan=tbl01[(tbl01['Ave_Current']<18)&(tbl01['Ave_Current']>15)&(tbl01['end_soc']>=99)&(tbl01['start_soc']<=60)&(tbl01['frame_numbers']>=100)]
    chakan=chakan[['vin','segment_id','State','start_soc','end_soc','start_mile','Cap(A·h)']]
    tbl=pd.read_csv('Processed data storage file_segments/'+name+'_tbl.csv')
    tbl=tbl[(tbl['segment_id'].isin(chakan['segment_id']))|(tbl['State']==30)|(tbl['State']==40)]
    tbl['State_diff1']=tbl['State'].diff(1)
    tbl['State_diff2']=tbl['State'].diff(2)
    tbl['State_diff3']=tbl['State'].diff(-1)
    tbl['State_diff4']=tbl['State'].diff(-2)
    tbl04=tbl[(tbl['State']==10)&(tbl['State_diff2']==-20)&(tbl['State_diff1']==-30)&(tbl['State_diff3']==-30)&(tbl['State_diff4']==-30)]
    bianhao=pd.DataFrame({'Start':tbl04['segment_id'].values-2,'End':tbl04['segment_id'].values+2})
    bianhaojihe=[]
    for i in tqdm(range(len(bianhao))):
        add=list(range(int(bianhao['Start'].iloc[i]),int(bianhao['End'].iloc[i]+1)))
        bianhaojihe=bianhaojihe+add
    data=pd.read_csv('Processed data storage file_segments/'+name+'_chulihou.csv')
    data=data[data['number'].isin(bianhaojihe)]
    data['Mileage']=data['Mileage']*0.1
    data['frame_cap_diff']=data['frame_cap_diff']*0.1
    data['TotalVoltage']=data['TotalVoltage']*0.1
    data['TotalCurrent']=data['TotalCurrent']*0.1
    data['Tem']=data['MaxTemp']/2+data['MinTemp']/2
    data['time']=pd.to_datetime(data['time'],format = '%Y-%m-%d %H:%M:%S')

    del tbl,chakan,bianhao,bianhaojihe,add
    
    data.to_csv('Extracted data storage file_segments/'+name+'data.csv')
    tbl04.to_csv('Extracted data storage file_segments_tbl/'+name+'tbl04.csv')