# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:56:24 2024

@author: RICO
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

name='vehicle69'
data=pd.read_csv('Extracted data storage file/'+name+'data.csv', index_col=0)
tbl04=pd.read_csv('Extracted data storage file_segments_tbl/'+name+'data.csv', index_col=0)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def r(data_identification,m):
    i=data_identification[data_identification['TotalCurrent'].diff(1)>9].index.values[-1]
    chakan=data_identification.loc[i-1:i]
    r=-(chakan['Cell_V'+str(m)].diff(1)/chakan['TotalCurrent'].diff(1)).mean()*0.001
    i=data_identification[data_identification['TotalCurrent'].diff(1)<-10].index.values[0]
    chakan=data_identification.loc[i-1:i]
    r1=-(chakan['Cell_V'+str(m)].diff(1)/chakan['TotalCurrent'].diff(1)).mean()*0.001

    jihua=data_identification['Cell_V'+str(m)].iloc[-1]*0.001-data_identification[data_identification['State']==10]['Cell_V'+str(m)].iloc[-1]*0.001-r*data_identification[data_identification['充电State']==1]['TotalCurrent'].iloc[-1]
    return r,r1,jihua


def r1(data_identification,m):
    i=data_identification[data_identification['TotalCurrent'].diff(1)>9].index.values[-1]
    chakan=data_identification.loc[i-1:i]
    dert_u=chakan['Cell_V'+str(m)].iloc[0]-data_identification['Cell_V'+str(m)].iloc[-1]
    dert_i=chakan['TotalCurrent'].diff(1).mean()
    r=dert_u/dert_i*0.001
    dert_t=(data_identification['time'].iloc[-1]-chakan['time'].iloc[0]).total_seconds() / 3600
    return r,dert_t,dert_u,dert_i


# Use cycle 1792 as a reference to compute Cell 1's OCV-SOC curve
# and analyze its evolution pattern and overall distribution compared to 4 selected cycles

def OCV_SOC(m,i,data,xia,shang):
    add=list(range(i-1,i+3))
    data['time']=pd.to_datetime(data['time'],format = '%Y-%m-%d %H:%M:%S')
    data_identification=data[data['number'].isin(add)]

    r0=r(data_identification,m)[0]/2+r(data_identification,m)[1]/2
    jihua=r(data_identification,m)[2]
    data_identification['jihua']=jihua/len(data_identification)
    data_identification['jihua']=data_identification['jihua'].cumsum()
    data_identification.loc[data_identification['State']==10,'r'+str(m)]=r0
    data_identification.loc[data_identification['State']==10,'uocv'+str(m)]=r0*data_identification['TotalCurrent']+data_identification['Cell_V'+str(m)]*0.001+data_identification['jihua']
    data_ocv=data_identification[data_identification['State']==10]
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
    # Extract the time difference between upper and lower thresholds for capacity calculation,
    # obtain SOH, then calculate SOC. This gives four key features: SOC, UOCV, internal resistance, and SOH.
    # Plot SOC and open-circuit voltage vs. time, and SOC-OCV curve with battery ID, vehicle info, and timestamp.
    
    # First, concatenate the padded data with the original data into a DataFrame for SOH and SOC calculation
     
    ocv=data_ocv[['time','x','uocv'+str(m),'Cell_V'+str(m),'TotalCurrent','frame_cap_diff']]
    ocv_a=pd.DataFrame({'uocv'+str(m):p[0:600],'frame_cap_diff':-10*ocv['TotalCurrent'].iloc[2]/3600})
    ocv_b=pd.DataFrame({'uocv'+str(m):pp[-401:-1],'frame_cap_diff':-10*ocv['TotalCurrent'].iloc[-2]/3600})
    ocv=pd.concat([ocv_a,ocv,ocv_b],axis=0)
    ocv=ocv[(ocv['uocv'+str(m)]>=xia)&(ocv['uocv'+str(m)]<=shang)]
    ocv['SOC']=ocv['frame_cap_diff'].cumsum()/(ocv['frame_cap_diff'].sum())*100
    
    soh1=ocv['frame_cap_diff'].sum()/150
    ocv['r']=(ocv['uocv'+str(m)]-ocv['Cell_V'+str(m)]*0.001)/ocv['TotalCurrent']
    # Plotting section

    t = data_ocv['uocv' + str(m)]
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    light_green = (0.7, 0.9, 0.7)  
    dark_blue = (0.0, 0.0, 0.8)  
    dark_orange = (0.6, 0.3, 0.0)  

    
    plt.plot(x, t, color=light_green, linewidth=3, label='Original curves')
    plt.plot(x2, p, color=dark_blue, linewidth=3, label='Extended curves')
    plt.plot(x3, pp, color=dark_orange, linewidth=3, label='Extended curves')
    plt.scatter(x_n, t_n, color='', marker='o', edgecolors=dark_blue, s=100, linewidth=3, label='Points used for simulation')
    plt.scatter(x_nn, t_nn, color='', marker='o', edgecolors=dark_orange, s=100, linewidth=3, label='Points used for simulation')
    
    xia = 2.8
    shang = 4.25
    plt.axhline(xia, color='black', ls='--', lw=3)
    plt.axhline(shang, color='black', ls='--', lw=3)
    plt.ylim(2.75, 4.4)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(u'Time Series (10s)', fontsize=16)
    plt.ylabel(u'Voltage (V)', fontsize=16)
    plt.legend(loc='lower right', bbox_to_anchor=(0.30, 0.58), bbox_transform=ax1.transAxes, ncol=1, prop={'size': 16})
    plt.text(366, 2.8, '2.8V, SOC=0', ha='center', va='bottom', fontsize=16)
    plt.text(366, 4.25, '4.25V, SOC=100', ha='center', va='bottom', fontsize=16)
    plt.show()
    # plt.close()

    return ocv,soh1,r(data_identification,m)[0],r(data_identification,m)[1]

ocv_soc_curve,soh,r01,r02=OCV_SOC(1,1792,data,2.8,4.25)
r0=ocv_soc_curve['r'].mean()