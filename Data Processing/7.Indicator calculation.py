# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:17:48 2024

@author: Rico



"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from sklearn.svm import SVR

#%%

fit_params = [2.79349202e+00,  1.32456734e-01, -1.06408510e-02,  4.87568943e-04,
       -1.21132612e-05,  1.21430353e-07,  1.19820875e-09, -4.82818206e-11,
        5.64142933e-13, -3.06234456e-15,  6.54961010e-18]

# Define 10th-order polynomial function
def polynomial_func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6 + a7*x**7 + a8*x**8 + a9*x**9 + a10*x**10

# Define a callable fitting function
def fitted_function(x,fit_params):
    return polynomial_func(x, *fit_params)


def EFC(soh, j, threshold,k):
    # Calculate the difference with respect to threshold
    diff = soh.iloc[:, j] - threshold  
    # Find the closest indices on both sides of the threshold
    try: 
        positive_idx = diff[diff >= 0].idxmin()
        negative_idx = diff[diff < 0].idxmax()  
        # Retrieve values at the two indices
        positive_j_values = soh.iloc[positive_idx, j]
        positive_k_values = soh.loc[positive_idx, k]
        negative_j_values = soh.iloc[negative_idx, j]
        negative_k_values = soh.loc[negative_idx, k]  

        x_values = [positive_j_values, negative_j_values]
        y_values = [positive_k_values, negative_k_values]
    
        # Interpolation target
        target_x = threshold
        # Linear interpolation
        coefficients = np.polyfit(x_values, y_values, 1)
    
       # Extract slope and intercept
        slope = coefficients[0]
        intercept = coefficients[1]
    
        # Compute EFC
        efc = slope * target_x + intercept
    except:
        idx = (soh.iloc[:, j] - threshold).abs().idxmin()
        efc = soh.loc[idx, k]
    
    return efc


# Build function: For a given vehicle and segment index i, iterate through all cells to construct SOC curve. 
# For each cell, locate k=0.9 and extract its voltage, current, delta to cutoff voltage, 
# then retrieve rp, cp, r0 and compute the maximum current.
# Finally, take the minimum I_max among all cells, calculate the maximum power of each cell, 
# and then compute the degradation indicator.
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
            data_identification.loc[:, 'soc'+str(m)] = I_soc + data_identification[data_identification['TotalVoltage']>=375]['frame_cap_diff'].cumsum() / (soh * 150) * 100
            yaosuzhen=data_identification[data_identification['soc'+str(m)]>=k*100].iloc[0]
            It=-yaosuzhen['TotalCurrent']
            vt=yaosuzhen['Cell_V'+str(m)]*0.001-r0*It-fitted_function(k*100,fit_params)
            fenzi=4.25-yaosuzhen['Cell_V'+str(m)]*0.001+r0*It+vt*(1-np.exp(-(dert_t/(rp*cp))))
            fenmu=r0+rp*(1-np.exp(-(dert_t/(rp*cp))))
            I_max=fenzi/fenmu    
            I_max_list.append(I_max)
        min_I_max=min(I_max_list)
        p_max_list=[]
        min_p_max_list=[]
        for m in range(1,96):
            r0=r0_list[(r0_list['0']==i)].iloc[:, m+97].values[0]
            I_max=I_max_list[m-1]
            It=-yaosuzhen['TotalCurrent']
            vt=yaosuzhen['Cell_V'+str(m)]*0.001
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
    # Assume r0_df contains the dataset, where r0_df[1] is the time series and r0_df[3] is the target for filtering
    time_series = r0_df['1'].values
    filter_data = r0_df['3'].values

    # Set window size and step size
    window_span = 10000
    step_size = 2000

    # Store filtered data
    filtered_time_series = []
    filtered_filter_data = []

    # Apply sliding window to detect and remove outliers
    start_time = time_series[0]
    end_time = start_time + window_span

    while end_time <= time_series[-1]:
        start_index = np.where(time_series >= start_time)[0][0]
        end_index = np.where(time_series <= end_time)[0][-1]

        window_time_series = time_series[start_index:end_index+1]
        window_filter_data = filter_data[start_index:end_index+1]

        z_scores = np.abs(stats.zscore(window_filter_data))

        threshold = 1.2
        filtered_window_time_series = window_time_series[z_scores < threshold]
        filtered_window_filter_data = window_filter_data[z_scores < threshold]

        filtered_time_series.extend(filtered_window_time_series)
        filtered_filter_data.extend(filtered_window_filter_data)

        start_time += step_size
        end_time = start_time + window_span

    filtered_time_series = np.array(filtered_time_series)
    filtered_filter_data = np.array(filtered_filter_data)

    data = {'Time_Series': filtered_time_series, 'Filtered_Filter_Data': filtered_filter_data}
    df = pd.DataFrame(data)

    df = df.drop_duplicates()
    return df['Time_Series']

# Use SVR to obtain smoothed curves: input includes best C/gamma, scaling factors, and filtered data;
# output includes the smoothed curve with mileage on x-axis and indicators (e.g., α4, α5) on y-axis.
def Smooth_curve(a, best_params, filted_df, n, n1, n2, indicator):
    X = filted_df['1'].values.reshape(-1, 1)
    X_pred = np.arange(int(X.min()), int(X.max()),100).reshape(-1, 1)
    X = X / n1
    X_pred = X_pred / n1

    X1 = a['1'].values.reshape(-1, 1)
    X1 = X1 / n1

    data = {'index_column': range(len(X_pred))}
    df = pd.DataFrame(data)
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
def indicator_cal(name):
    r0_df=pd.read_csv('Battery parameters/'+name+'r0_df'+'.csv')
    soh_df=pd.read_csv('Battery parameters_tem_fixed/{name}filtered_soh.csv')

   
    soc_df=pd.read_csv('Battery parameters/'+name+'soc_df'+'.csv')
    rp_df=pd.read_csv('Battery parameters/'+name+'rp_df'+'.csv')
    cp_df=pd.read_csv('Battery parameters/'+name+'cp_df'+'.csv')

    jiezhiSOC_df=pd.read_csv('Battery parameters/'+name+'jiezhiSOC_df'+'.csv')

   
    baoliulicheng =  lvbo(r0_df)
    filted_soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]
 
       
    
    tbl01=pd.read_csv('Processed data storage file_segments/'+name+'_tbl.csv',index_col=False)
    tbl01=tbl01[tbl01['State']==10]
    start_cycle= tbl01['start_mile'].iloc[0]*(tbl01['end_soc']-tbl01['start_soc']).cumsum().iloc[-1]/(tbl01['start_mile'].iloc[-1]-tbl01['start_mile'].iloc[0])/100
    tbl01['cycle_based on_soc']=(tbl01['end_soc']-tbl01['start_soc']).cumsum()/100+start_cycle
    start_cycle= tbl01['start_mile'].iloc[0]*(tbl01['Cap(A·h)']).cumsum().iloc[-1]/(tbl01['start_mile'].iloc[-1]-tbl01['start_mile'].iloc[0])/1500
    tbl01['cycle_based on_cap']=(tbl01['Cap(A·h)']).cumsum()/1500+start_cycle  
    tbl01=tbl01[tbl01['segment_id'].isin(soh_df['0'])]
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
    x=pd.read_csv('Extracted data storage file_segments/'+name+'data.csv', index_col=0)
    alpha5_list,I_max_list,min_p_max_list,p_max_list=FW(name,soh_df,0.85,10,soh_df,r0_df,rp_df,cp_df,soc_df,x)       
    soh_df['α5']=alpha5_list
    
    soh_df = soh_df.dropna()
    filted_soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]
    α5_soh_df, soh_df = Smooth_curve(soh_df, {'C': 10, 'gamma': 0.035}, filted_soh_df, 98, 10000, 100,'α5')
    
    soh_df['vin']=name
    b = soh_df.iloc[:, :3].join(soh_df.iloc[:, -12:-1])
    b = soh_df.iloc[:, -1:].join(b)
    b = b.rename(columns={'0': 'segment_id', '1': 'mileage', '2': 'temperature'})

    return b
#%% 
   
names=['vehiclexxx','vehiclexxx',...]

import warnings
# 
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)


for name in tqdm(names):       
    a = indicator_cal(name)
    a.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'α_df'+'.csv',index=False)
    
 



#%%

indicator_list=pd.DataFrame()

for name in tqdm(names):    
    a=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Indicators/'+name+'α_df'+'.csv')
    indicator_list = pd.concat([indicator_list, a], axis=0)
    

indicator_list_orgin = indicator_list.copy()  



#%% Calculate new_α3

def new_EFC(soh, j, threshold, k):
    # Calculate the difference from the threshold
    diff = soh.iloc[:, j] - threshold  
    # Find indices with the closest positive and negative differences
    try: 
        positive_idx = diff[diff >= 0].idxmin()
        negative_idx = diff[diff < 0].idxmax()  
        # Extract corresponding j and k column values
        positive_j_values = soh.iloc[positive_idx, j]
        positive_k_values = soh.loc[positive_idx, k]
        negative_j_values = soh.iloc[negative_idx, j]
        negative_k_values = soh.loc[negative_idx, k]  
        x_values = [positive_j_values, negative_j_values]
        y_values = [positive_k_values, negative_k_values]

        # Interpolation
        target_x = threshold
        coefficients = np.polyfit(x_values, y_values, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        efc = slope * target_x + intercept
    except:
        idx = (soh.iloc[:, j] - threshold).abs().idxmin()
        efc = soh.loc[idx, k]

    return efc

# α3: mileage at which first cell reaches 0.85 divided by mileage where average cell reaches 0.85
for name in tqdm(names):    
    soh_df = pd.read_csv('Battery parameters_tem_fixed/' + name + 'filtered_soh.csv')
    mean_value = soh_df.iloc[:, 193:288].mean(axis=1)
    soh_df['soh_mean'] = mean_value

    tbl01 = pd.read_csv('Processed data storage file_segments/' + name + '_tbl.csv', index_col=False)
    tbl01 = tbl01[tbl01['State'] == 10]

    # Calculate cycle count based on SOC and capacity
    start_cycle = tbl01['start_mile'].iloc[0] * (tbl01['end_soc'] - tbl01['start_soc']).cumsum().iloc[-1] / (tbl01['start_mile'].iloc[-1] - tbl01['start_mile'].iloc[0]) / 100
    tbl01['cycle_based on_soc'] = (tbl01['end_soc'] - tbl01['start_soc']).cumsum() / 100 + start_cycle
    start_cycle = tbl01['start_mile'].iloc[0] * (tbl01['Cap(A·h)']).cumsum().iloc[-1] / (tbl01['start_mile'].iloc[-1] - tbl01['start_mile'].iloc[0]) / 1500
    tbl01['cycle_based on_cap'] = (tbl01['Cap(A·h)']).cumsum() / 1500 + start_cycle  

    tbl01 = tbl01[tbl01['segment_id'].isin(soh_df['0'])]
    soh_df['cycle_based on_soc'] = tbl01['cycle_based on_soc'].values
    soh_df['cycle_based on_cap'] = tbl01['cycle_based on_cap'].values

    EFC_list = []
    for j in range(193, 289):
        efc_cell = new_EFC(soh_df, j, 0.85, 'cycle_based on_cap')
        EFC_list.append(efc_cell)

    indicator_list.loc[indicator_list['vin'] == name, 'new_α3_fixed'] = min(EFC_list) / EFC_list[-1]
    indicator_list.loc[indicator_list['vin'] == name, 'mile_treshold'] = soh_df[soh_df['cycle_based on_cap'] >= min(EFC_list)]['1'].iloc[0]

# If any vehicle has poor temperature correction performance
indicator_list = indicator_list.reset_index(drop=True)
indicator_list.loc[indicator_list['vin'] == 'xxx', 'fixed_α1'] = indicator_list['α1_svr']
indicator_list.loc[indicator_list['vin'] == 'xxx', 'fixed_α2'] = indicator_list['α2_svr']

indicator_list.to_csv('Battery parameters/' + 'new_α3' + '.csv', index=False)



#%% 最终的α_Overall capacity utilization

indicator_list=pd.read_csv('Battery parameters/'+'new_α3'+'.csv')
indicator_list=indicator_list[indicator_list['mileage']<=indicator_list['mile_treshold']]
for name in tqdm(names):  
    chakan=indicator_list[indicator_list['vin']==name]
    indicator_list.loc[indicator_list['vin']==name, 'α'] = chakan['fixed_α2'].mean()*chakan['new_α3_fixed'].mean()

indicator_list.to_csv('Battery parameters/'+'final_α'+'.csv',index=False)    
