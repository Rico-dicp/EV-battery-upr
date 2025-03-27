# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:09:31 2024

@author: Rico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times new Roman'] 
plt.rcParams['axes.unicode_minus'] = False	
from tqdm import tqdm
import pyswarms as ps
from scipy import stats
from sklearn.svm import SVR

#%%

fit_params = [2.79349202e+00,  1.32456734e-01, -1.06408510e-02,  4.87568943e-04,
       -1.21132612e-05,  1.21430353e-07,  1.19820875e-09, -4.82818206e-11,
        5.64142933e-13, -3.06234456e-15,  6.54961010e-18]


def r1(data_identification,m):
    i=data_identification[data_identification['TotalCurrent'].diff(1)>9].index.values[-1]
    chakan=data_identification.loc[i-1:i]
    dert_u=chakan['Cell_V'+str(m)].iloc[0]-data_identification['Cell_V'+str(m)].iloc[-1]
    dert_i=chakan['TotalCurrent'].diff(1).mean()
    r=dert_u/dert_i*0.001
    dert_t=(data_identification['time'].iloc[-1]-chakan['time'].iloc[0]).total_seconds() / 3600
    return r,dert_t,dert_u,dert_i

# Define 10th-degree polynomial function
def polynomial_func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6 + a7*x**7 + a8*x**8 + a9*x**9 + a10*x**10

# Define a callable fitting function
def fitted_function(x,fit_params):
    return polynomial_func(x, *fit_params)


#PSO
def system_function(particles, data_identification):
    a, b, c, Rp, Cp, Vp0 = particles[:, 0], particles[:, 1], particles[:, 2],particles[:, 3], particles[:, 4], particles[:, 5]
    cap = data_identification['cap'][:, np.newaxis]
    total_current = data_identification['TotalCurrent'][:, np.newaxis]
    y = fitted_function(a + cap / (b * 150)*100, fit_params) - (-total_current*Rp-Vp0)*np.exp(-np.divide(3600*cap, -Rp* Cp* total_current))-total_current*(c+Rp)
    return y


def objective_function(particles, n_particles, target_data, data_identification):
    predicted_data = system_function(particles, data_identification)
    target_data = np.tile(target_data, (n_particles, 1)).T
    mse = np.mean((target_data - predicted_data)**2, axis=0) 
    return mse

## For parameter identification, sample from 375 to 385, taking one every 10 cycles
def identify_parameters(data, i, m, n_particles, cishu):
    # First calculate two internal resistance values, output them, 
    # then take the average and multiply by a coefficient (0.8 to 1.0) as the reference range
    i = int(i)
    add = list(range(i - 1, i + 3))
    data_identification = data[data['number'].isin(add)]
    r0, dert_t, dert_u, dert_i = r1(data_identification, m)

    data_identification = data[data['number'].isin([i])][2:]
    data_identification = data_identification[(data_identification['TotalVoltage'] >= 375) & (data_identification['TotalVoltage'] <= 385)]
    data_identification['cap'] = data_identification['frame_cap_diff'].cumsum()
    data_identification['t'] = np.array(range(len(data_identification)))[:, np.newaxis] * 10
    data_identification = data_identification[::10]

    target_data = data_identification['Cell_V' + str(m)].values * 0.001

    options = {'c1': 0.1, 'c2': 0.1, 'w': 0.9}
    bounds = ([0, 0.7, r0 * 0.21, r0 * 0.06, 60000, 0.001], 
              [95, 1, 0.23 * r0, r0 * 0.08, 70000, 0.0015])    

    # Generate initial particle distribution within bounds
    initial_particles = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_particles, 6))
    initial_particles = np.clip(initial_particles, bounds[0], bounds[1])

    # Create optimizer object using fixed initial particles
    threshold = 1e-9
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, 
        dimensions=6, 
        options=options, 
        bounds=bounds, 
        init_pos=initial_particles,
        ftol=threshold,
        ftol_iter=10
    )
    cost, best_pos = optimizer.optimize(
        objective_function,
        iters=cishu, 
        n_particles=n_particles, 
        target_data=target_data, 
        data_identification=data_identification,
        verbose=0
    )
    return best_pos


def lvbo(r0_df):
    time_series = r0_df[1].values
    filter_data = r0_df[5].values

    window_span = 10000
    step_size = 2000

    # Store filtered data
    filtered_time_series = []
    filtered_filter_data = []

    # Use sliding window to filter outliers
    start_time = time_series[0]
    end_time = start_time + window_span

    while end_time <= time_series[-1]:
        # Determine start and end indices of the window
        start_index = np.where(time_series >= start_time)[0][0]
        end_index = np.where(time_series <= end_time)[0][-1]

        window_time_series = time_series[start_index:end_index + 1]
        window_filter_data = filter_data[start_index:end_index + 1]

        # Calculate z-score within the window
        z_scores = np.abs(stats.zscore(window_filter_data))

        # Filter values based on threshold
        threshold = 1.2  # Adjust threshold as needed
        filtered_window_time_series = window_time_series[z_scores < threshold]
        filtered_window_filter_data = window_filter_data[z_scores < threshold]

        # Append filtered values to result lists
        filtered_time_series.extend(filtered_window_time_series)
        filtered_filter_data.extend(filtered_window_filter_data)

        # Update window position
        start_time += step_size
        end_time = start_time + window_span

    # Convert results to arrays
    filtered_time_series = np.array(filtered_time_series)
    filtered_filter_data = np.array(filtered_filter_data)

    data = {'Time_Series': filtered_time_series, 'Filtered_Filter_Data': filtered_filter_data}
    df = pd.DataFrame(data)

    # Remove duplicate rows
    df = df.drop_duplicates()
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
    
    for xunhuan in tqdm(tbltbl['segment_id']):
      try: 
        data_identification = data[data['number'].isin([xunhuan])][2:]
        licheng=data_identification['Mileage'].iloc[0]
        tem=(data_identification['MaxTemp']/2+data_identification['MinTemp']/2).mean()
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
            jiezhi_SOC=identified_parameters[0]+data_identification[data_identification['TotalVoltage']>=375]['frame_cap_diff'].sum()/(identified_parameters[1]*150)*100
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


# Use SVR to obtain a smoothed curve:
# Input: optimal C and gamma, scaling factors, and denoised data
# Output: filtered curve with mileage on the x-axis, and capacity/internal resistance on the y-axis
def Smooth_curve(a, best_params, filted_df, n, n1, n2):
    X = filted_df[1].values.reshape(-1, 1)
    X_pred = np.arange(int(X.min()), int(X.max()), 100).reshape(-1, 1)
    X = X / n1
    X_pred = X_pred / n1

    X1 = a[1].values.reshape(-1, 1)
    X1 = X1 / n1

    data = {'index_column': range(len(X_pred))}
    df = pd.DataFrame(data)
    
    # Add 'mileage' column from scaled X_pred
    df['mileage'] = X_pred * n1

    for m in range(3, n):
        y = filted_df[m].values * n2
        best_svr = SVR(
            kernel='rbf', 
            degree=3, 
            C=best_params['C'], 
            gamma=best_params['gamma'], 
            epsilon=0.11, 
            verbose=1
        )
        best_svr.fit(X, y)
        y_pred = best_svr.predict(X_pred)
        y_pred = y_pred / n2
        df['flitered' + str(m - 2)] = y_pred
        a['svr' + str(m - 2)] = best_svr.predict(X1) / n2

    df = df.iloc[:, 1:]
    return df, a



#%%
names=['vehiclexxx','vehiclexxx',...]

for name in names:

    data=pd.read_csv('Extracted data storage file_segments/'+name+'data.csv', index_col=0)
    tbl04=pd.read_csv('Extracted data storage file_segments_tbl/'+name+'tbl04.csv', index_col=0)
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
    
    
    
 
    r0_df.to_csv('Battery parameters/'+name+'r0_df'+'.csv',index=False)
    soh_df.to_csv('Battery parameters/'+name+'soh_df'+'.csv',index=False)
    soc_df.to_csv('Battery parameters/'+name+'soc_df'+'.csv',index=False)
    rp_df.to_csv('Battery parameters/'+name+'rp_df'+'.csv',index=False)
    cp_df.to_csv('Battery parameters/'+name+'cp_df'+'.csv',index=False)
    Vp0_df.to_csv('Battery parameters/'+name+'Vp0_df'+'.csv',index=False)
    jiezhiSOC_df.to_csv('Battery parameters/'+name+'jiezhiSOC_df'+'.csv',index=False)
    svr_r0_df.to_csv('Battery parameters/'+name+'svr_r0_df'+'.csv',index=False)
    svr_soh_df.to_csv('Battery parameters/'+name+'svr_soh_df'+'.csv',index=False)
    wendu_svr_r0_df.to_csv('Battery parameters/'+name+'wendu_svr_r0_df'+'.csv',index=False)
    wendu_filted_r0_df.to_csv('Battery parameters/'+name+'wendu_filted_r0_df'+'.csv',index=False)

    
    #Figures
    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.family'] = 'Arial'
    plt.scatter(r0_df[1]/1000, r0_df[3]*1000, color='lightseagreen', label='Outlier', marker='o')
    plt.scatter(filted_r0_df[1]/1000, filted_r0_df[3]*1000,  color='deeppink', label='Outliers removed', marker='o')
    plt.scatter(wendu_filted_r0_df[1]/1000, wendu_filted_r0_df[3]*1000,  color='gold', label='23-25\u00b0C', marker='o')
    plt.plot(svr_r0_df['mileage']/1000, svr_r0_df['flitered1']*1000, marker='o', linestyle='-', color='darkblue', label='SVR')
    plt.plot(wendu_svr_r0_df['mileage']/1000, wendu_svr_r0_df['flitered1']*1000, marker='o', linestyle='-', color='orange', label='23-25\u00b0C SVR')
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('R0 (mΩ)', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.show()
    plt.close()
    
    

    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.family'] = 'Arial'
    for m in range(1, 96):

        plt.plot(svr_r0_df['mileage']/1000, svr_r0_df['flitered'+str(m)]*1000, linestyle='-', color='darkblue')
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('R0 (mΩ)', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    plt.close()
    

    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.family'] = 'Arial'
    for m in range(1, 96):
        plt.plot(wendu_svr_r0_df['mileage']/1000, wendu_svr_r0_df['flitered'+str(m)]*1000, linestyle='-', color='red')
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('R0 (mΩ)', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    plt.close()
    
    

    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.family'] = 'Arial'
    plt.scatter(filted_soh_df[1]/1000, filted_soh_df[3],  color='deeppink', label='SOH', marker='o')
    plt.plot(svr_soh_df['mileage']/1000, svr_soh_df['flitered1'], marker='o', linestyle='-', color='darkblue', label='SVR')
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('SOH', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.show()
    plt.close()
    

    plt.figure(figsize=(16, 8))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.family'] = 'Arial'
    for m in range(1, 96):
        plt.plot(svr_soh_df['mileage']/1000, svr_soh_df['flitered'+str(m)], linestyle='-', color='red')
    plt.xlabel('Mileage (10³ km)', fontsize=16, fontweight='bold')
    plt.ylabel('SOH', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    plt.close()
