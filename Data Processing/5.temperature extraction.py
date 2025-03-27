# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:54:54 2024

@author: Rico

Three tasks:
1) Merge sensor temperature data into the main `data` DataFrame
2) Recalculate and overwrite max, min, and average temperatures for each sensor in each segment
3) Load SOH and R0 tables, reformat into a single DataFrame, and analyze cell grouping structure,
   as well as the mapping between cell IDs and temperature sensor probes
"""

import pandas as pd
from tqdm import tqdm
import numpy as np

# Parse CellTemperatures string into a list of 34 temperature values, adjusted and cleaned
def unfold(a):
    b = (a.split('_'))
    b = pd.to_numeric(b, 'coerce')
    b = list(b)
    if len(b) == 34:
        b_array = np.array(b)
        b_array = b_array.astype(float)
        b_array -= 40
        b_array[b_array == -40] = np.nan
        b = list(b_array)
    else:
        b = list(np.full(34, np.nan))
    return b

# Expand temperature data for each sensor using sliding window through the DataFrame
def cell_temp(length, data_c):
    b = pd.DataFrame()
    for i in tqdm(range(int(len(data_c) / length + 1))):
        try:
            chakan = data_c.iloc[i * length:i * length + length]
            a1 = list(chakan['CellTemperatures'].apply(lambda x: unfold(x)))
            a1 = pd.DataFrame(a1)
            b = pd.concat([b, a1])
        except:
            return i
    return b

names = ['vehiclexxx', 'vehiclexxx', ...]

chuanganqinumber = 34
for name in names:

    tbl04 = pd.read_csv('Extracted data storage file_segments_tbl/' + name + 'tbl04.csv', index_col=0)
    data = pd.read_csv('Extracted data storage file_segments/' + name + 'data.csv', index_col=0)
    data['len_templist'] = data['CellTemperatures'].apply(lambda x: len(str(x)))
    
    # Extract sensor temperatures from CellTemperatures field
    sensor_temp = cell_temp(500000, data)
    
    new_col = []
    for k in range(1, chuanganqinumber + 1):
        new_col.append('Tensor_temp' + str(k))
    sensor_temp.columns = new_col
    sensor_temp = sensor_temp.reset_index(drop=True)
    data = pd.concat([data, sensor_temp], axis=1)
    del new_col
    del sensor_temp

    # Calculate max, min, mean temperature across all sensors
    data['MaxTemp'] = data.loc[:, 'Tensor_temp1':'Tensor_temp34'].max(axis=1)
    data['MinTemp'] = data.loc[:, 'Tensor_temp1':'Tensor_temp34'].min(axis=1)
    data['Tem'] = data.loc[:, 'Tensor_temp1':'Tensor_temp34'].mean(axis=1)
    data.to_csv('data_with_temperature/' + name + 'data.csv', index=False)

    # Update segment table with average temperatures per segment
    for i in tbl04['segment_id']:
        chakan = data[data['number'] == i]
        tbl04.loc[tbl04['segment_id'] == i, 'temp_max'] = chakan['MaxTemp'].mean()
        tbl04.loc[tbl04['segment_id'] == i, 'temp_min'] = chakan['MinTemp'].mean()
        tbl04.loc[tbl04['segment_id'] == i, 'temp_mean'] = chakan['Tem'].mean()
        for k in range(1, chuanganqinumber + 1):
            tbl04.loc[tbl04['segment_id'] == i, f'Tensor_temp{k}_max'] = chakan[f'Tensor_temp{k}'].max()
            tbl04.loc[tbl04['segment_id'] == i, f'Tensor_temp{k}_min'] = chakan[f'Tensor_temp{k}'].min()
            tbl04.loc[tbl04['segment_id'] == i, f'Tensor_temp{k}_mean'] = chakan[f'Tensor_temp{k}'].mean()

    tbl04['Ave_temp_diff'] = tbl04['temp_max'] - tbl04['temp_min']
    tbl04.to_csv('data_with_temperature/' + name + 'tbl04.csv', index=False)


# Final step: merge resistance and SOH data, observe and verify, then save

from scipy import stats

# Outlier filtering using sliding window + z-score
def lvbo(r0_df):
    time_series = r0_df['1'].values
    filter_data = r0_df['3'].values

    window_span = 10000
    step_size = 2000

    # Store filtered results
    filtered_time_series = []
    filtered_filter_data = []

    # Use sliding window to filter outliers
    start_time = time_series[0]
    end_time = start_time + window_span

    while end_time <= time_series[-1]:
        # Get window indices
        start_index = np.where(time_series >= start_time)[0][0]
        end_index = np.where(time_series <= end_time)[0][-1]

        window_time_series = time_series[start_index:end_index + 1]
        window_filter_data = filter_data[start_index:end_index + 1]

        # Apply Z-score
        z_scores = np.abs(stats.zscore(window_filter_data))
        threshold = 1.2
        filtered_window_time_series = window_time_series[z_scores < threshold]
        filtered_window_filter_data = window_filter_data[z_scores < threshold]

        # Append filtered data
        filtered_time_series.extend(filtered_window_time_series)
        filtered_filter_data.extend(filtered_window_filter_data)

        # Move window
        start_time += step_size
        end_time = start_time + window_span

    filtered_time_series = np.array(filtered_time_series)
    filtered_filter_data = np.array(filtered_filter_data)

    data = {'Time_Series': filtered_time_series, 'Filtered_Filter_Data': filtered_filter_data}
    df = pd.DataFrame(data)

    # Remove duplicates
    # Return mileage only
    return df['Time_Series']


merge_data = pd.DataFrame()
for name in tqdm(names):  
    r0_df = pd.read_csv(f'Battery parameters/{name}r0_df.csv')
    soh_df = pd.read_csv(f'Battery parameters/{name}soh_df.csv')
    tbl04 = pd.read_csv('data_with_temperature/' + name + 'tbl04.csv')

    # These lines determine whether filtering is applied
    baoliulicheng = lvbo(r0_df)
    r0_df = r0_df[r0_df['1'].isin(baoliulicheng)]
    soh_df = soh_df[soh_df['1'].isin(baoliulicheng)]

    # Rename resistance columns
    r0_df = r0_df.rename(columns={r0_df.columns[0]: 'segment_id'})
    new_column_names = [f'estimated_r0_cell{i}' for i in range(1, 96)]
    r0_df = r0_df.rename(columns=dict(zip(r0_df.columns[3:98], new_column_names)))
    selected_columns = ['segment_id'] + [f'estimated_r0_cell{i}' for i in range(1, 96)]
    r0_df = r0_df.loc[:, selected_columns]

    # Rename SOH columns
    soh_df = soh_df.rename(columns={soh_df.columns[0]: 'segment_id'})
    new_column_names = [f'estimated_soh_cell{i}' for i in range(1, 96)]
    soh_df = soh_df.rename(columns=dict(zip(soh_df.columns[3:98], new_column_names)))
    selected_columns = ['segment_id'] + [f'estimated_soh_cell{i}' for i in range(1, 96)]
    soh_df = soh_df.loc[:, selected_columns]

    tbl04 = tbl04[tbl04['segment_id'].isin(r0_df['segment_id'])] 
    a = tbl04.loc[:, 'Tensor_temp1_max':'Tensor_temp34_mean']
    tbl04 = tbl04[['segment_id', 'start_time', 'end_time', 'State', 'start_mile',
           'end_mile', 'Ave_speed', 'Ave_Current', 'fragment_numbers', 'start_soc', 'end_soc', 'Cap(A·h)',
           'temp_max', 'temp_min', 'temp_mean', 'Ave_temp_diff', 'Ave_voltage_diff ', 'vin']]
    tbl04 = pd.concat([tbl04, a], axis=1)

    # Merge R0, SOH, and temperature/segment info
    merged_df = pd.merge(r0_df, soh_df, on='segment_id', how='outer')
    merged_df = pd.merge(merged_df, tbl04, on='segment_id', how='outer')

    merge_data = pd.concat([merge_data, merged_df], axis=0)

# Unit conversion
merge_data['start_mile'] = merge_data['start_mile'] * 0.1
merge_data['end_mile'] = merge_data['end_mile'] * 0.1
merge_data['Ave_Current'] = merge_data['Ave_Current'] * 0.1
merge_data['Cap(A·h)'] = merge_data['Cap(A·h)'] * 0.1

# Related to battery pack structure
def identify_group(number):
    if number <= 10:
        return int((number - 1) / 5 + 1)
    elif 11 <= number <= 70:
        return int((number - 11) / 6 + 3)
    elif 71 <= number <= 95:
        return int((number - 71) / 5 + 13)

# Add corresponding sensor temperatures for each cell in merge_data
for i in tqdm(range(1, 96)):
    a = identify_group(i)
    b = 2 * a - 1
    c = 2 * a
    merge_data[f'cell{i}max_temp a'] = merge_data[f'Tensor_temp{b}_max']
    merge_data[f'cell{i}min_temp a'] = merge_data[f'Tensor_temp{b}_min']
    merge_data[f'cell{i}mean_temp a'] = merge_data[f'Tensor_temp{b}_mean']
    merge_data[f'cell{i}max_temp b'] = merge_data[f'Tensor_temp{c}_max']
    merge_data[f'cell{i}min_temp b'] = merge_data[f'Tensor_temp{c}_min']
    merge_data[f'cell{i}mean_temp b'] = merge_data[f'Tensor_temp{c}_mean']

merge_data.to_csv('Battery parameters/results need to be fixed_filted.csv', index=None)
