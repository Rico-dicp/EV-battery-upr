# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:49:15 2024

@author: Rico
Loop execution:
Read neural network processed data and convert it into evenly spaced format.
Fields include 'Mileage', 'Fixed', 'k', 'kk' (assumed known).
"""

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# %% Data Preparation
pickle_file_path = 'cap_compare.pickle'

# Open and load data from pickle file
with open(pickle_file_path, 'rb') as file:
    data_soh = pickle.load(file)

for name in tqdm(['vehiclexx', 'vehiclexxx', 'vehiclexxx', ...]):

    r0_df = pd.read_csv(f'Battery parameters/{name}r0_df.csv')
    soh_df = pd.read_csv(f'Battery parameters/{name}soh_df.csv')

    filted_soh_df = soh_df.copy()
    chakan = data_soh[name]

    # Step 1: Read min and max mileage from filted_soh_df['1']
    min_mileage = filted_soh_df['1'].min()
    max_mileage = filted_soh_df['1'].max()

    # Step 2: Process each cell (battery)
    for battery_number in range(1, 96):  # Battery numbers 1 to 95
        battery_status = chakan[battery_number]

        # Determine index range corresponding to mileage
        min_index = int(min_mileage / 5000)
        max_index = int(max_mileage / 5000)

        # Extract battery status within valid mileage range
        effective_status = battery_status[min_index:max_index + 1]
        effective_status = effective_status.ravel()

        # Calculate mileage values (recorded every 5000 km)
        mileage_values = np.arange(min_index * 5000, (max_index + 1) * 5000, 5000)

        # 10th-degree polynomial fitting
        coefficients = np.polyfit(mileage_values, effective_status, 10)

        # Create a polynomial function
        poly_func = np.poly1d(coefficients)

        # Add a new column to filted_soh_df
        column_name = f'{battery_number}Fixed'
        filted_soh_df[column_name] = filted_soh_df['1'].apply(poly_func)

    # Prepare data for plotting
    # First column: 5000 evenly spaced mileage points from min_mileage to max_mileage
    num_rows = 5000
    mileage_range = np.linspace(min_mileage, max_mileage, num_rows)
    new_df = pd.DataFrame(mileage_range, columns=['Mileage'])

    # For each battery, apply the polynomial function and add as a new column
    for battery_number in range(1, 96):
        battery_status = chakan[battery_number]
        min_index = int(min_mileage / 5000)
        max_index = int(max_mileage / 5000)
        effective_status = battery_status[min_index:max_index + 1]
        effective_status = effective_status.ravel()
        mileage_values = np.arange(min_index * 5000, (max_index + 1) * 5000, 5000)
        coefficients = np.polyfit(mileage_values, effective_status, 10)
        poly_func = np.poly1d(coefficients)

        column_name = f'{battery_number}Fixed'
        new_df[column_name] = poly_func(new_df['Mileage'])

    # Save results
    filted_soh_df.to_csv(f'Battery parameters_tem_fixed/{name}filtered_soh.csv', index=False)
    new_df.to_csv(f'Battery parameters_tem_fixed/{name}soh_for_line.csv', index=False)
