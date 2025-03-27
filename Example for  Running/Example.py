
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:22:36 2024

@author: 13121
所有图的大小需要改成(16,8)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Times New Roman'

# Load the data
line_soh_df = pd.read_csv('C:/Users/13121/Desktop/Example for  Running/soh_for_line2.csv')
line_r0_df = pd.read_csv('C:/Users/13121/Desktop/Example for  Running/r0_for_line2.csv')


# Create a custom color map
colors = LinearSegmentedColormap.from_list('custom_cmap', ['#81A9A9', '#D69C9C'], 95)


plt.rcParams['font.family'] = 'Times New Roman'

fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: SOH修正后
subset = line_soh_df.iloc[-1, 2:98]
sorted_values = subset.sort_values()
sorted_indices = np.argsort(sorted_values)
for m in range(1, 96):
    m1 = line_soh_df['一舟' + str(m) + 'Fixed'].iloc[-1]
    if m1 in sorted_values.values:  # Check if m1 exists in sorted_values
        x_position = 95 - np.where(sorted_indices == np.where(sorted_values == m1)[0][0])[0][0]
        color = colors(x_position / 95)
        axs[0].plot(line_soh_df['Mileage'] / 1000, line_soh_df['一舟' + str(m) + 'Fixed'], linestyle='-', color=color, linewidth=2.5)
axs[0].set_xlabel('Mileage (10³ km)', fontsize=26)
axs[0].set_ylabel('SOH', fontsize=26)
axs[0].tick_params(labelsize=26)

# Plot 2: R0修正后
subset = line_r0_df.iloc[-1, 2:98]
sorted_values = subset.sort_values()
sorted_indices = np.argsort(sorted_values)
for m in range(1, 96):
    m1 = line_r0_df['一舟' + str(m) + 'Fixed'].iloc[-1]
    if m1 in sorted_values.values:  # Check if m1 exists in sorted_values
        x_position = np.where(sorted_indices == np.where(sorted_values == m1)[0][0])[0][0]
        color = colors(x_position / 95)
        axs[1].plot(line_r0_df['Mileage'] / 1000, line_r0_df['一舟' + str(m) + 'Fixed'], linestyle='-', color=color, linewidth=2.5)
axs[1].set_xlabel('Mileage (10³ km)', fontsize=26)
axs[1].set_ylabel('R0 (mΩ)', fontsize=26)
axs[1].set_ylim(0.2, 0.6)
axs[1].tick_params(labelsize=26)
# Add space at the top for the colorbar
fig.subplots_adjust(top=0.9)

# Add a single color bar at the top
cbar_ax = fig.add_axes([0.2, 0.96, 0.6, 0.03])  # position: [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=1, vmax=95))
sm.set_array([])
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=26)
cbar.set_label('Cell Index', fontsize=26, labelpad=-52, loc='center')
# Show the plot
plt.show()

plt.subplots_adjust(hspace=0.3, wspace=0.3)
    

    


# 保存为PNG格式
plt.savefig('C:/Users/13121/Desktop/Example for  Running/'+'example2.png', dpi=1200)
# 保存为SVG格式
plt.savefig('C:/Users/13121/Desktop/Example for  Running/'+'example2.svg', format='svg')
# 保存为SVG格式
plt.savefig('C:/Users/13121/Desktop/Example for  Running/'+'example2.pdf', format='pdf')
