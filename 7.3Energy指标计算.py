# -*- coding: utf-8 -*-
"""
Created on Mon May 20 07:59:33 2024

@author: 13121
1）循环读取数据，参考soh指标，做svr处理，保存结果
2）指标计算，保存，简单出图观察下指标
3）在9.0里补充出图程序

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times new roman'] # 修改字体为宋体
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'
from tqdm import tqdm
import os
from scipy import stats
from sklearn.svm import SVR

#用SVR获得平滑曲线：我需要的是输入参数（最优化的C和gamma，缩放倍数，以及去噪后的数据),输出滤波后的曲线，横坐标为里程，纵坐标为容量与内阻
def Smooth_curve(a, best_params, filted_df, n, n1, n2):
    X = filted_df['1'].values.reshape(-1, 1)
    X_pred = np.arange(int(X.min()), int(X.max()),100).reshape(-1, 1)
    X = X / n1
    X_pred = X_pred / n1
    
    X1 = a['1'].values.reshape(-1, 1)
    X1 = X1 / n1

    # 创建一个示例 DataFrame，包含一个名为 'index_column' 的索引
    data = {'index_column': range(len(X_pred))}
    df = pd.DataFrame(data)
    # 将 X_pred 添加为 'mileage' 列
    df['mileage'] = X_pred * n1
    for m in range(3, n):
        #只有处理读取的结果才有str()

        y = filted_df[str(m)].values * n2
        best_svr = SVR(kernel='rbf', degree=3, C=best_params['C'], gamma=best_params['gamma'], epsilon=0.11, verbose=1)
        best_svr.fit(X, y)
        y_pred = best_svr.predict(X_pred)
        y_pred = y_pred / n2
        df['flitered' + str(m - 2)] = y_pred
        a['svr' + str(m - 2)] = best_svr.predict(X1) / n2
    df = df.iloc[:, 1:]
    return df, a



names=['LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
             'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
             'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
             'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']


energy_df_list=pd.DataFrame()

for name in tqdm(names):
    
    energy_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/'+name+'energy_df.csv')
    energy_df['vin']=name
    svr_energy_df, energy_df = Smooth_curve(energy_df, {'C': 10, 'gamma': 0.035}, energy_df, 98, 10000, 100)
    
    svr_energy_df.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy滤波结果，出图用/svr'+name+'.csv', index='False')
    energy_df.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy滤波结果，出图用/svr_origin'+name+'.csv', index='False')
    energy_df_list=pd.concat([energy_df_list,energy_df],axis=0)

    x = energy_df['1']
    fig, ax1 = plt.subplots()
    # 绘制第一个纵坐标轴
    ax1.set_xlabel('Mileage (km)')
    ax1.set_ylabel('Energy(kh)')
    for col in range(1, 96):
        ax1.plot(x, energy_df['svr'+str(col)],color='green')
    plt.show()
    plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy滤波结果，出图用/'+name+'.png')
    plt.close()


indicator_list=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'new_α3'+'.csv')
#α1
energy_df_list['α1'] = energy_df_list.iloc[:, 3:98].std(axis=1)
energy_df_list['α1_svr'] = energy_df_list.iloc[:, 99:194].std(axis=1)
# energy_df_list['fixed_α1'] = energy_df_list.iloc[:, 193:288].std(axis=1)
#α2    
energy_df_list['α2'] = energy_df_list.iloc[:, 3:98].min(axis=1)/energy_df_list.iloc[:, 3:98].mean(axis=1)
energy_df_list['α2_svr'] = energy_df_list.iloc[:, 99:194].min(axis=1)/energy_df_list.iloc[:, 99:194].mean(axis=1)
# energy_df_list['fixed_α2'] = energy_df_list.iloc[:, 193:288].min(axis=1)/energy_df_list.iloc[:, 193:288].mean(axis=1)

# 重置 energy_df_list 的索引
energy_df_list = energy_df_list.reset_index()

# 合并两个数据框
merged_df = indicator_list.merge(energy_df_list, left_on=['vin', '片段编号'], right_on=['vin', '0'], how='left')

# 更新原来的 indicator_list
indicator_list['energy_α1'] = merged_df['α1_y']
indicator_list['energy_α1_svr'] = merged_df['α1_svr_y']
indicator_list['energy_α2'] = merged_df['α2_y']
indicator_list['energy_α2_svr'] = merged_df['α2_svr_y']


indicator_list.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy滤波结果，出图用/New_indicators'+'.csv',index=False)


# #事先去除有空值的行即可
# plt.figure(figsize=(9, 6))

# # 获取一个由19种颜色组成的色条
# colors = plt.get_cmap('coolwarm', 19)
# # 存储每个name的关键fixed_α1值
# critical_values = []

# for name in names:
#     data = indicator_list[indicator_list['vin'] == name]
#     data = data.sort_values(by='mileage')
#     # 找到mileage接近200000的点
#     closest_row = data.iloc[(data['mileage'] - 200000).abs().argsort()[:1]]
#     critical_values.append((name, closest_row['energy_α1_svr'].values[0]))

# # 根据fixed_α1排序并映射颜色
# critical_values.sort(key=lambda x: x[1])
# name_color_map = {name: colors(i / (len(critical_values) - 1)) for i, (name, _) in enumerate(critical_values)}

# # 绘制每个name的折线图
# for name in names:
#     data = indicator_list[indicator_list['vin'] == name]
#     data = data.sort_values(by='mileage')
#     color = name_color_map[name]  # 使用映射后的颜色
#     plt.plot(data['mileage'] / 1000, data['energy_α1_svr'], linestyle='-', markersize=4, color=color,  linewidth=2.0)

# # 设置标签和字体属性
# plt.xlabel('Mileage (10³ km)',fontsize = 26)
# plt.ylabel('α_energy',fontsize = 26)
# # 增大坐标轴刻度数字的字体大小
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.tight_layout()
# plt.show()

# plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α1_energy.png', format='png', bbox_inches='tight')
# plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α1_energy.svg', format='svg', bbox_inches='tight')
# plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α1_energy.pdf', format='pdf', bbox_inches='tight')
# plt.close()


# ################################################


# plt.figure(figsize=(9, 6))

# # 获取一个由19种颜色组成的色条
# colors = plt.get_cmap('coolwarm', 19)
# # 存储每个name的关键fixed_α1值
# critical_values = []

# for name in names:
#     data = indicator_list[indicator_list['vin'] == name]
#     data = data.sort_values(by='mileage')
#     # 找到mileage接近200000的点
#     closest_row = data.iloc[(data['mileage'] - 200000).abs().argsort()[:1]]
#     critical_values.append((name, closest_row['energy_α2_svr'].values[0]))

# # 根据fixed_α1排序并映射颜色
# critical_values.sort(key=lambda x: x[1])
# name_color_map = {name: colors(i / (len(critical_values) - 1)) for i, (name, _) in enumerate(critical_values)}

# # 绘制每个name的折线图
# for name in names:
#     data = indicator_list[indicator_list['vin'] == name]
#     data = data.sort_values(by='mileage')
#     color = name_color_map[name]  # 使用映射后的颜色
#     plt.plot(data['mileage'] / 1000, data['energy_α2_svr'], linestyle='-', markersize=4, color=color,  linewidth=2.0)

# # 设置标签和字体属性
# plt.xlabel('Mileage (10³ km)',fontsize = 26)
# plt.ylabel('β_energy',fontsize = 26)
# # 增大坐标轴刻度数字的字体大小
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.tight_layout()
# plt.show()


# plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α2_energy.png', format='png', bbox_inches='tight')
# plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α2_energy.svg', format='svg', bbox_inches='tight')
# plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α2_energy.pdf', format='pdf', bbox_inches='tight')
# plt.close()
