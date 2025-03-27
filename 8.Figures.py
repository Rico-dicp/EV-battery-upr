# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:27:30 2024

@author: 13121
Created on Fri Mar 29 14:50:44 2024

@author: 13121



第一第二个指标各出一张图，不要颜色区分，要颜色渐变
第三个指标作小提琴图，与LFP合成一张
指标四不变，出箱线图
第五个指标尝试把时间对齐再出箱线图
第六个指标作小提琴图，与LFP合成一张
总计：2x2+1+2+2+1共计8张子图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times new Roman']


#%%indicators出图

names=['LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
       'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
       'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
       'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']


indicator_list=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'new_α3'+'.csv')

    
# Set font properties
# 设置字体属性
plt.rc('font', weight='normal',family='Times new Roman')
font_size = 26
font = {'size': font_size, 'family': 'Times new Roman'}
#########################################################α1
# indicator_list = pd.DataFrame(...)
# plt.figure(figsize=(12, 6))
plt.figure(figsize=(9, 6))

# 获取一个由19种颜色组成的色条
colors = plt.get_cmap('coolwarm', 19)
# 存储每个name的关键fixed_α1值
critical_values = []

for name in names:
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    # 找到mileage接近200000的点
    closest_row = data.iloc[(data['mileage'] - 200000).abs().argsort()[:1]]
    critical_values.append((name, closest_row['fixed_α1'].values[0]))

# 根据fixed_α1排序并映射颜色
critical_values.sort(key=lambda x: x[1])
name_color_map = {name: colors(i / (len(critical_values) - 1)) for i, (name, _) in enumerate(critical_values)}

# 绘制每个name的折线图
for name in names:
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    color = name_color_map[name]  # 使用映射后的颜色
    plt.plot(data['mileage'] / 1000, data['fixed_α1'], linestyle='-', markersize=4, color=color,  linewidth=2.0)

# 设置标签和字体属性
plt.xlabel('Mileage (10³ km)',fontsize = 26)
plt.ylabel('α',fontsize = 26)
# 增大坐标轴刻度数字的字体大小
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tight_layout()
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α1.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α1.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α1.pdf', format='pdf', bbox_inches='tight')
plt.close()

#########################################################α2
# 创建指定大小的图表
# 你的DataFrame数据和其他设置
# indicator_list = pd.DataFrame(...)
# plt.figure(figsize=(12, 6))
plt.figure(figsize=(9, 6))

# 获取一个由19种颜色组成的色条
colors = plt.get_cmap('coolwarm', 19)
# 存储每个name的关键fixed_α1值
critical_values = []

for name in names:
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    # 找到mileage接近200000的点
    closest_row = data.iloc[(data['mileage'] - 200000).abs().argsort()[:1]]
    critical_values.append((name, closest_row['fixed_α1'].values[0]))

# 根据fixed_α1排序并映射颜色
critical_values.sort(key=lambda x: x[1])
name_color_map = {name: colors(i / (len(critical_values) - 1)) for i, (name, _) in enumerate(critical_values)}

# 绘制每个name的折线图
for name in names:
    data = indicator_list[indicator_list['vin'] == name]
    data = data.sort_values(by='mileage')
    color = name_color_map[name]  # 使用映射后的颜色
    plt.plot(data['mileage'] / 1000, data['fixed_α2'], linestyle='-', markersize=4, color=color,  linewidth=2.0)

# 设置标签和字体属性
plt.xlabel('Mileage (10³ km)',fontsize = 26)
plt.ylabel('β',fontsize = 26)
# 增大坐标轴刻度数字的字体大小
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tight_layout()
plt.show()


plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α2.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α2.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α2.pdf', format='pdf', bbox_inches='tight')
plt.close()

#########################################################α3

import seaborn as sns

# 创建指定大小的图表
plt.figure(figsize=(12, 8))


# 绘制小提琴图
# 使用seaborn的violinplot来显示所有车辆的fixed_α3值的分布
sns.violinplot(y=indicator_list['new_α3_fixed'], color='royalblue')

# 设置坐标轴标签和字体属性
plt.xlabel('NMC_Vehicles', fontsize=26)  # 可以留空或者标记为'All Vehicles'
plt.ylabel('γ', fontsize=26)

plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
# 显示图表
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α3.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α3.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α3.pdf', format='pdf', bbox_inches='tight')
plt.close()

############################################################α3的补充图

VS=[chr(ord('a') + i) for i in range(19)]

alpha3_svr_values = indicator_list['new_α3_fixed'].unique().tolist()
# 创建指定大小的图表
plt.figure(figsize=(12, 8))

plt.bar(VS, alpha3_svr_values, color='royalblue')  # 绘制柱状图
plt.xlabel('Vehicles', fontdict=font)  # 设置横坐标标签
plt.ylabel('γ', fontdict=font)  # 设置纵坐标标签
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(0, 1)
# 显示柱状图
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()


plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α3.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α3.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α3.pdf', format='pdf', bbox_inches='tight')
plt.close()


#########################################################α4
#α4，建立出图用的表：

# 设置mileage范围和间隔

mileage_range = np.arange(50, 325, 25)
colors = ['#2E3D5C'] * 6 + ['#E93628'] * 5

# 创建vin到整数索引的映射
unique_vins = indicator_list['vin'].unique()
vin_to_index = {vin: i for i, vin in enumerate(unique_vins)}

# 初始化结果数组
result = np.empty((len(mileage_range), len(unique_vins)))
result[:] = np.nan

# 遍历mileage范围
for i, mileage in enumerate(mileage_range):
    # 过滤indicator_listFrame以获取符合mileage范围的行
    filtered_indicator_list = indicator_list[(indicator_list['mileage'] / 1000 >= (mileage - 10)) & (indicator_list['mileage'] / 1000 <= (mileage + 10))]
    
    # 遍历vin分组
    for vin, vin_group in filtered_indicator_list.groupby('vin'):
        if not vin_group.empty:
            # 获取最接近mileage的行
            closest_row = vin_group.loc[(vin_group['mileage'] / 1000 - mileage).abs().idxmin()]
            result[i, vin_to_index[vin]] = closest_row['α4']

result_list = result.tolist()
import seaborn as sns
# 创建箱线图
plt.figure(figsize=(12,8))
sns.boxplot(data=result_list, palette=colors, showfliers=False)

# 设置箱线图横坐标标签
plt.xticks(range(len(mileage_range)), mileage_range)

# 计算并绘制均值线
means = [np.nanmean(group) for group in result_list]
plt.plot(range(len(mileage_range)), means, marker='o', linestyle='-', markersize=6, color='green', label='Average Value', linewidth=2.5)

# # 添加均值线（水平虚线）
# overall_mean = np.nanmean(np.array(result_list))
# plt.axhline(overall_mean, color='black', linestyle='--', label='Overall Average Value', linewidth=2.5)

# 设置横坐标标签
plt.xticks(range(len(mileage_range)), mileage_range)
plt.ylim(0.97, 1)
# 显示图例
plt.legend(fontsize=26)


# 设置横坐标和纵坐标标签
plt.xlabel('Mileage (10³ km)', fontdict=font)
plt.ylabel('ζ', fontdict=font)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α4.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α4.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α4.pdf', format='pdf', bbox_inches='tight')
plt.close()




#########################################################α5

# 计算里程/1000，并划分到不同的区间
indicator_list['mileage_k'] = indicator_list['mileage'] / 1000  # 转换为千公里
mileage_range = np.arange(50, 325, 25)  # 定义区间

# 使用pd.cut将mileage_k划分到这些区间
indicator_list['mileage_category'] = pd.cut(indicator_list['mileage_k'], bins=mileage_range, include_lowest=True, right=False)

# 创建图表
plt.figure(figsize=(12, 8))

# 绘制小提琴图
sns.violinplot(x='mileage_category', y='svrα5', data=indicator_list, palette="coolwarm")

# 设置标签和字体属性
plt.xlabel('Mileage (10³ km)', fontsize=26)
plt.ylabel('η', fontsize=26)
plt.xticks(rotation=45)  # 旋转x轴标签以便更好阅读
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
# 显示图表
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()


plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α5.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α5.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α5.pdf', format='pdf', bbox_inches='tight')
plt.close()


#%% 最终的α_Overall capacity utilization
# Set font properties
indicator_list=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/5.0 Transform data/Results/'+'final_α'+'.csv')


# 绘制小提琴图
# 使用seaborn的violinplot来显示所有车辆的fixed_α3值的分布
sns.violinplot(y=indicator_list['α'], color='royalblue')

# 设置坐标轴标签和字体属性
plt.xlabel('NMC_Vehicles', fontsize=26)  # 可以留空或者标记为'All Vehicles'
plt.ylabel('θ', fontsize=26)

plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
# 显示图表
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Figure 3/'+'NMC_α.pdf', format='pdf', bbox_inches='tight')
plt.close()   


###补充出图
alpha_values = indicator_list['α'].unique().tolist()

# 创建指定大小的图表
plt.figure(figsize=(12, 8))  # 设置图形大小
# 计算α的平均值
average_utilization = np.mean(alpha_values)

# 添加平均值虚线
plt.axhline(average_utilization, color='red', linestyle='--', linewidth=3, label=f'Average Utilization Rate: {average_utilization:.4f}')



plt.bar(VS, alpha_values, color='royalblue')  # 绘制柱状图
plt.xlabel('Vehicles', fontdict=font)  # 设置横坐标标签
plt.ylabel('θ', fontdict=font)

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(0, 1)
plt.legend(fontsize=font_size)  # 显示图例
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()

plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α.png', format='png', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α.svg', format='svg', bbox_inches='tight')
plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/2.0 Figures for manuscript/Supplementary Figures/'+'NMC_α.pdf', format='pdf', bbox_inches='tight')
plt.close()   