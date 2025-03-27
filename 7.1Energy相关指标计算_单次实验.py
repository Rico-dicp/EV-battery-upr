# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:53:54 2024

@author: 13121

针对前两个指标补充计算能量，读取单次的完整数据，按次序获取：1）当次能量 2）估计的完整能量 3）出图观察指标一与指标二的结果
代码模板：计算参数指标的程序

对每个单体单次读数，补全三个参数：energy1(根据辨识结果加等效电路模型前期补全),energy2（直接计算），energy3(尾巴补全)的值，全部加和
可能续驶里程要重点分析

就用原始辨识数据，可以算完了再整体做修正

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times new Roman'] 
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'
from tqdm import tqdm
import pyswarms as ps
from scipy import stats
from sklearn.svm import SVR

#SOC函数
fit_params = [2.79349202e+00,  1.32456734e-01, -1.06408510e-02,  4.87568943e-04,
       -1.21132612e-05,  1.21430353e-07,  1.19820875e-09, -4.82818206e-11,
        5.64142933e-13, -3.06234456e-15,  6.54961010e-18]

# 定义十次多项式函数
def polynomial_func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6 + a7*x**7 + a8*x**8 + a9*x**9 + a10*x**10

# 定义可调用的拟合函数
def fitted_function(x,fit_params):
    return polynomial_func(x, *fit_params)



# names=['LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
#        'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
#        'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
#        'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']


# for name in names:
#     data=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'data.csv', index_col=0)
#     tbl04=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'tbl04.csv', index_col=0)


name='LB378Y4W1JA175086'

data=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'data.csv', index_col=0)
tbl04=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'tbl04.csv', index_col=0)
reference=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/results need to be fixed_filted.csv')
reference=reference[reference['vin']==name]

cp_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'cp_df.csv')
rp_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'rp_df.csv')
soc_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'soc_df.csv')
jiezhiSOC_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'jiezhiSoc_df.csv')
Vp0_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'Vp0_df.csv')

# for m in reference['片段编号']:

#预设单体电池编号n,循环次数编号m

m=548


# for n in range(95):
n=5

soh=reference[reference['片段编号']==m]['estimated_soh_cell'+str(n)].iloc[0]
capshangxian=soh*150

r0=reference[reference['片段编号']==m]['estimated_r0_cell'+str(n)].iloc[0]

cp=cp_df[cp_df['0']==m][str(n+2)].iloc[0]
rp=rp_df[rp_df['0']==m][str(n+2)].iloc[0]
qishiSOC=soc_df[soc_df['0']==m][str(n+2)].iloc[0]
jiezhiSOC=jiezhiSOC_df[jiezhiSOC_df['0']==m][str(n+2)].iloc[0]
#这个截止SOC，截止到哪？截止到充满：据我观察截止SOC在99到102，不需要考虑energy3了，只计算第一段和有数据的第二段就行,起始SOC对应375，自己往前捋，需要考虑rnergy2

Vp0=Vp0_df[Vp0_df['0']==m][str(n+2)].iloc[0]


#提取对应的片段参数
chakan=data[data['number']==m]
cap1=chakan[chakan['总电压']<375]['单帧容量变化(A·h)'].sum()


# 创建一个新DataFrame，包含600行,600不够，加大到1200
num_new_rows = 1200
new_data = pd.DataFrame(np.zeros((num_new_rows, len(chakan.columns))), columns=chakan.columns)
# 设置新行中‘总电流’的值为chakan['总电流'].iloc[2]
new_data['总电流'] = chakan['总电流'].iloc[2]
new_data['单帧容量变化(A·h)'] = -10*chakan['总电流'].iloc[2]/3600
# 将新行添加到原DataFrame的前面
chakan = pd.concat([new_data, chakan], ignore_index=True)
# 计算‘单帧容量变化(A·h)’的累计和并添加到‘cap’列
chakan['cap'] = chakan['单帧容量变化(A·h)'].cumsum()+soh*150*qishiSOC/100-cap1-new_data['单帧容量变化(A·h)'].sum()
chakan=chakan[chakan['cap']>0]

num_new_rows = 600
new_data = pd.DataFrame(np.zeros((num_new_rows, len(chakan.columns))), columns=chakan.columns)
# 设置新行中‘总电流’的值为chakan['总电流'].iloc[2]
new_data['总电流'] = chakan['总电流'].iloc[-2]
new_data['单帧容量变化(A·h)'] = -10*chakan['总电流'].iloc[-2]/3600
chakan = pd.concat([chakan,new_data], ignore_index=True)
chakan['cap'] = chakan['单帧容量变化(A·h)'].cumsum()
chakan=chakan[chakan['cap']<=soh*150]


# 将‘cap’和‘总电流’转换为2D数组
cap = chakan['cap'][:, np.newaxis]
total_current = chakan['总电流'][:, np.newaxis]
  


cell_voltage =  chakan['单体电压'+str(n)][:, np.newaxis]


u = fitted_function( cap / (soh * 150)*100, fit_params) - (-total_current*rp-Vp0)*np.exp(-np.divide(3600*cap, -rp* cp* total_current))-total_current*(r0+rp)

chakan['拟合cell电压']=u
chakan.loc[chakan['单体电压'+str(n)]==0,'计算用电压']=chakan['拟合cell电压']
chakan.loc[chakan['单体电压'+str(n)]>0,'计算用电压']=0.001*chakan['单体电压'+str(n)]
chakan['energy']=(chakan['计算用电压']*chakan['单帧容量变化(A·h)']/1000).cumsum()
filtered_energy = chakan[chakan['单体电压1'] == 0]
if not filtered_energy.empty:
   energy1 = (filtered_energy['计算用电压']*filtered_energy['单帧容量变化(A·h)']/1000).sum()


   
else:
    energy1 = 0  # 或者你想要的默认值
             
energy=chakan['energy'].iloc[-1]
energy2=energy-energy1
import matplotlib.pyplot as plt
import pandas as pd

# 假设你的DataFrame名为chakan
# chakan = pd.read_csv('path_to_your_file.csv')

# 创建一个序列作为横坐标
x = range(1, len(chakan) + 1)
fig, ax1 = plt.subplots()

# 绘制第一个纵坐标轴
ax1.set_xlabel('Index')
ax1.set_ylabel('Cell voltage'+str(n)+' (mV)', color='tab:blue')
ax1.plot(x, 1000*u, color='tab:orange')
ax1.plot(x, chakan['单体电压'+str(n)], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 创建第二个纵坐标轴，共享x轴
ax2 = ax1.twinx()
ax2.set_ylabel('Current (A)', color='tab:red')
ax2.plot(x, chakan['总电流'], color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')


# 显示图形
plt.show()



#循环出图验(不循环了，直接批量计算，出现问题倒查结果)


name='LB378Y4W1JA175086'
energy_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/'+name+'energy_df'+'.csv')
# energy_df1=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/'+name+'energy_df1'+'.csv')
# energy_df2=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/'+name+'energy_df2'+'.csv')
# plt.plot(energy_df['1'],energy_df['97'])
# plt.plot(energy_df1['1'],energy_df1['97'])
# plt.plot(energy_df2['1'],energy_df2['97'])
# 假设 energy_df 是你的 DataFrame，第一列是 x 轴数据
x = energy_df['1']

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 遍历列索引 3 到 97
for col in range(3, 98):
    plt.plot(x, energy_df[str(col)], label=f'Column {col}')

# 添加图例
plt.legend(loc='upper right')

# 添加标题和轴标签
plt.title('Energy Data Plot')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# 显示图形
plt.show()


####看一下随着老化的进行，这些e参数对SOH比值有没有长进
name='LB378Y4W0JA174527'
energy_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/'+name+'energy_df'+'.csv')

reference=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/results need to be fixed_filted.csv')
reference=reference[reference['vin']==name]


fenmu=energy_df['3']
fenzi=reference['estimated_soh_cell1']
chakan=1000*fenmu/(fenzi*150)

