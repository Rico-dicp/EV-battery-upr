# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:44:29 2024

@author: 13121

增补目标：读取SOH_df表，构建类似的energy_df表：要素为vin名，里程，energy总,energy1,energy2,便于后续进行指标计算/温度修正

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times new Roman'] 
plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'
from tqdm import tqdm


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


# 
names=[ 'LB378Y4W0JA174527', 'LB378Y4W1JA175086', 'LB378Y4W1JA179350', 'LB378Y4W3JA179379', 'LB378Y4W4JA177348', 
       'LB378Y4W4JA179259', 'LB378Y4W5JA179156', 'LB378Y4W6JA179408', 'LB378Y4W7JA175268', 'LB378Y4W7JA177862', 
       'LB378Y4W7JA178669', 'LB378Y4W7JA179725', 'LB378Y4W8JA175280', 'LB378Y4W8JA177207', 'LB378Y4W8JA179782', 
       'LB378Y4W9JA173988', 'LB378Y4W9JA174980', 'LB378Y4W9JA176518', 'LB378Y4WXJA173319']


# names=['LB378Y4W0JA174527', 'LB378Y4W1JA175086']


for name in tqdm(names):
    
    energy_df =[]
    energy_df1 =[]
    energy_df2 =[]
    
    data=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'data.csv', index_col=0)
    tbl04=pd.read_csv('G:/DBD4数据/DIHAO,参数提取片段与数据/'+name+'tbl04.csv', index_col=0)
    reference=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/results need to be fixed_filted.csv')
    reference=reference[reference['vin']==name]
    
    cp_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'cp_df.csv')
    rp_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'rp_df.csv')
    soc_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'soc_df.csv')
    jiezhiSOC_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'jiezhiSoc_df.csv')
    Vp0_df=pd.read_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Vehicles/'+name+'Vp0_df.csv')

    for m in tqdm(reference['片段编号']):
        energy_list=[m, reference[reference['片段编号']==m]['start_mile'].iloc[0], reference[reference['片段编号']==m]['temp_mean'].iloc[0]]
        energy_list1=[m, reference[reference['片段编号']==m]['start_mile'].iloc[0], reference[reference['片段编号']==m]['temp_mean'].iloc[0]]
        energy_list2=[m, reference[reference['片段编号']==m]['start_mile'].iloc[0], reference[reference['片段编号']==m]['temp_mean'].iloc[0]]
        for n in range(1,96):
          soh=reference[reference['片段编号']==m]['estimated_soh_cell'+str(n)].iloc[0]
          r0=reference[reference['片段编号']==m]['estimated_r0_cell'+str(n)].iloc[0]
          cp=cp_df[cp_df['0']==m][str(n+2)].iloc[0]
          rp=rp_df[rp_df['0']==m][str(n+2)].iloc[0]
          qishiSOC=soc_df[soc_df['0']==m][str(n+2)].iloc[0]
          jiezhiSOC=jiezhiSOC_df[jiezhiSOC_df['0']==m][str(n+2)].iloc[0]
          #这个截止SOC，截止到哪？截止到充满：据我观察截止SOC在99到102，不需要考虑energy3了，只计算第一段和有数据的第二段就行,起始SOC对应375，自己往前捋
          Vp0=Vp0_df[Vp0_df['0']==m][str(n+2)].iloc[0]


          #提取对应的片段参数
          chakan=data[data['number']==m]
          cap1=chakan[chakan['总电压']<375]['单帧容量变化(A·h)'].sum()
          # 创建一个新DataFrame，包含600行，600不够，加到1500吧，不然算的能量不对
          num_new_rows = 1500
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
          
          energy_list.append(energy)
          energy_list1.append(energy1)
          energy_list2.append(energy2)

        energy_df.append(energy_list)
        energy_df1.append(energy_list1)
        energy_df2.append(energy_list2)
        
    energy_df = pd.DataFrame(energy_df)
    energy_df1 = pd.DataFrame(energy_df1)
    energy_df2 = pd.DataFrame(energy_df2)
        
    energy_df.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/'+name+'energy_df'+'.csv',index=False)
    energy_df1.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/'+name+'energy_df1'+'.csv',index=False)
    energy_df2.to_csv('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/'+name+'energy_df2'+'.csv',index=False)
        
    # return energy_df,energy_df1,energy_df2


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
    
    plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/Estimated V,末次片段，95号单体'+name+'.png')
    plt.close()
    

    x = energy_df[1]
    fig, ax1 = plt.subplots()
    # 绘制第一个纵坐标轴
    ax1.set_xlabel('Mileage (km)')
    ax1.set_ylabel('Energy(kh)')
    for col in range(3, 98):
        ax1.plot(x, energy_df[col],color='green')
    # 创建第二个纵坐标轴，共享x轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('SOH')
    for col in range(1, 96):
        ax2.plot(x, reference['estimated_soh_cell'+str(col)],color='blue')
        # 显示图形
    plt.show()
    
    plt.savefig('E:/My Craft/Lab/论文_zlt/DBD 4/3.0 Source code and figures/Story 1/Energy增补数据/Energy Trend'+name+'.png')
    plt.close()