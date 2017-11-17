# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tushare as ts
from matplotlib.pylab import datestr2num
import culKDJ

        
        
if __name__ =='__main__':
    L=1
    count=0
    DM='000002'
    global csvInput
    csvInput=[]
    
    sns.set_style("whitegrid")
    end = datetime.today() #开始时间结束时间，选取最近一年的数据
    start = datetime(end.year,end.month-3,end.day)
    end = str(end)[0:10]
    start = str(start)[0:10]
    
    getData=ts.get_hist_data(DM,start,end)
    closePrice=getData.close #收盘价
    closePrice=closePrice[::-1] #按日期从低到高
    date1=getData.index #日期
    date1 = date1[::-1] #按日期从低到高
    date2 = [datestr2num(i) for i in date1] #将日期转为数字进行坐标表示

    plt.gcf().set_size_inches(12,4)
    plt.plot_date(date2,closePrice,'b-')

    #买卖的的判断画图
    chiyou = [] #预计持有日期
    uptrend = [] #预计上升日期
    downtrend = [] #预计下跌日期
    chiyouPrice = [] #持有价钱
    uptrendPrice = [] #上升价钱
    downtrendPrice = [] #下跌价钱
    for i in date1:
        cul,count1 = culKDJ.cul_KDJ(DM,i,9)
        count+=count1
        if cul==0:
            chiyou.append(i)
        elif cul==1:
            uptrend.append(i)
        else:
            downtrend.append(i)
            
    for i in chiyou:
        chiyouPrice.append(closePrice[i])
    for i in uptrend:
        uptrendPrice.append(closePrice[i])
    for i in downtrend:
        downtrendPrice.append(closePrice[i])
    plt.scatter(chiyou,chiyouPrice,c='k',marker='^')
    plt.scatter(uptrend,uptrendPrice,c='k',marker='+')
    plt.scatter(downtrend,downtrendPrice,c='k',marker='o')
    print('正确率：',round(count/len(date2),2))
    
    
        
       