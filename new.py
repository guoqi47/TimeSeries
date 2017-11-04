# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import json
from datetime import datetime
import tushare as ts
from matplotlib.pylab import datestr2num
#import numpy as np

def culDistence(x1,y1,x2,y2,x3,y3):
    A=y2-y1
    B=x1-x2
    C=x2*y1-x1*y2
    dis=abs(A*x3+B*y3+C)/math.sqrt(A*A+B*B)
    return dis
    
def a(x1,x2,L):
    d=[]
    
    for i in range(x1,x2):
        d.append(i)
        d.append(culDistence(x1,stockPrice[x1*2+1],x2,stockPrice[x2*2+1],i,stockPrice[i*2+1]))
#    index=d.index(max(d))
    if len(d)==0:
        return
    d1=[d[i] for i in range(1,len(d),2)] #不包含索引的list
    maxD = max(d1)
    if maxD<L:
        return
    else:
        index=d[d.index(maxD)-1]
        result.append(index)

        a(x1,index,L)
        a(index,x2,L)
        
def fun(x1,x2,L):
    global result
    result=[]
    a(x1,x2,L)
    
    result=[0]+sorted(result)+[int(len(stockPrice)/2)-1]
    # 画图部分
    for i in range(0,len(result)-1):
        plt.plot_date([datestr2num(date1[-result[i]-1]),datestr2num(date1[-result[i+1]-1])],
#                       [stockPrice[result[i]*2+1],stockPrice[result[i+1]*2+1]], 'r-')
                        [closePrice[-result[i]-1],closePrice[-result[i+1]-1]], 'r-')
    plt.show()
    
    profitRate = 1
    for i in range(0,len(result)-1):
        p1=stockPrice[result[i+1]*2+1]
        p0=stockPrice[result[i]*2+1]
        if p1-p0>0:
#                profitRate *= (2*p1-p0)*(1-Fee)/p1
            # 改进计算利润率算法
            profitRate *= 1+((1-Fee)*p1-(1+Fee)*p0)/((1+Fee)*p0)   
    print(profitRate)
        
if __name__ =='__main__':
    stockPrice=[]
    stockPrice1=[]
    L=1
    Fee=0.01 #每笔交易手续费
    
    sns.set_style("whitegrid")
    end = datetime.today() #开始时间结束时间，选取最近一年的数据
    start = datetime(end.year,end.month-6,end.day)
    end = str(end)[0:10]
    start = str(start)[0:10]
    
    stock = ts.get_hist_data('002253',start,end)#选取一支股票
    stock.to_json('stock.json',orient='records')#转化为json格式
    with open('stock.json', 'r') as f:
        data = json.load(f)
#    for p in data[:]:
#        stockPrice.append(p['open'])

    
    closePrice=ts.get_hist_data('002253',start,end).close
    date1=ts.get_hist_data('002253',start,end).index #日期

    date2 = [datestr2num(i) for i in date1] #将日期转为数字进行坐标表示

    for p in range(len(data)-1,-1,-1): #调换顺序，使成为随时间增长变化的曲线
        stockPrice1.append(data[p]['close'])
    for p in range(0,len(stockPrice1)):
        stockPrice.append(p)
        stockPrice.append(stockPrice1[p])
        
#    stock['close'].plot(legend=False ,figsize=(12,4)) #原画图
    plt.gcf().set_size_inches(12,4)
    plt.plot_date(date2,closePrice,'b-')
#    plt.plot([stockPrice[i] for i in range(0,22,2)],[stockPrice[i] for i in range(1,22,2)],'r-',label="point")
#    散点图
#    axes = plt.subplot(111)
#    type1 = axes.scatter([stockPrice[i] for i in range(0,42,2)],[stockPrice[i] for i in range(1,42,2)], s=20, c='red')

    fun(0,int(len(stockPrice)/2)-1,L)

   