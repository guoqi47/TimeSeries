# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import json
from datetime import datetime, timedelta
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
    x1=int(x1)
    x2=int(x2)
    for i in range(x1,x2):
        if(i not in date2):
            continue
        d.append(i)

        start1=datetime.strptime(start, "%Y-%m-%d")
        t1=str(start1+timedelta(x1-datestr2num(start)))[0:10]
        t2=str(start1+timedelta(x2-datestr2num(start)))[0:10]
        ti=str(start1+timedelta(i-datestr2num(start)))[0:10]

        d.append(culDistence(x1,closePrice[t1],x2,closePrice[t2],
                                           i,closePrice[ti]))
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
    
    result=[date2[0]]+sorted(result)+[date2[-1]]
    # 画图部分
    start1=datetime.strptime(start, "%Y-%m-%d")
    profitRate = 1
    for i in range(0,len(result)-1):
        ii=str(start1+timedelta(result[i]-datestr2num(start)))[0:10] #数字转化为日期
        ii1=str(start1+timedelta(result[i+1]-datestr2num(start)))[0:10]
        p0=closePrice[ii]
        p1=closePrice[ii1]
        plt.plot_date([result[i],result[i+1]],
                        [p0,p1], 'r-')
        if p1>p0:
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

    closePrice=ts.get_hist_data('002253',start,end).close #收盘价
    closePrice=closePrice[::-1] #按日期从低到高
    date1=ts.get_hist_data('002253',start,end).index #日期
    date1 = date1[::-1] #按日期从低到高
    date2 = [datestr2num(i) for i in date1] #将日期转为数字进行坐标表示

    plt.gcf().set_size_inches(12,4)
    plt.plot_date(date2,closePrice,'b-')

    fun(date2[0],date2[-1],L)

   