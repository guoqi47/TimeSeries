# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime, timedelta
import tushare as ts
from matplotlib.pylab import datestr2num
import csv
import numpy as np
import pandas as pd

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
#    fileHeader = ["start", "end","duration","slope"]
    datacsv = open("data.csv","w",newline="")
    csvwriter = csv.writer(datacsv,dialect = ("excel"))
    result=[date2[0]]+sorted(result)+[date2[-1]]
    # 画图部分
    start1=datetime.strptime(start, "%Y-%m-%d")
    for i in range(0,len(result)-1):
        ii=str(start1+timedelta(result[i]-datestr2num(start)))[0:10] #数字转化为日期
        
        ii1=str(start1+timedelta(result[i+1]-datestr2num(start)))[0:10]
        p0=closePrice[ii]
        p1=closePrice[ii1]
        plt.plot_date([result[i],result[i+1]],
                        [p0,p1], 'r-')
        #保存为csv格式
        csvwriter.writerow([ii,ii1,date2.index(result[i+1])-date2.index(result[i]),(p1-p0)/(result[i+1]-result[i])])
#        print(ii,ii1,date2.index(result[i+1])-date2.index(result[i]),(p1-p0)/(result[i+1]-result[i]))
        
        
if __name__ =='__main__':
    L=1
    
    sns.set_style("whitegrid")
    start='2015-10-31'
    end='2017-10-31'

            
    data1 = ts.get_h_data('002253', '2014-10-31', '2017-10-31')  # 三年
    data2 = ts.get_h_data('002253', '2011-10-31', '2014-10-30')  # 两年   
    
    data = pd.concat([data1, data2], axis=0)
    closePrice=data.close[::-1] #收盘价
#    closePrice2=data2.close[::-1] #收盘价
#    closePrice=closePrice2+closePrice1
    date11=data.index[::-1].tolist() #日期
#    date12=data2.index[::-1].tolist() #日期
    date11=[str(i)[:10] for i in date11]
#    date12=[str(i)[:10] for i in date12]
    
#    date11 = date12[:-1]+date11
    global date2
    date2 = [datestr2num(i) for i in date11] #将日期转为数字进行坐标表示

    plt.gcf().set_size_inches(12,4)
    plt.plot_date(date2,closePrice,'b-')

    fun(date2[0],date2[-1],L)

   