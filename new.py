# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import json
from datetime import datetime
import tushare as ts
#import numpy as np

def culDistence(x1,y1,x2,y2,x3,y3):
    A=y2-y1
    B=x1-x2
    C=x2*y1-x1*y2
    dis=abs(A*x3+B*y3+C)/math.sqrt(A*A+B*B)
    return dis
    
def a(x1,x2,L):
    d=[]
    
    
    for i in range(x1,x2,2):
        d.append(i)
        d.append(culDistence(stockPrice[x1],stockPrice[x1+1],stockPrice[x2],stockPrice[x2+1],stockPrice[i],stockPrice[i+1]))
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

        a(x1,int(index),L)
        a(int(index),x2,L)
        
def fun(x1,x2,L):
    global result
    result=[]
    a(x1,x2,L)
    
    result=[0]+sorted(result)+[int(len(stockPrice)/2)-1]
    # 画图部分
    for i in range(0,len(result)-1):
        plt.plot([result[i],result[i+1]], [stockPrice[result[i]*2+1],stockPrice[result[i+1]*2+1]], 'r-')
    plt.show()
    
    # 计算盈利率
    profitRate = 1
    for i in range(0,len(result)-1):
        p1=stockPrice[result[i+1]*2+1]
        p0=stockPrice[result[i]*2+1]
        if p1-p0>0:
            profitRate *= (2*p1-p0)*(1-Fee)/p1
            
    print(profitRate)
        
if __name__ =='__main__':
    stockPrice=[]
    stockPrice1=[]
    L=1
    Fee=0.01 #每笔交易手续费
#    global result #点集
#    result=[]
    
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

    for p in range(len(data)-1,-1,-1): #调换顺序，使成为随时间增长变化的曲线
        stockPrice1.append(data[p]['close'])
    for p in range(0,len(stockPrice1)):
        stockPrice.append(p)
        stockPrice.append(stockPrice1[p])
        
#    stock['close'].plot(legend=False ,figsize=(12,4)) #原画图
    plt.gcf().set_size_inches(12,4)
    plt.plot([stockPrice[i] for i in range(0,len(stockPrice),2)],[stockPrice[i] for i in range(1,len(stockPrice),2)],'b-')


    fun(0,int((len(stockPrice)+1)/2),L)
    
#    result=[0]+sorted(result)+[int(len(stockPrice)/2)-1]
#    # 画图部分
#    for i in range(0,len(result)-1):
#        plt.plot([result[i],result[i+1]], [stockPrice[result[i]*2+1],stockPrice[result[i+1]*2+1]], 'r-')
#    plt.show()
#    
#    # 计算盈利率
#    profitRate = 1
#    for i in range(0,len(result)-1):
#        p1=stockPrice[result[i+1]*2+1]
#        p0=stockPrice[result[i]*2+1]
#        if p1-p0>0:
#            profitRate *= (2*p1-p0)*(1-Fee)/p1
#    print(profitRate)

    
    