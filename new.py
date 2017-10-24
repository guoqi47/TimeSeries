# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
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
    
def a(x1,x2):
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
        
#        plt.plot([x1,index], [stockPrice[x1+1],maxD], 'r-')
#        plt.plot([index,x2],[maxD,stockPrice[x2+1]], 'r-')
        a(x1,int(index))
        a(int(index),x2)
        
        
if __name__ =='__main__':
    stockPrice=[]
    L=0.5
    global result #点集
    result=[]
    
    sns.set_style("whitegrid")
    end = datetime.today() #开始时间结束时间，选取最近一年的数据
    start = datetime(end.year-1,end.month,end.day)
    end = str(end)[0:10]
    start = str(start)[0:10]
    
    stock = ts.get_hist_data('002253',start,end)#选取一支股票
    stock.to_json('stock.json',orient='records')#转化为json格式
    with open('stock.json', 'r') as f:
        data = json.load(f)
#    for p in data[:]:
#        stockPrice.append(p['open'])
    for p in range(0,len(data)):
        stockPrice.append(p)
        stockPrice.append(data[p]['close'])
        
    stock['close'].plot(legend=True ,figsize=(12,4))
    
    a(0,len(stockPrice)-2)
    
#    resultSort=sorted(result,reverse=True)
#    for i in range(0,10):
#        X.append(stockPrice[result.index(resultSort[i])]) #时间的值
#        X.append(resultSort[i])
    
    result=[0]+sorted(result)+[len(stockPrice)-1]
    for i in range(0,len(result)-1):
        plt.plot([result[i],result[i+1]], [stockPrice[result[i]*2+1],stockPrice[result[i+1]*2+1]], 'r-')
    
    plt.show()
    
    
