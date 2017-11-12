# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime, timedelta
import tushare as ts
from matplotlib.pylab import datestr2num
import culKDJ


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
    for i in range(0,len(result)-1):
        ii=str(start1+timedelta(result[i]-datestr2num(start)))[0:10] #数字转化为日期
        ii1=str(start1+timedelta(result[i+1]-datestr2num(start)))[0:10]
        p0=closePrice[ii]
        p1=closePrice[ii1]
        plt.plot_date([result[i],result[i+1]],
                        [p0,p1], 'r-')
        
if __name__ =='__main__':
    L=1
    count=0
    DM='000625'
    
    sns.set_style("whitegrid")
    end = datetime.today() #开始时间结束时间，选取最近一年的数据
    start = datetime(end.year,end.month-2,end.day)
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

#    fun(date2[0],date2[-1],L)
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
    
    
        
       