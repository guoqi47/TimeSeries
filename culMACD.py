# -*- coding: utf-8 -*-
from datetime import datetime
import tushare as ts

def cul_MACD(DM,date):
    C,EMA12,EMA26=[],[],[] #每日收盘价
    DIFF,DEA,MACD=[0],[0],[0]
    data = ts.get_hist_data(DM)
    dateList = [i for i in data.index][::-1] #日期list
    C = [i for i in data['close'][::-1]] #第n日收盘价
    EMA12.append(C[0]) #EMA12初始化为第一天的收盘价
    EMA26.append(C[0])
    for i in range(1,len(C)):
        EMA12.append(EMA12[i-1]*11/13+C[i]*2/13)
        EMA26.append(EMA26[i-1]*25/27+C[i]*2/27)
        DIFF.append(EMA12[i]-EMA26[i])
        DEA.append(DEA[i-1]*0.8+DIFF[i]*0.2)
        MACD.append((DIFF[i]-DEA[i])*2)
        
    #要获取的数据在MACD list的索引
    index = dateList.index(str(datetime.strptime(date, "%Y-%m-%d"))[:10]) 
    print('股票代码:',DM,'日期：',date)
    MACD1=MACD[index]
    MACD2=MACD[index-1]
    DIFF1=DIFF[index]
    DEA1=DEA[index]
    print('MACD:',round(MACD1,2))
    print('DIFF:',round(DIFF1,2))
    print('DEA:',round(DEA1,2))
    
    #买入卖出判断，-1看跌，1看涨，0持有
    if MACD1<0 and MACD2>0:
        print(-1)
        return -1
    elif MACD1>0 and MACD2<0:
        print(1)
        return 1
    else:
        print(0)
        return 0
    

    
if __name__=="__main__":
#    #cul_KDJ(DM,date,days,num)
#    #DM:股票代码 date:目标日期 days:周期 num:KDJ均为100的天数
    cul_MACD('300691','2017-09-21')