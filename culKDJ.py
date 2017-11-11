# -*- coding: utf-8 -*-
from datetime import datetime
import tushare as ts

def cul_KDJ(DM,date,days):
    days -=1
    C=[] #周期内每日收盘价
    L=[] #周期内每日最低价
    H=[] #周期内每日最高价
    K=[]
    D=[]
    J=[]
    RSV=[] #股票开盘日起每日RSV
    data = ts.get_hist_data(DM)
    dateList = [i for i in data.index][::-1] #日期list
    C = [i for i in data['close'][::-1]] #第n日收盘价
    L = [i for i in data['low'][::-1]]
    H = [i for i in data['high'][::-1]]
    
    RSV.append((C[0]-L[0])/(H[0]-L[0])*100)
    for i in range(1,len(C)):
        if i<days:
            t=0
        else:
            t=i-days
        RSV.append((C[i]-min(L[t:i+1]))/(max(H[t:i+1])-min(L[t:i+1]))*100)
    
    for i in range(3):
        K.append(100)
        D.append(100)
        J.append(100)
    for i in range(3,len(C)):
        K.append(K[i-1]*2/3 + RSV[i]/3)
        D.append(D[i-1]*2/3 + K[i]/3)
        J.append(K[i]*3 - 2*D[i])
    
    #要获取的数据在Klist的索引
    index = dateList.index(str(datetime.strptime(date, "%Y-%m-%d"))[:10]) 
    print('股票代码:',DM,'日期：',date)
    print('K:',round(K[index],2))
    print('D:',round(D[index],2))
    print('J:',round(J[index],2))
    
    #买入卖出判断，-1卖出，1买进，0持有
    if J[index]-D[index]<5: #近金叉或者死叉
        if J[index-1]>D[index-1]: 
            print(-1)
            return -1
        elif J[index-1]<D[index-1]: 
            print(1)
            return 1
    elif J[index]>D[index]: #要涨，K线总是在中间,故简化
        print(1)
        return 1
    elif J[index]<D[index]: #要跌
        print(-1)
        return -1
    else:  #持有
         print(0)
         return 0
  
    
if __name__=="__main__":
    #cul_KDJ(DM,date,days,num)
    #DM:股票代码 date:目标日期 days:周期 num:KDJ均为100的天数
    cul_KDJ('002253','2017-11-01',9)

    