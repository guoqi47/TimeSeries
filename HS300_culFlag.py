from datetime import datetime
import tushare as ts

def culFlag(DM,days,cycle=9): 
    cycle -=1
    C,L,H=[],[],[] #周期内每日收盘价,每日最低价,每日最高价
    K,D,J=[],[],[]
    RSV=[] #股票开盘日起每日RSV
    threshold=8 #判断金死叉的阈值
    data = ts.get_hist_data(DM)
    dateList = [i for i in data.index][::-1] #日期list
    C = [i for i in data['close'][::-1]] #第n日收盘价
    L = [i for i in data['low'][::-1]]
    H = [i for i in data['high'][::-1]]
    
    RSV.append((C[0]-L[0])/(H[0]-L[0])*100)
    for i in range(1,len(C)):
        if i<cycle:
            t=0
        else:
            t=i-cycle
        RSV.append((C[i]-min(L[t:i+1]))/(max(H[t:i+1])-min(L[t:i+1]))*100)
    
    for i in range(3):
        K.append(100)
        D.append(100)
        J.append(100)
    for i in range(3,len(C)):
        K.append(K[i-1]*2/3 + RSV[i]/3)
        D.append(D[i-1]*2/3 + K[i]/3)
        J.append(K[i]*3 - 2*D[i])
    
    flag=[]
    for i in range(len(K)):
        K1=K[i]
        D1=D[i]
        J1=J[i]
        if abs(J1-D1)<threshold: #近金叉或者死叉
            if J1-D1>0: 
                if J[i-1]-D[i-1]>J1-D1: #即将死叉
                    flag.append(-1)
                    continue
                if J[i-1]-D[i-1]<threshold:
                    flag.append(1)
                    continue
                else:
                    flag.append(0)
                    continue
            elif J1<D1 and D[i-1]-J[i-1]>threshold: #即将金叉
                flag.append(1)
                continue
            else:
                flag.append(-1)
                continue
        if J1>D1: #要涨，K线总是在中间,故简化
            flag.append(1)
            continue
        elif J1<D1: #要跌
            flag.append(-1)
            continue
        else:  #持有
            flag.append(0)
            continue
    return C[-days:],flag[-days:]

if __name__=="__main__":
#    #cul_KDJ(DM,date,days)
#    #DM:股票代码 days:返回结果的天数
    C,Flag = culFlag('002253',500)
    print(len(C))
    

