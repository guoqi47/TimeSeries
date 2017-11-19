# -*- coding: utf-8 -*-
import HS300_culFlag
import tushare as ts

def train(HS300Code):
    finalRate=0 #300支股票总收益率
    finalRateDrawdown=0 #考虑回撤
    holdRate=0 #持有收益
    count=0 #买卖次数
    countCorrect=0 #成功买卖次数
    maxDrawdown=0 #最大回撤
    sumMaxDrawdown=0 #最大回撤和
    for code in HS300Code:
        profitRate = 1 #收益率
        profitRateDrawdown = 1 #考虑回撤收益率
        C,Flag = HS300_culFlag.culFlag(code,500) #取前500-400天
        C=C[-500:-50]
        Flag=Flag[-500:-50]
        chiyou=False
        chiyouDrawdown=False
        for i in range(len(Flag)):
            if not(chiyou) and Flag[i]==1: #买进
                chiyou=True
                buyPrice=C[i]
            elif chiyou and Flag[i]==-1: #卖出
                chiyou=False
                count+=1
                sellPrice=C[i]
#                fund = fund*sellPrice/buyPrice - fund*0.001 
                profitRate *= (sellPrice/buyPrice - 0.001)
                if sellPrice>buyPrice:
                    countCorrect+=1
            if i>0: #求最大回撤
                drawdown=C[i]-C[i-1]
                if drawdown>maxDrawdown:
                    maxDrawdown=drawdown
                    
            #计算加入回撤后的情形,i也要大于零
                threshold=sumMaxDrawdown/(2*len(HS300Code))
                if not(chiyouDrawdown) and Flag[i]==1: #买进
                    chiyouDrawdown=True
                    buyPrice=C[i]
                elif chiyouDrawdown and (C[i-1]-C[i]>threshold or Flag[i]==-1): #卖出
                    chiyouDrawdown=False
                    sellPrice=C[i]
                    profitRateDrawdown *= (sellPrice/buyPrice - 0.001)
            
        sumMaxDrawdown+=maxDrawdown 
        finalRate+=profitRate
        finalRateDrawdown+=profitRateDrawdown
        
        holdRate += (C[-1]/C[0])
    print('根据单一指标KDJ进行模拟交易（取300支股票即日起前500至前450个交易日的数据）：')
    print('300支股票平均收益率：',round(finalRate/len(HS300Code),2))
    print('成功交易率(卖出价格大于买入价格)：',round(countCorrect/count,2))
    print('300支股票持有收益率：',round(holdRate/len(HS300Code),2))
    print('300支股票平均最大回撤：',round(sumMaxDrawdown/len(HS300Code),2))
    print('考虑止损后的收益率：',round(finalRateDrawdown/len(HS300Code),2))
    print()
    return sumMaxDrawdown/(2*len(HS300Code))

def CV(HS300Code,Drawdown):
    finalRate=0 #300支股票总收益率
    finalRateDrawdown=0 #考虑回撤
    holdRate=0 #持有收益
    count=0 #买卖次数
    countCorrect=0 #成功买卖次数
    for code in HS300Code:
        profitRate = 1 #收益率
        profitRateDrawdown = 1 #考虑回撤收益率
        C,Flag = HS300_culFlag.culFlag(code,50) #验证近100天
        chiyou=False
        chiyouDrawdown=False
        for i in range(len(Flag)):
            if not(chiyou) and Flag[i]==1: #买进
                chiyou=True
                buyPrice=C[i]
            elif chiyou and Flag[i]==-1: #卖出
                chiyou=False
                count+=1
                sellPrice=C[i]
#                fund = fund*sellPrice/buyPrice - fund*0.001 
                profitRate *= (sellPrice/buyPrice - 0.001)
                if sellPrice>buyPrice:
                    countCorrect+=1
            if i>0: #求最大回撤
                if not(chiyouDrawdown) and Flag[i]==1: #买进
                    chiyouDrawdown=True
                    buyPrice=C[i]
                elif chiyouDrawdown and (C[i-1]-C[i]>Drawdown or Flag[i]==-1): #卖出
                    chiyouDrawdown=False
                    sellPrice=C[i]
                    profitRateDrawdown *= (sellPrice/buyPrice - 0.001)
        finalRate+=profitRate
        holdRate += (C[-1]/C[0])
        finalRateDrawdown+=profitRateDrawdown
    
    print('取300支股票即日起前50个交易日的数据进行计算：')
    print('300支股票收益率：',round(finalRate/len(HS300Code),2))
    print('成功交易率(卖出价格大于买入价格)：',round(countCorrect/count,2))
    print('300支股票持有收益率：',round(holdRate/len(HS300Code),2))
    print('考虑止损后的收益率：',round(finalRateDrawdown/len(HS300Code),2))

if __name__=="__main__":
    data=ts.get_hs300s()
    HS300Code=data.code.tolist() #300支股票代码
    Drawdown=train(HS300Code)
    CV(HS300Code,Drawdown)
