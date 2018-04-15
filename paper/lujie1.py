import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def countAvg(l): # 平均最大长度,结果为6.9
    count=0
    count1=0 #  ji suan duan shu
    i=0
    while i<len(l)-2:
        current = l[i] #当前值
        j=i+1
        while l[j]==current and j<len(l)-1:
            j+=1
        count+=(j-i)
        i=j
        count1+=1
    return count/count1
    
def guaidian_result(column,sequence): #返回拐点的横纵坐标
    L=7 #取阈值为7
    guaidain_X=[]
    guaidain_Y=[]
    for i in range(len(column)-1):
        if column[i] != column[i+1]:
            guaidain_X.append(i)
            guaidain_Y.append(sequence[i])
    return guaidain_X,guaidain_Y   
    
def result(guaidain_X,guaidain_Y,L,j): #返回最终的线段图的坐标
    
    resultL_X=[]
    resultL_Y=[]
    i=0
    while i<len(guaidain_X)-3:
        if guaidain_X[i+1]-guaidain_X[i]>L: # 第一段大于阈值
            # 第二段长度的两倍比第一段和第三段都小
            if (guaidain_X[i+3]-guaidain_X[i+2])>Times*(guaidain_X[i+2]-guaidain_X[i+1])and\
            (guaidain_X[i+2]-guaidain_X[i+1])*Times<(guaidain_X[i+1]-guaidain_X[i]): 
                resultL_X.append(N*(j-1)+guaidain_X[i])
                resultL_X.append(N*(j-1)+guaidain_X[i+3])
                resultL_Y.append(guaidain_Y[i])
                resultL_Y.append(guaidain_Y[i+3])
                i+=2
            elif guaidain_X[i+1]-guaidain_X[i]>1.5*L: # 第一段大于阈值的1.5倍,只要第一段
                resultL_X.append(N*(j-1)+guaidain_X[i])
                resultL_X.append(N*(j-1)+guaidain_X[i+1])
                resultL_Y.append(guaidain_Y[i])
                resultL_Y.append(guaidain_Y[i+1])
        else: # 第一段小于阈值
            pass
        i+=1
    return resultL_X, resultL_Y

def iteration_result(guaidain_X,guaidain_Y,L): #返回迭代一次的线段图的坐标
    
    iterate_X=[]
    iterate_Y=[]
    i=0
    while i<len(guaidain_X)-3:
        
        if guaidain_X[i+1]-guaidain_X[i]>L: # 第一段大于阈值
            # 第二段长度的两倍比第一段和第三段都小
            if (guaidain_X[i+3]-guaidain_X[i+2])>Times*(guaidain_X[i+2]-guaidain_X[i+1])and\
            (guaidain_X[i+2]-guaidain_X[i+1])*Times<(guaidain_X[i+1]-guaidain_X[i]): 
                iterate_X.append(guaidain_X[i+3])
                iterate_Y.append(guaidain_Y[i+3])
                i+=2
            else: 
                iterate_X.append(guaidain_X[i+1])
                iterate_Y.append(guaidain_Y[i+1])
        else: # 第一段小于阈值
            iterate_X.append(guaidain_X[i+1])
            iterate_Y.append(guaidain_Y[i+1])
        i+=1
    return iterate_X,iterate_Y
    
column=[]
close=[]
with open('hs300_lujie.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        column.append(row[8]) 
        close.append(row[3])
column = column[4:-1]
close = close[4:-1]
column = [int(i) for i in column]
close = [float(i) for i in close]
sequence = [column[0]]  # 新建序列,相当于价格
for i in range(1, len(column)):
    sequence.append(column[i] + sequence[i - 1])
X = [i-1 for i in range(1, len(sequence) + 1)]
#循环进行分段
for j in range(1,2):
    N=800 #每次展示的数据量
    # 数据压缩
    X = X[N*(j-1):N*j] 
    sequence = sequence[N*(j-1):N*j]
    column = column[N*(j-1):N*j]
#    close = close[N*(j-1):N*j]
    
    guaidain_X,guaidain_Y=guaidian_result(column,sequence) #拐点的横纵坐标
    
    L=6 #阈值
    Times=1.5#第二段的倍数
    iterate_X,iterate_Y=iteration_result(guaidain_X,guaidain_Y,L)
#    iterate_X,iterate_Y=iteration_result(iterate_X,iterate_Y,L)
    resultL_X, resultL_Y=result(iterate_X,iterate_Y,L,j)#返回最终的线段图的坐标

    # 画图
    # +1-1拟合图
    plt.subplot(311)
    plt.gcf().set_size_inches(15, 12)
    plt.plot(X, sequence, 'b', lw=2)
    
    for i in range(0,len(resultL_X)-1,2):
        plotX=resultL_X[i:i+2]
        plotY=resultL_Y[i:i+2]
        plt.plot(plotX,plotY,'r',lw=3)
    
    # 原始数据拟合图
#    plt.subplot(312)
#    plt.plot(X, close[N*(j-1):N*j], 'b', lw=2)
#    slope = [] # 线段的斜率
#    for i in range(0,len(resultL_X)-1,2):
#        plotX=resultL_X[i:i+2]
#        plotY=[close[resultL_X[i]],close[resultL_X[i+1]]]
#        plt.plot(plotX,plotY,'r',lw=3)
#        # 算斜率
#        slope.append((plotY[1]-plotY[0])/(plotX[1]-plotX[0]))
    
#    # 斜率聚类
#    plt.subplot(313)
#    X = np.array([slope,slope]).T
#    y_pred = KMeans(n_clusters=3, random_state=100).fit_predict(X)
#    plt.scatter(X[:, 0], X[:, 0], c=y_pred)

