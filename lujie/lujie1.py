import csv
import matplotlib.pyplot as plt

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
    
def result(guaidain_X,guaidain_Y,L): #返回最终的线段图的坐标
    
    resultL_X=[]
    resultL_Y=[]
    i=0
    while i<len(guaidain_X)-3:
        if guaidain_X[i+1]-guaidain_X[i]>L: # 第一段大于阈值
            # 第二段比第一段和第三段都小
            if (guaidain_X[i+3]-guaidain_X[i+2])>(guaidain_X[i+2]-guaidain_X[i+1])and\
            (guaidain_X[i+2]-guaidain_X[i+1])<(guaidain_X[i+1]-guaidain_X[i]): 
                resultL_X.append(guaidain_X[i])
                resultL_X.append(guaidain_X[i+3])
                resultL_Y.append(guaidain_Y[i])
                resultL_Y.append(guaidain_Y[i+3])
                i+=2
            else: # 只要第一段
                resultL_X.append(guaidain_X[i])
                resultL_X.append(guaidain_X[i+1])
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
            # 第二段比第一段和第三段都小
            if (guaidain_X[i+3]-guaidain_X[i+2])>(guaidain_X[i+2]-guaidain_X[i+1])and\
            (guaidain_X[i+2]-guaidain_X[i+1])<(guaidain_X[i+1]-guaidain_X[i]): 
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
    

with open('hs300_lujie.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[8] for row in reader]
column = column[4:-1]
column = [int(i) for i in column]
sequence = [column[0]]  # 新建序列,相当于价格
for i in range(1, len(column)):
    sequence.append(column[i] + sequence[i - 1])
X = [i-1 for i in range(1, len(sequence) + 1)]
#循环进行分段
for j in range(1,2):
    
    # 数据压缩
    X = X[800*(j-1):800*j] 
    sequence = sequence[800*(j-1):800*j]
    column = column[800*(j-1):800*j]
    
    guaidain_X,guaidain_Y=guaidian_result(column,sequence) #拐点的横纵坐标
    
    L=6 #阈值
    iterate_X,iterate_Y=iteration_result(guaidain_X,guaidain_Y,L)
    resultL_X, resultL_Y=result(iterate_X,iterate_Y,L)#返回最终的线段图的坐标

    # 画图
    plt.gcf().set_size_inches(15, 4)
    plt.plot(X, sequence, 'b', lw=2)
    
    for i in range(0,len(resultL_X)-1,2):
        plotX=resultL_X[i:i+2]
        plotY=resultL_Y[i:i+2]
        plt.plot(plotX,plotY,'r',lw=3)


