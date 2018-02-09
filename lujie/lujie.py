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
#    print(count)
#    print(count1)
    return count/count1
    
def result(column,sequence): #返回满足结果的点
    L=7 #取阈值为6
    result_X=[]
    result_Y=[]
    for i in range(len(column)-1):
        if column[i] != column[i+1]:
            result_X.append(i)
            result_Y.append(sequence[i])
    return result_X,result_Y   
    
    

with open('hs300_lujie.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[8] for row in reader]
column = column[4:-1]
column = [int(i) for i in column]
sequence = [column[0]]  # 新建序列,相当于价格
for i in range(1, len(column)):
    sequence.append(column[i] + sequence[i - 1])
X = [i for i in range(1, len(sequence) + 1)]
# 数据压缩
X = X[:800]
sequence = sequence[:800]
column = column[:800]
#print(countAvg(column))
result_X,result_Y=result(column,sequence) #guaidian de hengzongzuobiao
resultL=[] #dai yuzhi
resultL_X=[]
resultL_Y=[]
for i in range(1,len(result_X)):
    if (result_X[i]-result_X[i-1])>6:
#        resultL.append(i)
        resultL_X.append(result_X[i-1])
        resultL_X.append(result_X[i])
        resultL_Y.append(result_Y[i-1])
        resultL_Y.append(result_Y[i])

##liangduan lianzaiyiqi de hebing
#for i in range(1,len(resultL_X)-2,2):
#    if resultL_X[i]==resultL_X[i+1]:
#        resultL

# 画图
plt.gcf().set_size_inches(15, 4)
plt.plot(X, sequence, 'b', lw=2)

for i in range(0,len(resultL_X)-1,2):
    plotX=resultL_X[i:i+2]
    plotY=resultL_Y[i:i+2]
    plt.plot(plotX,plotY,'r',lw=3)









