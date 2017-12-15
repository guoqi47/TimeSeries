# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import math
from matplotlib.pylab import datestr2num
import csv
import os

def culDistence(x1,y1,x2,y2,x3,y3): #计算点（x3,y3）到直线（x1,y1,x2,y2）距离函数
    A=y2-y1
    B=x1-x2
    C=x2*y1-x1*y2
    dis=abs(A*x3+B*y3+C)/math.sqrt(A*A+B*B)
    return dis
    
def iterate(x1,x2,L,result,dateToNum,closePrice): #迭代函数，
    # x1,迭代起始横坐标 x2,迭代终止横坐标  L,阈值 
    # result,拐点横坐标 dateToNum,日期的数字list closePrice,收盘价
    d=[] #保存索引和点到直线的距离值
    x1,x2=int(x1),int(x2)
    for i in range(x1,x2):
        if(i not in dateToNum):
            continue
        d.append(i) #添加索引
        t1=int(dateToNum.index(x1))
        t2=int(dateToNum.index(x2))
        ti=int(dateToNum.index(i))
        d.append(culDistence(x1,closePrice[t1],x2,closePrice[t2],
                                           i,closePrice[ti])) #添加距离
    if len(d)==0: #d为空
        return
    d1=[d[i] for i in range(1,len(d),2)] #筛选出来距离值，成为d1
    maxD = max(d1) #求最大距离
    if maxD<L: #最大距离小于阈值
        return
    else:
        index=d[d.index(maxD)-1]
        result.append(index)
        
        iterate(x1,index,L,result,dateToNum,closePrice)
        iterate(index,x2,L,result,dateToNum,closePrice)
        
def pieceWise(dateToNum,L,closePrice,code):
    # dateToNum,日期转换为数字的列表  closePrice,收盘价 code 股票代码
    result=[] #用来存拐点的横坐标
    iterate(dateToNum[0],dateToNum[-1],L,result,dateToNum,closePrice)
#    fileHeader = ["start", "end","duration","slope"]
    datacsv = open("data_"+code+".csv","w",newline="")
    csvwriter = csv.writer(datacsv,dialect = ("excel"))
    result=[dateToNum[0]]+sorted(result)+[dateToNum[-1]] #拐点加起始和终止两点
    # 画图线性分段图
    for i in range(0,len(result)-1):
        ii=dateToNum.index(result[i]) #当前拐点对应的收盘价在closePrice中的索引
        ii1=dateToNum.index(result[i+1])#下一个拐点对应的收盘价在closePrice中的索引
        p0=closePrice[ii] #当前拐点对应的收盘价
        p1=closePrice[ii1] #下一个拐点对应的收盘价
        plt.plot_date([result[i],result[i+1]],
                        [p0,p1], 'r-')
        #保存为csv格式
        csvwriter.writerow([ii,ii1,dateToNum.index(result[i+1])-dateToNum.index(result[i]),(p1-p0)/(result[i+1]-result[i])])
    datacsv.close()
def mainPieceWise(L,path): #分段主函数
    files = os.listdir(path) #得到文件夹下的所有文件名称  
    for code in files: #遍历文件夹  
        closePrice = [] #收盘价
        date = [] #字符串日期
        filePath = path+"/"+code+"/"+code+".csv" #csv文件名  
        with open(filePath) as f:
            f_csv = csv.reader(f)
            headers = next(f_csv) # 略过第一行列名
            for row in f_csv:
                closePrice.append(row[3])
                date.append(row[0])
        os.chdir(path +"/"+code) #更改当前文件夹，进行文件的保存
        closePrice=[float(i) for i in closePrice][::-1] 
        date=[str(i)[:10] for i in date][::-1] #字符串日期
        dateToNum = [datestr2num(i) for i in date] #将日期转为数字进行坐标表示
        #画原始数据图
        plt.gcf().set_size_inches(12,4)
        plt.plot_date(dateToNum,closePrice,'b-')
        #调用线性分段函数
        pieceWise(dateToNum,L,closePrice,code)
        
if __name__ =='__main__':
    L=1 #阈值
    path = "D:/PythonCode/TimeSeries/xiaolintie/data" #data文件夹目录  
    mainPieceWise(L,path)
    