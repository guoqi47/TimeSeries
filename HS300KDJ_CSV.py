# -*- coding: utf-8 -*-
import os 
import csv
from HS300KDJ_CSV_culKDJ import cul_KDJ

def culFlag(K,D,J,index,K_value,D_value,J_value,CBigThanMA5): #传入所有KDJlist，index，当日KDJ,收盘价是否大于MA5
    threshold=8 #判断金死叉的阈值
    #买入卖出判断，-1看跌，1看涨，0持有
    if index<len(K)-1:
        if abs(J_value-D_value)<threshold: #近金叉或者死叉
            if J_value-D_value>0: 
                if J[index-1]-D[index-1]>J_value-D_value:
                    return -1
                if J[index-1]-D[index-1]<threshold and CBigThanMA5:
                    return 1
                else:
                    return 0   
            elif J_value<D_value and D[index-1]-J[index-1]>threshold and CBigThanMA5: #即将金叉
                return 1
                        
            else:
                return -1
                        
        if J_value>D_value and CBigThanMA5: #要涨，K线总是在中间,故简化
            return 1
                        
        elif J_value<D_value: #要跌
            return -1
                    
        else:  #持有
            return 0
    return 0
  

if __name__=="__main__":
    start='2015-10-31'
    end='2017-10-31'
    path = "D:/PythonCode/TimeSeries/300data_day" #文件夹目录  
    files= os.listdir(path) #得到文件夹下的所有文件名称  
    
    for file in files: #遍历文件夹  
        s = []  
        a = file.split('.')
        if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
            filePath = path+"/"+file #打开文件  
            K,D,J,dateList,CBigThanMA5 = cul_KDJ(a[0])
            with open(filePath) as f:
                f_csv = csv.reader(f)
                headers = next(f_csv)
                for row in f_csv:
                    #计算KDJ
                    if row[0] != '2016-11-01' and row[0] != '2016-10-31':
                        index = dateList.index(row[0])
                        K_value = K[index]
                        D_value = D[index]
                        J_value = J[index]
                        m = CBigThanMA5[index] #收盘价是否大于五日均线
                        d = {'K':K_value,'D':D_value,'J':J_value}
                        row.append(d)
                        row.append(culFlag(K,D,J,index,K_value,D_value,J_value,m))
                        s.append(row)
                    else:
                        d = {'K':0,'D':0,'J':0}
                        row.append(d)
                        row.append(0)
                        s.append(row)
            #重写文件
            with open(filePath,'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'open', 'high','close', 'low','volume', 'amount','KDJ_value', 'flag'])
                for item in s:
                     writer.writerow([item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8]])
            
                
                    
                       
         
                  
              
                       
               
                    
        