# -*- coding: utf-8 -*-
import os 
import csv
from HS300KDJ_CSV_culKDJ import cul_KDJ

path = "D:/PythonCode/TimeSeries/300data_day1" #文件夹目录  
files= os.listdir(path) #得到文件夹下的所有文件名称  
s = []  
for file in files: #遍历文件夹  
    a = file.split('.')
    if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
        filePath = path+"/"+file #打开文件  
#        open(filePath).readlines()
        with open(filePath) as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                s.append(cul_KDJ(a[0],row[0]))
        
            
                
                   
     
              
          
                   
           
                
    