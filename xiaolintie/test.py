# -*- coding: utf-8 -*-
import os
import csv

path = "D:/PythonCode/TimeSeries/xiaolintie/data" #文件夹目录  
files= os.listdir(path) #得到文件夹下的所有文件名称  

for file in files: #遍历文件夹  
#    if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
    filePath = path+"/"+file+"/"+file+".csv" #打开文件  
    with open(filePath) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            print(row[0])