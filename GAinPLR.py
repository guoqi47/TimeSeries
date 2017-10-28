# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import json
from datetime import datetime
import tushare as ts
#import numpy as np
import copy
from numpy import random

def culDistence(x1,y1,x2,y2,x3,y3):
    A=y2-y1
    B=x1-x2
    C=x2*y1-x1*y2
    dis=abs(A*x3+B*y3+C)/math.sqrt(A*A+B*B)
    return dis
    
def a(x1,x2,L):
    d=[]
    
    
    for i in range(x1,x2,2):
        d.append(i)
        d.append(culDistence(stockPrice[x1],stockPrice[x1+1],stockPrice[x2],stockPrice[x2+1],stockPrice[i],stockPrice[i+1]))
#    index=d.index(max(d))
    if len(d)==0:
        return
    d1=[d[i] for i in range(1,len(d),2)] #不包含索引的list
    maxD = max(d1)
    if maxD<L:
        return
    else:
        index=d[d.index(maxD)-1]
        result.append(index)

        a(x1,int(index),L)
        a(int(index),x2,L)
        

        
class Gas():
    #初始化一些变量
    def __init__(self,popsize,chrosize,xrangemin,xrangemax):
        self.popsize =popsize
        self.chrosize =chrosize
        self.xrangemin =xrangemin
        self.xrangemax =xrangemax
        self.crossrate =0.6 #交叉率
        self.mutationrate =0.01 #变异率
        
    #初始化种群
    def initialpop(self):
        pop = random.randint(0,2,size =(self.popsize,self.chrosize)) #pop为初始好的数据
        return pop
     
    #定义函数
    def fun(self,x1,x2,L):
        global result
        result=[]
        a(x1,x2,L)
        
        result=[0]+sorted(result)+[int(len(stockPrice)/2)-1]
        # 画图部分
#        for i in range(0,len(result)-1):
#            plt.plot([result[i],result[i+1]], [stockPrice[result[i]*2+1],stockPrice[result[i+1]*2+1]], 'r-')
#        plt.show()
        
        # 计算盈利率
        profitRate = 1
        for i in range(0,len(result)-1):
            p1=stockPrice[result[i+1]*2+1]
            p0=stockPrice[result[i]*2+1]
            if p1-p0>0:
                profitRate *= (2*p1-p0)*(1-Fee)/p1
                
        return profitRate

    #适应度函数，由于我们所求的是最大值，所以可以用函数值代替适应度
    def get_fitness(self,x1,x2,x): 
        fitness = []
        for l in x:
            fitness.append(self.fun(x1,x2,l))
        return fitness

    #输入参数为上一代的种群，和上一代种群的适应度列表
    def selection(self,popsel,fitvalue):
        new_fitvalue = []
        totalfit = sum(fitvalue)
        accumulator = 0.0
        for val in fitvalue: 
            #对每一个适应度除以总适应度，然后累加，这样可以使
            #适应度大的个体获得更大的比例空间。
            new_val =(val/totalfit)            
            accumulator += new_val
            new_fitvalue.append(accumulator)            
        ms = []
        for i in range(self.popsize):
            #随机生成0,1之间的随机数
            ms.append(random.random()) 
        ms.sort() #对随机数进行排序
        fitin = 0
        newin = 0
        newpop = popsel
        while newin < self.popsize:
            #随机投掷，选择落入个体所占轮盘空间的个体
            if(ms[newin] < new_fitvalue[fitin]):
                newpop[newin] = popsel[fitin]
                newin = newin + 1
            else:
                fitin = fitin + 1
        #适应度大的个体会被选择的概率较大
        #使得新种群中，会有重复的较优个体
        pop = newpop
        return pop

    def crossover(self,pop):
        for i in range(self.popsize-1):
            #近邻个体交叉，若随机数小于交叉率
            if(random.random()<self.crossrate):
                #随机选择交叉点
                singpoint =random.randint(0,self.chrosize)
                temp1 = []
                temp2 = []
                #对个体进行切片，重组
                temp1.extend(pop[i][0:singpoint])
                temp1.extend(pop[i+1][singpoint:self.chrosize])
                temp2.extend(pop[i+1][0:singpoint])
                temp2.extend(pop[i][singpoint:self.chrosize])
                pop[i]=temp1
                pop[i+1]=temp2
        return pop

    def mutation(self,pop):
        for i in range(self.popsize):
            #反转变异，随机数小于变异率，进行变异
            if (random.random()< self.mutationrate):
                mpoint = random.randint(0,self.chrosize-1)
                #将随机点上的基因进行反转。
                if(pop[i][mpoint]==1):
                    pop[i][mpoint] = 0
                else:
                    pop[mpoint] =1

        return pop


    def elitism(self,pop,popbest,nextbestfit,fitbest):
        #输入参数为上一代最优个体，变异之后的种群，
        #上一代的最优适应度，本代最优适应度。这些变量是在主函数中生成的。
        if nextbestfit-fitbest <0:  
            #满足精英策略后，找到最差个体的索引，进行替换。         
            pop_worst =nextfitvalue.index(min(nextfitvalue))
            pop[pop_worst] = popbest
        return pop


    #对十进制进行转换到求解空间中的数值
    def get_declist(self,chroms):
        step =(self.xrangemax - self.xrangemin)/float(2**self.chrosize-1)
        self.chroms_declist =[]
        for i in range(self.popsize):
            chrom_dec =self.xrangemin+step*self.chromtodec(chroms[i])  
            self.chroms_declist.append(chrom_dec)      
        return self.chroms_declist
    #将二进制数组转化为十进制
    def chromtodec(self,chrom):
        m = 1
        r = 0
        for i in range(self.chrosize):
            r = r + m * chrom[i]
            m = m * 2
        return r #0011001111




if __name__ =='__main__':
    stockPrice=[]
    stockPrice1=[]
    L=1
    Fee=0.01 #每笔交易手续费
#    global result #点集
#    result=[]
    
    sns.set_style("whitegrid")
    end = datetime.today() #开始时间结束时间，选取最近一年的数据
    start = datetime(end.year,end.month-6,end.day)
    end = str(end)[0:10]
    start = str(start)[0:10]
    
    stock = ts.get_hist_data('002253',start,end)#选取一支股票
    stock.to_json('stock.json',orient='records')#转化为json格式
    with open('stock.json', 'r') as f:
        data = json.load(f)
#    for p in data[:]:
#        stockPrice.append(p['open'])

    for p in range(len(data)-1,-1,-1): #调换顺序，使成为随时间增长变化的曲线
        stockPrice1.append(data[p]['close'])
    for p in range(0,len(stockPrice1)):
        stockPrice.append(p)
        stockPrice.append(stockPrice1[p])
        
#    stock['close'].plot(legend=False ,figsize=(12,4)) #原画图
#    plt.gcf().set_size_inches(12,4)
#    plt.plot([stockPrice[i] for i in range(0,len(stockPrice),2)],[stockPrice[i] for i in range(1,len(stockPrice),2)],'b-')

    
#    result=[0]+sorted(result)+[int(len(stockPrice)/2)-1]
#    # 画图部分
#    for i in range(0,len(result)-1):
#        plt.plot([result[i],result[i+1]], [stockPrice[result[i]*2+1],stockPrice[result[i+1]*2+1]], 'r-')
#    plt.show()
#    
#    # 计算盈利率
#    profitRate = 1
#    for i in range(0,len(result)-1):
#        p1=stockPrice[result[i+1]*2+1]
#        p0=stockPrice[result[i]*2+1]
#        if p1-p0>0:
#            profitRate *= (2*p1-p0)*(1-Fee)/p1
#    print(profitRate)

    #遗传算法部分
    generation = 50 # 遗传代数
    
    popsize=50 #每代的个体数
    chrosize=8 #变量编码长度
    xrangemin=0.1 #变量取值范围
    xrangemax=5
    #引入Gas类，传入参数：种群大小，编码长度，变量范围
    mainGas =Gas(popsize,chrosize,xrangemin,xrangemax) 
    pop =mainGas.initialpop()  #种群初始化
    pop_best = [] #每代最优个体
    for i in range(generation): 
        #在遗传代数内进行迭代
        declist =mainGas.get_declist(pop)#解码
        fitvalue =mainGas.get_fitness(0,int((len(stockPrice)+1)/2),declist)#适应度函数
        #选择适应度函数最高个体
        popbest = pop[fitvalue.index(max(fitvalue))]
        #对popbest进行深复制，以为后面精英选择做准备
        popbest =copy.deepcopy(popbest)
        #最高适应度
        fitbest = max(fitvalue)
        #保存每代最高适应度值
        pop_best.append(fitbest)        
        ################################进行算子操作，并不断更新pop
        mainGas.selection(pop,fitvalue)  #选择
        mainGas.crossover(pop) # 交叉
        mainGas.mutation(pop)  #变异
        ################################精英策略前的准备
        #对变异之后的pop，求解最大适应度
        nextdeclist = mainGas.get_declist(pop) 
        nextfitvalue =mainGas.get_fitness(0,int((len(stockPrice)+1)/2),nextdeclist)     
        nextbestfit = max(nextfitvalue) 
        ################################精英策略
        #比较深复制的个体适应度和变异之后的适应度
        mainGas.elitism(pop,popbest,nextbestfit,fitbest)

        
    t = [x for x in range(generation)]
    s = pop_best
    print(max(s))
    plt.plot(t,s)
    plt.show()
    