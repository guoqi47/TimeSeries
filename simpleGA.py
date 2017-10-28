# -*- coding: utf-8 -*-

from numpy import random
import matplotlib.pyplot as plt
from math import pi,sin
import copy

class Gas():
    #初始化一些变量
    def __init__(self,popsize,chrosize,xrangemin,xrangemax):
        self.popsize =popsize
        self.chrosize =chrosize
        self.xrangemin =xrangemin
        self.xrangemax =xrangemax
        self.crossrate =1
        self.mutationrate =0.01
        
    #初始化种群
    def initialpop(self):
        pop = random.randint(0,2,size =(self.popsize,self.chrosize))
        return pop
     
    #定义函数
    def fun(self,x1): 
        return x1*sin(10*pi*x1)+2.0

    #适应度函数
    def get_fitness(self,x): 
        fitness = []
        for x1 in x:
            fitness.append(self.fun(x1))
        return fitness

    #输入参数为上一代的种群，和上一代种群的适应度列表
    def selection(self,popsel,fitvalue):
        new_fitvalue = []
        totalfit = sum(fitvalue)
        accumulator = 0.0
        for val in fitvalue: 
            #对每一个适应度除以总适应度，然后累加，这样可以使适应度大
            #的个体获得更大的比例空间。
            new_val =(val*1.0/totalfit)            
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
        return r



if __name__ == '__main__':
    generation = 100 # 遗传代数
    #引入Gas类，传入参数：种群大小，编码长度，变量范围
    mainGas =Gas(100,10,-1,2) 
    pop =mainGas.initialpop()  #种群初始化
    pop_best = [] #每代最优个体
    for i in range(generation): 
        #在遗传代数内进行迭代
        declist =mainGas.get_declist(pop)#解码
        fitvalue =mainGas.get_fitness(declist)#适应度函数
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
#        对变异之后的pop，求解最大适应度
        nextdeclist = mainGas.get_declist(pop) 
        nextfitvalue =mainGas.get_fitness(nextdeclist)     
        nextbestfit = max(nextfitvalue) 
        ################################精英策略
        #比较深复制的个体适应度和变异之后的适应度
        mainGas.elitism(pop,popbest,nextbestfit,fitbest)

        
    t = [x for x in range(generation)]
    s = pop_best
    print(max(s))
    plt.plot(t,s)
    plt.show()

