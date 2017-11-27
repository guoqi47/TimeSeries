import tushare as ts
from pieceWise import pieceWise
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig

def analysisResult(result):  # 对分段数据进行分析
    # 求Window大小
    # maxWindow = 0
    # for i in range(1, len(result)):
    #     if result[i] - result[i - 1] > maxWindow:
    #         maxWindow = result[i] - result[i - 1]
    # print(maxWindow, result[0], result[-1])
    pass
def dataPlot(result, closePrice, date):
    fig = plt.figure() # 这两句后面remove lines用到
    ax = fig.add_subplot(1, 1, 1)
    for i in range(1, len(result)):
        x = []
        plotResult=[]
        indexInClose = date.index(result[i])  # 拐点在closePrince和date中的index
        if indexInClose >= Window-1:
            x.append(indexInClose - Window)  # 先加入窗口的开始值
            plotResult.append(closePrice[indexInClose - Window])
            for k in range(indexInClose-1, -1, -1):
                if result[k] < indexInClose - Window:
                    break
            for j in range(k+1, i+1):  # k+1:窗口值后面的第一个拐点的索引
                x.append(result[j]-result[0])
                plotResult.append(closePrice[result[j]-result[0]])
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot(x, plotResult, 'k')
            savefig('D:/PythonCode/TimeSeries/TS/figure/'+str(i)+'.jpg')



if __name__ == "__main__":
    Window = 50 #暂设50
    start = '2015-11-01'
    end = '2017-11-01'
    data = ts.get_hs300s()
    HS300Code = data.code.tolist()[:2]  # 300支股票代码

    for DM in HS300Code:
        result, closePrice, date2 = pieceWise(DM, start, end, L=1)
        analysisResult(result)
        dataPlot(result, closePrice, date2)
