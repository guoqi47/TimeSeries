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
    fig = plt.figure()  # 这两句后面remove lines用到
    ax = fig.add_subplot(1, 1, 1)
    for i in range(1, len(result)):
        x = []
        plotResult = []
        indexInClose = result[i] - result[0]  #
        if indexInClose >= Window - 1:
            x.append(indexInClose - Window + 1)  # 先加入窗口的开始值
            plotResult.append(closePrice[indexInClose - Window + 1])
            for k in range(i - 1, -1, -1):
                if (result[k] - result[0]) < indexInClose - Window + 1:
                    break
            for j in range(k + 1, i + 1):  # k+1:窗口值后面的第一个拐点的索引
                x.append(result[j] - result[0])
                plotResult.append(closePrice[result[j] - result[0]])
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            x, plotResult = normalization(x, plotResult)
            lines = ax.plot(x, plotResult, 'k')
            savefig('D:/PythonCode/TimeSeries/TS/figure/' + str(i) + '.jpg')


def normalization(x, y):
    x0 = x[0]
    x = [(i - x0) for i in x]  # 横坐标划到0-50
    y_mean = sum(y) / len(y)
    y_range = max(y) - min(y)
    f = lambda i: (i - y_mean) / y_range
    y = list(map(f, y))
    return x, y


if __name__ == "__main__":
    Window = 50  # 暂设50
    start = '2015-11-01'
    end = '2017-11-01'
    data = ts.get_hs300s()
    HS300Code = data.code.tolist()[1:2]  # 300支股票代码

    for DM in HS300Code:
        result, closePrice, date2 = pieceWise(DM, start, end, L=1)
        analysisResult(result)
        dataPlot(result, closePrice, date2)
