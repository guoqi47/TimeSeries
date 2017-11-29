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


def dataPlot(result, closePrice, date, volume):
    fig = plt.figure()  # 这两句后面remove lines用到
    ax = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    for i in range(1, len(result)):
        x_result, y_result, y_volume = [], [], []  # result折线图,volume柱状图
        x_volume = [a for a in range(Window)]
        indexInClose = result[i] - result[0]  #
        if indexInClose >= Window - 1:
            x_result.append(indexInClose - Window + 1)  # 先加入窗口的开始值
            y_result.append(closePrice[indexInClose - Window + 1])
            for n in range(Window):  # volume柱状图的y
                y_volume.append(volume[indexInClose - Window + 1 + n])
            for k in range(i - 1, -1, -1):
                if (result[k] - result[0]) < indexInClose - Window + 1:
                    break
            for j in range(k + 1, i + 1):  # k+1:窗口值后面的第一个拐点的索引
                x_result.append(result[j] - result[0])
                y_result.append(closePrice[result[j] - result[0]])
            try:
                ax.lines.remove(lines[0])
                # ax.lines.remove(bars[0])
            except Exception:
                pass
            x_result, y_result = normalization(x_result, y_result)
            lines = ax.plot(x_result, y_result, 'k')
            # bars = ax1.bar(x_volume, y_volume, width=0.5, facecolor='k')
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
    start = '2015-10-31'
    end = '2017-10-31'
    data = ts.get_hs300s()
    HS300Code = data.code.tolist()[0:1]  # 300支股票代码

    for DM in HS300Code:
        result, closePrice, date2, volume = pieceWise(DM, start, end, L=1)
        analysisResult(result)
        dataPlot(result, closePrice, date2, volume)
