import tushare as ts
from pieceWise import pieceWise


def analysisResult(result):  # 对分段数据进行分析
    # 求Window大小
    # maxWindow = 0
    # for i in range(1, len(result)):
    #     if result[i] - result[i - 1] > maxWindow:
    #         maxWindow = result[i] - result[i - 1]
    # print(maxWindow, result[0], result[-1])
    pass
def dataPlot(result):
    for i in range(len(result)):
        if result[i] >= Window:
            continue
        else:
            continue


if __name__ == "__main__":
    Window = 50 #暂设50
    start = '2015-11-01'
    end = '2017-11-01'
    data = ts.get_hs300s()
    HS300Code = data.code.tolist()  # 300支股票代码
    for DM in HS300Code:
        result = pieceWise(DM, start, end, L=0.5)
        analysisResult(result)
        dataPlot(result)
