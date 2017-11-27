# -*- coding: utf-8 -*-
import math
from datetime import datetime, timedelta
import tushare as ts
from matplotlib.pylab import datestr2num

def culDistence(x1, y1, x2, y2, x3, y3):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    dis = abs(A * x3 + B * y3 + C) / math.sqrt(A * A + B * B)
    return dis

def a(x1, x2, L):
    d = []
    x1 = int(x1)
    x2 = int(x2)
    for i in range(x1, x2):
        if (i not in date2):
            continue
        d.append(i)

        start1 = datetime.strptime(start, "%Y-%m-%d")
        t1 = str(start1 + timedelta(x1 - datestr2num(start)))[0:10]
        t2 = str(start1 + timedelta(x2 - datestr2num(start)))[0:10]
        ti = str(start1 + timedelta(i - datestr2num(start)))[0:10]

        d.append(culDistence(x1, closePrice[t1], x2, closePrice[t2],
                             i, closePrice[ti]))
    # index=d.index(max(d))
    if len(d) == 0:
        return
    d1 = [d[i] for i in range(1, len(d), 2)]  # 不包含索引的list
    maxD = max(d1)
    if maxD < L:
        return
    else:
        index = d[d.index(maxD) - 1]
        result.append(index)

        a(x1, index, L)
        a(index, x2, L)


def fun(x1, x2, L):
    global result
    result = []
    a(x1, x2, L)
    result = [int(date2[0])] + sorted(result) + [int(date2[-1])]
    return result

def pieceWise(DM,start1,end1,L=1):
    global closePrice, date2, start, end
    start = start1
    end = end1
    closePrice = ts.get_hist_data(DM, start, end).close  # 收盘价
    closePrice = closePrice[::-1]  # 按日期从低到高
    date1 = ts.get_hist_data(DM, start, end).index  # 日期
    date1 = date1[::-1]  # 按日期从低到高
    date2 = [datestr2num(i) for i in date1]  # 将日期转为数字进行坐标表示
    # print(date2,len(closePrice))
    return fun(date2[0], date2[-1], L), closePrice.tolist(), date2

if __name__ == '__main__':
    pieceWise('002253', '2015-11-01', '2017-11-01', L=1)




