# import outlabel
import pandas as pd
import matplotlib.finance as mpf
import matplotlib.pyplot as plt


def drawCandlestick(index):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0.3, 1, 0.7])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0, 0, 1, 0.3])
    # 画蜡烛图
    mpf.candlestick2_ochl(ax, data['open'][index - 20:index], data['close'][index - 20:index],
                          data['high'][index - 20:index], data['low'][index - 20:index],
                          width=0.5, colorup='r', colordown='g', alpha=1)
    # 画交易量
    mpf.volume_overlay(ax2, data['open'][index - 20:index], data['close'][index - 20:index],
                       data['volume'][index - 20:index],
                       colorup='r', colordown='g', width=0.5, alpha=0.8)
    # 去掉坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.savefig("img/img" + str(index) + ".jpg")
    plt.show()

def guaidian_result(y):  # 返回拐点的横纵坐标
    guaidian_x = []
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            guaidian_x.append(i)
    return guaidian_x

def culIndex(guaidian_x):
    index = []
    i = 1
    while i < len(guaidian_x) - 2:
        if y[guaidian_x[i - 1]] == 1 and guaidian_x[i] - guaidian_x[i - 1] >= 10:
            index.append(guaidian_x[i - 1])
        elif y[guaidian_x[i - 1]] == 1 and \
                (guaidian_x[i + 1] - guaidian_x[i]) / (guaidian_x[i + 2] - guaidian_x[i - 1]) >= 0.25:
            index.append(guaidian_x[i - 1])
            i += 2
        i += 1
    return index


data = pd.read_csv('sh1.csv')
open_p = data['open']
high_p = data['high']
close_p = data['close']
low_p = data['low']
volume = data['volume']
y = data['y']

guaidian_x = [0]+guaidian_result(y)
index = culIndex(guaidian_x)

for i in index:
    if i>=20:
        drawCandlestick(i)

