import matplotlib.pyplot as plt
import pandas as pd


def guaidian_result(y):  # 返回拐点的横纵坐标
    guaidian_x = []
    for i in range(1,len(y)):
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
close_p = data['close'][:300]
low_p = data['low']
volume = data['volume'][:300]
y = data['y'][:300]

guaidian_x = [0]+guaidian_result(y)
index = culIndex(guaidian_x)

# resultL_x = result(guaidian_x)
# resultL_x1 = [resultL_x[x] for x in range(0,len(resultL_x),2)]
resultL_y = [close_p[x] for x in index]

fig = plt.figure(figsize=(12, 5))
plt.plot([x for x in range(len(close_p))], close_p)
plt.scatter(index, resultL_y, c='r', marker='o')
plt.show()
