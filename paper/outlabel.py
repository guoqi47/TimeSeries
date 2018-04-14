import pandas as pd

def culLable():
    cma, y = [0] * 3, [0] * 3
    for i in range(3, len(close_p) - 3):
        totle = totle - close_p[i - 4] + close_p[i + 3] if i != 3 else sum(close_p[0:7])
        cma.append(totle / 7)

    for i in range(3, len(close_p) - 3 - 3):
        y.append(1 if cma[i] > close_p[i] and cma[i + 3] > cma[i + 1]  \
                     else -1 if cma[i] < close_p[i] and cma[i + 3] < cma[i + 1] else y[i - 1])
    return cma+[0]*3, y+[0]*6


df = pd.read_csv('sh.csv')
close_p = df['close']

cma, y = culLable()
df.insert(6,'cma',cma)
df.insert(7,'y',y)
df.to_csv('sh1.csv')



