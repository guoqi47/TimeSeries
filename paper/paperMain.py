# import outlabel
import pandas as pd
import matplotlib.finance as mpf
import matplotlib.pyplot as plt

def drawCandlestick():
    pass

data = pd.read_csv('sh1.csv')
num = 100 #画前num个交易日的图
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0,0.2,1,0.5])
ax2 = fig.add_axes([0,0,1,0.2])
mpf.candlestick2_ochl(ax, data['open'][:num], data['close'][:num], data['high'][:num], data['low'][:num],
                      width=0.5, colorup='r', colordown='g',alpha=1)
ax.set_xticks(range(0, num, 10))
ax.grid(True)

mpf.volume_overlay(ax2, data['open'][:num], data['close'][:num], data['volume'][:num],
                   colorup='r', colordown='g', width=0.5, alpha=0.8)
ax2.set_xticks(range(0, num, 10))
ax2.set_xticklabels(data['date'][::10], rotation=30)
ax2.grid(True)

#plt.subplots_adjust(hspace=0)
plt.show()