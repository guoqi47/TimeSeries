import numpy as np
from matplotlib import pyplot as plt
X = np.arange(50)
fig = plt.figure()  # 这两句后面remove lines用到
ax = fig.add_subplot(2, 1, 1)
#X是1,2,3,4,5,6,7,8,柱的个数
# numpy.random.uniform(low=0.0, high=1.0, size=None), normal
#uniform均匀分布的随机数，normal是正态分布的随机数，0.5-1均匀分布的数，一共有n个
Y1 = np.random.uniform(0.5,1.0,50)
# lines = ax.bar(X,Y1,width = 0.5,facecolor = 'k')
lines = ax.plot([1,2], [3,4], 'k')
ax.lines.remove(lines)
plt.show()