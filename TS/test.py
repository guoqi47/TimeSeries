import numpy as np

import matplotlib

matplotlib.use('Agg')

from matplotlib.pyplot import plot, savefig
x=np.linspace(-4,4,30)
y=np.sin(x);
plot(x,y,'k')
savefig('D:/PythonCode/TimeSeries/TS/figure/a.jpg')