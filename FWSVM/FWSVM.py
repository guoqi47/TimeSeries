# import tushare as ts
import csv
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.cluster import KMeans


# 数据集的熵
def calc_info():
    count_1 = 0
    for i in train_y:
        if i == 1:
            count_1 += 1
    p = count_1 / len(train_y)
    info = -p * np.log2(p) - (1 - p) * np.log2(1 - p)  # 0.99
    return info


# 某个属性划分后的熵
def calc_featureInfo():
    qqq = np.vstack((train_X[:, 1], train_X[:, 1])).T
    y_pred = KMeans(n_clusters=3, random_state=100).fit_predict(qqq)
    c0, c1, c2 = [], [], []
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            c0.append(i)
        elif y_pred[i] == 1:
            c1.append(i)
        else:
            c2.append(i)

    featureInfo = 0
    totle = len(y_pred)
    for c in [c0, c1, c2]:
        count_1 = 0
        for ci in c:
            if train_y[ci] == 1:
                count_1 += 1
        p = count_1 / len(c)
        featureInfo += (len(c) / totle) * (-p * np.log2(p) - (1 - p) * np.log2(1 - p))
    return featureInfo

def infoGain(info,featureInfo):
    return info - featureInfo

# 获取数据，存到csv
# sh = ts.get_hist_data('sh').to_csv('sh.csv')
# sz = ts.get_hist_data('sz').to_csv('sz.csv')

open_p, high_p, close_p, low_p, volume, ma5, ma10, ma20 = [], [], [], [], [], [], [], []
y = []  # label
step = 10
with open('sh.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(f)
    for row in reader:
        open_p.append(row[1])
        high_p.append(row[2])
        close_p.append(row[3])
        low_p.append(row[4])
        volume.append(row[5])
        ma5.append(row[8])
        ma10.append(row[9])
        ma20.append(row[10])
# 打label
for i in range(len(close_p) - step):
    if close_p[i + 10] >= close_p[i]:
        y.append(1)
    else:
        y.append(-1)

open_p = np.array(open_p)[:-step]
high_p = np.array(high_p)[:-step]
close_p = np.array(close_p)[:-step]
low_p = np.array(low_p)[:-step]
volume = np.array(volume)[:-step]
ma5 = np.array(ma5)[:-step]
ma10 = np.array(ma10)[:-step]
ma20 = np.array(ma20)[:-step]

data = np.vstack((open_p, high_p, close_p, low_p, volume, ma5, ma10, ma20)).T
train_X, train_y = data[:500, :], y[:500]
test_X, test_y = data[500:], y[500:]

# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
train_X = min_max_scaler.fit_transform(train_X)
test_X = min_max_scaler.transform(test_X)

clf = svm.SVC(kernel='rbf')
clf.fit(train_X, train_y)
count = 0
for i in range(len(test_X)):
    if clf.predict([test_X[i, :]]) == test_y[i]:
        count += 1
print('准确率:', count / len(test_X))
info = calc_info()
featureInfo = calc_featureInfo()
print(infoGain(info,featureInfo))
