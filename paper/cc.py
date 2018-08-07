# import outlabel
import pandas as pd
import matplotlib.finance as mpf
import matplotlib.image as mpimg  # mpimg 用于读取图片
import tensorflow as tf
import numpy
import os
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def drawCandlestick(index, path):
    fig = plt.figure(figsize=(2, 2))
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

    plt.savefig(path + str(int(index)) + ".jpg")

def guaidian_result(y):  # 返回拐点的横纵坐标
    guaidian_x = []
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            guaidian_x.append(i)
    return guaidian_x

def culIndex(guaidian_x):
    index, label = [], []
    if guaidian_x[0] > 10:
        index.append(guaidian_x[0])
        label.append(-y[guaidian_x[0]])
    i = 1
    while i < len(guaidian_x):
        if guaidian_x[i] - guaidian_x[i - 1] >= 10 or abs(sum(y[guaidian_x[i - 1]:guaidian_x[i - 1] + 10])) >= 4:
            if guaidian_x[i - 1] >= 20:
                index.append(guaidian_x[i - 1])
                label.append(-y[guaidian_x[i - 1]])
        i += 1

    return index, label

def getbatch_X(train_X, batch_size):
    for i in range(0, len(train_X), batch_size):
        yield train_X[i:i + batch_size]


def getbatch_y(train_y, batch_size):
    for i in range(0, len(train_y), batch_size):
        yield train_y[i:i + batch_size]

def one_hot(train_y):
    res = [[0, 1] if i < 0 else [1, 0] for i in train_y]
    return res

data = pd.read_csv('train1.csv')
open_p = data['open']
high_p = data['high']
close_p = data['close']
low_p = data['low']
volume = data['volume']
y = data['y']

train_num = 1500
train_x = y[:train_num]

guaidian_x = guaidian_result(train_x)
train_x_index, train_y = culIndex(guaidian_x)
train_y = y[:train_num]
# # train set 画图
path_train = 'img_train/'
train_x = []
gc.collect()
# for i in range(20+10000,20+train_num+10000):
#     drawCandlestick(i, path_train)
#     if i%400==0:
#         gc.collect()
files = os.listdir(path_train)  # 得到文件夹下的所有文件名称
files.sort(key=lambda x:int(x[:-4])) #对文件进行排序读取
for file in files:
    # print(file)
    train_x.append(mpimg.imread(path_train + file))  # 64*64*3
train_x = numpy.array(train_x)
train_x = numpy.reshape(train_x, (-1, 200, 200, 3))/255

train_y = one_hot(train_y)
train_y = numpy.array(train_y).astype(numpy.float32)

# # test set 画图
testNum = 100
test_x_index = [i for i in range(train_num,train_num+testNum)]
test_y = y[train_num:train_num+testNum]
#
path_test = 'img_test/'
test_x = []
for i in test_x_index:
    drawCandlestick(i, path_test)
files = os.listdir(path_test)  # 得到文件夹下的所有文件名称
for file in files:
    test_x.append(mpimg.imread(path_test + file))  # 64*64*3
test_x = numpy.array(test_x)
test_x = numpy.reshape(test_x, (-1, 200, 200, 3))/255
test_y = one_hot(test_y)
test_y = numpy.array(test_y).astype(numpy.float32)
# batch_X = getbatch_X(train_x, 64)
# batch_y = getbatch_y(train_y, 64)
#------------------------------------------------------------------------------------
batch_size = 32
x = tf.placeholder(tf.float32, shape=[None, 200, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

conv1 = tf.layers.conv2d(
    inputs=x,
    filters=16,
    kernel_size=2,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)
conv2 = tf.layers.conv2d(pool1, 32, 2, 1, 'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
flat = tf.reshape(pool2, [-1, 50*50*32])
output = tf.layers.dense(flat, 2)

cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=output)  # compute cost
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(y_, axis=1), predictions=tf.argmax(output, axis=1), )[1]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50):
    # x_train = batch_X.__next__()
    # y_train = batch_y.__next__()
    b_x = train_x[i * batch_size:(i + 1) * batch_size]
    b_y = train_y[i * batch_size:(i + 1) * batch_size]
    y_pred, _, loss,= sess.run([output, train_step, cross_entropy],
                               feed_dict={x: b_x, y_: b_y})
    if i % 30 == 0:
        accuracy_ = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        print('Step:', i, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy_)
