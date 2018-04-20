# import outlabel
import pandas as pd
import matplotlib.finance as mpf
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.image as mpimg  # mpimg 用于读取图片
import tensorflow as tf
import numpy


def drawCandlestick(index):
    fig = plt.figure(figsize=(3, 3))
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


def guaidian_result(y):  # 返回拐点的横纵坐标
    guaidian_x = []
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            guaidian_x.append(i)
    return guaidian_x


# 之前评测标准
# def culIndex(guaidian_x):
#     index = []
#     i = 1
#     while i < len(guaidian_x) - 2:
#         if y[guaidian_x[i - 1]] == 1 and guaidian_x[i] - guaidian_x[i - 1] >= 10:
#             index.append(guaidian_x[i - 1])
#         elif y[guaidian_x[i - 1]] == 1 and \
#                 (guaidian_x[i + 1] - guaidian_x[i]) / (guaidian_x[i + 2] - guaidian_x[i - 1]) >= 0.25:
#             index.append(guaidian_x[i - 1])
#             i += 2
#         i += 1
#     return index

# lujie的方法
def culIndex(guaidian_x):
    index, label = [], []
    i = 1
    while i < len(guaidian_x):
        if guaidian_x[i] - guaidian_x[i - 1] >= 10 or \
                abs(sum(y[guaidian_x[i - 1]:guaidian_x[i - 1] + 10])) >= 4:
            index.append(guaidian_x[i - 1])
            label.append(y[guaidian_x[i - 1]])
        i += 1

    return index, label


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    #strides第一个和最后一个值为1（约定），padding='SAME'表示提取一次后大小不变，即自己会补0
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def cnn(train_x,train_y,test_x,test_y):
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, 96 * 96, 3])
    y_ = tf.placeholder("float", shape=[None, 2])
    # 第一层卷积 + maxpooling
    # 前两个代表卷积核尺寸，第三个代表通道数，灰度图就是1，rgb图就是3，第四个是卷积核的个数，多少个卷积核代表提取多少种特征
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 96, 96, 3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # 第二层卷积 + maxpooling
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    W_fc1 = weight_variable([24 * 24 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 24 * 24 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout层，只能在训练中使用，而不能用于测试过程
    keep_prob = tf.placeholder(tf.float32)  # 表示一个神经元的输出在dropout时不被丢弃的概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax层
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 模型训练和测试
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(10):
        train_step.run(feed_dict={x: train_x, y_: train_y, keep_prob: 0.8})
        if i % 10 == 0:
            print('i:',i)
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: test_x, y_: test_y, keep_prob: 1.0}))

data = pd.read_csv('sh1.csv')
open_p = data['open']
high_p = data['high']
close_p = data['close']
low_p = data['low']
volume = data['volume']
y = data['y']

train_x = y[:100]

guaidian_x = guaidian_result(train_x)
train_x_index, train_y = culIndex(guaidian_x)

for i in train_x_index:
    if i >= 20:
        drawCandlestick(i)

# 图像压缩
path = 'img/img22.jpg'
path_out = 'img1/img22.jpg'
image = cv.imread(path)
new_image = cv.resize(image, (96, 96))
cv.imwrite(path_out, new_image)

lena = mpimg.imread(path_out)  # 96*96*3
l = [lena] * 3
l = numpy.reshape(l,(-1,96*96,3))
lena = numpy.reshape(lena,(-1,96*96,3))
label = [[1,0],[1,0],[1,0]]

cnn(l,label,l,label)

