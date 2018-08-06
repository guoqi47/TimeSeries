# import outlabel
import pandas as pd

import matplotlib.finance as mpf
import cv2 as cv
import matplotlib.image as mpimg  # mpimg 用于读取图片
import tensorflow as tf
import numpy
import os
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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides第一个和最后一个值为1（约定），padding='SAME'表示提取一次后大小不变，即自己会补0
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def getbatch_X(train_X, batch_size):
    for i in range(0, len(train_X), batch_size):
        yield train_X[i:i + batch_size]


def getbatch_y(train_y, batch_size):
    for i in range(0, len(train_y), batch_size):
        yield train_y[i:i + batch_size]


def cnn(train_x, train_y, test_x, test_y):
    #    print(train_y)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 200, 200, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    # 第一层卷积 + maxpooling
    # 前两个代表卷积核尺寸，第三个代表通道数，灰度图就是1，rgb图就是3，第四个是卷积核的个数，多少个卷积核代表提取多少种特征
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 200, 200, 3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # 第二层卷积 + maxpooling,24*24*32
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #    h_pool2 = max_pool_2x2(h_conv2)

    # 第三层卷积 + maxpooling,24*24*64
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # # 第四层卷积 + maxpooling
    # W_conv4 = weight_variable([5, 5, 64, 64])
    # b_conv4 = bias_variable([64])
    #
    # h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv3)
    # h_pool4 = max_pool_2x2(h_conv3)
    # 全连接层
    W_fc1 = weight_variable([50 * 50 * 128, 2])
    b_fc1 = bias_variable([2])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 50 * 50 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # dropout层，只能在训练中使用，而不能用于测试过程
    # keep_prob = tf.placeholder(tf.float32)  # 表示一个神经元的输出在dropout时不被丢弃的概率
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 模型训练和测试
    # y_ = tf.convert_to_tensor(y_, dtype=tf.float32)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=h_fc1)  # compute cost
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

    accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
        labels=tf.argmax(y_, axis=1), predictions=tf.argmax(h_fc1, axis=1), )[1]



    # cross_entropy = tf.reduce_sum(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=h_fc1))
    # train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(h_fc1, 1), tf.argmax(y_, 1))

    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    batch_X = getbatch_X(train_x, 64)
    batch_y = getbatch_y(train_y, 64)
    for i in range(50):
        y_pred, _, loss, real_y = sess.run([h_fc1, train_step, cross_entropy,y_],
                                   feed_dict={x: batch_X.__next__(), y_: batch_y.__next__()})
        if i % 1 == 0:
            print('i:', i)
            print(y_pred)
            # print("result={}".format(result))
            print("real_y={}".format(real_y))
            print("cross_entropy={}".format(loss))
            # print("train_y",train_y)

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: test_x, y_: test_y}))


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

train_num = 100
train_x = y[:train_num]

guaidian_x = guaidian_result(train_x)
train_x_index, train_y = culIndex(guaidian_x)
train_y = y[:train_num]
# # train set 画图
path_train = 'img_train/'
train_x = []
for i in range(20,20+train_num):
    drawCandlestick(i, path_train)
    train_x.append(mpimg.imread(path_train + str(int(i)) + ".jpg"))  # 64*64*3
# # train set图像压缩
# files = os.listdir(path_train)  # 得到文件夹下的所有文件名称
# for file in files:
#     path_out_train = 'img_train_compress/' + file
#     image = cv.imread(path_train + '/' + file)
#     new_image = cv.resize(image, (144, 144))
#     cv.imwrite(path_out_train, new_image)
# # train set图像读取
# path_out_train = 'img_train_compress'
# files = os.listdir(path_train)
# train_x = []
# for file in files:
#     train_x.append(mpimg.imread(path_train + '/' + file))  # 64*64*3

# train_x = numpy.reshape(numpy.array(train_x), (-1, 144, 144, 3))/255
train_x = numpy.array(train_x)
# print(train_x.shape)
train_x = numpy.reshape(train_x, (-1, 200, 200, 3))/255

train_y = one_hot(train_y)
train_y = numpy.array(train_y).astype(numpy.float32)
#train_y = OneHotEncoder().fit_transform(train_y).todense()

# # test set 画图
testNum = 5
test_x_index = [i for i in range(len(y) - testNum, len(y))]
test_y = y[train_num:train_num+testNum]
#
path_test = 'img_test/'
test_x = []
for i in test_x_index:
    drawCandlestick(i, path_test)
    test_x.append(mpimg.imread(path_test + str(int(i)) + ".jpg"))  # 64*64*3
# # test set图像压缩
# files = os.listdir(path_test)  # 得到文件夹下的所有文件名称
# for file in files:
#     path_out_test = 'img_test_compress/' + file
#     image = cv.imread(path_test + '/' + file)
#     new_image = cv.resize(image, (144, 144))
#     cv.imwrite(path_out_test, new_image)
# # train set图像读取
# path_out = 'img_test_compress'
# files = os.listdir(path_out)
# test_x = []
# for file in files:
#     test_x.append(mpimg.imread(path_out + '/' + file))  # 64*64*3
#
# # test_x = numpy.reshape(numpy.array(test_x), (-1, 144, 144, 3))/255
test_x = numpy.array(test_x)
test_x = numpy.reshape(test_x, (-1, 200, 200, 3))/255
test_y = one_hot(test_y)
test_y = numpy.array(test_y).astype(numpy.float32)

cnn(train_x, train_y, test_x, test_y)
