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

train_num = 2000
train_y = y[20:20+train_num]
# # train set 画图
path_train = 'img_train_use/'
train_x = []
gc.collect()
# for i in range(20+20500,20+train_num+20500):
#     drawCandlestick(i, path_train)
#     if i%400==0:
#         gc.collect()
files = os.listdir(path_train)  # 得到文件夹下的所有文件名称
files.sort(key=lambda x: int(x[:-4]))  # 对文件进行排序读取
count =0
for file in files:
    # print(file)
    train_x.append(mpimg.imread(path_train + file))  # 64*64*3
    count=count+1
    if count%200==0:
        gc.collect()
train_x = numpy.array(train_x)
train_x = numpy.reshape(train_x, (-1, 200, 200, 3)) / 255

train_y = one_hot(train_y)
train_y = numpy.array(train_y).astype(numpy.float32)

# # test set 画图
testNum = 100
test_x_index = [i for i in range(train_num, train_num + testNum)]
test_y = y[20+train_num:20+train_num + testNum]
#
path_test = 'img_test/'
test_x = []
#for i in test_x_index:
#    drawCandlestick(i, path_test)
files = os.listdir(path_test)  # 得到文件夹下的所有文件名称
files.sort(key=lambda x: int(x[:-4]))  # 对文件进行排序读取
for file in files:
    test_x.append(mpimg.imread(path_test + file))  # 64*64*3
test_x = numpy.array(test_x)
test_x = numpy.reshape(test_x, (-1, 200, 200, 3)) / 255
test_y = one_hot(test_y)
test_y = numpy.array(test_y).astype(numpy.float32)
# batch_X = getbatch_X(train_x, 64)
# batch_y = getbatch_y(train_y, 64)
# ------------------------------------------------------------------------------------
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w', shape=[kh, kw, n_in, n_out], dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = 'SAME')
        bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
        biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name = scope)
        p += [kernel, biases]
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name= scope)
        p += [kernel, biases]
        return activation

def inference_op(input_op,keep_prob):
    # 初始化参数p列表
    p = []

    with tf.device('/cpu:0'):
        # 第一段卷积的第一个卷积层 卷积核3*3，共64个卷积核（输出通道数），步长1*1
        # input_op：200*200*3 输出尺寸200*200*64
        conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1,
                          dw=1, p=p)

        # 第一段卷积的第2个卷积层 卷积核3*3，共64个卷积核（输出通道数），步长1*1
        # input_op：200*200*64 输出尺寸200*200*64
        conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1,
                          dw=1, p=p)

        # 第一段卷积的pooling层，核2*2，步长2*2
        # input_op：200*200*64 输出尺寸100*100*64
        pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

        # 第2段卷积的第一个卷积层 卷积核3*3，共128个卷积核（输出通道数），步长1*1
        # input_op：100*100*64 输出尺寸100*100*128
        conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1,
                          dw=1, p=p)

        # input_op：100*100*128 输出尺寸100*100*128
        conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1,
                          dw=1, p=p)

        # input_op：100*100*128 输出尺寸50*50*128
        pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

        # 第3段卷积的第一个卷积层 卷积核3*3，共256个卷积核（输出通道数），步长1*1
        # input_op：50*50*128 输出尺寸50*50*256
        conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1,
                          dw=1, p=p)

        # input_op：50*50*256 输出尺寸50*50*256
        conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1,
                          dw=1, p=p)

        # input_op：50*50*256 输出尺寸50*50*256
        conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1,
                          dw=1, p=p)

        # input_op：50*50*256 输出尺寸25*25*256
        pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

        # 第3段卷积的第一个卷积层 卷积核3*3，共512个卷积核（输出通道数），步长1*1
        # input_op：25*25*256 输出尺寸25*25*512
        conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1,
                          dw=1, p=p)

        # input_op：25*25*512 输出尺寸25*25*512
        conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1,
                          dw=1, p=p)

        # input_op：25*25*512 输出尺寸25*25*512
        conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1,
                          dw=1, p=p)

        # input_op：25*25*512 输出尺寸12*12*512
        pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

        # 第5段卷积的第一个卷积层 卷积核3*3，共512个卷积核（输出通道数），步长1*1
        # input_op：12*12*512 输出尺寸12*12*512
        conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1,
                          dw=1, p=p)

        # input_op：12*12*512 输出尺寸12*12*512
        conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1,
                          dw=1, p=p)

        # input_op：12*12*512 输出尺寸12*12*512
        conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1,
                          dw=1, p=p)

        # input_op：12*12*512 输出尺寸6*6*512
        pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)

        shp = pool5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        # tf.reshape(tensor, shape, name=None) 将tensor变换为参数shape的形式。
        resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

        # 全连接层，隐藏节点数为4096,后面接一个dropout层，训练时保留率为0.5，预测时为1.0
        fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
        fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

        # 全连接层，隐藏节点数为4096,后面接一个dropout层，训练时保留率为0.5，预测时为1.0
        fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
        fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

        # 最后是一个1000个输出节点的全连接层，
        # 利用softmax输出分类概率，argmax输出概率最大的类别。
        fc8 = fc_op(fc7_drop, name="fc8", n_out=2, p=p)
        softmax = tf.nn.softmax(fc8)
        predictions = tf.argmax(softmax, 1)
        # return predictions, softmax, fc8, p
        return predictions

batch_size=32
x = tf.placeholder(tf.float32, shape=[None, 200, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
predictions_train = inference_op(x,0.8)
predictions_test = inference_op(x,1)
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=predictions)  # compute cost
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(y_, axis=1), predictions=tf.argmax(predictions_train, axis=1), )[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init_op)


for i in range(8000):
    # x_train = batch_X.__next__()
    # y_train = batch_y.__next__()
    b_x = train_x[i * batch_size : (i + 1) * batch_size]
    b_y = train_y[i * batch_size : (i + 1) * batch_size]
    y_pred, _, loss, = sess.run([predictions_train, train_step, cross_entropy],
                                feed_dict={x: b_x, y_: b_y})
    # print(y_pred)
    if i % 50 == 0:
        accuracy_, _ = sess.run([accuracy,predictions_test], feed_dict={x: test_x, y_: test_y})
        print('Step:', i, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy_)
        # print(y_pred)
        # print("~~~loss: ", loss)
