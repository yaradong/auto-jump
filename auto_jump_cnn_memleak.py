# -*- coding: UTF-8 -*-
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import time
import math

def add_layer(inputs, in_size, out_size, activation_func = None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.random_normal([1,out_size]))
    wx_plus_b = tf.matmul(inputs,weights) + biases
    if activation_func is None:
        output = wx_plus_b
    else:
        output = activation_func(wx_plus_b)
    return output


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x,w):
    return tf.nn.conv2d(x,w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x,ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

def conv(x, w, b, Norm=False, ActiveFunc = None):
    feature_conv = conv2d(x, w) + b
    if Norm is False:
        result = feature_conv
    else:
        mean, variance = tf.nn.moments(feature_conv, [0, 1, 2])
        scale = tf.Variable(tf.ones(mean.get_shape()))
        shift = tf.Variable(tf.zeros(mean.get_shape()))
        epsilon = 0.001
        result = tf.nn.batch_normalization(feature_conv, mean, variance, shift, scale, epsilon)
    if ActiveFunc is None:
        output = result
    else:
        output = ActiveFunc(result)
    return output

def fc_layer(input, w, b, Norm = False, ActiveFunc= None):
    wx_plus_b = tf.matmul(input,w) + b
    if Norm is False:
        result = wx_plus_b
    else:
        mean, variance = tf.nn.moments(wx_plus_b, [0])
        scale = tf.Variable(tf.ones(mean.get_shape()))
        shift = tf.Variable(tf.zeros(mean.get_shape()))
        epsilon = 0.001
        result = tf.nn.batch_normalization(wx_plus_b, mean, variance, shift, scale, epsilon)
    if ActiveFunc is None:
        output = result
    else:
        output = ActiveFunc(result)
    return output

def get_press_time_array( num):
    train_path = './train_data' + str(num) + '/'
    press_time_file_str = train_path + 'press_time.npy'
    time_press_array = np.load(press_time_file_str)
    # del train_path
    # del press_time_file
    return time_press_array

def get_image(im_raw):
    im = tf.image.decode_jpeg(im_raw)
    im = tf.image.resize_images(im, (200, 200), method=np.random.randint(4))
    im = tf.image.rgb_to_grayscale(im)

    return [im]

def start_train(sess,output,loss,train_op, x, y, keep_prob, learning_rate, saver,press_time_array, num, get_im_data_op, img_raw):
    train_path = './train_data' + str(num) + '/'
    count = len(press_time_array)
    print(count)
    max_epoc = 1000
    for i in range(max_epoc):
        for img_id in range(count):
            im_file_str = train_path + str(img_id) + '.jpg'
            with open(im_file_str, "rb") as f:
                img_raw_str = f.read()
                im_data = sess.run(get_im_data_op,{img_raw: img_raw_str})
            if img_id % 1000 == 0:
                saver.save(sess,"./save/mode.mod")
            y_result = sess.run(output, feed_dict={x:im_data, keep_prob: 1})
            loss_result = sess.run(loss,feed_dict={output:y_result, y:[[np.asarray(press_time_array[img_id])]]})
            print(str(i), str(img_id), 'y_result:', y_result, 'press time:',  press_time_array[img_id], 'loss:',  loss_result)
            sess.run(train_op, feed_dict={x:im_data,output:y_result, y:[[np.asarray(press_time_array[img_id])]], keep_prob: 0.6, learning_rate:0.0002})
            del y_result
            del loss_result
            del im_data


def start_test(sess, output,loss, x, y, keep_prob, time_press_array, num, get_im_data_op, img_raw):
    print('测试结果：')
    test_path = './train_data' + str(num) + '/'
    count = len(time_press_array)
    print(count)
    for file_id in range(count):
        im_file_str = test_path + str(file_id) + '.jpg'
        with open(im_file_str, "rb") as f:
            img_raw_str = f.read()
            im_data = sess.run(get_im_data_op, {img_raw: img_raw_str})

        y_result = sess.run(output, feed_dict={x: im_data, keep_prob: 1})
        loss_result = sess.run(loss, feed_dict={output: y_result, y: [[np.asarray(time_press_array[file_id])]]})
        print(str(file_id), 'y_result:', y_result, 'press time:', time_press_array[file_id], 'loss:',loss_result)

def start_paly(sess, output, loss, x, y, keep_prob, get_im_data_op, img_raw):
    while True:
        process_screenshot()
        im_file_str = './screenshot.jpg'
        with open(im_file_str, "rb") as f:
            img_raw_str = f.read()
            im_data = sess.run(get_im_data_op, {img_raw: img_raw_str})

        y_result = sess.run(output, feed_dict={x: im_data, keep_prob: 1})
        jump(y_result)

def process_screenshot():
    cmd = 'adb shell screencap -p /sdcard/screenshot.png'
    os.system(cmd)
    cmd = 'adb pull /sdcard/screenshot.png ./screenshot.png'
    os.system(cmd)
    im = Image.open('./screenshot.png').convert('RGB')
    w = 1536
    h = 2560
    box = (w / 8, h / 2 - w * 3 / 8, w * 7 / 8, h / 2 + w * 3 / 8)
    im = im.crop(box)
    im.save('./screenshot.jpg')
    im_data = tf.image.decode_jpeg(tf.gfile.FastGFile('./screenshot.jpg','rb').read())
    im_data = tf.image.resize_images(im_data, (200, 200), method=np.random.randint(4))
    im_data = tf.image.rgb_to_grayscale(im_data)
    x_in = np.asarray(im_data.eval(), dtype='float32')

    # [0,255]转为[0,1]浮点
    for i in range(len(x_in)):
        for j in range(len(x_in[i])):
            x_in[i][j][0] /= 255

    # 因为输入shape有batch维度，所以还要套一层
    return [x_in]


def jump(press_time):
    #按压位置为开始游戏按钮的位置
    # w,h = im.size
    w=1536
    h=2560
    left = int(w/2)
    top = int(1536* (h /2560))
    right = int(random.uniform(left-200,left+200))
    bottom = int(random.uniform(top-200,top+200))
    cmd = 'adb shell input swipe ' + str(left)+ ' '+ str(top) +' '+str(right) +' '+ str(bottom) +' '+ str(int(press_time[0][0]))
    os.system(cmd)
    print(cmd)



def main(_):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default() as g:
        x = tf.placeholder(tf.float32, shape=[None,200,200,1])
        y = tf.placeholder(tf.float32,shape=[None,1])

        w_conv1 = weight_variable([5,5,1,32]) #filter size 5x5, 32个filter, 数据通道是1
        b_conv1 = bias_variable([32])
        output_conv1 = conv(x, w_conv1,b_conv1, Norm=True, ActiveFunc=tf.nn.relu)
        output_pool1 = max_pool_2x2(output_conv1) #输出变成100x100 32通道

        w_conv2 = weight_variable([5,5, 32, 32])
        b_conv2 = bias_variable([32])
        output_conv2 = conv(output_pool1, w_conv2, b_conv2, Norm=True, ActiveFunc=tf.nn.relu)
        output_pool2 = max_pool_2x2(output_conv2)#输出变成50x50 32通道

        w_conv3 = weight_variable([5, 5, 32, 32])
        b_conv3 = bias_variable([32])
        output_conv3 = conv(output_pool2, w_conv3, b_conv3, Norm=True, ActiveFunc=tf.nn.relu)
        output_pool3 = max_pool_2x2(output_conv3)#输出变成25x25 32通道

        w_fc1 = weight_variable([25*25*32, 32])#全链接层32个神经元
        b_fc1= bias_variable([32])
        output_flat1 = tf.reshape(output_pool3, [-1, 25*25*32])
        output_fc1 = fc_layer(output_flat1,w_fc1, b_fc1, Norm=False, ActiveFunc=tf.nn.relu)

        #dropout
        keep_prob = tf.placeholder(tf.float32)
        drop_fc1 = tf.nn.dropout(output_fc1, keep_prob)

        #输出层
        w_fc2 = weight_variable([32,1])
        b_fc2 = bias_variable([1])
        output = fc_layer(drop_fc1,w_fc2,b_fc2, Norm=False)

        loss = tf.reduce_mean(tf.square(output - y))
        l2_loss = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(w_conv3) + tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(w_fc2)
        delta = 0.01
        l2_loss = l2_loss * delta
        learning_rate = tf.placeholder(tf.float32)

        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        init_op = tf.global_variables_initializer()
        # saver = tf.train.Saver({"w_conv1":w_conv1, "b_conv1":b_conv1, "w_conv2":w_conv2, "b_conv2":b_conv2,
        #                         "w_conv3":w_conv3, "b_conv3":b_conv3, "w_fc1":w_fc1, "b_fc1":b_fc1,
        #                         "w_fc2": w_fc2})
        saver = tf.train.Saver(tf.global_variables())

        train_or_test = input('开始训练，请输出入1；开始测试，请输入2；在手机上玩，请输入3：')
        while int(train_or_test) != 1 and int(train_or_test) != 2 and int(train_or_test) != 3:
            train_or_test = input('请重新输入：')
        if int(train_or_test)==1:
            num = input('请输入训练数据序号：')
            press_time_array = get_press_time_array(num)
        elif int(train_or_test)==2:
            num = input('请输入测试数据序号：')
            press_time_array = get_press_time_array(num)

        img_raw = tf.placeholder(tf.string)
        get_im_data_op = get_image(img_raw)

    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        saver.restore(sess, "./save/mode.mod")

        if int(train_or_test)==1:
            start_train(sess,output,loss,train_op, x, y, keep_prob, learning_rate, saver, press_time_array, num,get_im_data_op,img_raw)
        elif int(train_or_test) == 2:
            # saver.restore(sess, "./save/mode.mod")
            start_test(sess, output, loss, x, y, keep_prob, press_time_array, num,get_im_data_op,img_raw)
        elif int(train_or_test)==3:
            # saver.restore(sess, "./save/mode.mod")
            start_paly(sess, output, loss, x, y, keep_prob, get_im_data_op, img_raw)


if __name__ == '__main__':
    tf.app.run()