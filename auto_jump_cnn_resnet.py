# -*- coding: UTF-8 -*-
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import time
import math
import resnet

def add_layer(inputs, in_size, out_size, activation_func = None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.random_normal([1,out_size]))
    wx_plus_b = tf.matmul(inputs,weights) + biases
    if activation_func is None:
        output = wx_plus_b
    else:
        output = activation_func(wx_plus_b)
    return output


def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape),name=name)

def conv2d(x,w):
    return tf.nn.conv2d(x,w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x,ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

def conv(x, w, b,cname, Norm=False, ActiveFunc = None):
    feature_conv = conv2d(x, w) + b
    if Norm is False:
        result = feature_conv
    else:
        # mean, variance = tf.nn.moments(feature_conv, [0, 1, 2])
        # scale = tf.Variable(tf.ones(mean.get_shape()))
        # shift = tf.Variable(tf.zeros(mean.get_shape()))
        # epsilon = 0.001
        # result = tf.nn.batch_normalization(feature_conv, mean, variance, shift, scale, epsilon)
        result = tf.layers.batch_normalization(feature_conv, training=True)
    if ActiveFunc is None:
        output = result
    else:
        output = ActiveFunc(result)
    return output

def fc_layer(input, w, b, fcname, Norm = False, ActiveFunc= None):
    wx_plus_b = tf.matmul(input,w) + b
    if Norm is False:
        result = wx_plus_b
    else:
        # mean, variance = tf.nn.moments(wx_plus_b, [0])
        # scale = tf.Variable(tf.ones(mean.get_shape()))
        # shift = tf.Variable(tf.zeros(mean.get_shape()))
        # epsilon = 0.001
        # result = tf.nn.batch_normalization(wx_plus_b, mean, variance, shift, scale, epsilon)
        result = tf.layers.batch_normalization(wx_plus_b, training=True)
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
    return time_press_array/[1000]

def get_image(im_raw):
    im = tf.image.decode_jpeg(im_raw)
    im = tf.image.resize_images(im, (64, 64), method=np.random.randint(4))
    #im = tf.image.rgb_to_grayscale(im)
    im = tf.image.per_image_standardization(im)
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
                saver.save(sess,"./save5/mode.mod")
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

def start_paly(sess, output, loss, x, y, get_im_data_op, img_raw, keep_prob=1):
    while True:
        process_screenshot()
        im_file_str = './screenshot.jpg'
        with open(im_file_str, "rb") as f:
            img_raw_str = f.read()
            im_data = sess.run(get_im_data_op, {img_raw: img_raw_str})

        y_result = sess.run(output, feed_dict={x: im_data})
        jump(y_result)
        time.sleep(random.uniform(1.2, 1.5))

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
    im_data = tf.image.resize_images(im_data, (64, 64), method=np.random.randint(4))
    #im_data = tf.image.rgb_to_grayscale(im_data)
    im_data = tf.image.per_image_standardization(im_data)
    return [im_data]

def jump(press_time):
    #按压位置为开始游戏按钮的位置
    w=1536
    h=2560
    left = int(w/2)
    top = int(1536* (h /2560))
    right = int(random.uniform(left-200,left+200))
    bottom = int(random.uniform(top-200,top+200))
    cmd = 'adb shell input swipe ' + str(left)+ ' '+ str(top) +' '+str(right) +' '+ str(bottom) +' '+ str(int(press_time[0][0] *1000))
    os.system(cmd)
    print(cmd)

def main(_):

    img_raw = tf.placeholder(tf.string)
    get_im_data_op = get_image(img_raw)

    H_size = 64
    W_size = 64
    C_size = 3
    classes = 1
    assert ((H_size, W_size, C_size) == (64, 64, 3))

    x = tf.placeholder(tf.float32, shape=(None, H_size, W_size, C_size), name='X')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='Y')

    output = resnet.ResNet50_reference(x)

    loss = tf.reduce_mean(tf.square(output - y))

    learning_rate = tf.placeholder(tf.float32)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    init_op = tf.global_variables_initializer()
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += g_list
    # saver = tf.train.Saver({}.fromkeys(var_list).keys())
    saver = tf.train.Saver(g_list)

    checkpoint_dir = './save11/'
    train_or_test = 3
    with tf.Session() as sess:
        sess.run(init_op)
        # saver.restore(sess, checkpoint_dir + "mode.mod")

        if int(train_or_test) == 3:
            start_paly(sess, output, loss, x, y, get_im_data_op, img_raw)
        # if int(train_or_test)==1:
        #     start_train(sess,output,loss,train_op, x, y, keep_prob, learning_rate, saver, press_time_array, num,get_im_data_op,img_raw)
        # elif int(train_or_test) == 2:
        #     # saver.restore(sess, "./save/mode.mod")
        #     start_test(sess, output, loss, x, y, keep_prob, press_time_array, num,get_im_data_op,img_raw)
        # elif int(train_or_test)==3:
        #     # saver.restore(sess, "./save/mode.mod")
        #     start_paly(sess, output, loss, x, y, keep_prob, get_im_data_op, img_raw)


if __name__ == '__main__':
    tf.app.run()