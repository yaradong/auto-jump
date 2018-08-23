# -*- coding: UTF-8 -*-
import sys
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
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


def weight_variable(shape,wname):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1),name=wname)


def bias_variable(shape,bname):
    return tf.Variable(tf.constant(0.1, shape=shape),name=bname)

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
        # scale = tf.Variable(tf.ones(mean.get_shape()),name=cname+"scale")
        # shift = tf.Variable(tf.zeros(mean.get_shape()),name=cname+"shift")
        # epsilon = 0.001
        # result = tf.nn.batch_normalization(feature_conv, mean, variance, shift, scale, epsilon)
        result = tf.layers.batch_normalization(feature_conv, training=True)
    if ActiveFunc is None:
        output = result
    else:
        output = ActiveFunc(result)
    return output

def fc_layer(input, w, b,fcname, Norm = False, ActiveFunc= None):
    wx_plus_b = tf.matmul(input,w) + b
    if Norm is False:
        result = wx_plus_b
    else:
        # mean, variance = tf.nn.moments(wx_plus_b, [0])
        # scale = tf.Variable(tf.ones(mean.get_shape()),name=fcname+"scale")
        # shift = tf.Variable(tf.zeros(mean.get_shape()),name=fcname+"shift")
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
    time_press_array = time_press_array.astype(np.float32)
    min_time, max_time = time_press_array.min(),time_press_array.max();
    #time_press_array = (time_press_array-min_time)/(max_time - min_time)
    time_press_array = time_press_array/[1000]
    # del train_path
    # del press_time_file
    return time_press_array

def get_image(im_raw):
    im = tf.image.decode_jpeg(im_raw)
    im = tf.image.resize_images(im, (64, 64), method=np.random.randint(4))
    #im = tf.image.rgb_to_grayscale(im)
    im = tf.image.per_image_standardization(im)
    return im

def get_img_id_list(devices_num, device_id, total_img_count):
    id_list = []
    for id in range(total_img_count):
        if id % devices_num == device_id:
            id_list.append(id)
    random.shuffle(id_list)
    return id_list

def start_train(sess,output,loss,train_op, x, y, learning_rate, saver,press_time_array, num, get_im_data_op, img_raw,batch_size_per_device,checkpoint_dir):
    train_path = './train_data' + str(num) + '/'
    count = len(press_time_array)
    print(count)
    # get img id list for this device
    devices_num = hvd.size()
    print("devices_num: ",devices_num)
    device_id = hvd.rank()
    print("device_id:",device_id)
    id_list = get_img_id_list(devices_num, device_id, count)
    print(id_list)

    iter_num_per_epoc = int(len(id_list) / batch_size_per_device)
    max_epoc = 300

    for i in range(max_epoc):
        for iter_id in range(iter_num_per_epoc):
            #get y_data todo
            y_data = []
            x_data = []
            for idx in range(batch_size_per_device):
                #get img id
                img_id = id_list[iter_id * batch_size_per_device + idx]
                #print(img_id)
                im_file_str = train_path + str(img_id) + '.jpg'
                with open(im_file_str, "rb") as f:
                    img_raw_str = f.read()
                    im_data = sess.run(get_im_data_op,{img_raw: img_raw_str})
                    x_data.append(im_data)
                y_data.append([np.asarray(press_time_array[img_id])])
            #if iter_id % (iter_num_per_epoc-1) == 0:
             #   saver.save(sess,"./save/mode.mod")
            y_result = sess.run(output, feed_dict={x:x_data})
            loss_result = sess.run(loss,feed_dict={output:y_result, y:y_data})
            print(str(i), str(iter_id), 'y_result:', y_result[9][0], 'press time:',  press_time_array[img_id], 'loss:',  loss_result)
           # print(str(i), str(iter_id),  'loss:',  loss_result)
            sess.run(train_op, feed_dict={x:x_data,output:y_result, y:y_data, learning_rate:0.000001})
            del y_result
            del loss_result
        #tf.Print(w_conv1,[w_conv1],message='before restore')
        if device_id==0 :
            #print("before save:",w_conv1.eval())
            saver.save(sess,checkpoint_dir+ "mode.mod")
            print("save:", device_id)


def start_test(sess, output,loss, x, y, time_press_array, num, get_im_data_op, img_raw):
    print('测试结果：')
    test_path = './train_data' + str(num) + '/'
    count = len(time_press_array)
    print(count)
    for file_id in range(count):
        im_file_str = test_path + str(file_id) + '.jpg'
        with open(im_file_str, "rb") as f:
            img_raw_str = f.read()
            im_data = sess.run(get_im_data_op, {img_raw: img_raw_str})

        y_result = sess.run(output, feed_dict={x: [im_data]})
        loss_result = sess.run(loss, feed_dict={output: y_result, y: [[np.asarray(time_press_array[file_id])]]})
        print(str(file_id), 'y_result:', y_result, 'press time:', time_press_array[file_id], 'loss:',loss_result)

def start_paly(sess, output, loss, x, y, get_im_data_op, img_raw):
    while True:
        process_screenshot()
        im_file_str = './screenshot.jpg'
        with open(im_file_str, "rb") as f:
            img_raw_str = f.read()
            im_data = sess.run(get_im_data_op, {img_raw: img_raw_str})

        y_result = sess.run(output, feed_dict={x: im_data})
        jump(y_result)

def process_screenshot():
    cmd = 'adb shell screencap -p /sdcard/screenshot.png'
    os.system(cmd)
    cmd = 'adb pull /sdcard/screenshot.png ./screenshot.png'
    os.system(cmd)
    im = Image.open('./screenshot.png').convert('RGB')
    im.save('./screenshot.jpg')
    im_data = tf.image.decode_jpeg(tf.gfile.FastGFile('./screenshot.jpg','rb').read())
    im_data = tf.image.resize_images(im_data, (64, 64), method=np.random.randint(4))
    #im_data = tf.image.rgb_to_grayscale(im_data)
    return [im_data]

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
    hvd.init()
    random.seed(10)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # tf.reset_default_graph()
    # graph = tf.Graph()
    # with graph.as_default() as g:
    checkpoint_dir = './save12/'

    H_size = 64
    W_size = 64
    C_size = 3
    classes = 1
    assert ((H_size, W_size, C_size) == (64, 64, 3))

    x = tf.placeholder(tf.float32, shape=(None, H_size, W_size, C_size), name='X')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='Y')
    output = resnet.ResNet50_reference(x)
    loss = tf.reduce_mean(tf.square(output - y))

    bcast_global_variables_op = hvd.broadcast_global_variables(0)

    learning_rate = tf.placeholder(tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate * hvd.size())
    # opt = tf.train.MomentumOptimizer(learning_rate * hvd.size(), 0.9,use_nesterov=True)
    opt = hvd.DistributedOptimizer(opt)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
    init_op = tf.global_variables_initializer()

    train_or_test = 1
    num = 21
    press_time_array = get_press_time_array(num)
    print("press_time_array:",press_time_array)
    batch_size_per_device = 10

    img_raw = tf.placeholder(tf.string)
    get_im_data_op = get_image(img_raw)

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += g_list
    saver = tf.train.Saver({}.fromkeys(var_list).keys())

    flag = 0
    if os.path.exists(checkpoint_dir) and os.path.isfile(checkpoint_dir +"mode.mod.meta" ):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt.model_checkpoint_path + '.meta')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        flag = 1
    else:
        if (not os.path.exists(checkpoint_dir)) and hvd.rank()==0:
            os.makedirs(checkpoint_dir)
  
    #saver = tf.train.import_meta_graph("./save/mode.mod.meta")
    with tf.Session(config=config) as sess:
        if flag==1:
            saver.restore(sess, checkpoint_dir+"mode.mod")
            #print("after restore:", w_conv1.eval())
            #sess.run(bcast_global_variables_op)
        else:
            sess.run(init_op)
            sess.run(bcast_global_variables_op)
	#tf.Print(w_conv1,[w_conv1],message='After restore')
        
        if int(train_or_test)==1:
            start_train(sess,output,loss,[train_op,update_ops], x, y, learning_rate, saver, press_time_array, num,get_im_data_op,img_raw,batch_size_per_device,checkpoint_dir)
        elif int(train_or_test) == 2:
            # saver.restore(sess, "./save/mode.mod")
            start_test(sess, output, loss, x, y, press_time_array, num,get_im_data_op,img_raw)
        elif int(train_or_test)==3:
            # saver.restore(sess, "./save/mode.mod")
            start_paly(sess, output, loss, x, y, get_im_data_op, img_raw)

if __name__ == '__main__':
    tf.app.run()
#./save/mode.mod.mate
