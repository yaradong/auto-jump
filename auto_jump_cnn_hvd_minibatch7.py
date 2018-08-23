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
    im = tf.image.resize_images(im, (320, 320), method=np.random.randint(4))
    im = tf.image.rgb_to_grayscale(im)
    im = tf.image.per_image_standardization(im)
    return im

def get_img_id_list(devices_num, device_id, total_img_count):
    id_list = []
    for id in range(total_img_count):
        if id % devices_num == device_id:
            id_list.append(id)
    random.shuffle(id_list)
    return id_list

def start_train(sess,output,loss,train_op, x, y, keep_prob, learning_rate, saver,press_time_array, num, get_im_data_op, img_raw,batch_size_per_device,checkpoint_dir):
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
            y_result = sess.run(output, feed_dict={x:x_data, keep_prob: 1})
            loss_result = sess.run(loss,feed_dict={output:y_result, y:y_data})
            print(str(i), str(iter_id), 'y_result:', y_result[9], 'press time:',  press_time_array[img_id], 'loss:',  loss_result)
           # print(str(i), str(iter_id),  'loss:',  loss_result)
            sess.run(train_op, feed_dict={x:x_data,output:y_result, y:y_data, keep_prob: 0.7, learning_rate:0.000002})
            del y_result
            del loss_result
        #tf.Print(w_conv1,[w_conv1],message='before restore')
        if device_id==0 :
            #print("before save:",w_conv1.eval())
            saver.save(sess,checkpoint_dir+ "mode.mod")
            print("save:", device_id)


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

        y_result = sess.run(output, feed_dict={x: [im_data], keep_prob: 1})
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
    im.save('./screenshot.jpg')
    im_data = tf.image.decode_jpeg(tf.gfile.FastGFile('./screenshot.jpg','rb').read())
    im_data = tf.image.resize_images(im_data, (320, 320), method=np.random.randint(4))
    im_data = tf.image.rgb_to_grayscale(im_data)
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
    checkpoint_dir = './save10/'
    #if not os.path.exists(checkpoint_dir):
    x = tf.placeholder(tf.float32, shape=[None,320,320,1],name="input")
    y = tf.placeholder(tf.float32,shape=[None,1],name="output")

    w_conv1 = weight_variable([5,5,1,32],"w1") #filter size 5x5, 32个filter, 数据通道是1
    b_conv1 = bias_variable([32],"b1")
    output_conv1 = conv(x, w_conv1,b_conv1,"con1", Norm=True, ActiveFunc=tf.nn.relu)
    output_pool1 = max_pool_2x2(output_conv1) #输出变成100x100 32通道

    w_conv2 = weight_variable([5,5, 32, 32],"w2")
    b_conv2 = bias_variable([32],"b2")
    output_conv2 = conv(output_pool1, w_conv2, b_conv2,"con2", Norm=True, ActiveFunc=tf.nn.relu)
    output_pool2 = max_pool_2x2(output_conv2)#输出变成50x50 32通道

    w_conv3 = weight_variable([5, 5, 32, 32],"w3")
    b_conv3 = bias_variable([32],"b3")
    output_conv3 = conv(output_pool2, w_conv3, b_conv3,"con3", Norm=True, ActiveFunc=tf.nn.relu)
    output_pool3 = max_pool_2x2(output_conv3)#输出变成25x25 32通道
    
    w_conv4 = weight_variable([5, 5, 32, 32],"w3")
    b_conv4 = bias_variable([32],"b3")
    output_conv4 = conv(output_pool3, w_conv4, b_conv4,"con4", Norm=True, ActiveFunc=tf.nn.relu)
    output_pool4 = max_pool_2x2(output_conv4)#轾S佇º住~X彈~P25x25 32轀~Z轁~S

    w_fc1 = weight_variable([20*20*32, 64],"wfc1")#全链接层32个神经元
    b_fc1= bias_variable([64],"bfc1")
    output_flat1 = tf.reshape(output_pool4, [-1, 20*20*32])
    output_fc1 = fc_layer(output_flat1,w_fc1, b_fc1,"fc1", Norm=False, ActiveFunc=tf.nn.relu)

    #dropout
    keep_prob = tf.placeholder(tf.float32,name="kp")
    drop_fc1 = tf.nn.dropout(output_fc1, keep_prob)

    #输出层
    w_fc2 = weight_variable([64,1],"wfc2")
    b_fc2 = bias_variable([1],"bfc2")
    output = fc_layer(drop_fc1,w_fc2,b_fc2,"fc2", Norm=False)

    loss = tf.reduce_mean(tf.square(output - y))
    l2_loss = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(w_conv3) + tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(w_fc2)
    delta = 0.01
    l2_loss = l2_loss * delta
    learning_rate = tf.placeholder(tf.float32)

    #global_step = tf.contrib.framework.get_or_create_global_step()
    #hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    bcast_global_variables_op = hvd.broadcast_global_variables(0)

    opt = tf.train.AdamOptimizer(learning_rate * hvd.size())
    #opt = tf.train.MomentumOptimizer(learning_rate * hvd.size(), 0.9,use_nesterov=True)
    opt = hvd.DistributedOptimizer(opt)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss,global_step=tf.train.get_global_step())
    init_op = tf.global_variables_initializer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
# input('开始训练，请输出入1；开始测试，请输入2；在手机上玩，请输入3：')
    #while int(train_or_test) != 1 and int(train_or_test) != 2 and int(train_or_test) != 3:
     #   train_or_test = input('请重新输入：')
    #if int(train_or_test)==1:
       # num = input('请输入训练数据序号：')
       # press_time_array = get_press_time_array(num)
   # elif int(train_or_test)==2:
       # num = input('请输入测试数据序号：')
       # press_time_array = get_press_time_array(num)
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
            start_train(sess,output,loss,[train_op,update_ops], x, y, keep_prob, learning_rate, saver, press_time_array, num,get_im_data_op,img_raw,batch_size_per_device,checkpoint_dir)
        elif int(train_or_test) == 2:
            # saver.restore(sess, "./save/mode.mod")
            start_test(sess, output, loss, x, y, keep_prob, press_time_array, num,get_im_data_op,img_raw)
        elif int(train_or_test)==3:
            # saver.restore(sess, "./save/mode.mod")
            start_paly(sess, output, loss, x, y, keep_prob, get_im_data_op, img_raw)

if __name__ == '__main__':
    tf.app.run()
#./save/mode.mod.mate
