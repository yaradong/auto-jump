# -*- coding: utf-8 -*-
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import math
import time
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#object detection import
from utils import label_map_util
from utils import visualization_utils as vis_util

import regression as rg
import auto_jump as aj

#model preparation
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'frozen_inference_graph_50000.pb'
#PATH_TO_CKPT = './model/frozen_inference_graph_frcnn_inception_v2_coco.pb'
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = './model/wechat_jump_label_map.pbtxt'
PATH_TO_LABELS = 'object-detection.pbtxt'

NUM_CLASSES = 7

#load model
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#     #serialized_graph = fid.read()
#     od_graph_def.ParseFromString(fid.read())
#     tf.import_graph_def(od_graph_def, name='')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
print("load model success")

#Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("load label success")

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def adb_get_screenshot():
    cmd = 'adb shell screencap -p /sdcard/screenshot.png'
    os.system(cmd)
    cmd = 'adb pull /sdcard/screenshot.png ./screenshot.png'
    os.system(cmd)
    im = Image.open('./screenshot.png').convert('RGB')
    return im


def get_position(boxes, scores, classes, category_index, num):
    chs_pos = [1, 1, 1, 1]
    tgt_pos = [1, 1, 1, 1]
    target_type = ''
    y_min = 1
    min_score_thresh = .5

    for i in range(int(num[0])):
        if scores[i] > min_score_thresh:
            if boxes[i][0] < 0.3 or boxes[i][2] > 0.8:
                continue
            if category_index[classes[i]]['name'] == 'chess' and scores[i] > 0.9:
                if chs_pos[0] == 1.0 or (boxes[i][3]-boxes[i][1]) < (chs_pos[3]-chs_pos[1]):
                    chs_pos = boxes[i]
            elif boxes[i][0] < y_min:
                tgt_pos = boxes[i]
                y_min = boxes[i][0]
                target_type = category_index[classes[i]]['name']

    return chs_pos, tgt_pos, target_type

def cal_distance(chs_pos,tgt_pos,im_width,im_height):
    chs_x = ((chs_pos[1] + chs_pos[3])/2 )* im_width
    chs_y = chs_pos[2] * im_height - 25

    tgt_x = ((tgt_pos[1] + tgt_pos[3])/2)* im_width
    tgt_y = ((tgt_pos[0] + tgt_pos[2])/2)* im_height

    #distance = math.sqrt((chs_x - tgt_x) ** 2 + (chs_y - tgt_y) ** 2)
    distance = math.sqrt(((chs_x - tgt_x) ** 2) * 2/3 + ((chs_y - tgt_y) ** 2) * 2)
    return distance

def save_data(press_time_array,distance_array,im_np, num, count):
    train_path = './train_data' + str(num) + '/'
    press_time_file = train_path + 'press_time.npy'
    distance_file = train_path + 'distance.npy'
    im_file = train_path + str(count) + '.jpg'

    cv2.imwrite(im_file, cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR))
    np.save(press_time_file, press_time_array)
    np.save(distance_file, distance_array)

with detection_graph.as_default():
    dir_num = input("请输入本次实验序号：")
    max_step = 1000

    distance_array = aj.get_distance_array(dir_num)
    press_time_array = aj.get_press_time_array(dir_num)
    count = np.size(press_time_array)

    with tf.Session() as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            im = adb_get_screenshot()
            (im_width, im_height) = im.size

            im_np = load_image_into_numpy_array(im)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(im_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes = np.reshape(boxes, (-1, boxes.shape[-1]))
            scores = np.reshape(scores, (-1))
            classes = np.reshape(classes, (-1)).astype(np.int32)

            chs_pos, tgt_pos, target_type = get_position(boxes, scores, classes, category_index, num)
            distance = cal_distance(chs_pos,tgt_pos,im_width,im_height)

            # linear ridge regression
            model = rg.ridg
            delta_y = np.random.uniform(-5,-2)

            coef, intercept = aj.get_coefts(press_time_array, distance_array, count, 0.94514905, delta_y, model)
            print('Coef: {}'.format(coef))
            print('Intercept: {}'.format(intercept))
            test = np.array([[distance]])
            # press_time = rg.get_pred(test)
            press_time = aj.get_press_time(distance, coef, intercept, count, model, delta_y)
            print('Press time:' + str(press_time))
            aj.jump(press_time, im)

            vis_util.visualize_boxes_and_labels_on_image_array(im_np, boxes, classes, scores, category_index,
                                                               use_normalized_coordinates=True, line_thickness=8)
            press_time_array.append(press_time)
            distance_array.append(distance)
            save_data(press_time_array, distance_array, im_np, dir_num, count)

            if target_type in ['waste', 'magic', 'shop', 'music']:
                time.sleep(np.random.uniform(1.5, 2))
            else:
                time.sleep(np.random.uniform(1.0, 1.2))

            count += 1







