import os
from PIL import Image
import numpy as np
import random
import time
import math
import regression as rg

def test_adb_devices():
    infor = os.popen('adb devices')
    outputs = infor.readlines()
    print('检查设备连接情况：')
    for line in outputs:
        print(line)

def adb_get_screenshot():
    cmd = 'adb shell screencap -p /sdcard/screenshot.png'
    os.system(cmd)
    cmd = 'adb pull /sdcard/screenshot.png ./screenshot.png'
    os.system(cmd)
    im = Image.open('./screenshot.png').convert('RGB')
    return im

def get_distance(im, count):
    print('get distance')
    w,h = im.size
    print(w)
    print(h)
    pix = im.load()
    #目标台面的中心坐标
    target_x = 0
    target_y = 0
    #小人位置坐标
    mov_x = 0
    mov_y = 0
    start_y = 0

    #缩小小人搜索范围
    step = 100
    for y in range(int(h/3), int(2*h/3), step):
        first_pix = pix[0, y]
        for x in range(1,w):
            if pix[x,y] != first_pix:
                start_y = y-step/2
                break
        if start_y:
            break
    points = []
    max_y = 0
    for y in range(int(start_y), int(2*h/3)):
        for x in range(int(w/8), int(w-w/8)):
            r,g,b = pix[x, y]
            if(r in range(50,60) and g in range(50,63) and b in range(80, 105)):
                points.append((x,y))
                max_y = max(max_y, y)

    bottom_x = [x for x,y in points if y==max_y]
    mov_x =int(sum(bottom_x)/len(bottom_x))
    mov_y = max_y - 30 #减去小人底座高度

    #画出小人位置
    # for x in range(mov_x-5, mov_x+5):
    #     for y in range(mov_y-5, mov_y+5):
    #         pix[x,y] = (255,0,0)
    #
    # for x in range(mov_x+50-5, mov_x+50+5):
    #     for y in range(mov_y-5, mov_y+5):
    #         pix[x,y] = (255,0,0)

        # 目标位置与上一个框框位置在一条直线上，斜率固定，而非与小人在一条直线上
        # 猜测一个中心位置，并不在图片中点
    # center_x = w / 2 + w / 45
    # center_y = h / 2 + h / 105

    center_x = w / 2 + 37
    center_y = h / 2 + 25

    center_x2 = w / 2
    center_y2 = h / 2

    # 画出中心点位置
    for x in range(int(center_x2) - 5, int(center_x2) + 5):
        for y in range(int(center_y2) - 5, int(center_y2) + 5):
            pix[x, y] = (255, 0, 0)



    # 画出中心点位置
    for x in range(int(center_x) - 5, int(center_x) + 5):
        for y in range(int(center_y) - 5, int(center_y) + 5):
            pix[x, y] = (0, 0, 255)

    #搜索目标方块位置，缩小搜索范围
    x_start = 0
    x_end = 0
    if mov_x < w/2:
        x_start = max(w/2+ w / 45, mov_x+55)
        x_end = w - w/8
    else:
        x_start = w/8
        x_end = min(w/2+ w / 45,mov_x-55)

    #根据目标上面划线，求平均获得target_x
    for y in  range(int(h/3), int(2*h/3)):
        first_pix = pix[0,y]
        if target_x or target_y:
            break
        x_sum = 0
        x_num = 0
        for x in range(int(x_start),int(x_end)):
            pix2 = pix[x+5, y + 30]
            pix1 = pix[x,y]
            if abs(pix1[0] - first_pix[0]) + abs(pix1[1] - first_pix[1]) + abs(pix1[2] - first_pix[2]) > 20 \
                and abs(pix2[0] - first_pix[0]) + abs(pix2[1] - first_pix[1]) + abs(pix2[2] - first_pix[2]) > 20:
                x_sum += x
                x_num += 1
        if x_sum:
            target_x = int(x_sum/x_num)
            #画目标框框最上面的线
            # for i in range(w):
            #     pix[i,y] = (0,0,0)
            break


    #根据中心位置坐标，以及目标x坐标，求出y坐标
    delta_y = 0
    if mov_x > w/2:
        target_y = round((25.5/43.5) * (target_x - center_x) + center_y) #直线斜率是25.5/43.5
        delta_y = mov_y - round((25.5/43.5) * (mov_x - center_x) + center_y)
        for x in range(w):
            for y in range(int(h / 3), int(2 * h / 3)):
                if y ==  round((25.5/43.5) * (x - center_x) + center_y) :
                    pix[x, y] = (0, 0, 0)
    else:
        target_y = round((-25.5 / 43.5) * (target_x - center_x) + center_y)
        delta_y = mov_y - round((-25.5 / 43.5) * (mov_x - center_x) + center_y)
        for x in range(w):
            for y in range(int(h / 3), int(2 * h / 3)):
                if y ==  round((-25.5/43.5) * (x - center_x) + center_y) :
                    pix[x, y] = (0, 0, 0)

    #画出目标坐标位置
    for x in range(int(target_x-5), target_x+5):
        for y in range(target_y-5, target_y+5):
            pix[x,y] = (0,255,0)

    #求距离
    distance = math.sqrt(((mov_x - target_x) ** 2) * 2 / 3 + ((mov_y - target_y) ** 2) * 2)
    #distance = math.sqrt((mov_x - target_x)**2 + (mov_y - target_y)**2)

    return im, distance, delta_y


def get_press_time(distance, coef, intercept,cout, model, delta_y):
    #计算按压时间
    if (cout < 20):
        press_time = int(distance * coef + intercept)
    else:
        press_time = rg.get_pred(distance, model=model)
    return max(int(press_time), 200)

def get_press_time_nonlinear(distance, coef, intercept, cout, model):
    distance_square = pow(distance, 2)
    distance_square_x = np.array([[distance, distance_square]])
    #计算按压时间
    if(cout<20):
        intercept = np.random.rand(-10,10)
        press_time = int(distance * coef + intercept)
    else:
        press_time = rg.get_pred(distance_square_x, model=model)
    return max(int(press_time), 200)


def get_press_time_array(num):
    press_time_array = []
    train_path = './train_data' + str(num) + '/'
    press_time_file = train_path + 'press_time.npy'
    if os.path.isfile(press_time_file) and os.path.exists(train_path):
        press_time_array = np.load(press_time_file).tolist()
    else:
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.isfile(press_time_file):
            np.save(press_time_file, press_time_array)

    return press_time_array


def get_distance_array(num):
    distance_array = []
    train_path = './train_data' + str(num) + '/'
    distance_file = train_path+'distance.npy'
    if os.path.isfile(distance_file) and os.path.exists(train_path):
        distance_array = np.load(distance_file).tolist()
    else:
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.isfile(distance_file):
            np.save(distance_file, distance_array)
    return distance_array


def save_data(press_time_array,distance_array,im, num, count):
    train_path = './train_data' + str(num) + '/'
    press_time_file = train_path + 'press_time.npy'
    distance_file = train_path + 'distance.npy'
    im_file = train_path + str(count) + '.jpg'

    w,h = im.size
    # box = (w / 8, h / 2 - w * 3 / 8, w * 7 / 8, h / 2 + w * 3 / 8)
    # im2 = im.crop(box)
    im.save(im_file)
    np.save(press_time_file, press_time_array)
    np.save(distance_file, distance_array)


def jump(press_time, im):
    #按压位置为开始游戏按钮的位置
    w,h = im.size
    left = int(w/2)
    top = int(1584* (h /1920))
    right = int(random.uniform(left-200,left+200))
    bottom = int(random.uniform(top-200,top+200))
    cmd = 'adb shell input swipe ' + str(left)+ ' '+ str(top) +' '+str(right) +' '+ str(bottom) +' '+ str(press_time)
    os.system(cmd)
    print(cmd)


def get_coefts(press_time_array, distance_array, count, defalt_coef, defalt_inter, model):
    coef = 0
    intercept = 0
    if count < 20:
        coef = defalt_coef
        intercept = defalt_inter
    elif count<100:
        coef, intercept = rg.get_coef( distance_array,press_time_array, model)
    else:
        local_press_time_array = press_time_array[-100:]
        local_distance_array = distance_array[-100:]
        coef, intercept = rg.get_coef(local_distance_array,local_press_time_array, model)
    return coef, intercept

def get_coefts_nonlinear(press_time_array, distance_array, count, defalt_coef, defalt_inter, model):
    distance_array = np.array(distance_array)
    distance_square = pow(distance_array,2)
    coef=[]
    intercept =[]
    size = len(distance_array)
    print(size)
    temp_square_x = np.append(distance_array, distance_square)
    temp_square_x = np.array(temp_square_x).reshape(2,size)
    if count < 20:
        coef = defalt_coef
        intercept = defalt_inter
    else:
        rg.get_coef(temp_square_x, press_time_array, model=model)
    return coef, intercept


def main():
    print("Start auto jump!")
    num = input("请输入本次实验序号：")
    max_step = 1000

    distance_array = get_distance_array(num)
    press_time_array = get_press_time_array(num)
    count = np.size(press_time_array)

    # coef, intercept = get_coefts(press_time_array, distance_array, count, 1.38, 200)
    # print(coef)
    # print(intercept)

    # for i in range(10):
    #     im = Image.open('./temp/image_'+str(count)+'.png').convert('RGB')
    #     distance, delta_y = get_distance(im,count)
    #     count += 1
    #     coef, intercept = get_coefts(press_time_array, distance_array, count, 1.38, delta_y)
    #     press_time = get_press_time(distance, coef, intercept)

    #jump(press_time, im)

    #print(distance)
    for i in range(max_step):
        im = adb_get_screenshot()
        print('Count: '+ str(count))
        im, distance, delta_y = get_distance(im, count)
        print('Distance:'+ str(distance))

        #linear ridge regression
        model = rg.ridg
        coef, intercept = get_coefts(press_time_array, distance_array, count, 0.949, delta_y, model)
        print('Coef: {}'.format(coef))
        print('Intercept: {}'.format(intercept))
        test = np.array([[distance]])
        #press_time = rg.get_pred(test)
        press_time = get_press_time(distance, coef, intercept,count, model, delta_y)
        print('Press time:' + str(press_time))

        #nonlinear
        #get_coefts_nonlinear(press_time_array, distance_array, count, 1.38, delta_y,model)
        #coef, intercept = get_coefts_nonlinear(press_time_array, distance_array, count, 0.99, delta_y)
        # print('Coef: {}'.format(coef))
        # print('Intercept: {}'.format(intercept))

        #press_time = get_press_time_nonlinear(distance, coef, intercept, count, model)
        #print('Press time:' + str(press_time))

        jump(press_time,im)

        press_time_array.append(press_time)
        distance_array.append(distance)

        save_data(press_time_array, distance_array, im, num, count)

        time.sleep(random.uniform(1.2, 1.5))
        count += 1


if __name__ == '__main__':
    #test_adb_devices()
    main()
