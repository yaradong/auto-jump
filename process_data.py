from PIL import Image
import numpy as np
import os

def crop_save_img(im, num, idx):
    train_path = './train_data' + str(num) + '/'
    im_file = train_path + str(idx) + '.jpg'

    w = 1080
    h = 1920
    #w,h = im.size
    box = (w / 8, h / 2 - w * 3 / 8, w * 7 / 8, h / 2 + w * 3 / 8)
    im2 = im.crop(box)
    im2.save(im_file)


def get_img(num, idx):
    train_path = './train_data' + str(num) + '/'
    im_file = train_path + str(idx) + '.jpg'
    im = Image.open(im_file)
    return im

def get_time_array(num):
    train_path = './train_data' + str(num) + '/'
    press_time_file = train_path + 'press_time.npy'
    press_time_array = np.load(press_time_file)
    return press_time_array

def save_press_time_array(num, press_time_array):
    train_path = './train_data' + str(num) + '/'
    press_time_file = train_path + 'press_time.npy'
    np.save(press_time_file, press_time_array)

def save_img(target_num,merge_num, idx, count):
    merge_path = './train_data' + str(merge_num) + '/'
    train_path = './train_data' + str(target_num) + '/'
    im_file = merge_path + str(idx) + '.jpg'
    im = Image.open(im_file)
    new_file = train_path + str(idx + count) + '.jpg'
    im.save(new_file)
    os.remove(im_file)

def remove_img(num, idx):
    train_path = './train_data' + str(num) + '/'
    im_file = train_path + str(idx) + '.jpg'
    os.remove(im_file)

def main():
    #打开目标文件夹，获得time array
    target_num = input('please input the target dir num:')
    press_time_array = get_time_array(target_num)
    target_count = len(press_time_array)
    print("target count:",target_count)

    #删除中间几张图片
    for idx in range(2):
        press_time_array = np.delete(press_time_array,5302-idx)
        remove_img(target_num, 5302-idx)
    #将后面的图片向前移动
    # for idx in range(275,283,1):
    #     save_img(target_num,target_num,idx, -2)


    # press_time_array = np.delete(press_time_array,27)
    # press_time_array = np.delete(press_time_array, 573)
    # # save_press_time_array(target_num, press_time_array)
    # remove_img(target_num, 27)
    # remove_img(target_num, 574)

    # 打开要合并的文件夹，获得要融合的数据
    # merge_num = input('please input the num of dir that you want to be merged:')
    # # for idx in range(23):
    # #     save_img(target_num,merge_num,idx, 684)
    # merge_time_array = get_time_array(merge_num)
    # merge_count = len(merge_time_array)
    # print("merge count:", merge_count)

    # merge_time_array = np.delete(merge_time_array,merge_count-1)
    #merge_time_array = np.delete(merge_time_array, merge_count - 2)
    # remove_img(merge_num, merge_count-1)
    #remove_img(merge_num, merge_count - 2)


    # for idx in range(merge_count):
    #     im = get_img(merge_num, idx)
    #     crop_save_img(im, merge_num, idx)
    # for i in range(2):
    #     merge_time_array = np.delete(merge_time_array,25)
    #     remove_img(merge_num, 25+i)
    #
    # merge_count = len(merge_time_array)
    # print("merge count:", merge_count)
    #

    # # #从文件夹merge_num 移动图片到 文件夹target_num
    # for i in range(merge_count):
    #      save_img(target_num, merge_num, i, target_count)
    # # #
    # # #合并time array
    # press_time_array = np.append(press_time_array, merge_time_array)
    print(len(press_time_array))

    save_press_time_array(target_num, press_time_array)

    # merge_path = './train_data' + str(merge_num) + '/'
    # merge_press_time_file = merge_path + 'press_time.npy'
    #
    # merge_time_array = np.load(merge_press_time_file)
    # print(len(merge_time_array))
    #
    # tem = 29
    # for idx in range(3):
    #      press_time_array = np.delete(press_time_array,189+29)
    #     im_file = merge_path + str(tem + idx) + '.jpg'
    #     os.remove(im_file)
    #
    # for idx in range(32, 125):
    #     im_file = merge_path + str(idx) + '.jpg'
    #     im = Image.open(im_file)
    #     new_file = merge_path + str(idx-3) + '.jpg'
    #     im.save(new_file)
    #     os.remove(im_file)
    #
    # for idx in range(0, 122):
    #     save_img(target_num, merge_num, idx, target_count)
    #     im_file = merge_path + str(idx) + '.jpg'
    #     im = Image.open(im_file)
    #     new_file = train_path + str(idx + count) + '.jpg'
    #     im.save(new_file)
    #






if __name__ == '__main__':
    main()