import math
import random
import tensorflow as tf
from glob import glob
import threading
import numpy as np
from util import get_color_array, get_features_array, preprocess_data, \
    get_grad_array , hdr_compression
import os


outputDir = 'E:\\Python3 Projects\\GradNet\\data\\train\\'
scenelist = ['bathroom', 'bathroom2', 'bedroom', 'classroom', 'diningroom', 'living-room', 'living-room-2','staircase','kitchen']

index_lock = threading.Lock()
index = 0
count = 0

def write_record(data_list, index):
    tfrecord_dir = (outputDir + 'train_%s.tfrecords') % (index)
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        for data in data_list:
            feature = {  # 建立 tf.train.Feature 字典
                'data': tf.train.Feature(float_list=tf.train.FloatList(value=data.reshape(-1)))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
            writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件

def get_data(f):
    file_name = f.split("/")[-1].split("\\")[-1].split('.')[0]
    dir = f.split(file_name)[0]

    input_color = get_color_array(f)
    height = input_color.shape[0]
    width = input_color.shape[1]
    # 特征的维度
    input_shape = (height, width, 7)
    # 读入特征
    features_path = dir + file_name + '_features.txt'
    input_features = get_features_array(input_shape, features_path)
    # preprocessing the data
    data = preprocess_data(input_color, input_features, input_shape)
    grad = get_grad_array(dir, file_name)
    grad = hdr_compression(grad)
    return np.concatenate((data, grad), axis=2)


def find_all_file(shuffle=True):
    file_list = []
    for i in scenelist:
        path = 'E:\\Rendering\\scene\\fortrain\\data\\%s\\' % (i)
        list = np.array(glob(os.path.join(path, '*.exr')))
        file_list.append(list)

    file_list = np.concatenate(file_list, axis=0)
    if shuffle:
        np.random.shuffle(file_list)
    return file_list



def create_record(list):
    global index, count
    patch_count = 0
    patch_list = []
    np.random.shuffle(list)
    for i in list:
        data = get_data(i)
        for _ in range (5):
            x = random.randint(0, data.shape[0] - 257)
            y = random.randint(0, data.shape[1] - 257)
            patch_list.append(np.array([data[x:x + 256, y:y + 256, :]]))
            patch_count += 1

            if patch_count == 256:
                index_lock.acquire()
                index += 1
                count += 256
                print(count, '/', total)
                index_ = index
                index_lock.release()
                np.random.shuffle(patch_list)
                write_record(patch_list, index_)

                patch_count = 0
                patch_list = []

    if patch_count > 0:
        index_lock.acquire()
        index += 1
        index_ = index
        count += patch_count
        print(count, '/', total)
        index_lock.release()
        np.random.shuffle(patch_list)
        write_record(patch_list, index_)


list = find_all_file()
total = len(list) * 15

for _ in range(3):
    t = threading.Thread(target=create_record, args=[list])
    t.start()

