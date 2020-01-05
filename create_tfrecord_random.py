import math
import random
import time

import tensorflow as tf
from glob import glob
import threading
import numpy as np
from util import get_color_array, get_features_array, preprocess_data, \
    get_grad_array
import os
output_dir = 'data/train/'
data_dir = 'scenes/'
scene_list = ['bathroom', 'bathroom2', 'bedroom', 'classroom', 'diningroom', 'living-room', 'living-room-2', 'staircase', 'kitchen']

# index_lock = threading.Lock()
index = 0
count = 0

def get_data(f):
    file_name = f.split("/")[-1].split("\\")[-1].split('.')[0]
    dir = f.split(file_name)[0]
    input_color = get_color_array(f)
    height = input_color.shape[0]
    width = input_color.shape[1]
    input_shape = (height, width, 7)
    features_path = dir + file_name + '_features.txt'
    input_features = get_features_array(input_shape, features_path)
    grad = get_grad_array(dir, file_name)
    # preprocessing the data
    r = preprocess_data(input_color[:,:,:1], input_features, np.concatenate((grad[:,:, :1],grad[:,:,3:4]),axis=-1), input_shape)
    g = preprocess_data(input_color[:,:,1:2], input_features, np.concatenate((grad[:,:, 1:2],grad[:,:,4:5]),axis=-1), input_shape)
    b = preprocess_data(input_color[:,:,2:3], input_features, np.concatenate((grad[:,:, 2:3],grad[:,:,5:6]),axis=-1), input_shape)

    return r,g,b


def find_all_file(shuffle=True):
    file_list = []
    for i in scene_list:
        path = data_dir + i + '/'
        list = np.array(glob(os.path.join(path, '*.exr')))
        file_list.append(list)
    file_list = np.concatenate(file_list, axis=0)
    if shuffle:
        np.random.shuffle(file_list)
    return file_list

def write_record(data_list, index):
    tfrecord_dir = (output_dir + 'train_%s.tfrecords') % (index)
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        for data in data_list:
            feature = {
                'data': tf.train.Feature(float_list=tf.train.FloatList(value=data.reshape(-1)))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def create_record(list):
    global index, count
    patch_count = 0
    patch_list = []
    np.random.shuffle(list)
    for i in list:
        r,g,b = get_data(i)
        for _ in range (15):
            x = random.randint(0, r.shape[0] - 257)
            y = random.randint(0, r.shape[1] - 257)
            patch_list.append(np.array([r[x:x + 256, y:y + 256, :]]))
            patch_list.append(np.array([g[x:x + 256, y:y + 256, :]]))
            patch_list.append(np.array([b[x:x + 256, y:y + 256, :]]))
            patch_count += 3

            if patch_count >= 256:
                index += 1
                count += patch_count
                print(count, '/', total)
                index_ = index
                np.random.shuffle(patch_list)
                write_record(patch_list, index_)

                patch_count = 0
                patch_list = []

    if patch_count > 0:
        index += 1
        index_ = index
        count += patch_count
        print(count, '/', total)
        np.random.shuffle(patch_list)
        write_record(patch_list, index_)


if __name__ == '__main__':
    list = find_all_file()
    total = len(list) * 15 * 3
    print('Creating tfrecords...')
    create_record(list)
    print('Finished, total patches: %d'%total)
    # for _ in range(3):
    #     t = threading.Thread(target=create_record, args=[list])
    #     t.start()

