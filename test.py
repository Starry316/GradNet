from glob import glob

import cv2
import tensorflow as tf
import numpy as np
import time
from numpy import inf
from data_generator import data_generator
from util import data_loss, first_order_loss, grad_loss, calc_grad_x, calc_grad_y, \
    get_features_array, get_color_array, preprocess_data, get_grad_array, revert_hdr, hdr_compression, revert_hdr_tf, \
    hdr_compression_tf,writeEXR
from model import GradNet
import os

data_dir = 'data/test/'
filepath_weights = 'results/xxx'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def get_testdata():
    test_data_list = []
    dir_list = []
    colorfiles = glob(os.path.join(data_dir, '*.exr'))
    for f in colorfiles:

        scene = f.split("/")[-1].split("\\")[-1].split('.')[0]
        print(scene)

        input_color =get_color_array(f)
        height = input_color.shape[0]
        width = input_color.shape[1]
        input_shape = (height, width, 7)

        features_path = data_dir+scene+'_features.txt'
        input_features = get_features_array(input_shape, features_path)


        grad = get_grad_array(data_dir, scene)
        # preprocessing the data
        data = preprocess_data( input_color,input_features,grad, input_shape)
        outputDir = 'data/test/res/%s/'%scene

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
            writeEXR(input_color, (outputDir + 'origin.exr'))
            writeEXR(input_features[:, :, 0:3], (outputDir + 'normal.exr'))
            writeEXR(input_features[:, :, 4:7], (outputDir + 'albedo.exr'))
            writeEXR(grad[:, :, 0:3], (outputDir + 'gx.exr'))
            writeEXR(grad[:, :, 3:6], (outputDir + 'gy.exr'))


        test_data_list.append(data)
        # test_grad_list.append(grad)
        dir_list.append(outputDir)

    return  test_data_list,dir_list


def denoiseImage(model,weight_name):
    for data,outputDir in zip(test_data,outputdir):
        r = np.concatenate((data[:,:,:1], data[:,:,3:10],data[:,:,10:11],data[:,:,13:14]), axis=-1)
        g = np.concatenate((data[:,:,1:2], data[:,:,3:10],data[:,:,11:12],data[:,:,14:15]), axis=-1)
        b = np.concatenate((data[:,:,2:3], data[:,:,3:10],data[:,:,12:13],data[:,:,15:16]), axis=-1)

        r = model.predict(tf.expand_dims(r, 0), steps=1)
        g = model.predict(tf.expand_dims(g, 0), steps=1)
        b = model.predict(tf.expand_dims(b, 0), steps=1)
        prediction = tf.concat((r[0],g[0],b[0]),axis=-1)
        prediction = revert_hdr(prediction)
        writeEXR(prediction.numpy(),outputDir + weight_name + '.exr')
        # cv2.imwrite(outputDir + weight_name + '.exr', cv2.cvtColor(prediction.numpy().astype(np.float32), cv2.COLOR_RGB2BGR))
        print('writing to ' +outputDir + weight_name + '.exr')


if __name__ == '__main__':
    test_data, outputdir = get_testdata()

    weight_name = filepath_weights.split('/')[-1].split('.ckpt')[0]
    print(weight_name)
    model = GradNet()
    tf.train.Checkpoint(grad_model = model).restore(filepath_weights)
    denoiseImage(model,weight_name)