import array
import os
from glob import glob

import Imath
import OpenEXR
import numpy as np
import tensorflow as tf



@tf.function
def calc_grad_x(data):
    dX = data[:, :, 1:, :] - data[:, :, :tf.shape(data)[2] - 1, :]
    dX = tf.concat((dX, tf.zeros([tf.shape(data)[0], tf.shape(data)[1], 1, tf.shape(data)[3]])), axis=2)
    return dX

@tf.function
def calc_grad_y(data):
    dY = data[:, 1:, :, :] - data[:, :tf.shape(data)[1] - 1, :, :]
    dY = tf.concat((dY, tf.zeros([tf.shape(data)[0], 1, tf.shape(data)[2], tf.shape(data)[3]])), axis=1)
    return dY


@tf.function
def data_loss(pred, gt):
    return tf.keras.losses.mean_absolute_error(gt, pred)


@tf.function
def grad_loss(pred_gx, pred_gy, g_x, g_y):
    return tf.keras.losses.mean_absolute_error(pred_gx, g_x) \
           + tf.keras.losses.mean_absolute_error(pred_gy, g_y)

@tf.function
def cal_suround(p):
    batch_size =tf.shape(p)[0]
    h = tf.shape(p)[1]
    w = tf.shape(p)[2]
    c = tf.shape(p)[3]

    bottom_p = p[:, 1:, :, :] - p[:, :h - 1, :, :]
    top_p = p[:, :h - 1, :, :] - p[:, 1:, :, :]

    right_p = p[:, :, 1:, :] - p[:, :, :w - 1, :]
    left_p = p[:, :, :w - 1, :] - p[:, :, 1:, :]
    #
    bottom_p = tf.concat((tf.zeros([batch_size, 1, w, c]), bottom_p), axis=1)
    top_p = tf.concat((top_p, tf.zeros([batch_size, 1, w, c])), axis=1)
    right_p = tf.concat((tf.zeros([batch_size, h, 1, c]), right_p), axis=2)
    left_p = tf.concat((left_p, tf.zeros([batch_size, h, 1, c])), axis=2)

    return top_p, bottom_p, left_p, right_p


@tf.function
def first_order_loss(pred, features, ib, G):

    top_i, bottom_i, left_i, right_i = cal_suround(pred)
    top_f, bottom_f, left_f, right_f = cal_suround(features)
    top_ib, bottom_ib, left_ib, right_ib = cal_suround(ib)

    top_w = tf.exp(-9 * tf.square(top_ib))
    bottom_w = tf.exp(-9 * tf.square(bottom_ib))
    left_w = tf.exp(-9 * tf.square(left_ib))
    right_w = tf.exp(-9 * tf.square(right_ib))


    res_top = tf.reduce_sum((G * top_f), -1 )
    res_bottom = tf.reduce_sum((G * bottom_f), -1)
    res_left = tf.reduce_sum((G *  left_f), -1)
    res_right = tf.reduce_sum((G *  right_f), -1)

    res_top = tf.expand_dims(res_top,-1)
    res_bottom = tf.expand_dims(res_bottom, -1)
    res_left = tf.expand_dims(res_left, -1)
    res_right = tf.expand_dims(res_right, -1)



    loss = tf.reduce_mean(top_w * tf.abs(top_i - res_top))
    loss += tf.reduce_mean(bottom_w *tf.abs(bottom_i - res_bottom))
    loss += tf.reduce_mean(left_w * tf.abs(left_i - res_left))
    loss += tf.reduce_mean(right_w * tf.abs(right_i - res_right))


    return loss

@tf.function
def hdr_compression_tf(I):

    return tf.sign(I)*tf.math.log(1+tf.abs(I)*16)/tf.math.log(17.0)

@tf.function
def revert_hdr_tf(I):
    sign = tf.sign(I)
    return sign * ((tf.exp(tf.abs(I)*tf.math.log(17.0))-1)/16)

def hdr_compression(I):
    return np.sign(I)*np.log(1+np.abs(I)*16)/np.log(17.0)

def revert_hdr(I):
    sign = np.sign(I)
    return sign * ((np.exp(np.abs(I)*np.log(17.0))-1)/16)

def data_normalize(data_unnormalized, min_feature, max_feature):
    data_normalized = (data_unnormalized - min_feature) / (max_feature - min_feature)
    return data_normalized

def get_color_array(path):
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    file = OpenEXR.InputFile(path)
    data_window = file.header()['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    r_str = file.channel('R', pixel_type)
    g_str = file.channel('G', pixel_type)
    b_str = file.channel('B', pixel_type)
    r = np.frombuffer(r_str, dtype=np.float32)
    r.shape = (size[1], size[0], 1)
    g = np.frombuffer(g_str, dtype=np.float32)
    g.shape = (size[1], size[0], 1)
    b = np.frombuffer(b_str, dtype=np.float32)
    b.shape = (size[1], size[0], 1)
    rgb = np.concatenate([r, g, b], axis=2)
    return rgb

def get_grad_array(path, name):
    gx = get_color_array(path + 'grad/' + name + '_dx.exr')
    gy = get_color_array(path + 'grad/' + name + '_dy.exr')
    return np.concatenate((gx, gy), axis=2)

def get_features_array(input_shape, path):
    f = open(path, 'r')
    data = np.zeros(input_shape, dtype=np.float32)
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            line = f.readline()
            line = line.split(' ')
            data[i][j] = line[:-1]
    f.close()
    return data


def writeEXR(image, name):
    R = image[:, :, 0].reshape(-1)
    G = image[:, :, 1].reshape(-1)
    B = image[:, :, 2].reshape(-1)
    (Rs, Gs, Bs) = [array.array('f', Chan).tobytes() for Chan in (R, G, B)]
    out = OpenEXR.OutputFile(name, OpenEXR.Header(image.shape[1], image.shape[0]))
    out.writePixels({'R': Rs, 'G': Gs, 'B': Bs})


def preprocess_data(input_color,input_features, input_grad, input_shape):
    input_features[input_features > 1e4] = 1e4
    feature_normal = input_features[:, :, 0:3].reshape(input_shape[0], input_shape[1], 3)
    feature_depth = input_features[:, :, 3:4].reshape(input_shape[0], input_shape[1], 1)
    feature_albedo = input_features[:, :, 4:7].reshape(input_shape[0], input_shape[1], 3)
    input_grad = hdr_compression(input_grad)
    input_color = hdr_compression(input_color)
    feature_normal = (feature_normal + 1) / 2

    data = np.concatenate((input_color,  # 1
                           feature_depth,  # 1
                           feature_normal,  # 3
                           feature_albedo,  # 3
                           input_grad # 2
                           ), axis=2)
    return data


