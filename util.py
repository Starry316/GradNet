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
    # 计算上下左右四个元素和中间元素的差，返回四个矩阵
    batch_size =tf.shape(p)[0]
    h = tf.shape(p)[1]
    w = tf.shape(p)[2]
    c = tf.shape(p)[3]

    bottom_p = p[:, 1:, :, :] - p[:, :h - 1, :, :]
    top_p = p[:, :h - 1, :, :] - p[:, 1:, :, :]

    right_p = p[:, :, 1:, :] - p[:, :, :w - 1, :]
    left_p = p[:, :, :w - 1, :] - p[:, :, 1:, :]

    bottom_p = tf.concat((tf.zeros([batch_size, 1, w, c]), bottom_p), axis=1)
    top_p = tf.concat((top_p, tf.zeros([batch_size, 1, w, c])), axis=1)
    right_p = tf.concat((tf.zeros([batch_size, h, 1, c]), right_p), axis=2)
    left_p = tf.concat((left_p, tf.zeros([batch_size, h, 1, c])), axis=2)

    return top_p, bottom_p, left_p, right_p


@tf.function
def first_order_loss(pred, features, ib, G):
    h = tf.shape(pred)[1]
    w = tf.shape(pred)[2]

    top_i, bottom_i, left_i, right_i = cal_suround(pred)
    top_f, bottom_f, left_f, right_f = cal_suround(features)
    top_ib, bottom_ib, left_ib, right_ib = cal_suround(ib)

    top_w = tf.exp(-9 * tf.square(tf.norm(top_ib, axis=[3, -1])))
    bottom_w = tf.exp(-9 * tf.square(tf.norm(bottom_ib, axis=[3, -1])))
    left_w = tf.exp(-9 * tf.square(tf.norm(left_ib, axis=[3, -1])))
    right_w = tf.exp(-9 * tf.square(tf.norm(right_ib, axis=[3, -1])))

    G = tf.reshape(G, [-1, h, w, 3, 7])

    # feature 的差值, 1e-9为正则项,防止出现NaN
    top_f = tf.expand_dims(top_f, -1) + 1e-9
    bottom_f = tf.expand_dims(bottom_f, -1) + 1e-9
    left_f =  tf.expand_dims(left_f, -1) + 1e-9
    right_f = tf.expand_dims(right_f, -1) + 1e-9

    res_top = tf.matmul(G, top_f)
    res_bottom = tf.matmul(G, bottom_f)
    res_left = tf.matmul(G, left_f)
    res_right = tf.matmul(G, right_f)

    res_top = tf.reshape(res_top, [-1, h, w, 3])
    res_bottom = tf.reshape(res_bottom, [-1, h, w, 3])
    res_left = tf.reshape(res_left, [-1, h, w, 3])
    res_right = tf.reshape(res_right, [-1, h, w, 3])

    loss = top_w * tf.norm(top_i - res_top, axis=[3, -1]) + \
           bottom_w *tf.norm(bottom_i - res_bottom, axis=[3, -1]) + \
           left_w * tf.norm(left_i - res_left, axis=[3, -1]) + \
           right_w * tf.norm(right_i - res_right, axis=[3, -1])

    return tf.reduce_mean(loss) / 4



def hdr_compression(I):
    # log(17) is 2.833213344056216
    return np.sign(I)*np.log(1+np.abs(I)*16)/2.833213344056216

def revert_hdr(I):
    sign = np.sign(I)
    return sign * ((np.exp(np.abs(I)*2.833213344056216)-1)/16)

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

def preprocess_data(input_color,input_features, input_shape):
    feature_normal = input_features[:, :, 0:3].reshape(input_shape[0], input_shape[1], 3)
    feature_depth = input_features[:, :, 3:4].reshape(input_shape[0], input_shape[1], 1)
    feature_albedo = input_features[:, :, 4:7].reshape(input_shape[0], input_shape[1], 3)

    input_color = hdr_compression(input_color)
    feature_normal = (feature_normal + 1) / 2
    feature_albedo = data_normalize(feature_albedo, np.min(feature_albedo), np.max(feature_albedo))
    data = np.concatenate((input_color,  # 3
                           feature_depth,  # 1
                           feature_normal,  # 3
                           feature_albedo,  # 3
                           ), axis=2)
    return data