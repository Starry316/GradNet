from glob import glob

import tensorflow as tf
import numpy as np
import time
from numpy import inf
from data_generator import data_generator
from util import data_loss, first_order_loss, grad_loss, calc_grad_x, calc_grad_y, \
    get_features_array, get_color_array, preprocess_data, get_grad_array, revert_hdr, hdr_compression, revert_hdr_tf, \
    hdr_compression_tf
from model import GradNet, GNet
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


save_dir = 'results/tf/'
log_dir = 'log/log-all'

train_dir = 'data/train/'
val_dir = 'data/val/'

g_weights = 'results/tf/g_net-41-26.337116.ckpt-41'
grad_weights = 'results/tf/grad_net-41-55.127800.ckpt-41'

max_epochs = 2000
batch_size = 32
steps_per_epoch = 34425 // batch_size
val_steps = 60 * 10 // batch_size

initial_lrate = 1e-4


val = False
restore_model = False

alpha = 1.0

def train():
    learning_rate = tf.Variable(initial_lrate, trainable=False)
    g_net = GNet()
    grad_net = GradNet()
    g_checkpoint = tf.train.Checkpoint(g_model=g_net)
    grad_checkpoint = tf.train.Checkpoint(grad_model=grad_net)
    # 恢复权重
    if restore_model:
        # tf.train.Checkpoint(grad_model=grad_net).restore(grad_weights)
        tf.train.Checkpoint(g_model=g_net).restore(g_weights)
        print('restore grad model from %s, g model from %s'%(grad_weights, g_weights))
        # print('restore model from %s'%grad_weights)
    g_net.build(input_shape=(batch_size, 256, 256, 10))
    grad_net.build(input_shape=(batch_size, 256, 256, 10))

    # Adam 优化器
    g_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, lr=learning_rate)
    grad_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, lr=learning_rate)
    # tensorboard
    summary_writer = tf.summary.create_file_writer(log_dir)

    best_g_loss = inf
    best_grad_loss = inf
    # 训练集和验证集
    training_data = iter(data_generator("%s%s_*.tfrecords" % (train_dir, 'train'), batch_size))
    val_data = iter(data_generator("%s%s_*.tfrecords" % (val_dir, 'train'), batch_size))

    # train
    for epoch in range(1, max_epochs):
        # log loss
        all_loss, g_loss, val_loss, fst_loss = [], [], [], []

        # =========================================================
        # grad branch
        # =========================================================
        epoch_start_time = time.time()
        for _ in range(steps_per_epoch):
            start_time = time.time()
            batch_x = next(training_data)
            # input image
            origin_i = batch_x[:,:,:,:1]
            # input grad x and grad y
            grad_x = batch_x[:, :, :, 8:9]
            grad_y = batch_x[:, :, :, 9:10]

            with tf.GradientTape() as tape:

                pred = grad_net(batch_x, training=True)

                # revert hdr compression to calculate gradient
                re_hdr = revert_hdr_tf(pred)
                dx, dy = hdr_compression_tf(calc_grad_x(re_hdr)), hdr_compression_tf(calc_grad_y(re_hdr))


                if epoch > 5:
                    # epoch > 5  add first_order_loss to loss
                    lamda = min(0.1 * (1.1 ** epoch), 2)
                    # depth normal albedo
                    features = batch_x[:, :, :, 1:8]
                    # color loss and grad loss
                    loss = alpha * data_loss(pred, origin_i) +\
                        grad_loss(dx, dy, grad_x, grad_y)

                    # first order loss
                    G = g_net(batch_x, training=False)
                    f_loss =lamda * first_order_loss(pred, features, origin_i, G)
                    loss = loss + f_loss

                    fst_loss.append(tf.reduce_mean(f_loss))


                else:
                    # only color loss and grad loss
                    loss = alpha * data_loss(pred, origin_i) + \
                            grad_loss(dx, dy, grad_x, grad_y)
                    fst_loss.append(0)

            grads = tape.gradient(loss, grad_net.trainable_variables)
            grad_optimizer.apply_gradients(zip(grads, grad_net.trainable_variables))
            all_loss.append(tf.reduce_mean(loss))
            print('grad step:%d/%d all_loss:%f 1st_loss:%f %fs'%(_, steps_per_epoch, all_loss[-1], fst_loss[-1], time.time()-start_time))

        print('epoch: %d grad_loss: %f 1st_loss:%f time: %fs' % (epoch, tf.reduce_mean(all_loss), tf.reduce_mean(fst_loss), (time.time() - epoch_start_time),))

        # image = denoiseImage(grad_net, test_data, epoch, outputDir)


        if best_grad_loss > tf.reduce_mean(all_loss):
            print('grad loss improve from %f to %f' % (best_grad_loss, tf.reduce_mean(all_loss)))
            best_grad_loss = tf.reduce_mean(all_loss)
        else:
            print('grad loss did not improve from %f' % (best_grad_loss))

        # save model weight
        grad_checkpoint.save(save_dir+'grad_net-%d-%f.ckpt' % (epoch, tf.reduce_mean(all_loss)))
        print('saving checkpoint to %sgrad_net-%d-%f.ckpt' % (save_dir, epoch, tf.reduce_mean(all_loss)))


        # =========================================================
        # g branch
        # =========================================================
        # train g branch
        epoch_start_time = time.time()
        for _ in range(steps_per_epoch):
            start_time = time.time()
            batch_x= next(training_data)
            # depth normal albedo
            features = batch_x[:, :, :, 1:8]
            origin_i = batch_x[:, :, :, :1]

            with tf.GradientTape() as tape:
                G = g_net(batch_x, training=True)
                loss = first_order_loss(origin_i, features, origin_i, G)
            grads = tape.gradient(loss, g_net.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, g_net.trainable_variables))
            g_loss.append(tf.reduce_mean(loss))
            print('g step:%d/%d g_loss:%f %fs' % (_, steps_per_epoch, g_loss[-1], time.time() - start_time))

        print('epoch: %d g_loss: %f time: %fs' % (epoch, tf.reduce_mean(g_loss), (time.time() - epoch_start_time)))
        if best_g_loss > tf.reduce_mean(g_loss):
            print('g_loss improve from %f to %f' % (best_g_loss, tf.reduce_mean(g_loss)))
            best_g_loss = tf.reduce_mean(g_loss)
        else:
            print('g_loss did not improve from %f' % (best_g_loss))
        # save g branch model
        g_checkpoint.save('%sg_net-%d-%f.ckpt' % (save_dir,epoch, tf.reduce_mean(g_loss)))
        print('saving checkpoint to %sg_net-%d-%f.ckpt' % (save_dir,epoch, tf.reduce_mean(g_loss)))

        # =========================================================
        # val
        # =========================================================
        if val :
            val_loss_tmp = []
            for _ in range(val_steps):
                batch_x = next(val_data)
                # input image
                origin_i = batch_x[:, :, :, :1]
                # input grad x and grad y
                grad_x = batch_x[:, :, :, 8:9]
                grad_y = batch_x[:, :, :, 9:10]

                pred = grad_net(batch_x, training=False)

                re_hdr = revert_hdr_tf(pred)
                dx, dy = hdr_compression_tf(calc_grad_x(re_hdr)), hdr_compression_tf(calc_grad_y(re_hdr))
                if epoch > 5:
                    lamda = min(0.1 * (1.1 ** epoch), 2)
                    features = batch_x[:, :, :, 1:8]
                    G = g_net(batch_x, training=False)
                    loss = alpha * data_loss(pred, origin_i) + \
                           grad_loss(dx, dy, grad_x, grad_y) + \
                           lamda * first_order_loss(pred, features, origin_i, G)
                else:
                    loss = alpha * data_loss(pred, origin_i) + \
                           grad_loss(dx, dy, grad_x, grad_y)
                val_loss_tmp.append(loss)
            val_loss.append(tf.reduce_mean(val_loss_tmp))
            print('val_loss: %f' % (tf.reduce_mean(val_loss)))


        # =========================================================
        # tensorboard
        # =========================================================
        with summary_writer.as_default():
            tf.summary.scalar("g loss", tf.reduce_mean(g_loss), step=epoch)
            tf.summary.scalar("all loss", tf.reduce_mean(all_loss), step=epoch)
            tf.summary.scalar('learning_rate', learning_rate, step=epoch)
            # tf.summary.image("image_%s" % epoch, image, step=epoch)
            tf.summary.scalar('1st loss', tf.reduce_mean(fst_loss), step=epoch)
            if val:
                tf.summary.scalar("val loss", tf.reduce_mean(val_loss), step=epoch)

        # update learning rate
        lrate = initial_lrate * np.math.pow(0.95, epoch)
        # if lrate < 1e-6:
        #     lrate = 1e-6
        tf.keras.backend.set_value(learning_rate, lrate)



if __name__ == '__main__':
    train()
