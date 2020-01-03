import tensorflow as tf
import numpy as np
import time
from numpy import inf
from GradNet.results.data_generator import data_generator
from util import data_loss, first_order_loss, grad_loss, calc_grad_x, calc_grad_y
from model import GradNet, GNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 模型保存目录
save_dir = 'results/'
# tensorboard log 目录
log_dir = 'log-all'
# 模型加载路径
g_weights = 'results/g_net-51-0.053565.ckpt-51'
grad_weights = 'results/g_net-51-0.053565.ckpt-51'
# 最大训练epoch
max_epochs = 2000
# batch size
batch_size = 32
# 初始学习率
initial_lrate = 1e-4
learning_rate = tf.Variable(initial_lrate, trainable=False)
# 每轮训练的step数
steps_per_epoch = 60 * 189 // batch_size
val_steps = 60 * 10 // batch_size
# 是否验证
val = False
# 是否恢复模型权重
restore_model = False

def train():
    g_net = GNet()
    grad_net = GradNet()
    g_checkpoint = tf.train.Checkpoint(g_model=g_net)
    grad_checkpoint = tf.train.Checkpoint(grad_model=grad_net)
    # 恢复权重
    if restore_model:
        tf.train.Checkpoint(grad_model=grad_net).restore(grad_weights)
        tf.train.Checkpoint(g_model=g_net).restore(g_weights)

    g_net.build(input_shape=(batch_size, 256, 256, 16))
    grad_net.build(input_shape=(batch_size, 256, 256, 16))

    # Adam 优化器
    g_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, lr=learning_rate)
    grad_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, lr=learning_rate)
    # tensorboard
    summary_writer = tf.summary.create_file_writer(log_dir)

    best_g_loss = inf
    best_grad_loss = inf
    # 训练集和验证集
    training_data = iter(data_generator("%s%s_*.tfrecords" % ('/data/train/', 'train'), batch_size))
    val_data = iter(data_generator("%s%s_*.tfrecords" % ('/data/val/', 'train'), batch_size))

    # train
    for epoch in range(1, max_epochs):
        # 记录每轮loss
        all_loss, g_loss, val_loss, fst_loss = [], [], [], []

        # =========================================================
        # grad branch
        # =========================================================
        epoch_start_time = time.time()
        for _ in range(steps_per_epoch):
            start_time = time.time()
            # batch
            batch_x = next(training_data)

            origin_i = batch_x[:, :, :, :3]
            with tf.GradientTape() as tape:
                pred = grad_net(batch_x, training=True)
                gx, gy = calc_grad_x(pred), calc_grad_y(pred)
                alpha = 1
                # epoch 超过5加入first order loss
                if epoch > 5:
                    lamda = min(0.1 * (1.1 ** epoch), 2)

                    # normal depth albedo
                    features = batch_x[:, :, :, 3:10]
                    # data and grad loss
                    loss = alpha * data_loss(pred, origin_i) +\
                        grad_loss(gx, gy, batch_x[:, :, :, 10:13], batch_x[:, :, :, 13:16])


                    # first order loss
                    G = g_net(batch_x, training=False)
                    f_loss =lamda * first_order_loss(pred, features, origin_i, G)
                    loss = loss + f_loss
                    # 记录loss
                    fst_loss.append(tf.reduce_mean(f_loss))
                # epoch不超过5 计算data 和 grad loss
                else:
                    loss = alpha * data_loss(pred, origin_i) + \
                           grad_loss(gx, gy, batch_x[:, :, :, 10:13], batch_x[:, :, :, 13:16])
                    fst_loss.append(0)
            # 更新权重
            grads = tape.gradient(loss, grad_net.trainable_variables)
            grad_optimizer.apply_gradients(zip(grads, grad_net.trainable_variables))
            # 记录loss
            all_loss.append(tf.reduce_mean(loss))

            print('grad step:%d/%d all_loss:%f 1st_loss:%f %fs'%(_, steps_per_epoch,all_loss[-1], fst_loss[-1], time.time()-start_time))

        print('epoch: %d grad_loss: %f 1st_loss:%f time: %fs' % (epoch, tf.reduce_mean(all_loss),tf.reduce_mean(fst_loss), (time.time() - epoch_start_time),))
        if best_grad_loss > tf.reduce_mean(all_loss):
            print('grad loss improve from %f to %f' % (best_grad_loss, tf.reduce_mean(all_loss)))
            best_grad_loss = tf.reduce_mean(all_loss)
        else:
            print('grad loss did not improve from %f' % (best_grad_loss))

        # 保存模型
        grad_checkpoint.save(save_dir+'grad_net-%d-%f.ckpt' % (epoch, tf.reduce_mean(all_loss)))
        print('saving checkpoint to %sgrad_net-%d-%f.ckpt' % (save_dir, epoch, tf.reduce_mean(all_loss)))


        # =========================================================
        # g branch
        # =========================================================
        # 交替训练g branch
        epoch_start_time = time.time()
        for _ in range(steps_per_epoch):
            start_time = time.time()
            batch_x = next(training_data)
            origin_i = batch_x[:, :, :, :3]
            features = batch_x[:, :, :, 3:10]
            with tf.GradientTape() as tape:
                G = g_net(batch_x, training=True)
                loss = first_order_loss(origin_i, features, origin_i, G)
            grads = tape.gradient(loss, g_net.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, g_net.trainable_variables))
            # 记录loss
            g_loss.append(tf.reduce_mean(loss))
            # 输出信息
            print('g step:%d/%d g_loss:%f %fs' % (_, steps_per_epoch, g_loss[-1], time.time() - start_time))

        print('epoch: %d g_loss: %f time: %fs' % (epoch, tf.reduce_mean(g_loss), (time.time() - epoch_start_time)))
        if best_g_loss > tf.reduce_mean(g_loss):
            print('g_loss improve from %f to %f' % (best_g_loss, tf.reduce_mean(g_loss)))
            best_g_loss = tf.reduce_mean(g_loss)
        else:
            print('g_loss did not improve from %f' % (best_g_loss))

        g_checkpoint.save('results/tfnew/g_net-%d-%f.ckpt' % (epoch, tf.reduce_mean(g_loss)))
        print('saving checkpoint to %sg_net-%d-%f.ckpt' % (save_dir,epoch, tf.reduce_mean(g_loss)))

        # =========================================================
        # val
        # =========================================================
        if val :
            val_loss_tmp = []
            for _ in range(val_steps):
                batch_x = next(val_data)
                pred = grad_net(batch_x, training=False)
                gx, gy = calc_grad_x(pred), calc_grad_y(pred)
                if epoch > 5:
                    lamda = min(0.1 * (1.1 ** epoch), 2)
                    origin_i = batch_x[:, :, :, :3]
                    features = batch_x[:, :, :, 3:10]
                    G = g_net(batch_x, training=False)
                    loss = alpha * data_loss(pred, origin_i) + \
                           grad_loss(gx, gy, batch_x[:, :, :, 10:13], batch_x[:, :, :, 13:16]) + \
                           lamda * first_order_loss(pred, features, origin_i, G)
                else:
                    loss = alpha * data_loss(pred, origin_i) + \
                           grad_loss(gx, gy, batch_x[:, :, :, 10:13], batch_x[:, :, :, 13:16])
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
            tf.summary.scalar('1st loss', tf.reduce_mean(fst_loss), step=epoch)
            if val:
                tf.summary.scalar("val loss", tf.reduce_mean(val_loss), step=epoch)

        # update learning rate
        lrate = initial_lrate * np.math.pow(0.95, epoch)
        tf.keras.backend.set_value(learning_rate, lrate)



train()