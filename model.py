import tensorflow as tf


initializer = tf.keras.initializers.glorot_normal(seed=None)

# Gbranch
class GNet(tf.keras.Model):
    def __init__(self):
        super(GNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer)
        # self.lrelu1 = tf.keras.layers.LeakyReLU()
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer)
        # self.lrelu2 = tf.keras.layers.LeakyReLU()
        self.conv3 = tf.keras.layers.Conv2D(21, (3, 3), padding='same', kernel_initializer=initializer)

    def call(self, inputs, training=None):
        x = inputs
        x = tf.nn.leaky_relu(self.conv1(x, training=training))
        x = tf.nn.leaky_relu(self.conv2(x, training=training))
        x = self.conv3(x, training=training)
        return x





class GradNet(tf.keras.Model):

    def __init__(self):
        super(GradNet, self).__init__()
        # data branch [image, features]
        self.dconv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same', kernel_initializer=initializer)
        self.dconv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer=initializer)
        self.dconv3 = tf.keras.layers.Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer=initializer)

        # grad branch [gx, gy]
        self.gconv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same', kernel_initializer=initializer)
        self.gconv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer=initializer)
        self.gconv3 = tf.keras.layers.Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer=initializer)

        # res blocks
        self.res_layers = self.build_resblock(256, 4)

        # deconvs
        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, (4, 4), padding='same', strides=2, kernel_initializer=initializer)
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64, (4, 4), padding='same', strides=2, kernel_initializer=initializer)
        self.deconv3 = tf.keras.layers.Conv2DTranspose(32, (4, 4), padding='same', strides=2, kernel_initializer=initializer)
        self.deconv4 = tf.keras.layers.Conv2DTranspose(3, (3, 3), padding='same', strides=1, kernel_initializer=initializer)

    def call(self, inputs, training=None):
        d = inputs[:, :, :, :10]
        g = inputs[:, :, :, 10:]
        # data branch
        d1 = tf.nn.leaky_relu(self.dconv1(d, training=training))
        d2 = tf.nn.leaky_relu(self.dconv2(d1, training=training))
        d3 = tf.nn.leaky_relu(self.dconv3(d2, training=training))
        # grad branch
        g1 = tf.nn.leaky_relu(self.gconv1(g, training=training))
        g2 = tf.nn.leaky_relu(self.gconv2(g1, training=training))
        g3 = tf.nn.leaky_relu(self.gconv3(g2, training=training))
        # concat
        x = tf.keras.layers.Concatenate()([d3, g3])
        # resblocks
        x = self.res_layers(x, training=training)
        # deconv
        x = tf.concat([x, d3, g3], axis=-1)
        x = tf.nn.leaky_relu(self.deconv1(x, training=training))
        x = tf.concat([x, d2, g2], axis=-1)
        x = tf.nn.leaky_relu(self.deconv2(x, training=training))
        x = tf.concat([x, d1, g1], axis=-1)
        x = tf.nn.leaky_relu(self.deconv3(x, training=training))
        x = tf.nn.relu(self.deconv4(x, training=training))
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = tf.keras.Sequential()
        res_blocks.add(ResBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(ResBlock(filter_num, stride=1))
        return res_blocks


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same', kernel_initializer=initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.lrelu1 = tf.keras.layers.LeakyReLU()
        self.conv2 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=1, padding='same', kernel_initializer=initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        output = tf.keras.layers.add([out, identity])
        output = tf.nn.leaky_relu(output)
        return output