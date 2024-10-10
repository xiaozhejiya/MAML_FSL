import tensorflow as tf
from tensorflow.keras.layers import *


class ResNet(tf.keras.Model):
    def __init__(self, input_shape, n_way):
        super(ResNet, self).__init__()
        self.n_way = n_way
        self.build_model(input_shape)

    def build_model(self, input_shape):
        # 卷积层
        self.conv1 = Conv2D(64, 3, strides=1, padding="same", activation="relu", input_shape=input_shape, name="conv1")
        self.conv2 = Conv2D(64, 3, strides=1, padding="same", activation="relu", name="conv2")
        # 降采样
        self.conv3 = Conv2D(64, 3, strides=2, padding="same", activation="relu", name="conv3")
        # 残差块1
        self.identity_block1_conv1 = Conv2D(64, 3, strides=1, padding="same", activation="relu",
                                            name="identity/block1/conv1")
        self.identity_block1_conv2 = Conv2D(64, 3, strides=1, padding="same", activation=None,
                                            name="identity/block1/conv2")
        # 卷积块1,使用降采样
        self.conv_block1_conv1 = Conv2D(64, 3, strides=2, padding="same", activation="relu", name="conv/block1/conv1")
        self.conv_block1_conv2 = Conv2D(64, 3, strides=1, padding="same", activation=None, name="conv/block1/conv2")
        self.conv_block1_shortcut = Conv2D(64, 3, strides=2, padding="same", activation=None, name="conv/block1/shortcut")
        # 卷积块2,使用降采样
        self.conv_block2_conv1 = Conv2D(64, 3, strides=2, padding="same", activation="relu", name="conv/block2/conv1")
        self.conv_block2_conv2 = Conv2D(64, 3, strides=1, padding="same", activation=None, name="conv/block2/conv2")
        self.conv_block2_shortcut = Conv2D(64, 3, strides=2, padding="same", activation=None, name="conv/block2/shortcut")
        # 卷积块4,使用降采样
        self.conv_block3_conv1 = Conv2D(64, 3, strides=2, padding="same", activation="relu", name="conv/block3/conv1")
        self.conv_block3_conv2 = Conv2D(64, 3, strides=1, padding="same", activation=None, name="conv/block3/conv2")
        self.conv_block3_shortcut = Conv2D(64, 3, strides=2, padding="same", activation=None, name="conv/block3/shortcut")

        # 全局平均池化和全连接层
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name='global_pool')
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(self.n_way, name='dense2')


    def call(self, inputs, weights=None):
        if weights is None:
            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.conv3(x)

            # 恒等残差块
            shortcut = x
            x = self.identity_block1_conv1(x)
            x = self.identity_block1_conv2(x)
            x = tf.add(x, shortcut)
            x = tf.nn.relu(x)

            # 卷积块1
            shortcut = self.conv_block1_shortcut(x)
            x = self.conv_block1_conv1(x)
            x = self.conv_block1_conv2(x)
            x = tf.add(x, shortcut)
            x = tf.nn.relu(x)

            # 卷积块2
            shortcut = self.conv_block2_shortcut(x)
            x = self.conv_block2_conv1(x)
            x = self.conv_block2_conv2(x)
            x = tf.add(x, shortcut)
            x = tf.nn.relu(x)

            # 卷积块3
            shortcut = self.conv_block3_shortcut(x)
            x = self.conv_block3_conv1(x)
            x = self.conv_block3_conv2(x)
            x = tf.add(x, shortcut)
            x = tf.nn.relu(x)

            # 全连接
            x = self.global_pool(x)
            x = self.dense1(x)
            x = self.dense2(x)

        else:
            # 自定义向前传播参数
            x = tf.nn.conv2d(inputs, weights['conv1/kernel'], strides=1, padding='SAME')
            x = tf.nn.bias_add(x, weights['conv1/bias'])
            x = tf.nn.relu(x)

            x = tf.nn.conv2d(x, weights['conv2/kernel'], strides=1, padding='SAME')
            x = tf.nn.bias_add(x, weights['conv2/bias'])
            x = tf.nn.relu(x)

            x = tf.nn.conv2d(x, weights['conv3/kernel'], strides=2, padding='SAME')
            x = tf.nn.bias_add(x, weights['conv3/bias'])
            x = tf.nn.relu(x)

            # 恒等残差块
            x = self.__identity_block(x, weights, "identity/block1")
            # 卷积块1
            x = self.__conv_block_custom(x, weights, "conv/block1", downsample=True)
            # 卷积块2
            x = self.__conv_block_custom(x, weights, "conv/block2", downsample=True)
            # 卷积块3
            x = self.__conv_block_custom(x, weights, "conv/block3", downsample=True)

            x = tf.reduce_mean(x, axis=[1, 2])  # 全局平均池化
            x = tf.matmul(x, weights['dense1/kernel']) + weights['dense1/bias']
            x = tf.nn.relu(x)
            x = tf.matmul(x, weights['dense2/kernel']) + weights['dense2/bias']
        return x

    def __identity_block(self, x, weights, block_name):
        shortcut = x

        # 第一层卷积
        x = tf.nn.conv2d(x, weights[f"{block_name}/conv1/kernel"], strides=1, padding="SAME")
        x = tf.nn.bias_add(x, weights[f"{block_name}/conv1/bias"])
        x = tf.nn.relu(x)

        # 第二层卷积
        x = tf.nn.conv2d(x, weights[f"{block_name}/conv2/kernel"], strides=1, padding="SAME")
        x = tf.nn.bias_add(x, weights[f"{block_name}/conv2/bias"])

        # 残差连接
        x = tf.add(x, shortcut)
        x = tf.nn.relu(x)
        return x

    def __conv_block_custom(self, x, weights, block_name, downsample=False):
        shortcut = x
        strides = 2 if downsample else 1

        # 第一层卷积
        x = tf.nn.conv2d(x, weights[f"{block_name}/conv1/kernel"], strides=strides, padding="SAME")
        x = tf.nn.bias_add(x, weights[f"{block_name}/conv1/bias"])
        x = tf.nn.relu(x)

        # 第二层卷积
        x = tf.nn.conv2d(x, weights[f"{block_name}/conv2/kernel"], strides=1, padding="SAME")
        x = tf.nn.bias_add(x, weights[f"{block_name}/conv2/bias"])

        # 降采样
        if downsample:
            shortcut = tf.nn.conv2d(shortcut, weights[f"{block_name}/shortcut/kernel"], strides=strides, padding="SAME")
            shortcut = tf.nn.bias_add(shortcut, weights[f"{block_name}/shortcut/bias"])

        # 残差连接
        x = tf.add(x, shortcut)
        x = tf.nn.relu(x)
        return x
