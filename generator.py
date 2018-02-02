import tensorflow as tf
from layers import dense, reshape, batch_norm, activation, conv2d
from blocks import residual_block, conv_block


class Generator:
    def __init__(self,
                 input_shape,
                 last_activation='tanh',
                 color_mode='rgb',
                 normalization='batch',
                 upsampling='subpixel',
                 is_training=True):
        self.input_shape = input_shape
        self.last_activation = last_activation
        self.name = 'model/generator'
        assert color_mode in ['grayscale', 'gray', 'rgb']
        self.channel = 1 if color_mode in ['grayscale', 'gray'] else 3
        self.normalization = normalization
        self.upsampling = upsampling
        self.is_training = is_training

        self.conv_kwargs = {'activation_': 'relu',
                            'is_training': self.is_training,
                            'normalization': self.normalization}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


class UNet(Generator):
    def __init__(self,
                 input_shape,
                 last_activation='tanh',
                 color_mode='rgb',
                 normalization='batch',
                 upsampling='subpixel',
                 is_training=True):

        super().__init__(input_shape,
                         last_activation,
                         color_mode,
                         normalization,
                         upsampling,
                         is_training)

    def __call__(self, x, reuse=False):
        feature_maps = []
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            with tf.variable_scope('Encoder'):
                _x = conv_block(x, 16, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 16, sampling='down', **self.conv_kwargs)
                feature_maps.append(_x)

                _x = conv_block(_x, 32, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 32, sampling='down', **self.conv_kwargs)
                feature_maps.append(_x)

                _x = conv_block(_x, 64, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 64, sampling='down', **self.conv_kwargs)
                feature_maps.append(_x)

                _x = conv_block(_x, 128, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 128, sampling='down', **self.conv_kwargs)
                feature_maps.append(_x)

                _x = conv_block(_x, 256, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 256, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 256, sampling='down', **self.conv_kwargs)
                feature_maps.append(_x)

                _x = conv_block(_x, 512, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 512, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 512, sampling='down', **self.conv_kwargs)
                feature_maps.append(_x)

                _x = conv_block(_x, 512, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 512, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 512, sampling='down', **self.conv_kwargs)

            with tf.variable_scope('Decoder'):
                _x = conv_block(_x, 512, sampling=self.upsampling, **self.conv_kwargs)
                _x = conv_block(_x, 512, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 512, sampling='same', **self.conv_kwargs)

                _x = tf.concat([_x, feature_maps.pop()], axis=-1)
                _x = conv_block(_x, 512, sampling=self.upsampling, **self.conv_kwargs)
                _x = conv_block(_x, 512, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 256, sampling='same', **self.conv_kwargs)

                _x = tf.concat([_x, feature_maps.pop()], axis=-1)
                _x = conv_block(_x, 256, sampling=self.upsampling, **self.conv_kwargs)
                _x = conv_block(_x, 256, sampling='same', **self.conv_kwargs)
                _x = conv_block(_x, 128, sampling='same', **self.conv_kwargs)

                _x = tf.concat([_x, feature_maps.pop()], axis=-1)
                _x = conv_block(_x, 128, sampling=self.upsampling, **self.conv_kwargs)
                _x = conv_block(_x, 64, sampling='same', **self.conv_kwargs)

                _x = tf.concat([_x, feature_maps.pop()], axis=-1)
                _x = conv_block(_x, 64, sampling=self.upsampling, **self.conv_kwargs)
                _x = conv_block(_x, 32, sampling='same', **self.conv_kwargs)

                _x = tf.concat([_x, feature_maps.pop()], axis=-1)
                _x = conv_block(_x, 32, sampling=self.upsampling, **self.conv_kwargs)
                _x = conv_block(_x, 16, sampling='same', **self.conv_kwargs)

                _x = tf.concat([_x, feature_maps.pop()], axis=-1)
                _x = conv_block(_x, 16, sampling=self.upsampling, **self.conv_kwargs)
                _x = conv_block(_x, self.channel, sampling='same',
                                normalization=None, activation_=self.last_activation)
        return _x


class ResidualUNet(Generator):
    def __init__(self,
                 input_shape,
                 last_activation='tanh',
                 color_mode='rgb',
                 normalization='batch',
                 upsampling='subpixel',
                 is_training=True):

        super().__init__(input_shape,
                         last_activation,
                         color_mode,
                         normalization,
                         upsampling,
                         is_training)

    def __call__(self, x, reuse=False):
        pass
