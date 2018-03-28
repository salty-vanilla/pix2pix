import tensorflow as tf
from layers import average_pool2d
from blocks import residual_block, conv_block


def _down_block(x,
                filters,
                conv_iteration,
                **conv_params):
    with tf.variable_scope(None, 'Down'):
        for i in range(conv_iteration):
            x = conv_block(x,
                           filters,
                           sampling='same',
                           **conv_params)
        _x = average_pool2d(x)
    return x, _x


def _up_block(x,
              x_e,
              filters,
              upsampling,
              conv_iteration,
              **conv_params):
    with tf.variable_scope(None, 'Up'):
        x = conv_block(x,
                       filters,
                       sampling=upsampling,
                       **conv_params)
        x = tf.concat([x, x_e], axis=-1)

        for i in range(conv_iteration):
            _filters = filters//2 if i+1 == conv_iteration else filters
            x = conv_block(x,
                           _filters,
                           sampling='same',
                           **conv_params)
    return x


class Generator:
    def __init__(self, input_shape,
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
    def __init__(self, input_shape,
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

    def __call__(self, x,
                 reuse=False):
        feature_maps = []
        conv_iterations = [2, 2, 2, 2, 3, 3, 3]
        s = 16
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            # Encoder
            with tf.variable_scope('Encoder'):
                for bi, ci in enumerate(conv_iterations):
                    _x, x = _down_block(x, s*(2**bi), ci, **self.conv_kwargs)
                    feature_maps.append(_x)

            # Decoder
            with tf.variable_scope('Decoder'):
                # 1 - 6th block
                for bi, ci in enumerate(conv_iterations[:0:-1]):
                    x = _up_block(x, feature_maps.pop(),
                                  s*(2**(len(conv_iterations)-bi-1)),
                                  self.upsampling,
                                  ci, **self.conv_kwargs)
                # last block
                with tf.variable_scope(None, 'Up'):
                    x = conv_block(x, s, sampling=self.upsampling, **self.conv_kwargs)
                    x = tf.concat([x, feature_maps.pop()], axis=-1)
                    x = conv_block(x, s, sampling='same', **self.conv_kwargs)
                    x = conv_block(x, self.channel, sampling='same',
                                   normalization=None, activation_=self.last_activation)
        return x


class ResidualUNet(Generator):
    def __init__(self, input_shape,
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

    def __call__(self, x,
                 reuse=False):
        pass
