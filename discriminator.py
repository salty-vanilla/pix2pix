import tensorflow as tf
from blocks import conv_block
from layers import conv2d, dense, flatten, activation, global_average_pool2d, layer_norm


def discriminator_block(x,
                        filters,
                        activation_='lrelu',
                        kernel_size=(3, 3),
                        is_training=True,
                        normalization=None,
                        residual=True):
    with tf.variable_scope(None, discriminator_block.__name__):
        _x = conv_block(x, filters, activation_,
                        kernel_size, is_training, 'same',
                        normalization, 0., 'conv_first')
        _x = conv_block(_x, filters, None,
                        kernel_size, is_training,
                        'same', None, 0., 'conv_first')
        if residual:
            _x += x
        _x = activation(_x, activation_)
        if normalization == 'layer':
          _x = layer_norm(_x, is_training=is_training)
        return _x


class Discriminator:
    def __init__(self, input_shape,
                 normalization=None,
                 is_training=True):
        self.input_shape = input_shape
        self.name = 'model/discriminator'
        self.is_training = is_training
        self.normalization = normalization

        self.conv_kwargs = {'activation_': 'lrelu',
                            'is_training': self.is_training,
                            'normalization': self.normalization}

    def __call__(self, x, reuse=True, is_feature=False):
        raise NotImplementedError

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


class ResidualDiscriminator(Discriminator):
    def __init__(self, input_shape,
                 normalization=None,
                 is_training=True):
        super().__init__(input_shape,
                         normalization,
                         is_training)

    def __call__(self, x,
                 reuse=True,
                 is_feature=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            x = conv_block(x, 32,
                           kernel_size=(4, 4), sampling='down',
                           **self.conv_kwargs)

            for i in range(2):
                x = discriminator_block(x, 32, **self.conv_kwargs)
            x = conv_block(x, 64,
                           kernel_size=(4, 4), sampling='down',
                           **self.conv_kwargs)

            for i in range(4):
                x = discriminator_block(x, 64, **self.conv_kwargs)
            x = conv_block(x, 128,
                           kernel_size=(4, 4), sampling='down',
                           **self.conv_kwargs)

            for i in range(4):
                x = discriminator_block(x, 128, **self.conv_kwargs)
            x = conv_block(x, 256,
                           kernel_size=(4, 4), sampling='down',
                           **self.conv_kwargs)

            for i in range(4):
                x = discriminator_block(x, 256, **self.conv_kwargs)
            x = conv_block(x, 512,
                           kernel_size=(4, 4), sampling='down',
                           **self.conv_kwargs)

            for i in range(4):
                x = discriminator_block(x, 512, **self.conv_kwargs)
            x = conv_block(x, 1024,
                           kernel_size=(4, 4), sampling='down',
                           **self.conv_kwargs)

            if is_feature:
                return x

            x = global_average_pool2d(x)
            x = dense(x, units=1, activation_=None)
            return x
