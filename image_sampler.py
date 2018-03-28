from tensorflow.python.keras.preprocessing.image import Iterator
import os
import numpy as np
from PIL import Image


class ImageSampler:
    def __init__(self,
                 is_flip=True,
                 target_size=None,
                 color_mode_x='rgb',
                 color_mode_y='rgb',
                 normalization_x='tanh',
                 normalization_y='tanh'):
        self.is_flip = is_flip
        self.target_size = target_size
        self.color_mode_x = color_mode_x
        self.color_mode_y = color_mode_y
        self.normalization_x = normalization_x
        self.normalization_y = normalization_y

    def flow_from_directory(self,
                            image_x_dir,
                            image_y_dir,
                            batch_size=32,
                            shuffle=True,
                            seed=None):
        nb_sample = len(get_image_paths(image_x_dir))
        return DirectoryIterator(image_x_dir,
                                 image_y_dir,
                                 nb_sample,
                                 batch_size,
                                 shuffle,
                                 seed,
                                 **self.__dict__)


class MyIterator(Iterator):
    def __init__(self,
                 nb_sample,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 is_flip=True,
                 target_size=None,
                 color_mode_x='rgb',
                 color_mode_y='rgb',
                 normalization_x='tanh',
                 normalization_y='tanh'):
        self.is_flip = is_flip
        self.target_size = target_size
        self.color_mode_x = color_mode_x
        self.color_mode_y = color_mode_y
        self.normalization_x = normalization_x
        self.normalization_y = normalization_y

        self.paths_x = None
        self.paths_y = None

        super().__init__(nb_sample, batch_size, shuffle, seed)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def x_to_image(self, x):
        return denormalize(x, self.normalization_x)

    def y_to_image(self, y):
        return denormalize(y, self.normalization_y)


class DirectoryIterator(MyIterator):
    def __init__(self,
                 image_x_dir,
                 image_y_dir,
                 nb_sample,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 is_flip=True,
                 target_size=None,
                 color_mode_x='rgb',
                 color_mode_y='rgb',
                 normalization_x='tanh',
                 normalization_y='tanh'):

        super().__init__(nb_sample,
                         batch_size,
                         shuffle,
                         seed,
                         is_flip,
                         target_size,
                         color_mode_x,
                         color_mode_y,
                         normalization_x,
                         normalization_y)

        self.x_dir = image_x_dir
        self.y_dir = image_y_dir

        self.names = [os.path.basename(p) for p in get_image_paths(self.x_dir)]

        self.x_paths = np.array([os.path.join(self.x_dir, name) for name in self.names])
        self.y_paths = np.array([os.path.join(self.y_dir, name) for name in self.names])

    def __call__(self, *args, **kwargs):
        with self.lock:
            index_array = next(self.index_generator)
        x_path_batch = self.x_paths[index_array]
        y_path_batch = self.y_paths[index_array]
        image_x_batch = np.array([load_image(path,
                                             self.is_flip,
                                             self.target_size,
                                             self.color_mode_x,
                                             self.normalization_x)
                                  for path in x_path_batch])
        image_y_batch = np.array([load_image(path,
                                             self.is_flip,
                                             self.target_size,
                                             self.color_mode_y,
                                             self.normalization_y)
                                  for path in y_path_batch])
        self.current_paths = (x_path_batch, y_path_batch)
        return image_x_batch, image_y_batch


def load_image(path,
               is_flip=True,
               target_size=None,
               color_mode='rgb',
               normalization='tanh'):
    assert color_mode in ['grayscale', 'gray', 'rgb']
    image = Image.open(path)

    if color_mode in ['grayscale', 'gray']:
        image = image.convert('L')

    if target_size is not None and target_size != image.size:
        image = image.resize(target_size, Image.BILINEAR)

    image_array = np.asarray(image)
    image_array = normalize(image_array, normalization)

    if len(image_array.shape) == 2:
        image_array = np.expand_dims(image_array, axis=-1)

    if is_flip and np.random.uniform() < 0.5:
        image_array = image_array[:, ::-1, :]

    return image_array


def normalize(x, mode='tanh'):
    if mode == 'tanh':
        return (x.astype('float32') / 255 - 0.5) / 0.5
    elif mode == 'sigmoid':
        return x.astype('float32') / 255
    else:
        raise NotImplementedError


def denormalize(x, mode='tanh'):
    if mode == 'tanh':
        return ((x + 1.) / 2 * 255).astype('uint8')
    elif mode == 'sigmoid':
        return (x * 255).astype('uint8')
    else:
        raise NotImplementedError


def get_image_paths(src_dir):
    def get_all_paths():
        for root, dirs, files in os.walk(src_dir):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def is_image(path):
        if 'png' in path or 'jpg' in path or 'bmp' in path:
            return True
        else:
            return False

    return [path for path in get_all_paths() if is_image(path)]