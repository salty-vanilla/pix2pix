import tensorflow as tf
import numpy as np
import os
import csv
import time
from PIL import Image


class Pix2Pix:
    def __init__(self,
                 generator,
                 discriminator,
                 l1_weight=1.,
                 gradient_penalty_weight=10.,
                 is_training=True):

        tf.reset_default_graph()
        self.discriminator = discriminator
        self.generator = generator
        self.image_shape = self.generator.input_shape
        self.image_x = tf.placeholder(tf.float32, [None] + list(self.image_shape), name='x')
        self.image_y = tf.placeholder(tf.float32, [None] + list(self.image_shape), name='y')
        self.image_y_ = self.generator(self.image_x)
        self.true_pair = tf.concat([self.image_x, self.image_y], axis=-1)
        self.fake_pair = tf.concat([self.image_x, self.image_y_], axis=-1)

        self.discriminate_real = self.discriminator(self.true_pair, reuse=False)
        self.discriminate_fake = self.discriminator(self.fake_pair, reuse=True)

        with tf.name_scope('loss'):
            self.loss_g = -tf.reduce_mean(self.discriminate_fake)
            self.loss_d = -(tf.reduce_mean(self.discriminate_real)
                            - tf.reduce_mean(self.discriminate_fake))

            self.l1_loss = tf.reduce_mean(tf.abs(self.image_y - self.image_y_))

            # Gradient Penalty
            with tf.name_scope('GradientPenalty'):
                self.epsilon_first_dim = tf.placeholder(tf.int32, shape=[])
                if len(self.image_shape) == 3:
                    epsilon = tf.random_uniform(shape=[self.epsilon_first_dim, 1, 1, 1],
                                                minval=0., maxval=1.)
                elif len(self.image_shape) == 1:
                    epsilon = tf.random_uniform(shape=[self.epsilon_first_dim, 1],
                                                minval=0., maxval=1.)
                else:
                    raise ValueError
                differences = self.image_y_ - self.image_y
                interpolates = self.image_y + (epsilon * differences)
                gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                self.gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))

            # Optimizer
            if is_training:
                with tf.name_scope('Optimizer'):
                    self.opt_d = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
                        .minimize(self.loss_d + gradient_penalty_weight*self.gradient_penalty,
                                  var_list=self.discriminator.vars)
                    self.opt_g = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
                        .minimize(self.loss_g + l1_weight*self.l1_loss,
                                  var_list=self.generator.vars)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.model_dir = None
        self.sess.run(tf.global_variables_initializer())
        self.is_training = False
        tf.summary.FileWriter('../logs', graph=self.sess.graph)

    def fit(self, image_sampler,
            nb_epoch=1000,
            initial_steps=20,
            initial_critics=100,
            normal_critics=5,
            visualize_steps=1,
            save_steps=1,
            result_dir='result',
            model_dir='model'):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample
        self.model_dir = model_dir

        # prepare for csv
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        f = open(os.path.join(result_dir, 'learning_log.csv'), 'w')
        writer = csv.writer(f, lineterminator='\n')

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        # for display and csv
        loss_g = 0
        w_dis = 0
        writer.writerow(['loss_d', 'loss_g', 'gp', 'w_distance', 'l1_loss'])

        # fit loop
        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            n_critics = initial_critics if epoch < initial_steps else normal_critics
            for iter_ in range(1, steps_per_epoch + 1):
                image_x_batch, image_y_batch = image_sampler()
                _, loss_d, gradient_penalty = \
                    self.sess.run([self.opt_d, self.loss_d, self.gradient_penalty],
                                  feed_dict={self.image_x: image_x_batch,
                                             self.image_y: image_y_batch,
                                             self.epsilon_first_dim: image_x_batch.shape[0],
                                             })
                if iter_ % n_critics == 0:
                    # print noise_batch.shape, self.noise, self.generate, self.image
                    _, loss_g, loss_l1 = self.sess.run([self.opt_g, self.loss_g, self.l1_loss],
                                                       feed_dict={self.image_x: image_x_batch,
                                                                  self.image_y: image_y_batch,
                                                                  })
                    w_dis = self.sess.run(self.loss_d,
                                          feed_dict={self.image_x: image_x_batch,
                                                     self.image_y: image_y_batch,
                                                     })
                    w_dis = -w_dis
                print('iter : {} / {}  {:.1f}[s]  loss_d : {:.4f}  loss_g : {:.4f}  gp : {:.4f}, w_dis : {:.4f}\r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              loss_d, loss_g, gradient_penalty, w_dis), end='')
                writer.writerow([loss_d, loss_g, gradient_penalty, w_dis, loss_l1])
            if epoch % visualize_steps == 0:
                self.visualize(os.path.join(result_dir, 'epoch_{}'.format(epoch)),
                               image_x_batch, image_sampler.data_to_image)
            if epoch % save_steps == 0:
                self.save(epoch)
        print('\nTraining is done ...\n')

    def restore(self, file_path):
        reader = tf.train.NewCheckpointReader(file_path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        var_dict = dict(zip(map(lambda x:
                                x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                current_var = var_dict[saved_var_name]
                var_shape = current_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(current_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, file_path)

    def visualize(self, dst_dir, image_x_batch, convert_function):
        generated_data = self.predict_on_batch(image_x_batch)
        generated_images = convert_function(generated_data)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for i, image in enumerate(generated_images):
            if image.shape[2] == 1:
                image = image.reshape(image.shape[:2])
            dst_path = os.path.join(dst_dir, "{}.png".format(i))
            pil_image = Image.fromarray(np.uint8(image))
            pil_image.save(dst_path)

    def save(self, epoch):
        dst_dir = os.path.join(self.model_dir, "epoch_{}".format(epoch))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        return self.saver.save(self.sess, save_path=os.path.join(dst_dir, 'model.ckpt'))

    def predict(self, x, batch_size=16):
        outputs = np.empty([0] + list(self.image_shape))
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.predict_on_batch(x_batch)
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def predict_on_batch(self, x):
        return self.sess.run(self.image_y_,
                             feed_dict={self.image_x: x})
