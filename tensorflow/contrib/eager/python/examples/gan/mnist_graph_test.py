# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python.examples.gan import mnist

NOISE_DIM = 100
# Big enough so that summaries are never recorded.
# Lower this value if would like to benchmark with some summaries.
SUMMARY_INTERVAL = 10000
SUMMARY_FLUSH_MS = 100  # Flush summaries every 100ms


def data_format():
  return 'channels_first' if tf.test.is_gpu_available() else 'channels_last'


class MnistGraphGanBenchmark(tf.test.Benchmark):

  def _create_graph(self, batch_size):
    # Generate some random data.
    images_data = np.random.randn(batch_size, 784).astype(np.float32)
    dataset = tf.data.Dataset.from_tensors(images_data)
    images = dataset.repeat().make_one_shot_iterator().get_next()

    # Create the models and optimizers
    generator = mnist.Generator(data_format())
    discriminator = mnist.Discriminator(data_format())
    with tf.variable_scope('generator'):
      generator_optimizer = tf.train.AdamOptimizer(0.001)
    with tf.variable_scope('discriminator'):
      discriminator_optimizer = tf.train.AdamOptimizer(0.001)

    # Run models and compute loss
    noise_placeholder = tf.placeholder(tf.float32,
                                       shape=[batch_size, NOISE_DIM])
    generated_images = generator(noise_placeholder)
    tf.contrib.summary.image('generated_images',
                             tf.reshape(generated_images, [-1, 28, 28, 1]),
                             max_images=10)
    discriminator_gen_outputs = discriminator(generated_images)
    discriminator_real_outputs = discriminator(images)
    generator_loss = mnist.generator_loss(discriminator_gen_outputs)
    discriminator_loss = mnist.discriminator_loss(discriminator_real_outputs,
                                                  discriminator_gen_outputs)
    # Get train ops
    with tf.variable_scope('generator'):
      generator_train = generator_optimizer.minimize(
          generator_loss, var_list=generator.variables)
    with tf.variable_scope('discriminator'):
      discriminator_train = discriminator_optimizer.minimize(
          discriminator_loss, var_list=discriminator.variables)

    return (generator_train, discriminator_train, noise_placeholder)

  def _report(self, test_name, start, num_iters, batch_size):
    avg_time = (time.time() - start) / num_iters
    dev = 'gpu' if tf.test.is_gpu_available() else 'cpu'
    name = 'graph_%s_%s_batch_%d_%s' % (test_name, dev, batch_size,
                                        data_format())
    extras = {'examples_per_sec': batch_size / avg_time}
    self.report_benchmark(
        iters=num_iters, wall_time=avg_time, name=name, extras=extras)

  def benchmark_train(self):
    for batch_size in [64, 128, 256]:
      with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        increment_global_step = tf.assign_add(global_step, 1)
        with tf.contrib.summary.create_file_writer(
            tempfile.mkdtemp(), flush_millis=SUMMARY_FLUSH_MS).as_default(), (
                tf.contrib.summary.record_summaries_every_n_global_steps(
                    SUMMARY_INTERVAL)):
          (generator_train, discriminator_train, noise_placeholder
          ) = self._create_graph(batch_size)

          with tf.Session() as sess:
            tf.contrib.summary.initialize(graph=tf.get_default_graph(),
                                          session=sess)

            sess.run(tf.global_variables_initializer())

            num_burn, num_iters = (3, 100)
            for _ in range(num_burn):
              noise = np.random.uniform(-1.0, 1.0, size=[batch_size, NOISE_DIM])
              # Increment global step before evaluating summary ops to avoid
              # race condition.
              sess.run(increment_global_step)
              sess.run([generator_train, discriminator_train,
                        tf.contrib.summary.all_summary_ops()],
                       feed_dict={noise_placeholder: noise})

            # Run and benchmark 2 epochs
            start = time.time()
            for _ in range(num_iters):
              noise = np.random.uniform(-1.0, 1.0, size=[batch_size, NOISE_DIM])
              sess.run(increment_global_step)
              sess.run([generator_train, discriminator_train,
                        tf.contrib.summary.all_summary_ops()],
                       feed_dict={noise_placeholder: noise})
            self._report('train', start, num_iters, batch_size)

  def benchmark_generate(self):
    for batch_size in [64, 128, 256]:
      with tf.Graph().as_default():
        # Using random weights. This will generate garbage.
        generator = mnist.Generator(data_format())
        noise_placeholder = tf.placeholder(tf.float32,
                                           shape=[batch_size, NOISE_DIM])
        generated_images = generator(noise_placeholder)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
          sess.run(init)
          noise = np.random.uniform(-1.0, 1.0, size=[batch_size, NOISE_DIM])
          num_burn, num_iters = (30, 1000)
          for _ in range(num_burn):
            sess.run(generated_images, feed_dict={noise_placeholder: noise})

          start = time.time()
          for _ in range(num_iters):
            # Comparison with the eager execution benchmark in mnist_test.py
            # isn't entirely fair as the time here includes the cost of copying
            # the feeds from CPU memory to GPU.
            sess.run(generated_images, feed_dict={noise_placeholder: noise})
          self._report('generate', start, num_iters, batch_size)


if __name__ == '__main__':
  tf.test.main()
