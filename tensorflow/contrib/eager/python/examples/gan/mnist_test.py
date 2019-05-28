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

import tensorflow as tf

from tensorflow.contrib.eager.python.examples.gan import mnist

NOISE_DIM = 100
# Big enough so that summaries are never recorded.
# Lower this value if would like to benchmark with some summaries.
SUMMARY_INTERVAL = 10000
SUMMARY_FLUSH_MS = 100  # Flush summaries every 100ms


def data_format():
  return 'channels_first' if tf.test.is_gpu_available() else 'channels_last'


def device():
  return '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'


class MnistEagerGanBenchmark(tf.test.Benchmark):

  def _report(self, test_name, start, num_iters, batch_size):
    avg_time = (time.time() - start) / num_iters
    dev = 'gpu' if tf.test.is_gpu_available() else 'cpu'
    name = 'eager_%s_%s_batch_%d_%s' % (test_name, dev, batch_size,
                                        data_format())
    extras = {'examples_per_sec': batch_size / avg_time}
    self.report_benchmark(
        iters=num_iters, wall_time=avg_time, name=name, extras=extras)

  def benchmark_train(self):
    for batch_size in [64, 128, 256]:
      # Generate some random data.
      burn_batches, measure_batches = (3, 100)
      burn_images = [tf.random_normal([batch_size, 784])
                     for _ in range(burn_batches)]
      burn_dataset = tf.data.Dataset.from_tensor_slices(burn_images)
      measure_images = [tf.random_normal([batch_size, 784])
                        for _ in range(measure_batches)]
      measure_dataset = tf.data.Dataset.from_tensor_slices(measure_images)

      step_counter = tf.train.get_or_create_global_step()
      with tf.device(device()):
        # Create the models and optimizers
        generator = mnist.Generator(data_format())
        discriminator = mnist.Discriminator(data_format())
        with tf.variable_scope('generator'):
          generator_optimizer = tf.train.AdamOptimizer(0.001)
        with tf.variable_scope('discriminator'):
          discriminator_optimizer = tf.train.AdamOptimizer(0.001)

        with tf.contrib.summary.create_file_writer(
            tempfile.mkdtemp(), flush_millis=SUMMARY_FLUSH_MS).as_default():

          # warm up
          mnist.train_one_epoch(generator, discriminator, generator_optimizer,
                                discriminator_optimizer,
                                burn_dataset, step_counter,
                                log_interval=SUMMARY_INTERVAL,
                                noise_dim=NOISE_DIM)
          # measure
          start = time.time()
          mnist.train_one_epoch(generator, discriminator, generator_optimizer,
                                discriminator_optimizer,
                                measure_dataset, step_counter,
                                log_interval=SUMMARY_INTERVAL,
                                noise_dim=NOISE_DIM)
          self._report('train', start, measure_batches, batch_size)

  def benchmark_generate(self):
    for batch_size in [64, 128, 256]:
      with tf.device(device()):
        # Using random weights. This will generate garbage.
        generator = mnist.Generator(data_format())

        num_burn, num_iters = (30, 1000)
        for _ in range(num_burn):
          noise = tf.random_uniform(shape=[batch_size, NOISE_DIM],
                                    minval=-1., maxval=1.)
          generator(noise)

        start = time.time()
        for _ in range(num_iters):
          noise = tf.random_uniform(shape=[batch_size, NOISE_DIM],
                                    minval=-1., maxval=1.)
          generator(noise)
        self._report('generate', start, num_iters, batch_size)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
