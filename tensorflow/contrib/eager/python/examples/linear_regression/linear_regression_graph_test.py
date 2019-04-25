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
"""Graph benchmark for linear regression, to contrast with eager execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.linear_regression import linear_regression


class GraphLinearRegressionBenchmark(tf.test.Benchmark):

  def benchmarkGraphLinearRegression(self):
    num_epochs = 10
    num_batches = 200
    batch_size = 64
    dataset = linear_regression.synthetic_dataset_helper(
        w=tf.random_uniform([3, 1]),
        b=tf.random_uniform([1]),
        num_features=3,
        noise_level=0.01,
        batch_size=batch_size,
        num_batches=num_batches)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    x, y = iterator.get_next()

    model = linear_regression.LinearModel()

    if tf.test.is_gpu_available():
      use_gpu = True
      device = "/device:GPU:0"
    else:
      use_gpu = False
      device = "/device:CPU:0"

    with tf.device(device):
      loss = linear_regression.mean_square_loss(model, x, y)
      optimization_step = tf.train.GradientDescentOptimizer(
          learning_rate=0.1).minimize(loss)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      def train(num_epochs):
        for _ in range(num_epochs):
          sess.run(iterator.initializer)
          try:
            while True:
              _, _ = sess.run([optimization_step, loss])
          except tf.errors.OutOfRangeError:
            pass

      # Warmup: a single epoch.
      train(1)

      start_time = time.time()
      train(num_epochs)
      wall_time = time.time() - start_time

      examples_per_sec = num_epochs * num_batches * batch_size / wall_time
      self.report_benchmark(
          name="graph_train_%s" %
          ("gpu" if use_gpu else "cpu"),
          iters=num_epochs * num_batches,
          extras={"examples_per_sec": examples_per_sec},
          wall_time=wall_time)


if __name__ == "__main__":
  tf.test.main()
