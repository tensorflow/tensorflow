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
"""Unit tests for linear regression example under TensorFlow eager execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import shutil
import tempfile
import time

import tensorflow as tf

import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python.examples.linear_regression import linear_regression


def device():
  return "/device:GPU:0" if tfe.num_gpus() > 0 else "/device:CPU:0"


class LinearRegressionTest(tf.test.TestCase):

  def setUp(self):
    super(LinearRegressionTest, self).setUp()
    self._tmp_logdir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._tmp_logdir)
    super(LinearRegressionTest, self).tearDown()

  def testSyntheticDataset(self):
    true_w = tf.random_uniform([3, 1])
    true_b = [1.0]
    batch_size = 10
    num_batches = 2
    noise_level = 0.
    dataset = linear_regression.synthetic_dataset(true_w, true_b, noise_level,
                                                  batch_size, num_batches)

    it = tfe.Iterator(dataset)
    for _ in range(2):
      (xs, ys) = it.next()
      self.assertEqual((batch_size, 3), xs.shape)
      self.assertEqual((batch_size, 1), ys.shape)
      self.assertEqual(tf.float32, xs.dtype)
      self.assertEqual(tf.float32, ys.dtype)
    with self.assertRaises(StopIteration):
      it.next()

  def testLinearRegression(self):
    true_w = [[1.0], [-0.5], [2.0]]
    true_b = [1.0]

    model = linear_regression.LinearModel()
    dataset = linear_regression.synthetic_dataset(
        true_w, true_b, noise_level=0., batch_size=64, num_batches=40)

    with tf.device(device()):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
      linear_regression.fit(model, dataset, optimizer, logdir=self._tmp_logdir)

      self.assertAllClose(true_w, model.variables[0].numpy(), rtol=1e-2)
      self.assertAllClose(true_b, model.variables[1].numpy(), rtol=1e-2)
      self.assertTrue(glob.glob(os.path.join(self._tmp_logdir, "events.out.*")))


class EagerLinearRegressionBenchmark(tf.test.Benchmark):

  def benchmarkEagerLinearRegression(self):
    num_batches = 200
    batch_size = 64
    dataset = linear_regression.synthetic_dataset(
        w=tf.random_uniform([3, 1]),
        b=tf.random_uniform([1]),
        noise_level=0.01,
        batch_size=batch_size,
        num_batches=num_batches)
    burn_in_dataset = dataset.take(10)

    model = linear_regression.LinearModel()

    with tf.device(device()):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

      # Perform burn-in.
      linear_regression.fit(model, burn_in_dataset, optimizer)

      start_time = time.time()
      linear_regression.fit(model, dataset, optimizer)
      wall_time = time.time() - start_time

      examples_per_sec = num_batches * batch_size / wall_time
      self.report_benchmark(
          name="eager_train_%s" %
          ("gpu" if tfe.num_gpus() > 0 else "cpu"),
          iters=num_batches,
          extras={"examples_per_sec": examples_per_sec},
          wall_time=wall_time)


if __name__ == "__main__":
  tfe.enable_eager_execution()
  tf.test.main()
