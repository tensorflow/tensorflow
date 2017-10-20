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

import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python.examples import linear_regression
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.platform import tf_logging as logging


def _create_data_gen_for_test():
  true_w = np.array([[1.0], [-0.5], [2.0]], dtype=np.float32)
  true_b = np.array([1.0], dtype=np.float32)
  noise_level = 0
  batch_size = 64
  return (
      true_w, true_b, noise_level, batch_size,
      linear_regression.DataGenerator(true_w, true_b, noise_level, batch_size))


class LinearRegressionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(LinearRegressionTest, self).setUp()
    self._tmp_logdir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._tmp_logdir)
    super(LinearRegressionTest, self).tearDown()

  def testSyntheticBatch(self):
    _, _, _, batch_size, data_gen = _create_data_gen_for_test()

    xs, ys = data_gen.next_batch()
    self.assertEqual((batch_size, 3), xs.shape)
    self.assertEqual((batch_size, 1), ys.shape)
    self.assertEqual(tf.float32, xs.dtype)
    self.assertEqual(tf.float32, ys.dtype)

  def testLinearRegression(self):
    true_w, true_b, _, _, data_gen = _create_data_gen_for_test()

    learning_rate = 0.1
    num_iters = 40

    device = "gpu:0" if context.context().num_gpus() > 0 else "cpu:0"
    logging.info("device = %s", device)
    with context.device(device):
      linear_model = linear_regression.LinearModel()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      linear_model.fit(data_gen.next_batch, optimizer, num_iters,
                       logdir=self._tmp_logdir)

      self.assertAllClose(true_w, linear_model.weights, rtol=1e-2)
      self.assertAllClose(true_b, linear_model.biases, rtol=1e-2)
      self.assertTrue(glob.glob(os.path.join(self._tmp_logdir, "events.out.*")))


class EagerLinearRegressionBenchmark(test.Benchmark):

  def benchmarkEagerLinearRegression(self):
    _, _, _, _, data_gen = _create_data_gen_for_test()

    learning_rate = 0.1
    num_burnin_iters = 10
    num_iters = 200

    device = "gpu:0" if context.context().num_gpus() > 0 else "cpu:0"
    logging.info("device = %s", device)
    with context.device(device):
      linear_model = linear_regression.LinearModel()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)

      # Perform burn-in.
      linear_model.fit(data_gen.next_batch, optimizer, num_burnin_iters)

      start_time = time.time()
      linear_model.fit(data_gen.next_batch, optimizer, num_iters)
      wall_time = time.time() - start_time

      self.report_benchmark(
          name="EagerLinearRegression",
          iters=num_iters,
          wall_time=wall_time)


if __name__ == "__main__":
  test.main()
