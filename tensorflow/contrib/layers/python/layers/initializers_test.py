# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class InitializerTest(tf.test.TestCase):

  def test_xavier_wrong_dtype(self):
    with self.assertRaisesRegexp(
        TypeError,
        'Cannot create initializer for non-floating point type.'):
      tf.contrib.layers.xavier_initializer(dtype=tf.int32)

    self.assertIsNone(tf.contrib.layers.l1_regularizer(0.)(None))

  def _test_xavier(self, initializer, shape, variance, uniform):
    with tf.Session() as sess:
      var = tf.get_variable(name='test', shape=shape, dtype=tf.float32,
                            initializer=initializer(uniform=uniform, seed=1))
      sess.run(tf.initialize_all_variables())
      values = var.eval()
      self.assertAllClose(np.var(values), variance, 1e-3, 1e-3)

  def test_xavier_uniform(self):
    self._test_xavier(tf.contrib.layers.xavier_initializer,
                      [100, 40], 2. / (100. + 40.), True)

  def test_xavier_normal(self):
    self._test_xavier(tf.contrib.layers.xavier_initializer,
                      [100, 40], 2. / (100. + 40.), False)

  def test_xavier_conv2d_uniform(self):
    self._test_xavier(tf.contrib.layers.xavier_initializer_conv2d,
                      [100, 40, 5, 7], 2. / (100. * 40 * (5 + 7)), True)

  def test_xavier_conv2d_normal(self):
    self._test_xavier(tf.contrib.layers.xavier_initializer_conv2d,
                      [100, 40, 5, 7], 2. / (100. * 40 * (5 + 7)), False)


class VarianceScalingInitializerTest(tf.test.TestCase):

  def test_wrong_dtype(self):
    with self.assertRaisesRegexp(
        TypeError,
        'Cannot create initializer for non-floating point type.'):
      tf.contrib.layers.variance_scaling_initializer(dtype=tf.int32)

  def _test_variance(self, initializer, shape, variance, factor, mode, uniform):
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        var = tf.get_variable(name='test', shape=shape, dtype=tf.float32,
                              initializer=initializer(factor=factor,
                                                      mode=mode,
                                                      uniform=uniform,
                                                      seed=1))
        sess.run(tf.initialize_all_variables())
        values = var.eval()
        self.assertAllClose(np.var(values), variance, 1e-3, 1e-3)

  def test_fan_in(self):
    for uniform in [False, True]:
      self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                          shape=[100, 40],
                          variance=2. / 100.,
                          factor=2.0,
                          mode='FAN_IN',
                          uniform=uniform)

  def test_fan_out(self):
    for uniform in [False, True]:
      self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                          shape=[100, 40],
                          variance=2. / 40.,
                          factor=2.0,
                          mode='FAN_OUT',
                          uniform=uniform)

  def test_fan_avg(self):
    for uniform in [False, True]:
      self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                          shape=[100, 40],
                          variance=4. / (100. + 40.),
                          factor=2.0,
                          mode='FAN_AVG',
                          uniform=uniform)

  def test_conv2d_fan_in(self):
    for uniform in [False, True]:
      self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                          shape=[100, 40, 5, 7],
                          variance=2. / (100. * 40. * 5.),
                          factor=2.0,
                          mode='FAN_IN',
                          uniform=uniform)

  def test_conv2d_fan_out(self):
    for uniform in [False, True]:
      self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                          shape=[100, 40, 5, 7],
                          variance=2. / (100. * 40. * 7.),
                          factor=2.0,
                          mode='FAN_OUT',
                          uniform=uniform)

  def test_conv2d_fan_avg(self):
    for uniform in [False, True]:
      self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                          shape=[100, 40, 5, 7],
                          variance=2. / (100. * 40. * (5. + 7.)),
                          factor=2.0,
                          mode='FAN_AVG',
                          uniform=uniform)

  def test_xavier_uniform(self):
    self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                        shape=[100, 40],
                        variance=2. / (100. + 40.),
                        factor=1.0,
                        mode='FAN_AVG',
                        uniform=True)

  def test_xavier_normal(self):
    self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                        shape=[100, 40],
                        variance=2. / (100. + 40.),
                        factor=1.0,
                        mode='FAN_AVG',
                        uniform=False)

  def test_xavier_conv2d_uniform(self):
    self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                        shape=[100, 40, 5, 7],
                        variance=2. / (100. * 40. * (5. + 7.)),
                        factor=1.0,
                        mode='FAN_AVG',
                        uniform=True)

  def test_xavier_conv2d_normal(self):
    self._test_variance(tf.contrib.layers.variance_scaling_initializer,
                        shape=[100, 40, 5, 7],
                        variance=2. / (100. * 40. * (5. + 7.)),
                        factor=1.0,
                        mode='FAN_AVG',
                        uniform=True)

if __name__ == '__main__':
  tf.test.main()
