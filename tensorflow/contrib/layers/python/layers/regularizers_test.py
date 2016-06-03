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
"""Tests for regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class RegularizerTest(tf.test.TestCase):

  def test_l1(self):
    with self.assertRaises(ValueError):
      tf.contrib.layers.l1_regularizer(2.)
    with self.assertRaises(ValueError):
      tf.contrib.layers.l1_regularizer(-1.)
    with self.assertRaises(ValueError):
      tf.contrib.layers.l1_regularizer(0)

    self.assertIsNone(tf.contrib.layers.l1_regularizer(0.)(None))

    values = np.array([1., -1., 4., 2.])
    weights = tf.constant(values)
    with tf.Session() as sess:
      result = sess.run(tf.contrib.layers.l1_regularizer(.5)(weights))

    self.assertAllClose(np.abs(values).sum() * .5, result)

  def test_l2(self):
    with self.assertRaises(ValueError):
      tf.contrib.layers.l2_regularizer(2.)
    with self.assertRaises(ValueError):
      tf.contrib.layers.l2_regularizer(-1.)
    with self.assertRaises(ValueError):
      tf.contrib.layers.l2_regularizer(0)

    self.assertIsNone(tf.contrib.layers.l2_regularizer(0.)(None))

    values = np.array([1., -1., 4., 2.])
    weights = tf.constant(values)
    with tf.Session() as sess:
      result = sess.run(tf.contrib.layers.l2_regularizer(.42)(weights))

    self.assertAllClose(np.power(values, 2).sum() / 2.0 * .42, result)

  def test_sum_regularizer(self):
    l1_function = tf.contrib.layers.l1_regularizer(.1)
    l2_function = tf.contrib.layers.l2_regularizer(.2)
    self.assertIsNone(tf.contrib.layers.sum_regularizer([]))
    self.assertIsNone(tf.contrib.layers.sum_regularizer([None]))

    values = np.array([-3.])
    weights = tf.constant(values)
    with tf.Session() as sess:
      l1_reg1 = tf.contrib.layers.sum_regularizer([l1_function])
      l1_result1 = sess.run(l1_reg1(weights))

      l1_reg2 = tf.contrib.layers.sum_regularizer([l1_function, None])
      l1_result2 = sess.run(l1_reg2(weights))

      l1_l2_reg = tf.contrib.layers.sum_regularizer([l1_function, l2_function])
      l1_l2_result = sess.run(l1_l2_reg(weights))

    self.assertAllClose(.1 * np.abs(values).sum(), l1_result1)
    self.assertAllClose(.1 * np.abs(values).sum(), l1_result2)
    self.assertAllClose(.1 * np.abs(values).sum() +
                        .2 * np.power(values, 2).sum() / 2.0,
                        l1_l2_result)

  def test_apply_regularization(self):
    dummy_regularizer = lambda x: tf.reduce_sum(2 * x)
    array_weights_list = [[1.5], [2, 3, 4.2], [10, 42, 666.6]]
    tensor_weights_list = [tf.constant(x) for x in array_weights_list]
    expected = sum([2 * x for l in array_weights_list for x in l])
    with self.test_session():
      result = tf.contrib.layers.apply_regularization(dummy_regularizer,
                                                      tensor_weights_list)
      self.assertAllClose(expected, result.eval())

  def test_apply_regularization_invalid_regularizer(self):
    non_scalar_regularizer = lambda x: tf.tile(x, [2])
    tensor_weights_list = [tf.constant(x)
                           for x in [[1.5], [2, 3, 4.2], [10, 42, 666.6]]]
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.layers.apply_regularization(non_scalar_regularizer,
                                               tensor_weights_list)


if __name__ == '__main__':
  tf.test.main()
