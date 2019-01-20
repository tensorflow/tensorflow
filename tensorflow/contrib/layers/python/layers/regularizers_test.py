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

from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RegularizerTest(test.TestCase):

  def test_l1(self):
    with self.assertRaises(ValueError):
      regularizers.l1_regularizer(-1.)
    with self.assertRaises(ValueError):
      regularizers.l1_regularizer(0)

    self.assertIsNone(regularizers.l1_regularizer(0.)(None))

    values = np.array([1., -1., 4., 2.])
    weights = constant_op.constant(values)
    with session.Session() as sess:
      result = sess.run(regularizers.l1_regularizer(.5)(weights))

    self.assertAllClose(np.abs(values).sum() * .5, result)

  def test_l2(self):
    with self.assertRaises(ValueError):
      regularizers.l2_regularizer(-1.)
    with self.assertRaises(ValueError):
      regularizers.l2_regularizer(0)

    self.assertIsNone(regularizers.l2_regularizer(0.)(None))

    values = np.array([1., -1., 4., 2.])
    weights = constant_op.constant(values)
    with session.Session() as sess:
      result = sess.run(regularizers.l2_regularizer(.42)(weights))

    self.assertAllClose(np.power(values, 2).sum() / 2.0 * .42, result)

  def test_l1_l2(self):
    with self.assertRaises(ValueError):
      regularizers.l1_l2_regularizer(-1., 0.5)
    with self.assertRaises(ValueError):
      regularizers.l1_l2_regularizer(0.5, -1.)
    with self.assertRaises(ValueError):
      regularizers.l1_l2_regularizer(0, 0.5)
    with self.assertRaises(ValueError):
      regularizers.l1_l2_regularizer(0.5, 0)

    with self.cached_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = constant_op.constant(1.0, shape=shape)
      loss = regularizers.l1_l2_regularizer(1.0, 1.0)(tensor)
      self.assertEquals(loss.op.name, 'l1_l2_regularizer')
      self.assertAlmostEqual(loss.eval(), num_elem + num_elem / 2, 5)

  def test_l1_l2_scale_l1Zero(self):
    shape = [5, 5, 5]
    num_elem = 5 * 5 * 5
    tensor = constant_op.constant(1.0, shape=shape)
    loss = regularizers.l1_l2_regularizer(0.0, 1.0)(tensor)
    with self.cached_session():
      self.assertEquals(loss.op.name, 'l1_l2_regularizer')
      self.assertAlmostEqual(loss.eval(), num_elem / 2, 5)

  def test_l1_l2_scale_l2Zero(self):
    shape = [5, 5, 5]
    num_elem = 5 * 5 * 5
    tensor = constant_op.constant(1.0, shape=shape)
    loss = regularizers.l1_l2_regularizer(1.0, 0.0)(tensor)
    with self.cached_session():
      self.assertEquals(loss.op.name, 'l1_l2_regularizer')
      self.assertAlmostEqual(loss.eval(), num_elem, 5)

  def test_l1_l2_scales_Zero(self):
    shape = [5, 5, 5]
    tensor = constant_op.constant(1.0, shape=shape)
    loss = regularizers.l1_l2_regularizer(0.0, 0.0)(tensor)
    self.assertEquals(loss, None)

  def testL1L2RegularizerWithScope(self):
    with self.cached_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = constant_op.constant(1.0, shape=shape)
      with ops.name_scope('foo'):
        loss = regularizers.l1_l2_regularizer(1.0, 1.0, scope='l1_l2')(tensor)
      self.assertEquals(loss.op.name, 'foo/l1_l2')
      self.assertAlmostEqual(loss.eval(), num_elem + num_elem / 2, 5)

  def test_sum_regularizer(self):
    l1_function = regularizers.l1_regularizer(.1)
    l2_function = regularizers.l2_regularizer(.2)
    self.assertIsNone(regularizers.sum_regularizer([]))
    self.assertIsNone(regularizers.sum_regularizer([None]))

    values = np.array([-3.])
    weights = constant_op.constant(values)
    with session.Session() as sess:
      l1_reg1 = regularizers.sum_regularizer([l1_function])
      l1_result1 = sess.run(l1_reg1(weights))

      l1_reg2 = regularizers.sum_regularizer([l1_function, None])
      l1_result2 = sess.run(l1_reg2(weights))

      l1_l2_reg = regularizers.sum_regularizer([l1_function, l2_function])
      l1_l2_result = sess.run(l1_l2_reg(weights))

    self.assertAllClose(.1 * np.abs(values).sum(), l1_result1)
    self.assertAllClose(.1 * np.abs(values).sum(), l1_result2)
    self.assertAllClose(
        .1 * np.abs(values).sum() + .2 * np.power(values, 2).sum() / 2.0,
        l1_l2_result)

  def test_apply_regularization(self):
    dummy_regularizer = lambda x: math_ops.reduce_sum(2 * x)
    array_weights_list = [[1.5], [2, 3, 4.2], [10, 42, 666.6]]
    tensor_weights_list = [constant_op.constant(x) for x in array_weights_list]
    expected = sum(2 * x for l in array_weights_list for x in l)
    with self.cached_session():
      result = regularizers.apply_regularization(dummy_regularizer,
                                                 tensor_weights_list)
      self.assertAllClose(expected, result.eval())

  def test_apply_zero_regularization(self):
    regularizer = regularizers.l2_regularizer(0.0)
    array_weights_list = [[1.5], [2, 3, 4.2], [10, 42, 666.6]]
    tensor_weights_list = [constant_op.constant(x) for x in array_weights_list]
    with self.cached_session():
      result = regularizers.apply_regularization(regularizer,
                                                 tensor_weights_list)
      self.assertAllClose(0.0, result.eval())

  def test_apply_regularization_invalid_regularizer(self):
    non_scalar_regularizer = lambda x: array_ops.tile(x, [2])
    tensor_weights_list = [
        constant_op.constant(x) for x in [[1.5], [2, 3, 4.2], [10, 42, 666.6]]
    ]
    with self.cached_session():
      with self.assertRaises(ValueError):
        regularizers.apply_regularization(non_scalar_regularizer,
                                          tensor_weights_list)


if __name__ == '__main__':
  test.main()
