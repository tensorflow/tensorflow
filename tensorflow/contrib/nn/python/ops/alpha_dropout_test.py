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
"""Tests for sampling_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.nn.python.ops.alpha_dropout import alpha_dropout
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import test


class AlphaDropoutTest(test.TestCase):

  def testAlphaDropout(self):
    x_dim, y_dim = 40, 30
    for keep_prob in [0.1, 0.5, 0.8]:
      with self.cached_session():
        t = random_ops.random_normal([x_dim, y_dim])
        output = alpha_dropout(t, keep_prob)
        self.assertEqual([x_dim, y_dim], output.get_shape())
        t_mean, t_std = nn_impl.moments(t, axes=[0, 1])
        output_mean, output_std = nn_impl.moments(output, axes=[0, 1])
        self.assertLess(abs(t_mean.eval() - output_mean.eval()), 0.1)
        self.assertLess(abs(t_std.eval() - output_std.eval()), 0.1)

  def testShapedDropoutShapeError(self):
    # Runs shaped dropout and verifies an error is thrown on misshapen noise.
    x_dim = 40
    y_dim = 30
    keep_prob = 0.5
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      _ = alpha_dropout(t, keep_prob, noise_shape=[x_dim, y_dim + 10])
    with self.assertRaises(ValueError):
      _ = alpha_dropout(t, keep_prob, noise_shape=[x_dim, y_dim, 5])
    with self.assertRaises(ValueError):
      _ = alpha_dropout(t, keep_prob, noise_shape=[x_dim + 3])
    with self.assertRaises(ValueError):
      _ = alpha_dropout(t, keep_prob, noise_shape=[x_dim])

    # test that broadcasting proceeds
    _ = alpha_dropout(t, keep_prob, noise_shape=[y_dim])
    _ = alpha_dropout(t, keep_prob, noise_shape=[1, y_dim])
    _ = alpha_dropout(t, keep_prob, noise_shape=[x_dim, 1])
    _ = alpha_dropout(t, keep_prob, noise_shape=[1, 1])

  def testInvalidKeepProb(self):
    x_dim, y_dim = 40, 30
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      alpha_dropout(t, -1.0)
    with self.assertRaises(ValueError):
      alpha_dropout(t, 1.1)
    with self.assertRaises(ValueError):
      alpha_dropout(t, [0.0, 1.0])
    with self.assertRaises(ValueError):
      alpha_dropout(t, array_ops.placeholder(dtypes.float64))
    with self.assertRaises(ValueError):
      alpha_dropout(t, array_ops.placeholder(dtypes.float32, shape=[2]))

  def testNoDropoutFast(self):
    x = array_ops.zeros((5,))
    for p in 1, constant_op.constant(1.0):
      y = alpha_dropout(x, keep_prob=p)
      self.assertTrue(x is y)


if __name__ == '__main__':
  test.main()
