# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.ops.special_math_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class LBetaTest(tf.test.TestCase):
  _use_gpu = False

  def test_one_dimensional_arg(self):
    # Should evaluate to 1 and 1/2.
    x_one = [1, 1.]
    x_one_half = [2, 1.]
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllClose(1, tf.exp(tf.lbeta(x_one)).eval())
      self.assertAllClose(0.5, tf.exp(tf.lbeta(x_one_half)).eval())
      self.assertEqual([], tf.lbeta(x_one).get_shape())

  def test_one_dimensional_arg_dynamic_alloc(self):
    # Should evaluate to 1 and 1/2.
    x_one = [1, 1.]
    x_one_half = [2, 1.]
    with self.test_session(use_gpu=self._use_gpu):
      ph = tf.placeholder(tf.float32)
      beta_ph = tf.exp(tf.lbeta(ph))
      self.assertAllClose(1, beta_ph.eval(feed_dict={ph: x_one}))
      self.assertAllClose(0.5, beta_ph.eval(feed_dict={ph: x_one_half}))

  def test_two_dimensional_arg(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllClose([0.5, 0.5], tf.exp(tf.lbeta(x_one_half)).eval())
      self.assertEqual((2,), tf.lbeta(x_one_half).get_shape())

  def test_two_dimensional_arg_dynamic_alloc(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=self._use_gpu):
      ph = tf.placeholder(tf.float32)
      beta_ph = tf.exp(tf.lbeta(ph))
      self.assertAllClose([0.5, 0.5], beta_ph.eval(feed_dict={ph: x_one_half}))

  def test_two_dimensional_proper_shape(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllClose([0.5, 0.5], tf.exp(tf.lbeta(x_one_half)).eval())
      self.assertEqual((2,), tf.shape(tf.lbeta(x_one_half)).eval())
      self.assertEqual(tf.TensorShape([2]), tf.lbeta(x_one_half).get_shape())

  def test_complicated_shape(self):
    with self.test_session(use_gpu=self._use_gpu):
      x = tf.convert_to_tensor(np.random.rand(3, 2, 2))
      self.assertAllEqual((3, 2), tf.shape(tf.lbeta(x)).eval())
      self.assertEqual(tf.TensorShape([3, 2]), tf.lbeta(x).get_shape())

  def test_length_1_last_dimension_results_in_one(self):
    # If there is only one coefficient, the formula still works, and we get one
    # as the answer, always.
    x_a = [5.5]
    x_b = [0.1]
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllClose(1, tf.exp(tf.lbeta(x_a)).eval())
      self.assertAllClose(1, tf.exp(tf.lbeta(x_b)).eval())
      self.assertEqual((), tf.lbeta(x_a).get_shape())

  def test_empty_rank2_or_greater_input_gives_empty_output(self):
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllEqual([], tf.lbeta([[]]).eval())
      self.assertEqual((0,), tf.lbeta([[]]).get_shape())
      self.assertAllEqual([[]], tf.lbeta([[[]]]).eval())
      self.assertEqual((1, 0), tf.lbeta([[[]]]).get_shape())

  def test_empty_rank2_or_greater_input_gives_empty_output_dynamic_alloc(self):
    with self.test_session(use_gpu=self._use_gpu):
      ph = tf.placeholder(tf.float32)
      self.assertAllEqual([], tf.lbeta(ph).eval(feed_dict={ph: [[]]}))
      self.assertAllEqual([[]], tf.lbeta(ph).eval(feed_dict={ph: [[[]]]}))

  def test_empty_rank1_input_raises_value_error(self):
    with self.test_session(use_gpu=self._use_gpu):
      with self.assertRaisesRegexp(ValueError, 'rank'):
        tf.lbeta([])

  def test_empty_rank1_dynamic_alloc_input_raises_op_error(self):
    with self.test_session(use_gpu=self._use_gpu):
      ph = tf.placeholder(tf.float32)
      with self.assertRaisesOpError('rank'):
        tf.lbeta(ph).eval(feed_dict={ph: []})


class LBetaTestGpu(LBetaTest):
  _use_gpu = True


if __name__ == '__main__':
  tf.test.main()
