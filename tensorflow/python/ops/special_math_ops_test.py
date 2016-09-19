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


class EinsumTest(tf.test.TestCase):

  # standard cases
  simple_cases = [
    'ij,jk->ik',
    'ijk,jklm->il',
    'ij,jk,kl->il',
    'ijk->i',
  ]

  # where axes are not in order
  misordered_cases = [
    'ji,kj->ik',
    'ikl,kji->kl',
    'klj,lki->ij',
  ]

  # more than two arguments
  multiarg_cases = [
    'ijk,ijl,ikl->i',
    'i,ijk,j->k',
    'ij,ij,jk,kl->il',
  ]

  invalid_cases = [
    # bad formats
    'ijk ijk',
    'ij,jk,kl'
    'ij->',

    # axis in output that does not exist
    'ij,jk->im',

    # incorrect number of dimensions
    'ij,jkl->kl',
  ]

  dim_mismatch_cases = [
    ('ijk,jkl->il',
     [(2,3,4), (3,5,6)]),

  ]

  def test_simple(self):
    for case in self.simple_cases:
      self.run_test(case)

  def test_misordered(self):
    for case in self.misordered_cases:
      self.run_test(case)

  def test_multiarg(self):
    for case in self.multiarg_cases:
      self.run_test(case)

  def test_invalid(self):
    for axes in self.invalid_cases:
      result = None
      inputs = [
        tf.placeholder(tf.float32, shape=(3,4)),
        tf.placeholder(tf.float32, shape=(3,4)),
      ]

      try:
        result = tf.einsum(axes, *inputs)
      except AssertionError as e:
        print(e)
      assert result is None, \
        "An exception should have been thrown."

  def test_dim_mismatch(self):
    for axes, input_shapes in self.dim_mismatch_cases:
      inputs = [
        tf.placeholder(tf.float32, shape=shape)
        for shape in input_shapes
      ]
      result = None
      try:
        result = tf.einsum(axes, *inputs)
      except AssertionError:
        pass
      assert result is None, "An exception should have been thrown."

  def run_test(self, axes):
    all_axes = {ax: np.random.randint(4, 12)
                for ax in axes if ax.isalpha()}

    input_vals = []
    input_axes, _, _ = axes.partition('->')

    for idx in input_axes.split(','):
      shape = [all_axes[ax] for ax in idx]
      input_vals.append(np.random.random(shape))

    input_tensors = [tf.constant(val) for val in input_vals]
    output_tensor = tf.einsum(axes, *input_tensors)

    with self.test_session():
      output_value = output_tensor.eval()

    correct_value = np.einsum(axes, *input_vals)

    err = np.abs(correct_value - output_value).max()
    print(axes, err)
    assert err < 1e-8


if __name__ == '__main__':
  tf.test.main()
