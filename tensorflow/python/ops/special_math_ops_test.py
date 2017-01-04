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

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import test


class LBetaTest(test.TestCase):
  _use_gpu = False

  def test_one_dimensional_arg(self):
    # Should evaluate to 1 and 1/2.
    x_one = [1, 1.]
    x_one_half = [2, 1.]
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllClose(1, math_ops.exp(special_math_ops.lbeta(x_one)).eval())
      self.assertAllClose(
          0.5, math_ops.exp(special_math_ops.lbeta(x_one_half)).eval())
      self.assertEqual([], special_math_ops.lbeta(x_one).get_shape())

  def test_one_dimensional_arg_dynamic_alloc(self):
    # Should evaluate to 1 and 1/2.
    x_one = [1, 1.]
    x_one_half = [2, 1.]
    with self.test_session(use_gpu=self._use_gpu):
      ph = array_ops.placeholder(dtypes.float32)
      beta_ph = math_ops.exp(special_math_ops.lbeta(ph))
      self.assertAllClose(1, beta_ph.eval(feed_dict={ph: x_one}))
      self.assertAllClose(0.5, beta_ph.eval(feed_dict={ph: x_one_half}))

  def test_two_dimensional_arg(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllClose(
          [0.5, 0.5], math_ops.exp(special_math_ops.lbeta(x_one_half)).eval())
      self.assertEqual((2,), special_math_ops.lbeta(x_one_half).get_shape())

  def test_two_dimensional_arg_dynamic_alloc(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=self._use_gpu):
      ph = array_ops.placeholder(dtypes.float32)
      beta_ph = math_ops.exp(special_math_ops.lbeta(ph))
      self.assertAllClose([0.5, 0.5], beta_ph.eval(feed_dict={ph: x_one_half}))

  def test_two_dimensional_proper_shape(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllClose(
          [0.5, 0.5], math_ops.exp(special_math_ops.lbeta(x_one_half)).eval())
      self.assertEqual(
          (2,), array_ops.shape(special_math_ops.lbeta(x_one_half)).eval())
      self.assertEqual(
          tensor_shape.TensorShape([2]),
          special_math_ops.lbeta(x_one_half).get_shape())

  def test_complicated_shape(self):
    with self.test_session(use_gpu=self._use_gpu):
      x = ops.convert_to_tensor(np.random.rand(3, 2, 2))
      self.assertAllEqual(
          (3, 2), array_ops.shape(special_math_ops.lbeta(x)).eval())
      self.assertEqual(
          tensor_shape.TensorShape([3, 2]),
          special_math_ops.lbeta(x).get_shape())

  def test_length_1_last_dimension_results_in_one(self):
    # If there is only one coefficient, the formula still works, and we get one
    # as the answer, always.
    x_a = [5.5]
    x_b = [0.1]
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllClose(1, math_ops.exp(special_math_ops.lbeta(x_a)).eval())
      self.assertAllClose(1, math_ops.exp(special_math_ops.lbeta(x_b)).eval())
      self.assertEqual((), special_math_ops.lbeta(x_a).get_shape())

  def test_empty_rank2_or_greater_input_gives_empty_output(self):
    with self.test_session(use_gpu=self._use_gpu):
      self.assertAllEqual([], special_math_ops.lbeta([[]]).eval())
      self.assertEqual((0,), special_math_ops.lbeta([[]]).get_shape())
      self.assertAllEqual([[]], special_math_ops.lbeta([[[]]]).eval())
      self.assertEqual((1, 0), special_math_ops.lbeta([[[]]]).get_shape())

  def test_empty_rank2_or_greater_input_gives_empty_output_dynamic_alloc(self):
    with self.test_session(use_gpu=self._use_gpu):
      ph = array_ops.placeholder(dtypes.float32)
      self.assertAllEqual(
          [], special_math_ops.lbeta(ph).eval(feed_dict={ph: [[]]}))
      self.assertAllEqual(
          [[]], special_math_ops.lbeta(ph).eval(feed_dict={ph: [[[]]]}))

  def test_empty_rank1_input_raises_value_error(self):
    with self.test_session(use_gpu=self._use_gpu):
      with self.assertRaisesRegexp(ValueError, 'rank'):
        special_math_ops.lbeta([])

  def test_empty_rank1_dynamic_alloc_input_raises_op_error(self):
    with self.test_session(use_gpu=self._use_gpu):
      ph = array_ops.placeholder(dtypes.float32)
      with self.assertRaisesOpError('rank'):
        special_math_ops.lbeta(ph).eval(feed_dict={ph: []})


class LBetaTestGpu(LBetaTest):
  _use_gpu = True


class EinsumTest(test.TestCase):

  simple_cases = [
      'ij,jk->ik',
      'ijk,jklm->il',
      'ij,jk,kl->il',
      'ijk->i',
      'ijk->kji',
      'ji,kj->ik',

      'ikl,kji->kl',
      'klj,lki->ij',
      'ijk,ilj->kli',
      'kij,mkb->ijmb',
      'ijk,ijl,ikl->i',
      'i,ijk,j->k',
      'ij,ij,jk,kl->il',
      'ij,kj,il,jm->ml',

      'a,ab,abc->abc',
      'a,b,ab->ab',
      'ab,ab,c->',
      'ab,ab,c->c',
      'ab,ab,cd,cd->',
      'ab,ab,cd,cd->ac',
      'ab,ab,cd,cd->cd',
      'ab,ab,cd,cd,ef,ef->',

      'ab,cd,ef->abcdef',
      'ab,cd,ef->acdf',
      'ab,cd,de->abcde',
      'ab,cd,de->be',
      'ab,bcd,cd->abcd',
      'ab,bcd,cd->abd',

      'eb,cb,fb->cef',
      'abcd,ad',
      'bd,db,eac->ace',
      'ba,ac,da->bcd',

      'ab,ab',
      'ab,ba',
      'abc,abc',
      'abc,bac',
      'abc,cba',

      'dba,ead,cad->bce',
      'aef,fbc,dca->bde',
  ]

  long_cases = [
      'bca,cdb,dbf,afc->',
      'efc,dbc,acf,fd->abe',
      'ea,fb,gc,hd,abcd->efgh',
      'ea,fb,abcd,gc,hd->efgh',
      'abhe,hidj,jgba,hiab,gab',
  ]

  invalid_cases = [
      # bad formats
      '',
      'ijk ijk',
      'ij.jk->ik',
      'ij...,jk...->ik...',

      # axis in output that does not exist
      'ij,jk->im',

      # incorrect number of dimensions
      'ij,jkl->kl',

      # this is allowed in numpy but not implemented here yet
      'iij,jk'
  ]

  dim_mismatch_cases = [('ijk,jkl->il', [(2, 3, 4), (3, 5, 6)])]

  def test_simple(self):
    for case in self.simple_cases:
      self.run_test(case)

  def test_long(self):
    for case in self.long_cases:
      self.run_test(case)

  def test_invalid(self):
    for axes in self.invalid_cases:
      inputs = [
          array_ops.placeholder(
              dtypes.float32, shape=(3, 4)),
          array_ops.placeholder(
              dtypes.float32, shape=(3, 4)),
      ]
      with self.assertRaises(ValueError):
        _ = special_math_ops.einsum(axes, *inputs)

  def test_dim_mismatch(self):
    for axes, input_shapes in self.dim_mismatch_cases:
      inputs = [
          array_ops.placeholder(
              dtypes.float32, shape=shape) for shape in input_shapes
      ]
      with self.assertRaises(ValueError):
        _ = special_math_ops.einsum(axes, *inputs)

  def run_test(self, axes):
    all_axes = {ax: np.random.randint(4, 12) for ax in axes if ax.isalpha()}

    input_vals = []
    input_axes, _, _ = axes.partition('->')

    for idx in input_axes.split(','):
      shape = [all_axes[ax] for ax in idx]
      input_vals.append(np.random.random(shape))

    input_tensors = [constant_op.constant(val) for val in input_vals]
    output_tensor = special_math_ops.einsum(axes, *input_tensors)

    with self.test_session():
      output_value = output_tensor.eval()

    correct_value = np.einsum(axes, *input_vals)

    err = np.abs(correct_value - output_value).max()
    print(axes, err)
    assert err < 1e-8

  def test_input_is_placeholder(self):
    with ops.Graph().as_default():
      m0 = array_ops.placeholder(dtypes.int32, shape=(1, None))
      m1 = array_ops.placeholder(dtypes.int32, shape=(None, 1))
      out = special_math_ops.einsum('ij,jk->ik', m0, m1)
      with session.Session() as sess:
        feed_dict = {
            m0: [[1, 2, 3]],
            m1: [[2], [1], [1]],
        }
        np.testing.assert_almost_equal(
            [[7]], sess.run(out, feed_dict=feed_dict))

    with ops.Graph().as_default():
      m0 = array_ops.placeholder(dtypes.int32, shape=(None, 3))
      m1 = array_ops.placeholder(dtypes.int32, shape=(3,))
      out = special_math_ops.einsum('ij,j->i', m0, m1)
      with session.Session() as sess:
        feed_dict = {
            m0: [[1, 2, 3]],
            m1: [2, 1, 1],
        }
        np.testing.assert_almost_equal([7], sess.run(out, feed_dict=feed_dict))


if __name__ == '__main__':
  test.main()
