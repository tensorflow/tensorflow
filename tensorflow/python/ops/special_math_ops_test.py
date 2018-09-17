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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

class LBetaTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_one_dimensional_arg(self):
    # Should evaluate to 1 and 1/2.
    x_one = [1, 1.]
    x_one_half = [2, 1.]
    with self.test_session(use_gpu=True):
      self.assertAllClose(
          1, self.evaluate(math_ops.exp(special_math_ops.lbeta(x_one))))
      self.assertAllClose(
          0.5, self.evaluate(math_ops.exp(special_math_ops.lbeta(x_one_half))))
      self.assertEqual([], special_math_ops.lbeta(x_one).get_shape())

  def test_one_dimensional_arg_dynamic(self):
    # Should evaluate to 1 and 1/2.
    x_one = [1, 1.]
    x_one_half = [2, 1.]
    with self.test_session(use_gpu=True):
      ph = array_ops.placeholder(dtypes.float32)
      beta_ph = math_ops.exp(special_math_ops.lbeta(ph))
      self.assertAllClose(1, beta_ph.eval(feed_dict={ph: x_one}))
      self.assertAllClose(0.5,
                          beta_ph.eval(feed_dict={ph: x_one_half}))

  def test_four_dimensional_arg_with_partial_shape_dynamic(self):
    x_ = np.ones((3, 2, 3, 4))
    # Gamma(1) = 0! = 1
    # Gamma(1 + 1 + 1 + 1) = Gamma(4) = 3! = 6
    # ==> Beta([1, 1, 1, 1])
    #     = Gamma(1) * Gamma(1) * Gamma(1) * Gamma(1) / Gamma(1 + 1 + 1 + 1)
    #     = 1 / 6
    expected_beta_x = 1 / 6 * np.ones((3, 2, 3))
    with self.test_session(use_gpu=True):
      x_ph = array_ops.placeholder(dtypes.float32, [3, 2, 3, None])
      beta_ph = math_ops.exp(special_math_ops.lbeta(x_ph))
      self.assertAllClose(expected_beta_x,
                          beta_ph.eval(feed_dict={x_ph: x_}))

  @test_util.run_in_graph_and_eager_modes
  def test_two_dimensional_arg(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=True):
      self.assertAllClose(
          [0.5, 0.5],
          self.evaluate(math_ops.exp(special_math_ops.lbeta(x_one_half))))
      self.assertEqual((2,), special_math_ops.lbeta(x_one_half).get_shape())

  def test_two_dimensional_arg_dynamic(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=True):
      ph = array_ops.placeholder(dtypes.float32)
      beta_ph = math_ops.exp(special_math_ops.lbeta(ph))
      self.assertAllClose([0.5, 0.5],
                          beta_ph.eval(feed_dict={ph: x_one_half}))

  @test_util.run_in_graph_and_eager_modes
  def test_two_dimensional_proper_shape(self):
    # Should evaluate to 1/2.
    x_one_half = [[2, 1.], [2, 1.]]
    with self.test_session(use_gpu=True):
      self.assertAllClose(
          [0.5, 0.5],
          self.evaluate(math_ops.exp(special_math_ops.lbeta(x_one_half))))
      self.assertEqual(
          (2,),
          self.evaluate(array_ops.shape(special_math_ops.lbeta(x_one_half))))
      self.assertEqual(
          tensor_shape.TensorShape([2]),
          special_math_ops.lbeta(x_one_half).get_shape())

  @test_util.run_in_graph_and_eager_modes
  def test_complicated_shape(self):
    with self.test_session(use_gpu=True):
      x = ops.convert_to_tensor(np.random.rand(3, 2, 2))
      self.assertAllEqual(
          (3, 2), self.evaluate(array_ops.shape(special_math_ops.lbeta(x))))
      self.assertEqual(
          tensor_shape.TensorShape([3, 2]),
          special_math_ops.lbeta(x).get_shape())

  @test_util.run_in_graph_and_eager_modes
  def test_length_1_last_dimension_results_in_one(self):
    # If there is only one coefficient, the formula still works, and we get one
    # as the answer, always.
    x_a = [5.5]
    x_b = [0.1]
    with self.test_session(use_gpu=True):
      self.assertAllClose(
          1, self.evaluate(math_ops.exp(special_math_ops.lbeta(x_a))))
      self.assertAllClose(
          1, self.evaluate(math_ops.exp(special_math_ops.lbeta(x_b))))
      self.assertEqual((), special_math_ops.lbeta(x_a).get_shape())

  @test_util.run_in_graph_and_eager_modes
  def test_empty_rank1_returns_negative_infinity(self):
    with self.test_session(use_gpu=True):
      x = constant_op.constant([], shape=[0])
      lbeta_x = special_math_ops.lbeta(x)
      expected_result = constant_op.constant(-np.inf, shape=())

      self.assertAllEqual(self.evaluate(expected_result),
                          self.evaluate(lbeta_x))
      self.assertEqual(expected_result.get_shape(), lbeta_x.get_shape())

  @test_util.run_in_graph_and_eager_modes
  def test_empty_rank2_with_zero_last_dim_returns_negative_infinity(self):
    with self.test_session(use_gpu=True):
      event_size = 0
      for batch_size in [0, 1, 2]:
        x = constant_op.constant([], shape=[batch_size, event_size])
        lbeta_x = special_math_ops.lbeta(x)
        expected_result = constant_op.constant(-np.inf, shape=[batch_size])

        self.assertAllEqual(self.evaluate(expected_result),
                            self.evaluate(lbeta_x))
        self.assertEqual(expected_result.get_shape(), lbeta_x.get_shape())

  @test_util.run_in_graph_and_eager_modes
  def test_empty_rank2_with_zero_batch_dim_returns_empty(self):
    with self.test_session(use_gpu=True):
      batch_size = 0
      for event_size in [0, 1, 2]:
        x = constant_op.constant([], shape=[batch_size, event_size])
        lbeta_x = special_math_ops.lbeta(x)

        expected_result = constant_op.constant([], shape=[batch_size])

        self.assertAllEqual(self.evaluate(expected_result),
                            self.evaluate(lbeta_x))
        self.assertEqual(expected_result.get_shape(), lbeta_x.get_shape())


class BesselTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_bessel_i0(self):
    x_single = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    x_double = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      self.assertAllClose(special.i0(x_single),
                          self.evaluate(special_math_ops.bessel_i0(x_single)))
      self.assertAllClose(special.i0(x_double),
                          self.evaluate(special_math_ops.bessel_i0(x_double)))
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @test_util.run_in_graph_and_eager_modes
  def test_bessel_i1(self):
    x_single = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    x_double = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      self.assertAllClose(special.i1(x_single),
                          self.evaluate(special_math_ops.bessel_i1(x_single)))
      self.assertAllClose(special.i1(x_double),
                          self.evaluate(special_math_ops.bessel_i1(x_double)))
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))


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
      'iJ,Jk->ik',
      'iJ,Ki->JK',
      'iJk,Jklm->Jk'
      'ij, jk, kl -> il',
      'a, ab, abc -> abc',
      'ab, ab, cd, cd, ef, ef -> ',
      'abc, bac',
      'iJ, Ki -> JK',
      'iJk, Jklm -> Jk'
  ]

  long_cases = [
      'bca,cdb,dbf,afc->',
      'efc,dbc,acf,fd->abe',
      'ea,fb,gc,hd,abcd->efgh',
      'ea,fb,abcd,gc,hd->efgh',
      'abhe,hidj,jgba,hiab,gab',
      'efc, dbc, acf, fd -> abe',
      'abhe, hidj, jgba, hiab, gab',
  ]

  invalid_cases = [
      # bad formats
      '',
      'ijk ijk',
      'ij.jk->ik',
      'ij...,jk...->ik...',
      'ij,k ->kji',
      'ij,k-> kji',

      # axis in output that does not exist
      'ij,jk->im',

      # incorrect number of dimensions
      'ij,jkl->kl',

      # this is allowed in numpy but not implemented here yet
      'iij,jk'
  ]

  dim_mismatch_cases = [('ijk,jkl->il', [(2, 3, 4), (3, 5, 6)])]

  def disabled_test_simple(self):
    for case in self.simple_cases:
      self.run_test(case)

  def test_long(self):
    for case in self.long_cases:
      self.run_test(case)

  def test_invalid(self):
    for axes in self.invalid_cases:
      inputs = [
          array_ops.placeholder(dtypes.float32, shape=(3, 4)),
          array_ops.placeholder(dtypes.float32, shape=(3, 4)),
      ]
      with self.assertRaises(ValueError):
        _ = special_math_ops.einsum(axes, *inputs)

  def test_invalid_keyword_arguments(self):
    m0 = array_ops.placeholder(dtypes.int32, shape=(1, None))
    m1 = array_ops.placeholder(dtypes.int32, shape=(None, 1))
    with self.assertRaisesRegexp(
        TypeError,
        'invalid keyword arguments for this function: invalid1, invalid2'):
      _ = special_math_ops.einsum(
          'ij,jk->ik',
          m0,
          m1,
          name='name',
          invalid1='value1',
          invalid2='value2')

  def test_dim_mismatch(self):
    for axes, input_shapes in self.dim_mismatch_cases:
      inputs = [
          array_ops.placeholder(dtypes.float32, shape=shape)
          for shape in input_shapes
      ]
      with self.assertRaises(ValueError):
        _ = special_math_ops.einsum(axes, *inputs)

  def run_test(self, axes):
    all_axes = {ax: np.random.randint(4, 12) for ax in axes if ax.isalpha()}

    input_vals = []
    input_axes, _, _ = axes.partition('->')

    for idx in input_axes.split(','):
      shape = [all_axes[ax] for ax in idx if ax.isalpha()]
      input_vals.append(np.random.random(shape))

    input_tensors = [constant_op.constant(val) for val in input_vals]
    output_tensor = special_math_ops.einsum(axes, *input_tensors)

    with self.test_session(use_gpu=True):
      output_value = self.evaluate(output_tensor)

    correct_value = np.einsum(axes, *input_vals)

    err = np.abs(correct_value - output_value).max()
    # print(axes, err)
    self.assertLess(err, 1e-8)

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
        self.assertAllClose([[7]], sess.run(out, feed_dict=feed_dict))

    with ops.Graph().as_default():
      m0 = array_ops.placeholder(dtypes.int32, shape=(None, 3))
      m1 = array_ops.placeholder(dtypes.int32, shape=(3,))
      out = special_math_ops.einsum('ij,j->i', m0, m1)
      with session.Session() as sess:
        feed_dict = {
            m0: [[1, 2, 3]],
            m1: [2, 1, 1],
        }
        self.assertAllClose([7], sess.run(out, feed_dict=feed_dict))

    # Tests for placeholders which have two or more None values
    with ops.Graph().as_default():
      m0 = array_ops.placeholder(dtypes.int32, shape=(None, None, 2))
      m1 = array_ops.placeholder(dtypes.int32, shape=(2, 1))
      out = special_math_ops.einsum('ijk,kl->ijl', m0, m1)
      with session.Session() as sess:
        feed_dict = {
            m0: [[[1, 2]]],
            m1: [[3], [2]],
        }
        self.assertAllClose([[[7]]], sess.run(out, feed_dict=feed_dict))

    with ops.Graph().as_default():
      m0 = array_ops.placeholder(dtypes.int32, shape=(2, 1))
      m1 = array_ops.placeholder(dtypes.int32, shape=(None, None, 2))
      out = special_math_ops.einsum('kl,ijk->ijl', m0, m1)
      with session.Session() as sess:
        feed_dict = {
            m0: [[3], [2]],
            m1: [[[1, 2]]],
        }
        self.assertAllClose([[[7]]], sess.run(out, feed_dict=feed_dict))

    with ops.Graph().as_default():
      m0 = array_ops.placeholder(dtypes.int32, shape=(None, None, 2))
      m1 = array_ops.placeholder(dtypes.int32, shape=(2,))
      out = special_math_ops.einsum('ijk,k->ij', m0, m1)
      with session.Session() as sess:
        feed_dict = {
            m0: [[[1, 2]]],
            m1: [3, 2],
        }
        self.assertAllClose([[7]], sess.run(out, feed_dict=feed_dict))

    with ops.Graph().as_default():
      m0 = array_ops.placeholder(dtypes.int32, shape=(None, 2, None, 2))
      m1 = array_ops.placeholder(dtypes.int32, shape=(None, 2))
      out = special_math_ops.einsum('ijkl,ij->ikl', m0, m1)
      with session.Session() as sess:
        feed_dict = {
            m0: [[[[1, 2]], [[2, 1]]]],
            m1: [[3, 2]],
        }
        self.assertAllClose([[[7, 8]]], sess.run(out, feed_dict=feed_dict))


if __name__ == '__main__':
  test.main()
