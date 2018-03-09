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
"""Tests for tf.contrib.kfac.fisher_factors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr

from tensorflow.contrib.kfac.python.ops import fisher_blocks as fb
from tensorflow.contrib.kfac.python.ops import fisher_factors as ff
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import test


def make_damping_func(damping):
  return fb._package_func(lambda: damping, damping)


class FisherFactorTestingDummy(ff.FisherFactor):
  """Dummy class to test the non-abstract methods on ff.FisherFactor."""

  @property
  def _var_scope(self):
    return 'dummy/a_b_c'

  @property
  def _cov_shape(self):
    raise NotImplementedError

  @property
  def _num_sources(self):
    return 1

  @property
  def _dtype(self):
    return dtypes.float32

  def _compute_new_cov(self):
    raise NotImplementedError

  def instantiate_covariance(self):
    pass

  def make_inverse_update_ops(self):
    return []

  def get_cov(self):
    return NotImplementedError

  def left_multiply(self, x, damping):
    return NotImplementedError

  def right_multiply(self, x, damping):
    return NotImplementedError

  def left_multiply_matpower(self, x, exp, damping):
    return NotImplementedError

  def right_multiply_matpower(self, x, exp, damping):
    return NotImplementedError

  def instantiate_inv_variables(self):
    return NotImplementedError


class InverseProvidingFactorTestingDummy(ff.InverseProvidingFactor):
  """Dummy class to test the non-abstract methods on ff.InverseProvidingFactor.
  """

  def __init__(self, shape):
    self._shape = shape
    super(InverseProvidingFactorTestingDummy, self).__init__()

  @property
  def _var_scope(self):
    return 'dummy/a_b_c'

  @property
  def _cov_shape(self):
    return self._shape

  @property
  def _num_sources(self):
    return 1

  @property
  def _dtype(self):
    return dtypes.float32

  def _compute_new_cov(self):
    raise NotImplementedError

  def instantiate_covariance(self):
    pass


class NumericalUtilsTest(test.TestCase):

  def testComputeCovAgainstNumpy(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      npr.seed(0)
      random_seed.set_random_seed(200)

      x = npr.randn(100, 3)
      cov = ff.compute_cov(array_ops.constant(x))
      np_cov = np.dot(x.T, x) / x.shape[0]

      self.assertAllClose(sess.run(cov), np_cov)

  def testComputeCovAgainstNumpyWithAlternativeNormalizer(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      npr.seed(0)
      random_seed.set_random_seed(200)

      normalizer = 10.
      x = npr.randn(100, 3)
      cov = ff.compute_cov(array_ops.constant(x), normalizer=normalizer)
      np_cov = np.dot(x.T, x) / normalizer

      self.assertAllClose(sess.run(cov), np_cov)

  def testAppendHomog(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      npr.seed(0)

      m, n = 3, 4
      a = npr.randn(m, n)
      a_homog = ff.append_homog(array_ops.constant(a))
      np_result = np.hstack([a, np.ones((m, 1))])

      self.assertAllClose(sess.run(a_homog), np_result)


class NameStringUtilFunctionTest(test.TestCase):

  def _make_tensor(self):
    x = array_ops.placeholder(dtypes.float64, (3, 1))
    w = array_ops.constant(npr.RandomState(0).randn(3, 3))
    y = math_ops.matmul(w, x)
    g = gradients_impl.gradients(y, x)[0]
    return g

  def testScopeStringFromParamsSingleTensor(self):
    with tf_ops.Graph().as_default():
      g = self._make_tensor()
      scope_string = ff.scope_string_from_params(g)
      self.assertEqual('gradients_MatMul_grad_MatMul_1', scope_string)

  def testScopeStringFromParamsMultipleTensors(self):
    with tf_ops.Graph().as_default():
      x = array_ops.constant(1,)
      y = array_ops.constant(2,)
      scope_string = ff.scope_string_from_params((x, y))
      self.assertEqual('Const_Const_1', scope_string)

  def testScopeStringFromParamsMultipleTypes(self):
    with tf_ops.Graph().as_default():
      x = array_ops.constant(1,)
      y = array_ops.constant(2,)
      scope_string = ff.scope_string_from_params([[1, 2, 3], 'foo', True, 4,
                                                  (x, y)])
      self.assertEqual('1-2-3_foo_True_4_Const__Const_1', scope_string)

  def testScopeStringFromParamsUnsupportedType(self):
    with tf_ops.Graph().as_default():
      x = array_ops.constant(1,)
      y = array_ops.constant(2,)
      unsupported = 1.2  # Floats are not supported.
      with self.assertRaises(ValueError):
        ff.scope_string_from_params([[1, 2, 3], 'foo', True, 4, (x, y),
                                     unsupported])

  def testScopeStringFromName(self):
    with tf_ops.Graph().as_default():
      g = self._make_tensor()
      scope_string = ff.scope_string_from_name(g)
      self.assertEqual('gradients_MatMul_grad_MatMul_1', scope_string)

  def testScalarOrTensorToString(self):
    with tf_ops.Graph().as_default():
      self.assertEqual(ff.scalar_or_tensor_to_string(5.), repr(5.))

      g = self._make_tensor()
      scope_string = ff.scope_string_from_name(g)
      self.assertEqual(ff.scalar_or_tensor_to_string(g), scope_string)


class FisherFactorTest(test.TestCase):

  def testMakeInverseUpdateOps(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      factor = FisherFactorTestingDummy()

      self.assertEqual(0, len(factor.make_inverse_update_ops()))


class InverseProvidingFactorTest(test.TestCase):

  def testRegisterDampedInverse(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      shape = [2, 2]
      factor = InverseProvidingFactorTestingDummy(shape)
      factor_var_scope = 'dummy/a_b_c'

      damping_funcs = [make_damping_func(0.1),
                       make_damping_func(0.1),
                       make_damping_func(1e-5),
                       make_damping_func(1e-5)]
      for damping_func in damping_funcs:
        factor.register_inverse(damping_func)

      factor.instantiate_inv_variables()

      inv = factor.get_inverse(damping_funcs[0])
      self.assertEqual(inv, factor.get_inverse(damping_funcs[1]))
      self.assertNotEqual(inv, factor.get_inverse(damping_funcs[2]))
      self.assertEqual(factor.get_inverse(damping_funcs[2]),
                       factor.get_inverse(damping_funcs[3]))
      factor_vars = tf_ops.get_collection(tf_ops.GraphKeys.GLOBAL_VARIABLES,
                                          factor_var_scope)
      self.assertEqual(set([inv, factor.get_inverse(damping_funcs[2])]),
                       set(factor_vars))
      self.assertEqual(shape, inv.get_shape())

  def testRegisterMatpower(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      shape = [3, 3]
      factor = InverseProvidingFactorTestingDummy(shape)
      factor_var_scope = 'dummy/a_b_c'

      # TODO(b/74201126): Change to using the same func for both once
      # Topohash is in place.
      damping_func_1 = make_damping_func(0.5)
      damping_func_2 = make_damping_func(0.5)

      factor.register_matpower(-0.5, damping_func_1)
      factor.register_matpower(2, damping_func_2)

      factor.instantiate_inv_variables()

      factor_vars = tf_ops.get_collection(tf_ops.GraphKeys.GLOBAL_VARIABLES,
                                          factor_var_scope)
      matpower1 = factor.get_matpower(-0.5, damping_func_1)
      matpower2 = factor.get_matpower(2, damping_func_2)

      self.assertEqual(set([matpower1, matpower2]), set(factor_vars))

      self.assertEqual(shape, matpower1.get_shape())
      self.assertEqual(shape, matpower2.get_shape())

  def testMakeInverseUpdateOps(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      factor = FisherFactorTestingDummy()

      self.assertEqual(0, len(factor.make_inverse_update_ops()))

  def testMakeInverseUpdateOpsManyInversesEigenDecomp(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      cov = np.array([[1., 2.], [3., 4.]])
      factor = InverseProvidingFactorTestingDummy(cov.shape)
      factor._cov = array_ops.constant(cov, dtype=dtypes.float32)

      damping_funcs = []
      for i in range(1, ff.EIGENVALUE_DECOMPOSITION_THRESHOLD + 1):
        damping_funcs.append(make_damping_func(1./i))

      for i in range(ff.EIGENVALUE_DECOMPOSITION_THRESHOLD):
        factor.register_inverse(damping_funcs[i])

      factor.instantiate_inv_variables()
      ops = factor.make_inverse_update_ops()
      self.assertEqual(1, len(ops))

      sess.run(tf_variables.global_variables_initializer())
      new_invs = []
      sess.run(ops)
      for i in range(ff.EIGENVALUE_DECOMPOSITION_THRESHOLD):
        # The inverse op will assign the damped inverse of cov to the inv var.
        new_invs.append(sess.run(factor.get_inverse(damping_funcs[i])))

      # We want to see that the new invs are all different from each other.
      for i in range(len(new_invs)):
        for j in range(i + 1, len(new_invs)):
          # Just check the first element.
          self.assertNotEqual(new_invs[i][0][0], new_invs[j][0][0])

  def testMakeInverseUpdateOpsMatPowerEigenDecomp(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      cov = np.array([[6., 2.], [2., 4.]])
      factor = InverseProvidingFactorTestingDummy(cov.shape)
      factor._cov = array_ops.constant(cov, dtype=dtypes.float32)
      exp = 2  # NOTE(mattjj): must be int to test with np.linalg.matrix_power
      damping = 0.5
      damping_func = make_damping_func(damping)

      factor.register_matpower(exp, damping_func)
      factor.instantiate_inv_variables()
      ops = factor.make_inverse_update_ops()
      self.assertEqual(1, len(ops))

      sess.run(tf_variables.global_variables_initializer())
      sess.run(ops[0])
      matpower = sess.run(factor.get_matpower(exp, damping_func))
      matpower_np = np.linalg.matrix_power(cov + np.eye(2) * damping, exp)
      self.assertAllClose(matpower, matpower_np)

  def testMakeInverseUpdateOpsNoEigenDecomp(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      cov = np.array([[5., 2.], [2., 4.]])  # NOTE(mattjj): must be symmetric
      factor = InverseProvidingFactorTestingDummy(cov.shape)
      factor._cov = array_ops.constant(cov, dtype=dtypes.float32)

      damping_func = make_damping_func(0)

      factor.register_inverse(damping_func)
      factor.instantiate_inv_variables()
      ops = factor.make_inverse_update_ops()
      self.assertEqual(1, len(ops))

      sess.run(tf_variables.global_variables_initializer())
      # The inverse op will assign the damped inverse of cov to the inv var.
      old_inv = sess.run(factor.get_inverse(damping_func))
      self.assertAllClose(
          sess.run(ff.inverse_initializer(cov.shape, dtypes.float32)), old_inv)

      sess.run(ops)
      new_inv = sess.run(factor.get_inverse(damping_func))
      self.assertAllClose(new_inv, np.linalg.inv(cov))


class FullFactorTest(test.TestCase):

  def testFullFactorInit(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), name='a/b/c')
      factor = ff.FullFactor((tensor,), 32)
      factor.instantiate_cov_variables()
      self.assertEqual([6, 6], factor.get_cov().get_shape().as_list())

  def testFullFactorInitFloat64(self):
    with tf_ops.Graph().as_default():
      dtype = dtypes.float64_ref
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), dtype=dtype, name='a/b/c')
      factor = ff.FullFactor((tensor,), 32)
      factor.instantiate_cov_variables()
      cov = factor.get_cov()
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([6, 6], cov.get_shape().as_list())

  def testMakeCovarianceUpdateOp(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = array_ops.constant([1., 2.], name='a/b/c')
      factor = ff.FullFactor((tensor,), 2)
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[0.75, 0.5], [0.5, 1.5]], new_cov)


class NaiveDiagonalFactorTest(test.TestCase):

  def testNaiveDiagonalFactorInit(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), name='a/b/c')
      factor = ff.NaiveDiagonalFactor((tensor,), 32)
      factor.instantiate_cov_variables()
      self.assertEqual([6, 1], factor.get_cov_var().get_shape().as_list())

  def testNaiveDiagonalFactorInitFloat64(self):
    with tf_ops.Graph().as_default():
      dtype = dtypes.float64_ref
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), dtype=dtype, name='a/b/c')
      factor = ff.NaiveDiagonalFactor((tensor,), 32)
      factor.instantiate_cov_variables()
      cov = factor.get_cov_var()
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([6, 1], cov.get_shape().as_list())

  def testMakeCovarianceUpdateOp(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = array_ops.constant([1., 2.], name='a/b/c')
      factor = ff.NaiveDiagonalFactor((tensor,), 2)
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[0.75], [1.5]], new_cov)


class EmbeddingInputKroneckerFactorTest(test.TestCase):

  def testInitialization(self):
    with tf_ops.Graph().as_default():
      input_ids = array_ops.constant([[0], [1], [4]])
      vocab_size = 5
      factor = ff.EmbeddingInputKroneckerFactor(input_ids, vocab_size)
      factor.instantiate_cov_variables()
      cov = factor.get_cov_var()
      self.assertEqual(cov.shape.as_list(), [vocab_size])

  def testCovarianceUpdateOp(self):
    with tf_ops.Graph().as_default():
      input_ids = array_ops.constant([[0], [1], [4]])
      vocab_size = 5
      factor = ff.EmbeddingInputKroneckerFactor(input_ids, vocab_size)
      factor.instantiate_cov_variables()
      cov_update_op = factor.make_covariance_update_op(0.0)

      with self.test_session() as sess:
        sess.run(tf_variables.global_variables_initializer())
        new_cov = sess.run(cov_update_op)
        self.assertAllClose(np.array([1., 1., 0., 0., 1.]) / 3., new_cov)


class FullyConnectedKroneckerFactorTest(test.TestCase):

  def _testFullyConnectedKroneckerFactorInit(self,
                                             has_bias,
                                             final_shape,
                                             dtype=dtypes.float32_ref):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), dtype=dtype, name='a/b/c')
      factor = ff.FullyConnectedKroneckerFactor((tensor,), has_bias=has_bias)
      factor.instantiate_cov_variables()
      cov = factor.get_cov()
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual(final_shape, cov.get_shape().as_list())

  def testFullyConnectedKroneckerFactorInitNoBias(self):
    for dtype in (dtypes.float32_ref, dtypes.float64_ref):
      self._testFullyConnectedKroneckerFactorInit(False, [3, 3], dtype=dtype)

  def testFullyConnectedKroneckerFactorInitWithBias(self):
    for dtype in (dtypes.float32_ref, dtypes.float64_ref):
      self._testFullyConnectedKroneckerFactorInit(True, [4, 4], dtype=dtype)

  def testMakeCovarianceUpdateOpWithBias(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = array_ops.constant([[1., 2.], [3., 4.]], name='a/b/c')
      factor = ff.FullyConnectedKroneckerFactor((tensor,), has_bias=True)
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[3, 3.5, 1], [3.5, 5.5, 1.5], [1, 1.5, 1]], new_cov)

  def testMakeCovarianceUpdateOpNoBias(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = array_ops.constant([[1., 2.], [3., 4.]], name='a/b/c')
      factor = ff.FullyConnectedKroneckerFactor((tensor,))
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[3, 3.5], [3.5, 5.5]], new_cov)


class ConvInputKroneckerFactorTest(test.TestCase):

  def testConvInputKroneckerFactorInitNoBias(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), name='a/b/c')
      factor = ff.ConvInputKroneckerFactor(
          tensor, (1, 2, 3, 4), 3, 2, has_bias=False)
      factor.instantiate_cov_variables()
      self.assertEqual([1 * 2 * 3, 1 * 2 * 3],
                       factor.get_cov().get_shape().as_list())

  def testConvInputKroneckerFactorInit(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), name='a/b/c')
      factor = ff.ConvInputKroneckerFactor(
          tensor, (1, 2, 3, 4), 3, 2, has_bias=True)
      factor.instantiate_cov_variables()
      self.assertEqual([1 * 2 * 3 + 1, 1 * 2 * 3 + 1],
                       factor.get_cov().get_shape().as_list())

  def testConvInputKroneckerFactorInitFloat64(self):
    with tf_ops.Graph().as_default():
      dtype = dtypes.float64_ref
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), dtype=dtype, name='a/b/c')
      factor = ff.ConvInputKroneckerFactor(
          tensor, (1, 2, 3, 4), 3, 2, has_bias=True)
      factor.instantiate_cov_variables()
      cov = factor.get_cov()
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([1 * 2 * 3 + 1, 1 * 2 * 3 + 1],
                       cov.get_shape().as_list())

  def testMakeCovarianceUpdateOpWithBias(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = array_ops.constant(
          np.arange(1., 17.).reshape(2, 2, 2, 2), dtype=dtypes.float32)
      factor = ff.ConvInputKroneckerFactor(
          tensor, (1, 2, 1, 1), [1, 1, 1, 1], 'SAME', has_bias=True)
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[34.375, 37, 3.125], [37, 41, 3.5], [3.125, 3.5, 1]],
                          new_cov)

  def testMakeCovarianceUpdateOpNoBias(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = array_ops.constant(
          np.arange(1., 17.).reshape(2, 2, 2, 2), dtype=dtypes.float32)
      factor = ff.ConvInputKroneckerFactor(tensor, (1, 2, 1, 1),
                                           [1, 1, 1, 1], 'SAME')
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[34.375, 37], [37, 41]], new_cov)


class ConvOutputKroneckerFactorTest(test.TestCase):

  def testConvOutputKroneckerFactorInit(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3, 4, 5), name='a/b/c')
      factor = ff.ConvOutputKroneckerFactor((tensor,))
      factor.instantiate_cov_variables()
      self.assertEqual([5, 5], factor.get_cov().get_shape().as_list())

  def testConvOutputKroneckerFactorInitFloat64(self):
    with tf_ops.Graph().as_default():
      dtype = dtypes.float64_ref
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3, 4, 5), dtype=dtype, name='a/b/c')
      factor = ff.ConvOutputKroneckerFactor((tensor,))
      factor.instantiate_cov_variables()
      cov = factor.get_cov()
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([5, 5], cov.get_shape().as_list())

  def testConvOutputKroneckerFactorInitNotEnoughDims(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), name='a/b/c')
      with self.assertRaises(IndexError):
        ff.ConvOutputKroneckerFactor((tensor,))

  def testMakeCovarianceUpdateOp(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = np.arange(1, 17).reshape(2, 2, 2, 2).astype(np.float32)
      factor = ff.ConvOutputKroneckerFactor((array_ops.constant(tensor),))
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[43, 46.5], [46.5, 51.5]], new_cov)


class FullyConnectedMultiKFTest(test.TestCase):

  def testFullyConnectedMultiKFInit(self):
    with tf_ops.Graph().as_default():
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), name='a/b/c')
      tensor_list = [tensor]
      factor = ff.FullyConnectedMultiKF((tensor_list,), has_bias=False)
      factor.instantiate_cov_variables()
      self.assertEqual([3, 3], factor.get_cov().get_shape().as_list())

  def testFullyConnectedMultiKFInitFloat64(self):
    with tf_ops.Graph().as_default():
      dtype = dtypes.float64_ref
      random_seed.set_random_seed(200)
      tensor = array_ops.ones((2, 3), dtype=dtype, name='a/b/c')
      tensor_list = [tensor]
      factor = ff.FullyConnectedMultiKF((tensor_list,), has_bias=False)
      factor.instantiate_cov_variables()
      cov = factor.get_cov()
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([3, 3], cov.get_shape().as_list())

  def testMakeCovarianceUpdateOpWithBias(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = array_ops.constant([[1., 2.], [3., 4.]], name='a/b/c')
      tensor_list = [tensor]
      factor = ff.FullyConnectedMultiKF((tensor_list,), has_bias=True)
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[3, 3.5, 1], [3.5, 5.5, 1.5], [1, 1.5, 1]], new_cov)

  def testMakeCovarianceUpdateOpNoBias(self):
    with tf_ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      tensor = array_ops.constant([[1., 2.], [3., 4.]], name='a/b/c')
      tensor_list = [tensor]
      factor = ff.FullyConnectedMultiKF((tensor_list,))
      factor.instantiate_cov_variables()

      sess.run(tf_variables.global_variables_initializer())
      new_cov = sess.run(factor.make_covariance_update_op(.5))
      self.assertAllClose([[3, 3.5], [3.5, 5.5]], new_cov)


if __name__ == '__main__':
  test.main()
