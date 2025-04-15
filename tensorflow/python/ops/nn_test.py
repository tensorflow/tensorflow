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
"""Tests for miscellaneous functionality in tensorflow.ops.nn."""

import functools
import math
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test as test_lib


class ZeroFractionTest(test_lib.TestCase):

  def _ZeroFraction(self, x):
    assert x.shape
    total_elements = np.prod(x.shape)
    nonzeros = np.count_nonzero(x.flatten())
    return 1.0 - nonzeros / total_elements

  @test_util.run_deprecated_v1
  def testZeroFraction(self):
    x_shape = [5, 17]
    x_np = np.random.randint(0, 2, size=x_shape).astype(np.float32)
    y_np = self._ZeroFraction(x_np)

    x_tf = constant_op.constant(x_np)
    x_tf.set_shape(x_shape)
    y_tf = nn_impl.zero_fraction(x_tf)
    y_tf_np = self.evaluate(y_tf)

    eps = 1e-8
    self.assertAllClose(y_tf_np, y_np, eps)

  @test_util.run_deprecated_v1
  def testZeroFractionEmpty(self):
    x = np.zeros(0)
    y = self.evaluate(nn_impl.zero_fraction(x))
    self.assertTrue(np.isnan(y))

  @test_util.run_deprecated_v1
  def testZeroFraction2_27Zeros(self):
    sparsity = nn_impl.zero_fraction(
        array_ops.zeros([int(2**27 * 1.01)], dtype=dtypes.int8))
    self.assertAllClose(1.0, self.evaluate(sparsity))

  @test_util.run_deprecated_v1
  def testZeroFraction2_27Ones(self):
    sparsity = nn_impl.zero_fraction(
        array_ops.ones([int(2**27 * 1.01)], dtype=dtypes.int8))
    self.assertAllClose(0.0, self.evaluate(sparsity))

  @test_util.run_deprecated_v1
  def testUnknownSize(self):
    value = array_ops.placeholder(dtype=dtypes.float32)
    sparsity = nn_impl.zero_fraction(value)
    with self.cached_session() as sess:
      self.assertAllClose(0.25,
                          sess.run(sparsity, {value: [[0., 1.], [0.3, 2.]]}))


@test_util.run_all_in_graph_and_eager_modes
class SoftmaxTest(test_lib.TestCase, parameterized.TestCase):

  def _softmax(self, x):
    assert len(x.shape) == 2
    if x.shape[1] == 0:
      return x
    m = x.max(1)[:, np.newaxis]
    u = np.exp(x - m)
    z = u.sum(1)[:, np.newaxis]
    return u / z

  def testSoftmax(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._softmax(x_np)
    x_tf = constant_op.constant(x_np)
    y_tf = nn_ops.softmax_v2(x_tf)
    y_tf_last_dim = nn_ops.softmax_v2(x_tf, 1)
    y_tf_np = self.evaluate(y_tf)
    y_tf_last_dim_np = self.evaluate(y_tf_last_dim)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)
    self.assertAllClose(y_tf_last_dim_np, y_np, eps)

  def testSoftmaxAxes(self):
    arr = np.linspace(0., 1, 12).reshape(3, 4)
    x_neg_axis = nn_ops.softmax_v2(arr, axis=-2)
    y_pos_axis = nn_ops.softmax_v2(arr, axis=0)
    z_gt_axis = nn_ops.softmax_v2(arr, axis=0)
    x_neg_axis_tf = self.evaluate(x_neg_axis)
    y_pos_axis_tf = self.evaluate(y_pos_axis)
    z_gt_axis_tf = self.evaluate(z_gt_axis)
    eps = 1e-3
    self.assertAllClose(x_neg_axis_tf, y_pos_axis_tf, eps)
    self.assertAllClose(y_pos_axis_tf, z_gt_axis_tf, eps)

  def testSoftmaxExtendType(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)

    x_f32_tf = constant_op.constant(x_np)
    x_bf16_tf = math_ops.cast(x_f32_tf, dtypes.bfloat16)
    y_f32_tf = self.evaluate(nn_ops.softmax(x_f32_tf))
    y_bf16_tf = self.evaluate(nn_ops.softmax(x_bf16_tf))
    expected = math_ops.cast(y_f32_tf, dtypes.bfloat16)
    tol = x_shape[1] * 1e-3
    self.assertAllClose(y_bf16_tf, expected, rtol=tol, atol=tol)

  @parameterized.parameters(((5, 10),), ((2, 3, 4),))
  def testGradient(self, x_shape):
    x_np = np.random.randn(*x_shape).astype(np.float64)
    x_tf = constant_op.constant(x_np)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        nn_ops.softmax_v2, [x_tf])
    self.assertAllClose(theoretical, numerical)


@test_util.run_all_in_graph_and_eager_modes
class LogPoissonLossTest(test_lib.TestCase):

  def _log_poisson_loss(self, x, z, compute_full_loss=False):
    lpl = np.exp(x) - z * x
    if compute_full_loss:
      stirling_approx = z * np.log(z) - z + 0.5 * np.log(2. * np.pi * z)
      lpl += np.ma.masked_array(stirling_approx, mask=(z <= 1)).filled(0.)
    return lpl

  def testLogPoissonLoss(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    z_np = np.random.randint(0, 5, size=x_shape).astype(np.float32)
    y_np = self._log_poisson_loss(x_np, z_np, compute_full_loss=False)
    y_np_stirling = self._log_poisson_loss(x_np, z_np, compute_full_loss=True)
    y_tf = nn_impl.log_poisson_loss(z_np, x_np, compute_full_loss=False)
    y_tf_stirling = nn_impl.log_poisson_loss(z_np, x_np, compute_full_loss=True)
    y_tf_np = self.evaluate(y_tf)
    y_tf_np_stirling = self.evaluate(y_tf_stirling)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)
    self.assertAllClose(y_tf_np_stirling, y_np_stirling, eps)

  def testGradient(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float64)
    z_np = np.random.randint(0, 5, size=x_shape).astype(np.float64)
    with self.cached_session():
      x_tf = constant_op.constant(x_np)
      # TODO(b/241834841): Test with `compute_full_loss` set as True
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_impl.log_poisson_loss, [z_np, x_tf])
      self.assertAllClose(theoretical, numerical)


@test_util.run_all_in_graph_and_eager_modes
class LogSoftmaxTest(test_lib.TestCase, parameterized.TestCase):

  def _log_softmax(self, x):
    assert len(x.shape) == 2
    m = x.max(1)[:, np.newaxis]
    u = x - m
    return u - np.log(np.sum(np.exp(u), 1, keepdims=True))

  def testLogSoftmax(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._log_softmax(x_np)
    x_tf = constant_op.constant(x_np)
    y_tf = nn_ops.log_softmax_v2(x_tf)
    y_tf_np = self.evaluate(y_tf)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)

  def testLogSoftmaxAxes(self):
    arr = np.linspace(0., 1, 12).reshape(3, 4)
    x_neg_axis = nn_ops.log_softmax_v2(arr, axis=-2)
    y_pos_axis = nn_ops.log_softmax_v2(arr, axis=0)
    z_gt_axis = nn_ops.log_softmax_v2(arr, axis=0)
    x_neg_axis_tf = self.evaluate(x_neg_axis)
    y_pos_axis_tf = self.evaluate(y_pos_axis)
    z_gt_axis_tf = self.evaluate(z_gt_axis)
    eps = 1e-3
    self.assertAllClose(x_neg_axis_tf, y_pos_axis_tf, eps)
    self.assertAllClose(y_pos_axis_tf, z_gt_axis_tf, eps)

  @parameterized.parameters(((5, 10),), ((2, 3, 4),))
  def testGradient(self, x_shape):
    x_np = np.random.randn(*x_shape).astype(np.float64)
    with self.cached_session():
      x_tf = constant_op.constant(x_np)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_ops.log_softmax_v2, [x_tf])
      self.assertAllClose(theoretical, numerical)


@test_util.run_all_in_graph_and_eager_modes
class L2LossTest(test_lib.TestCase):

  def testL2Loss(self):
    for dtype in [dtypes.float32, dtypes.float64] + \
                 [dtypes.bfloat16] if test_util.is_gpu_available(
                                          cuda_only=True) else []:
      x = constant_op.constant([1.0, 0.0, 3.0, 2.0],
                               shape=[2, 2],
                               name="x",
                               dtype=dtype)
      l2loss = nn_ops.l2_loss(x)
      value = self.evaluate(l2loss)
      self.assertAllClose(7.0, value)

  def testGradient(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    with self.cached_session():
      x = constant_op.constant(x_val, name="x")
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_ops.l2_loss, [x])
      self.assertAllClose(theoretical, numerical)


@test_util.run_all_in_graph_and_eager_modes
class L2NormalizeTest(test_lib.TestCase):

  def _l2Normalize(self, x, dim):
    if isinstance(dim, list):
      norm = np.linalg.norm(x, axis=tuple(dim))
      for d in dim:
        norm = np.expand_dims(norm, d)
      return x / norm
    else:
      norm = np.apply_along_axis(np.linalg.norm, dim, x)
      return x / np.expand_dims(norm, dim)

  def testL2Normalize(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float32)
    for dim in range(len(x_shape)):
      y_np = self._l2Normalize(x_np, dim)
      x_tf = constant_op.constant(x_np, name="x")
      y_tf = nn_impl.l2_normalize(x_tf, dim)
      self.assertAllClose(y_np, self.evaluate(y_tf))

  def testL2NormalizeDimArray(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float32)
    dim = [1, 2]
    y_np = self._l2Normalize(x_np, dim)
    x_tf = constant_op.constant(x_np, name="x")
    y_tf = nn_impl.l2_normalize(x_tf, dim)
    self.assertAllClose(y_np, self.evaluate(y_tf))

  def testL2NormalizeGradient(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float64)
    with self.cached_session():
      x_tf = constant_op.constant(x_np, name="x")
      # TODO(b/241834841): Test l2_normalize with `axis` set to other dims
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_impl.l2_normalize, [x_tf])
      self.assertAllClose(theoretical, numerical)

  def testL2NormalizeComplex(self):
    x_shape = [20, 7, 3]
    for dtype in [np.complex64, np.complex128]:
      np.random.seed(1)
      x_np = (
          np.random.random_sample(x_shape).astype(dtype) +
          np.random.random_sample(x_shape).astype(dtype) * 1j)
      for dim in range(len(x_shape)):
        y_np = self._l2Normalize(x_np, dim)
        x_tf = constant_op.constant(x_np, name="x")
        y_tf = nn_impl.l2_normalize(x_tf, dim)
        self.assertAllClose(y_np, self.evaluate(y_tf))


DROPOUT_FNS = [
    ("stateful_v1", nn_ops.dropout),
    ("stateful_v2", nn_ops.dropout_v2),
    ("stateless", functools.partial(nn_ops.stateless_dropout, seed=(1, 2))),
    ("stateless_philox", functools.partial(
        nn_ops.stateless_dropout, seed=(1, 2), rng_alg="philox")),
    ("generator", functools.partial(  # pylint: disable=g-long-lambda
        nn_ops.general_dropout, uniform_sampler=lambda shape, dtype: (  # pylint: disable=g-long-lambda
            stateful_random_ops.Generator.from_seed(1).uniform(
                shape=shape, dtype=dtype)))),
    ]


class DropoutTest(test_lib.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("_%s_%s_%s" % (case_name, use_noise_shape, keep_prob), case_name,  # pylint: disable=g-complex-comprehension
       dropout_fn, use_noise_shape, keep_prob)
      for keep_prob in [0.1, 0.5, 0.8]
      for use_noise_shape in ["no", "concrete", "partial"]
      for case_name, dropout_fn in DROPOUT_FNS)
  def testDropout(self, case_name, dropout_fn, use_noise_shape, keep_prob):
    # Runs dropout with 0-1 tensor 10 times, sum the number of ones and validate
    # that it is producing approximately the right number of ones over a large
    # number of samples, based on the keep probability.
    if "generator" in case_name and not context.executing_eagerly():
      self.skipTest("tf.random.Generator can only be used in TF2.")
    if use_noise_shape == "no":
      x_dim = 70
      y_dim = 30
    else:
      x_dim = 70 * 30
      y_dim = 3
    num_iter = 10
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    if use_noise_shape == "no":
      noise_shape = None
    elif use_noise_shape == "concrete":
      noise_shape = [x_dim, 1]
    else:
      noise_shape = [None, 1]
    dropout = dropout_fn(t, rate=(1 - keep_prob), noise_shape=noise_shape)
    final_count = 0
    self.assertEqual([x_dim, y_dim], dropout.get_shape())
    for _ in range(0, num_iter):
      value = self.evaluate(dropout)
      final_count += np.count_nonzero(value)
      # Verifies that there are only two values: 0 and 1/keep_prob.
      sorted_value = np.unique(np.sort(value))
      self.assertEqual(0, sorted_value[0])
      self.assertAllClose(1 / keep_prob, sorted_value[1])

    # Check that we are in the 15% error range
    expected_count = x_dim * y_dim * keep_prob * num_iter
    rel_error = math.fabs(final_count - expected_count) / expected_count
    self.assertLess(rel_error, 0.15)

  @parameterized.named_parameters(
      ("_%s_%s" % (case_name, keep_prob), case_name, dropout_fn, keep_prob)  # pylint: disable=g-complex-comprehension
      for keep_prob in [0.1, 0.5, 0.8]
      for case_name, dropout_fn in DROPOUT_FNS)
  def testShapedDropoutCorrelation(self, case_name, dropout_fn, keep_prob):
    # Runs a shaped dropout and tests that the correlations are correct.
    if "generator" in case_name and not context.executing_eagerly():
      self.skipTest("tf.random.Generator can only be used in TF2.")
    x_dim = 40
    y_dim = 30
    num_iter = 10
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    dropout = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim, 1])
    self.assertEqual([x_dim, y_dim], dropout.get_shape())
    for _ in range(0, num_iter):
      value = self.evaluate(dropout)
      # Verifies that each row has only one type of activation.
      for i in range(x_dim):
        sorted_value = np.unique(np.sort(value[i, :]))
        self.assertEqual(sorted_value.size, 1)

  @parameterized.named_parameters(
      ("_%s_%s_%s" % (case_name, keep_prob, use_keep_prob), case_name,  # pylint: disable=g-complex-comprehension
       dropout_fn, keep_prob, use_keep_prob)
      for use_keep_prob in [False, True]
      for keep_prob in [0.1, 0.5, 0.8]
      for case_name, dropout_fn in DROPOUT_FNS)
  def testDropoutPlaceholderRateAndKeepProb(self, case_name, dropout_fn,
                                            keep_prob, use_keep_prob):
    # Runs dropout with 0-1 tensor 10 times, sum the number of ones and validate
    # that it is producing approximately the right number of ones over a large
    # number of samples, based on the keep probability.
    if use_keep_prob and case_name != "stateful_v1":
      self.skipTest("Only V1 `dropout` has the `keep_prob` argument.")
    if "generator" in case_name and not context.executing_eagerly():
      self.skipTest("tf.random.Generator can only be used in TF2.")
    x_dim = 70
    y_dim = 30
    num_iter = 10
    t = constant_op.constant(
        1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    final_count = 0
    for _ in range(0, num_iter):
      if use_keep_prob:
        dropout = dropout_fn(t, keep_prob=keep_prob)
      else:
        dropout = dropout_fn(t, rate=1 - keep_prob)
      self.assertEqual([x_dim, y_dim], dropout.get_shape())
      value = self.evaluate(dropout)
      final_count += np.count_nonzero(value)
      # Verifies that there are only two values: 0 and 1/keep_prob.
      sorted_value = np.unique(np.sort(value))
      self.assertEqual(0, sorted_value[0])
      self.assertAllClose(1 / keep_prob, sorted_value[1])
    # Check that we are in the 15% error range
    expected_count = x_dim * y_dim * keep_prob * num_iter
    rel_error = math.fabs(final_count - expected_count) / expected_count
    self.assertLess(rel_error, 0.15)

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  @test_util.run_deprecated_v1
  def testShapedDropoutUnknownShape(self, dropout_fn):
    x_dim = 40
    y_dim = 30
    keep_prob = 0.5
    x = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    dropout_x = dropout_fn(
        x,
        rate=(1 - keep_prob),
        noise_shape=array_ops.placeholder(dtypes.int32))
    self.assertEqual(x.get_shape(), dropout_x.get_shape())

  @parameterized.named_parameters(
      ("_%s_%s" % (case_name, use_keep_prob), case_name, dropout_fn,  # pylint: disable=g-complex-comprehension
       use_keep_prob)
      for use_keep_prob in [False, True]
      for case_name, dropout_fn in DROPOUT_FNS)
  @test_util.run_deprecated_v1
  def testInvalidRateAndKeepProb(self, case_name, dropout_fn, use_keep_prob):
    if use_keep_prob and case_name != "stateful_v1":
      self.skipTest("Only V1 `dropout` has the `keep_prob` argument.")
    if use_keep_prob:
      fn = lambda x, y: dropout_fn(x, keep_prob=y)
    else:
      fn = lambda x, y: dropout_fn(x, rate=y)
    x_dim = 40
    y_dim = 30
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      fn(t, -1.0)
    with self.assertRaises(ValueError):
      fn(t, 1.1)
    with self.assertRaises(ValueError):
      fn(t, [0.0, 1.0])
    with self.assertRaises(ValueError):
      fn(t, array_ops.placeholder(dtypes.float64))
    with self.assertRaises(ValueError):
      fn(t, array_ops.placeholder(dtypes.float32, shape=[2]))

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  def testLargeRate(self, dropout_fn):
    x_dim = 40
    y_dim = 30
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    _ = dropout_fn(t, rate=0.9)

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  def testVariableRef(self, dropout_fn):
    x = variable_scope.get_variable("x", shape=[10, 10], dtype=dtypes.float32)
    _ = dropout_fn(x, rate=0.1)

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  @test_util.run_deprecated_v1
  def testShapedDropoutShapeError(self, dropout_fn):
    # Runs shaped dropout and verifies an error is thrown on misshapen noise.
    x_dim = 40
    y_dim = 30
    keep_prob = 0.5
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      _ = dropout_fn(
          t, rate=(1 - keep_prob), noise_shape=[x_dim, y_dim + 10])
    with self.assertRaises(ValueError):
      _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim, y_dim, 5])
    with self.assertRaises(ValueError):
      _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim + 3])
    with self.assertRaises(ValueError):
      _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim])
    # test that broadcasting proceeds
    _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[y_dim])
    _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[1, y_dim])
    _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim, 1])
    _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[1, 1])

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  def testNoDropout(self, dropout_fn):
    x = array_ops.zeros((5,))
    y = dropout_fn(x, rate=0)
    self.assertAllEqual(x, y)

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  def testDropoutWithIntegerInputs(self, dropout_fn):
    x = constant_op.constant([1, 1, 1, 1, 1])
    with self.assertRaises(ValueError):
      _ = dropout_fn(x, rate=0.5)


@test_util.run_all_without_tensor_float_32(
    "Tests _compute_sampled_logits and related functions, which call matmul")
class ComputeSampledLogitsTest(test_lib.TestCase):

  def setUp(self):
    self._eps = 1e-3

  def _GenerateTestData(self, num_classes, dim, batch_size, num_true, labels,
                        sampled, subtract_log_q):
    """Randomly generates input/output data for a single test case.

    This function returns numpy constants for use in a test case.

    Args:
      num_classes: An int. The number of embedding classes in the test case.
      dim: An int. The dimension of the embedding.
      batch_size: An int. The batch size.
      num_true: An int. The number of target classes per training example.
      labels: A list of batch_size * num_true ints. The target classes.
      sampled: A list of indices in [0, num_classes).
      subtract_log_q: A bool corresponding to the parameter in
        _compute_sampled_logits().

    Returns:
      weights: Embedding weights to use as test input. It is a numpy array
          of shape [num_classes, dim]
      biases: Embedding biases to use as test input. It is a numpy array
          of shape [num_classes].
      hidden_acts: Forward activations of the network to use as test input.
          It is a numpy array of shape [batch_size, dim].
      sampled_vals: A tuple based on `sampled` to use as test input in the
          format returned by a *_candidate_sampler function.
      exp_logits: The output logits expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
      exp_labels: The output labels expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
    """
    weights = np.random.randn(num_classes, dim).astype(np.float32)
    biases = np.random.randn(num_classes).astype(np.float32)
    hidden_acts = np.random.randn(batch_size, dim).astype(np.float32)

    true_exp = np.full([batch_size, 1], fill_value=0.5, dtype=np.float32)
    sampled_exp = np.full([len(sampled)], fill_value=0.5, dtype=np.float32)
    sampled_vals = (sampled, true_exp, sampled_exp)

    sampled_w, sampled_b = weights[sampled], biases[sampled]
    true_w, true_b = weights[labels], biases[labels]

    true_logits = np.sum(
        hidden_acts.reshape((batch_size, 1, dim)) * true_w.reshape(
            (batch_size, num_true, dim)),
        axis=2)
    true_b = true_b.reshape((batch_size, num_true))
    true_logits += true_b
    sampled_logits = np.dot(hidden_acts, sampled_w.T) + sampled_b

    if subtract_log_q:
      true_logits -= np.log(true_exp)
      sampled_logits -= np.log(sampled_exp[np.newaxis, :])

    exp_logits = np.concatenate([true_logits, sampled_logits], axis=1)
    exp_labels = np.hstack(
        (np.ones_like(true_logits) / num_true, np.zeros_like(sampled_logits)))

    return weights, biases, hidden_acts, sampled_vals, exp_logits, exp_labels

  def _ShardTestEmbeddings(self, weights, biases, num_shards):
    """Shards the weights and biases returned by _GenerateTestData.

    Args:
      weights: The weights returned by _GenerateTestData.
      biases: The biases returned by _GenerateTestData.
      num_shards: The number of shards to create.

    Returns:
      sharded_weights: A list of size `num_shards` containing all the weights.
      sharded_biases: A list of size `num_shards` containing all the biases.
    """
    with ops.Graph().as_default() as g:
      sharded_weights = variable_scope.get_variable(
          "w",
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards),
          initializer=constant_op.constant(weights))
      sharded_biases = variable_scope.get_variable(
          "b",
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards),
          initializer=constant_op.constant(biases))
      with self.session(graph=g) as sess:
        self.evaluate(variables.global_variables_initializer())
        return self.evaluate([list(sharded_weights), list(sharded_biases)])

  def testShapes(self):
    np.random.seed(0)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=[1, 0, 2, 3],
           subtract_log_q=False)
      logits_tensor, labels_tensor = _compute_sampled_logits(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=False,
          remove_accidental_hits=False,
          partition_strategy="div",
          name="sampled_logits_basic_num_true_%d" % num_true)
      got_logits, got_labels = self.evaluate([logits_tensor, labels_tensor])
      self.assertEqual(exp_logits.shape, got_logits.shape, self._eps)
      self.assertEqual(exp_labels.shape, got_labels.shape, self._eps)

  def testBasic(self):
    """Without accidental hit removal or subtract_log_q."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=[1, 0, 2, 3],
           subtract_log_q=False)
      logits_tensor, labels_tensor = _compute_sampled_logits(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=False,
          remove_accidental_hits=False,
          partition_strategy="div",
          name="sampled_logits_basic_num_true_%d" % num_true)
      got_logits, got_labels = self.evaluate([logits_tensor, labels_tensor])
      self.assertAllClose(exp_logits, got_logits, self._eps)
      self.assertAllClose(exp_labels, got_labels, self._eps)

  def testAccidentalHitRemoval(self):
    """With accidental hit removal, no subtract_log_q."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    sampled = [1, 0, 2, 3]

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, _,
       _) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=sampled,
           subtract_log_q=False)
      logits_tensor, _ = _compute_sampled_logits(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=len(sampled),
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=False,
          remove_accidental_hits=True,
          partition_strategy="div",
          name="sampled_logits_accidental_hit_removal_num_true_%d" % num_true)
      # Test that the exponentiated logits of accidental hits are near 0.
      # First we need to find the hits in this random test run:
      labels_reshape = labels.reshape((batch_size, num_true))
      got_logits = self.evaluate(logits_tensor)
      for row in range(batch_size):
        row_labels = labels_reshape[row, :]
        for col in range(len(sampled)):
          if sampled[col] in row_labels:
            # We need to add the num_true_test offset into logits_*
            self.assertNear(
                np.exp(got_logits[row, col + num_true]), 0., self._eps)

  def testSubtractLogQ(self):
    """With subtract_log_q, no accidental hit removal."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=[1, 0, 2, 3],
           subtract_log_q=True)
      logits_tensor, labels_tensor = _compute_sampled_logits(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=True,
          remove_accidental_hits=False,
          partition_strategy="div",
          name="sampled_logits_subtract_log_q_num_true_%d" % num_true)
      got_logits, got_labels = self.evaluate([logits_tensor, labels_tensor])
      self.assertAllClose(exp_logits, got_logits, self._eps)
      self.assertAllClose(exp_labels, got_labels, self._eps)

  def testSharded(self):
    """With sharded weights and sharded biases."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=[1, 0, 2, 3],
           subtract_log_q=False)
      weight_shards, bias_shards = self._ShardTestEmbeddings(
          weights, biases, num_shards=3)
      logits_tensor, labels_tensor = _compute_sampled_logits(
          weights=[constant_op.constant(shard) for shard in weight_shards],
          biases=[constant_op.constant(shard) for shard in bias_shards],
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=False,
          remove_accidental_hits=False,
          partition_strategy="div",
          name="sampled_logits_sharded_num_true_%d" % num_true)
      got_logits, got_labels = self.evaluate([logits_tensor, labels_tensor])
      self.assertAllClose(exp_logits, got_logits, self._eps)
      self.assertAllClose(exp_labels, got_labels, self._eps)

  def testNCELoss(self):
    # A simple test to verify the numerics.

    def _SigmoidCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      pred = 1. / (1. + np.exp(-logits))
      eps = 0.0001
      pred = np.minimum(np.maximum(pred, eps), 1 - eps)
      return -targets * np.log(pred) - (1. - targets) * np.log(1. - pred)

    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals, exp_logits,
     exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=10,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=[1, 0, 2, 3],
         subtract_log_q=True)
    exp_nce_loss = np.sum(
        _SigmoidCrossEntropyWithLogits(exp_logits, exp_labels), 1)

    got_nce_loss = nn_impl.nce_loss_v2(
        weights=constant_op.constant(weights),
        biases=constant_op.constant(biases),
        labels=constant_op.constant(labels, shape=(batch_size, 1)),
        inputs=constant_op.constant(hidden_acts),
        num_sampled=4,
        num_classes=num_classes,
        num_true=1,
        sampled_values=sampled_vals)

    self.assertAllClose(exp_nce_loss, self.evaluate(got_nce_loss), 1e-4)

    # Test with sharded weights and sharded biases.
    weight_shards, bias_shards = self._ShardTestEmbeddings(
        weights, biases, num_shards=3)
    got_nce_loss = nn_impl.nce_loss_v2(
        weights=[constant_op.constant(shard) for shard in weight_shards],
        biases=[constant_op.constant(shard) for shard in bias_shards],
        labels=constant_op.constant(labels, shape=(batch_size, 1)),
        inputs=constant_op.constant(hidden_acts),
        num_sampled=4,
        num_classes=num_classes,
        num_true=1,
        sampled_values=sampled_vals)

    self.assertAllClose(exp_nce_loss, self.evaluate(got_nce_loss), 1e-4)

  def testSampledSoftmaxLoss(self):
    # A simple test to verify the numerics.

    def _SoftmaxCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      stable_exp_logits = np.exp(logits -
                                 np.amax(logits, axis=1, keepdims=True))
      pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
      return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals, exp_logits,
     exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=10,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=[1, 0, 2, 3],
         subtract_log_q=True)
    exp_sampled_softmax_loss = _SoftmaxCrossEntropyWithLogits(
        exp_logits, exp_labels)

    got_sampled_softmax_loss = nn_impl.sampled_softmax_loss_v2(
        weights=constant_op.constant(weights),
        biases=constant_op.constant(biases),
        labels=constant_op.constant(labels, shape=(batch_size, 1)),
        inputs=constant_op.constant(hidden_acts),
        num_sampled=4,
        num_classes=num_classes,
        num_true=1,
        sampled_values=sampled_vals,
        remove_accidental_hits=False)

    self.assertAllClose(exp_sampled_softmax_loss,
                        self.evaluate(got_sampled_softmax_loss), 1e-4)

    # Test with sharded weights and sharded biases.
    weight_shards, bias_shards = self._ShardTestEmbeddings(
        weights, biases, num_shards=3)
    got_sampled_softmax_loss = nn_impl.sampled_softmax_loss_v2(
        weights=[constant_op.constant(shard) for shard in weight_shards],
        biases=[constant_op.constant(shard) for shard in bias_shards],
        labels=constant_op.constant(labels, shape=(batch_size, 1)),
        inputs=constant_op.constant(hidden_acts),
        num_sampled=4,
        num_classes=num_classes,
        num_true=1,
        sampled_values=sampled_vals,
        remove_accidental_hits=False)

    self.assertAllClose(exp_sampled_softmax_loss,
                        self.evaluate(got_sampled_softmax_loss), 1e-4)

  def testSampledSoftmaxLossBf16(self):
    # A simple test to verify the numerics for bfloat16.
    def _SoftmaxCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      stable_exp_logits = np.exp(logits -
                                 np.amax(logits, axis=1, keepdims=True))
      pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
      return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    sampled = [1, 0, 2, 3]
    (weights, biases, hidden_acts, _, exp_logits,
     exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=10,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=sampled,
         subtract_log_q=True)
    exp_sampled_softmax_loss = _SoftmaxCrossEntropyWithLogits(
        exp_logits, exp_labels)

    true_exp_bf16 = np.full([batch_size, 1],
                            fill_value=0.5,
                            dtype=dtypes.bfloat16.as_numpy_dtype)
    sampled_exp_bf16 = np.full([len(sampled)],
                               fill_value=0.5,
                               dtype=dtypes.bfloat16.as_numpy_dtype)
    sampled_vals_bf16 = (sampled, true_exp_bf16, sampled_exp_bf16)

    got_sampled_softmax_loss = math_ops.cast(
        nn_impl.sampled_softmax_loss_v2(
            weights=constant_op.constant(weights, dtype=dtypes.bfloat16),
            biases=constant_op.constant(biases, dtype=dtypes.bfloat16),
            labels=constant_op.constant(
                labels, shape=(batch_size, 1), dtype=dtypes.bfloat16),
            inputs=constant_op.constant(hidden_acts, dtype=dtypes.bfloat16),
            num_sampled=4,
            num_classes=num_classes,
            num_true=1,
            sampled_values=sampled_vals_bf16,
            remove_accidental_hits=False), dtypes.float32)

    self.assertAllClose(exp_sampled_softmax_loss,
                        self.evaluate(got_sampled_softmax_loss), 1e-1)


class CReluTest(test_lib.TestCase):

  def test(self):
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.concatenate([x * (x > 0), -x * (x < 0)], axis=1)

    z = self.evaluate(nn_ops.crelu(constant_op.constant(x)))
    self.assertAllClose(y, z, 1e-4)


class ReluTest(test_lib.TestCase):

  def test(self):
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.maximum(x, 0.0)

    z = self.evaluate(nn_ops.relu(constant_op.constant(x)))
    self.assertAllEqual(y, z)

  @test_util.disable_xla(
      "This test relies on undefined behavior that XLA does not replicate")
  @test_util.run_deprecated_v1
  def testNaNs(self):
    # Test that relu(nan) = nan for various sizes.
    for i in range(18):
      x = np.zeros(i) + np.nan
      # TODO(b/178335491): This is broken on GPU today.
      with self.cached_session(use_gpu=False):
        z = nn_ops.relu(constant_op.constant(x)).eval()
        self.assertTrue(np.isnan(z).all())


class LeakyReluTest(test_lib.TestCase):

  def testRange(self):
    batch_size = 3
    height, width = 4, 4
    np.random.seed(1)  # Make it reproducible.
    inputs = np.random.uniform(size=(batch_size, height, width,
                                     3)).astype(np.float32)
    inputs = constant_op.constant(inputs)

    outputs = nn_ops.leaky_relu(inputs)
    self.assertEqual(inputs.shape, outputs.shape)

    inputs, outputs = self.evaluate([inputs, outputs])

    self.assertGreaterEqual(outputs.min(), 0.0)
    self.assertLessEqual(outputs.max(), 1.0)
    self.assertAllClose(inputs, outputs)

  @test_util.run_deprecated_v1
  def testValues(self):
    for dtype in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      np_values = np.array([-2, -1, 0, 1, 2], dtype=dtype)
      outputs = nn_ops.leaky_relu(constant_op.constant(np_values))

      outputs = self.evaluate(outputs)

      tol = 2e-3 if dtype == np.float16 else 1e-6
      self.assertAllClose(
          outputs, [-0.4, -0.2, 0.0, 1.0, 2.0], rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testName(self):
    np_values = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    outputs_with_name_set = nn_ops.leaky_relu(
        constant_op.constant(np_values), name="test_relu_op")
    self.assertEqual(outputs_with_name_set.name, "test_relu_op:0")
    outputs_without_name_set = nn_ops.leaky_relu(
        constant_op.constant(np_values))
    self.assertEqual(outputs_without_name_set.name, "LeakyRelu:0")


class GeluTest(test_lib.TestCase):

  def test(self):

    def gelu(x, approximate=False):
      if approximate:
        return 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
      else:
        from scipy.stats import norm  # pylint: disable=g-import-not-at-top
        return x * norm.cdf(x)

    # Make sure we test for negative arguments where GeLU is difficult to
    # evaluate accurately.
    x = np.linspace(-12, 5, 1000).astype(np.float32)
    y = gelu(x)
    z = self.evaluate(nn_ops.gelu(x))
    self.assertAllClose(y, z, atol=0, rtol=2e-5)

    y = gelu(x, True)
    z = self.evaluate(nn_ops.gelu(x, True))
    self.assertAllClose(y, z)


@test_util.run_all_in_graph_and_eager_modes
class SwishTest(test_lib.TestCase):

  def testValues(self):
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    tf_values = constant_op.constant(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values)
    expected_tf_outputs = tf_values * math_ops.sigmoid(tf_values)

    actual_outputs, expected_outputs = self.evaluate(
        [actual_tf_outputs, expected_tf_outputs])

    self.assertAllClose(actual_outputs, expected_outputs)

  def testValuesWithBeta(self):
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    tf_values = constant_op.constant(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values, beta=0.5)
    expected_tf_outputs = tf_values * math_ops.sigmoid(0.5 * tf_values)

    actual_outputs, expected_outputs = self.evaluate(
        [actual_tf_outputs, expected_tf_outputs])

    self.assertAllClose(actual_outputs, expected_outputs)

  def testGradients(self):
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = constant_op.constant(input_values)
    with self.cached_session():
      def f(x):  # pylint: disable=invalid-name
        return nn_impl.swish(x)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [x_tf])
      self.assertAllClose(theoretical, numerical)

  def testGradientsWithBeta(self):
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = constant_op.constant(input_values)
    with self.cached_session():
      def f(x):  # pylint: disable=invalid-name
        return nn_impl.swish(x, beta=0.5)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [x_tf])
      self.assertAllClose(theoretical, numerical)


class MomentsTest(test_lib.TestCase):

  def doOutputTest(self,
                   input_shape,
                   moments_axes,
                   tol=1e-4,
                   check_gradients=False):
    for mu in [0.0, 1.0, 1e3]:
      for sigma in [1.0, 0.1]:
        for keep_dims in [True, False]:
          input_values = np.random.rand(*input_shape) * sigma + mu
          expected_mean = np.mean(
              input_values, axis=moments_axes, keepdims=keep_dims)
          expected_var = np.var(
              input_values, axis=moments_axes, keepdims=keep_dims)
          with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
              inputs = constant_op.constant(
                  input_values, shape=input_shape, dtype=dtypes.float32)
              mean, variance = nn_impl.moments_v2(
                  inputs, moments_axes, keepdims=keep_dims)

              if check_gradients:
                err = gradient_checker.compute_gradient_error(
                    inputs, input_shape, mean, mean.shape.as_list())
                self.assertLess(err, 1e-3)
                err = gradient_checker.compute_gradient_error(
                    inputs, input_shape, variance, variance.shape.as_list())
                self.assertLess(err, 1e-3)

              # Evaluate.
              [mean, variance] = self.evaluate([mean, variance])
              # Make sure that there are no NaNs
              self.assertFalse(np.isnan(mean).any())
              self.assertFalse(np.isnan(variance).any())
              self.assertAllClose(mean, expected_mean, rtol=tol, atol=tol)
              self.assertAllClose(variance, expected_var, rtol=tol, atol=tol)

  def testOutputAndGradient2DInput0(self):
    self.doOutputTest((10, 10), (0,), check_gradients=True)

  def testOutputAndGradient2DInput01(self):
    self.doOutputTest((10, 10), (0, 1), check_gradients=True)

  def testOutput2DInput0(self):
    self.doOutputTest((10, 300), (0,))

  def testOutput2DInput1(self):
    self.doOutputTest((10, 300), (1,))

  def testOutput2DInput01(self):
    self.doOutputTest((10, 300), (0, 1))

  def testOutput4DInput0(self):
    self.doOutputTest((10, 10, 10, 30), (0,))

  def testOutput4DInput1(self):
    self.doOutputTest((10, 10, 10, 30), (1,))

  def testOutput4DInput3(self):
    self.doOutputTest((10, 10, 10, 30), (3,))

  def testOutput4DInput012(self):
    self.doOutputTest((10, 10, 10, 30), (0, 1, 2))

  def testOutput4DInput123(self):
    self.doOutputTest((10, 10, 10, 30), (1, 2, 3))


class DataFormatDimMapTest(test_lib.TestCase):

  def _test(self, x_val, y_val_expected):
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x)

    y_val = self.evaluate(y)
    self.assertAllEqual(y_val, y_val_expected)

  def test(self):
    self._test(0, 0)
    self._test(1, 2)
    self._test(2, 3)
    self._test(3, 1)
    self._test(-1, 1)
    self._test(-2, 3)
    self._test(-3, 2)
    self._test(-4, 0)
    self._test([1, 3], [2, 1])
    self._test([1, 3, -2], [2, 1, 3])
    self._test([1, -3, -2], [2, 2, 3])
    self._test([[1, -3], [1, -1]], [[2, 2], [2, 1]])

  def testNHWCtoNCHW(self):
    x_val = [1, -3, -2]
    y_val_expected = [2, 2, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="NCHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNHWCtoHWNC(self):
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [2, 0, 1, 3, 2, 0, 1, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNHWCtoWHCN(self):
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [3, 1, 0, 2, 3, 1, 0, 2]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNDHWCtoNCDHW(self):
    x_val = [1, -4, -3, -2]
    y_val_expected = [2, 2, 3, 4]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNDHWCtoDHWNC(self):
    x_val = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    y_val_expected = [3, 0, 1, 2, 4, 3, 0, 1, 2, 4]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="DHWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testDNHWCtoWHDCN(self):
    x_val = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    y_val_expected = [4, 2, 1, 0, 3, 4, 2, 1, 0, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="WHDCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testArbitraryASCII(self):
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [3, 2, 1, 0, 3, 2, 1, 0]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="qwer", dst_format="rewq")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testInvalidLength(self):
    x = [-4, -3, -2, -1, 0, 1, 2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Source format must be of length 4 or 5"):
      op = nn_ops.data_format_dim_map(
          x, src_format="12345678", dst_format="87654321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testDuplicateSrc(self):
    x = [-4, -3, -2, -1, 0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_dim_map(x, src_format="1233", dst_format="4321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testDuplicateDst(self):
    x = [-4, -3, -2, -1, 0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_dim_map(x, src_format="1234", dst_format="3321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testExtraSpecifiers(self):
    x = [-4, -3, -2, -1, 0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_dim_map(x, src_format="1234", dst_format="5321")
      with test_util.use_gpu():
        self.evaluate(op)


class DataFormatVectorPermuteTest(test_lib.TestCase):

  def testNHWCToNCHW(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 3, 4, 9])

  def testNHWCToNCHW_Size2(self):
    x_val = [4, 9]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [4, 9])

  def testNDHWCtoNCDHW(self):
    x_val = [7, 4, 9, 3, 5]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 5, 4, 9, 3])

  def testNDHWCtoNCDHW_Size3(self):
    x_val = [4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [4, 9, 3])

  @test_util.disable_xla("unsupported data format")
  def testNHWCToWHCN(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 4, 3, 7])

  @test_util.disable_xla("unsupported data format")
  def testNHWCToWHCN_Size2(self):
    x_val = [4, 9]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 4])

  @test_util.disable_xla("unsupported data format")
  def testNDHWCToWHDCN(self):
    x_val = [7, 4, 9, 3, 5]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="WHDCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [3, 9, 4, 5, 7])

  @test_util.disable_xla("unsupported data format")
  def testNDHWCToWHDCN_Size3(self):
    x_val = [4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="WHDCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [3, 9, 4])

  def testNCHWToNHWC(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NCHW", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 9, 3, 4])

  def testNCHWToNHWC_Size2(self):
    x_val = [9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 3])

  def testNCDHWToNDHWC(self):
    x_val = [7, 4, 9, 3, 5]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NCDHW", dst_format="NDHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 9, 3, 5, 4])

  def testNCDHWToNDHWC_Size3(self):
    x_val = [9, 3, 5]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NCDHW", dst_format="NDHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 3, 5])

  def testNHWCToHWNC(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [4, 9, 7, 3])

  def testHWNCToNHWC(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="HWNC", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 7, 4, 3])

  def testNHWCToNCHW2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [5, 1], [9, 3], [4, 5]])

  def testNHWCToHWNC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[9, 3], [4, 5], [7, 4], [5, 1]])

  def testHWNCToNHWC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="HWNC", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[4, 5], [7, 4], [9, 3], [5, 1]])

  def testNCHWToNHWC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NCHW", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [4, 5], [5, 1], [9, 3]])

  def testNDHWCToNCDHW2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1], [8, 2]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [8, 2], [9, 3], [4, 5], [5, 1]])

  @test_util.disable_xla("unsupported data format")
  def testNDHWCToDHWNC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1], [8, 2]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="DHWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[9, 3], [4, 5], [5, 1], [7, 4], [8, 2]])

  @test_util.disable_xla("unsupported data format")
  def testDHWNCToNDHWC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1], [8, 2]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="DHWNC", dst_format="NDHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[5, 1], [7, 4], [9, 3], [4, 5], [8, 2]])

  def testNCDHWToNDHWC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1], [8, 2]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NCDHW", dst_format="NDHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [4, 5], [5, 1], [8, 2], [9, 3]])

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testInvalidLength(self):
    x = [0, 1, 2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Source format must be of length 4 or 5"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="12345678", dst_format="87654321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testDuplicateSrc(self):
    x = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="1233", dst_format="4321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testDuplicateDst(self):
    x = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="1234", dst_format="3321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testExtraSpecifiers(self):
    x = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="1234", dst_format="5321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def test2DNoWH(self):
    x = [[0, 1], [2, 3]]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Format specifier must contain H and W for 2D case"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="1234", dst_format="4321")
      with test_util.use_gpu():
        self.evaluate(op)


@test_util.run_all_in_graph_and_eager_modes
class AvgPoolTest(test_lib.TestCase):

  def test1DTensor(self):
    x = array_ops.ones([3, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool1d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DNumpy(self):
    # explicitly use float32 for ROCm, as MIOpen does not yet support float64
    # np.ones defaults to using float64 when dtype is not explicitly specified
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.ones([3, 6, 5], dtype=dtype)
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool1d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DNumpyWithGolden(self):
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.array([[[3], [6], [5]], [[1], [0], [1]]], dtype=dtype)
    ksize = 2
    strides = 1
    y = nn_ops.avg_pool1d(x, ksize, strides, "SAME")
    expected_y = np.array([[[4.5], [5.5], [5.0]], [[0.5], [0.5], [1.0]]],
                          dtype=dtype)
    self.assertAllEqual(self.evaluate(y), expected_y)

  def test2DTensor(self):
    x = array_ops.ones([3, 6, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test2DNumpy(self):
    # explicitly use float32 for ROCm, as MIOpen does not yet support float64
    # np.ones defaults to using float64 when dtype is not explicitly specified
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.ones([3, 6, 6, 5], dtype=dtype)
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DTensor(self):
    x = array_ops.ones([3, 7, 6, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool3d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DNumpy(self):
    x = np.ones([3, 7, 6, 6, 5], dtype=np.float32)
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool3d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))


@test_util.run_all_in_graph_and_eager_modes
class MaxPoolTest(test_lib.TestCase):

  def test1DTensor(self):
    x = array_ops.ones([3, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool1d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DNumpy(self):
    # explicitly use float32 for ROCm, as MIOpen does not yet support float64
    # np.ones defaults to using float64 when dtype is not explicitly specified
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.ones([3, 6, 5], dtype=dtype)
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool1d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DNumpyWithGolden(self):
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.array([[[3], [6], [5]], [[1], [0], [1]]], dtype=dtype)
    ksize = 2
    strides = 1
    y = nn_ops.max_pool1d(x, ksize, strides, "SAME")
    expected_y = np.array([[[6], [6], [5]], [[1], [1], [1]]], dtype=dtype)
    self.assertAllEqual(self.evaluate(y), expected_y)

  def test2DTensor(self):
    x = array_ops.ones([3, 6, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test2DNumpy(self):
    # explicitly use float32 for ROCm, as MIOpen does not yet support float64
    # np.ones defaults to using float64 when dtype is not explicitly specified
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.ones([3, 6, 6, 5], dtype=dtype)
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DTensor(self):
    x = array_ops.ones([3, 7, 6, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool3d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DNumpy(self):
    x = np.ones([3, 7, 6, 6, 5], dtype=np.float32)
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool3d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def testIncorrectSizeInputSmall(self):
    x = array_ops.ones([3, 4])
    with self.assertRaisesRegex(
        ValueError,
        "`input.shape.rank` must be 3, 4 or 5.*of rank 2."):
      nn_ops.max_pool_v2(x, 2, 2, "SAME")

  def testIncorrectSizeInput(self):
    x = array_ops.ones([3, 4, 1, 2, 1, 2])
    with self.assertRaisesRegex(
        ValueError,
        "`input.shape.rank` must be 3, 4 or 5.*of rank 6."):
      nn_ops.max_pool_v2(x, 2, 2, "SAME")

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testIncoorectKSize(self):
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "Sliding window ksize must be positive."
    ):
      op = nn_ops.max_pool_v2(
          array_ops.ones([3, 4, 4, 5]), [1, -1, -1, 1], 2, "SAME"
      )
      with test_util.use_gpu():
        self.evaluate(op)

    ksize = sys.maxsize + 100  # Set to a value larger than sys.maxsize
    with self.assertRaises(
        OverflowError if context.executing_eagerly() else ValueError
    ):
      op = nn_ops.max_pool_v2(
          array_ops.ones([3, 4, 4, 5]), ksize=ksize, strides=2, padding="SAME"
      )
      with test_util.use_gpu():
        self.evaluate(op)


@test_util.run_all_in_graph_and_eager_modes
class ConvolutionTest(test_lib.TestCase):

  def testUnknownSize(self):
    x = tensor_spec.TensorSpec(None, dtypes.float32, name="x")
    k = np.ones([3, 6, 6, 5], dtype=np.float32)

    @def_function.function
    def F(value):
      return nn_ops.convolution(value, k, "SAME")

    F.get_concrete_function(x)


class ConvTransposeTest(test_lib.TestCase):

  def test1D(self):
    t = array_ops.ones([2, 4, 3])
    v = array_ops.ones([2, 5, 3])
    strides = 2

    y1 = nn_ops.conv1d_transpose(t, v, [2, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, [2, 8, 5], strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DTensor(self):
    t = array_ops.ones([2, 4, 3])
    v = array_ops.ones([2, 5, 3])
    strides = 2

    y1 = nn_ops.conv1d_transpose(t, v, [2, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, constant_op.constant([2, 8, 5]), strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test2D(self):
    t = array_ops.ones([2, 4, 4, 3])
    v = array_ops.ones([2, 2, 5, 3])
    strides = 2

    y1 = nn_ops.conv2d_transpose_v2(t, v, [2, 8, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, [2, 8, 8, 5], strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test2DTensor(self):
    t = array_ops.ones([2, 4, 4, 3])
    v = array_ops.ones([2, 2, 5, 3])
    strides = 2

    y1 = nn_ops.conv2d_transpose_v2(t, v, [2, 8, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, constant_op.constant([2, 8, 8, 5]),
                               strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3D(self):
    t = array_ops.ones([2, 4, 4, 4, 3])
    v = array_ops.ones([2, 2, 2, 5, 3])
    strides = 2

    y1 = nn_ops.conv3d_transpose_v2(t, v, [2, 8, 8, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, [2, 8, 8, 8, 5], strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DTensor(self):
    t = array_ops.ones([2, 4, 4, 4, 3])
    v = array_ops.ones([2, 2, 2, 5, 3])
    strides = 2

    y1 = nn_ops.conv3d_transpose_v2(t, v, [2, 8, 8, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, constant_op.constant([2, 8, 8, 8, 5]),
                               strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def testIncorrectSizeInputSmall(self):
    with self.assertRaisesRegex(
        ValueError,
        "`output_shape` must be of length 3, 4 or 5.*of length 2."):
      nn_ops.conv_transpose(None, 2, [2, 3], "SAME")

  def testIncorrectSizeInput(self):
    with self.assertRaisesRegex(
        ValueError,
        "`output_shape` must be of length 3, 4 or 5.* of length 6."):
      nn_ops.conv_transpose(None, 2, [2, 3, 4, 2, 5, 1], "SAME")

  def testTensorsNoShape(self):
    with self.assertRaisesRegex(
        ValueError, "`output_shape` must be a tensor or sized collection"):
      nn_ops.conv_transpose(None, None, None, None)


class RaggedEmbeddingTest(test_lib.TestCase):

  def testRaggedTensor(self):
    weights = constant_op.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    ragged_ids = ragged_factory_ops.constant([[1, 2, 3], [0], [1, 2]],
                                             ragged_rank=1)

    embedded_ragged = nn.embedding_lookup(weights, ragged_ids)
    expected_output = ragged_factory_ops.constant(
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[0, 0, 0]], [[1, 1, 1], [2, 2, 2]]
        ],
        ragged_rank=1)

    self.assertAllEqual(expected_output, embedded_ragged)

  def testMultipleRaggedDimTensor(self):
    weights = constant_op.constant([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                                    [5, 5], [6, 6]])
    ragged_ids = ragged_factory_ops.constant(
        [[[[3, 4], [0, 6]], []], [[[2, 1], [1, 0]], [[2, 5], [2, 3]]], [[[1, 0]]
                                                                       ]],
        ragged_rank=2)

    embedded_ragged = nn.embedding_lookup(weights, ragged_ids)
    expected_output = ragged_factory_ops.constant(
        [[[[[3, 3], [4, 4]], [[0, 0], [6, 6]]], []],
         [[[[2, 2], [1, 1]], [[1, 1], [0, 0]]],
          [[[2, 2], [5, 5]], [[2, 2], [3, 3]]]], [[[[1, 1], [0, 0]]]]],
        ragged_rank=2)

    self.assertAllEqual(expected_output, embedded_ragged)

  def testMissingWeights(self):
    ragged_ids = ragged_factory_ops.constant([[1, 2, 3], [0], [1, 2]])

    with self.assertRaisesRegex(ValueError, "params must be specified.*"):
      nn.embedding_lookup(None, ragged_ids)

  def testEmptyWeights(self):
    ragged_ids = ragged_factory_ops.constant([[1, 2, 3], [0], [1, 2]])

    with self.assertRaisesRegex(ValueError, "params should not be empty.*"):
      nn.embedding_lookup([], ragged_ids)

  def testInvalidIndicesType(self):
    weights = constant_op.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    ragged_ids = ragged_factory_ops.constant([[1., 2., 3.], [1., 2.]])

    with self.assertRaisesRegex(
        ValueError, "The values contained by the inputs have type*"):
      nn.embedding_lookup(weights, ragged_ids)

  def testMaxNormForEmbeddings(self):
    weights = constant_op.constant(
        [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        dtype=dtypes.float32)
    ragged_ids = ragged_factory_ops.constant([[1, 2, 3], [0], [1, 2]],
                                             ragged_rank=1)

    actual_embeddings = [
        nn.embedding_lookup(weights, ragged_ids, max_norm=max_norm)
        for max_norm in [1, 2, 5]
    ]

    expected_embeddings = (
        # max_norm = 1
        [[[.5, .5, .5, .5], [.5, .5, .5, .5], [.5, .5, .5, .5]], [[0, 0, 0, 0]],
         [[.5, .5, .5, .5], [.5, .5, .5, .5]]],
        # max_norm = 2
        [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[0, 0, 0, 0]],
         [[1, 1, 1, 1], [1, 1, 1, 1]]],
        # max_norm = 5
        [[[1, 1, 1, 1], [2, 2, 2, 2], [2.5, 2.5, 2.5, 2.5]], [[0, 0, 0, 0]],
         [[1, 1, 1, 1], [2, 2, 2, 2]]],
    )

    for expected, actual in zip(expected_embeddings, actual_embeddings):
      self.assertAllClose(
          ragged_factory_ops.constant(expected, dtype=float, ragged_rank=1),
          actual)


class IsotonicTest(parameterized.TestCase, test_lib.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_and_decreasing(self):
    x = constant_op.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                             dtype=dtypes.float64)
    y, segments = nn_ops.isotonic_regression(x, decreasing=False)
    self.assertAllClose(y, x)
    self.assertAllClose(segments, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

    y, segments = nn_ops.isotonic_regression(x, decreasing=True)
    self.assertAllClose(
        y,
        [
            [2, 2, 2, 2, 2],  # Average of the inputs.
            [7, 7, 7, 7, 7]
        ])
    self.assertAllClose(segments, array_ops.zeros((2, 5)))

    # pylint: disable=invalid-unary-operand-type
    y, segments = nn_ops.isotonic_regression(-x, decreasing=True)
    self.assertAllClose(segments, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

    self.assertAllClose(y, -x)
    y, segments = nn_ops.isotonic_regression(-x, decreasing=False)
    self.assertAllClose(
        -y,
        [
            [2, 2, 2, 2, 2],  # Average of the inputs.
            [7, 7, 7, 7, 7]
        ])
    self.assertAllClose(segments, array_ops.zeros((2, 5)))

  @test_util.run_in_graph_and_eager_modes
  def test_different_axis(self):
    x = constant_op.constant([[0, 6, 2, 8, 4], [5, 1, 7, 3, 9]],
                             dtype=dtypes.float64)
    y, segments = nn_ops.isotonic_regression(x, decreasing=True, axis=0)
    self.assertAllClose(
        y,
        [
            [2.5, 6, 4.5, 8, 6.5],  # Either identity or average.
            [2.5, 1, 4.5, 3, 6.5]
        ])
    self.assertAllClose(segments, [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0]])

  @test_util.run_v2_only
  def testGradientV2(self, dtype=np.float64, batch_size=30, dimensions=50):

    @def_function.function
    def ComputeIsotonicFn(x):
      y, _ = nn_ops.isotonic_regression(x)  # No gradient wrt segments.
      return y

    np.random.seed(0)
    x_init = np.random.randn(batch_size, dimensions).astype(dtype)
    grad_theoretical, grad_numerical = gradient_checker_v2.compute_gradient(
        ComputeIsotonicFn, [x_init], delta=1e-5)
    self.assertAllClose(grad_theoretical, grad_numerical)

  @test_util.run_v1_only("compute_gradient_error is v1 only")
  def testGradientV1(self, dtype=np.float64, batch_size=30, dimensions=50):
    np.random.seed(0)
    x_init = np.random.randn(batch_size, dimensions).astype(dtype)
    with self.cached_session():
      x = array_ops.placeholder(dtype, (batch_size, dimensions))
      y, _ = nn_ops.isotonic_regression(x)  # Segments have no gradient.
      max_error = gradient_checker.compute_gradient_error(
          x, (batch_size, dimensions), y, (batch_size, dimensions), x_init)
    self.assertAllClose(max_error, 0.)

  @parameterized.parameters([[dtypes.half, dtypes.half],
                             [dtypes.bfloat16, dtypes.bfloat16],
                             [dtypes.float32, dtypes.float32],
                             [dtypes.float64, dtypes.float64],
                             [dtypes.int32, dtypes.float64],
                             [dtypes.int16, dtypes.float32]])
  def testTypePromotion(self, dtype_in, expected_dtype_out):
    x = constant_op.constant([[0, 6, 2, 8, 4], [5, 1, 7, 3, 9]], dtype=dtype_in)
    y, segments = nn_ops.isotonic_regression(x)
    self.assertEqual(y.dtype, expected_dtype_out)
    self.assertEqual(segments.dtype, dtypes.int32)


if __name__ == "__main__":
  test_lib.main()
