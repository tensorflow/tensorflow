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
"""Functional tests for reduction ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numbers

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# The maximum input rank to test.
_MAX_RANK = 5


def _powerset(iterable):
  """Helper for generating all possible reduction_axes arguments.

  Example:
  powerset([0,1,2]): () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)

  Args:
    iterable: An iterable of items to generate the powerset of.

  Returns:
    The powerset of all items in iterable.
  """
  s = list(iterable)
  return itertools.chain.from_iterable(
      itertools.combinations(s, r) for r in range(len(s)+1))


class ReducedShapeTest(test.TestCase):

  def _check(self, shape, axes, result):
    output = math_ops.reduced_shape(shape, axes=axes)
    self.assertAllEqual(output.eval(), result)

  def testSimple(self):
    with self.test_session():
      self._check([3], [], [3])
      self._check([3], [0], [1])
      self._check([5, 3], [], [5, 3])
      self._check([5, 3], [0], [1, 3])
      self._check([5, 3], [1], [5, 1])
      self._check([5, 3], [0, 1], [1, 1])

  def testZeros(self):
    """Check that reduced_shape does the right thing with zero dimensions."""
    with self.test_session():
      self._check([0], [], [0])
      self._check([0], [0], [1])
      self._check([0, 3], [], [0, 3])
      self._check([0, 3], [0], [1, 3])
      self._check([0, 3], [1], [0, 1])
      self._check([0, 3], [0, 1], [1, 1])
      self._check([3, 0], [], [3, 0])
      self._check([3, 0], [0], [1, 0])
      self._check([3, 0], [1], [3, 1])
      self._check([3, 0], [0, 1], [1, 1])

  def testNegAxes(self):
    with self.test_session():
      self._check([10, 10, 10], [-1], [10, 10, 1])
      self._check([10, 10, 10], [-1, 2], [10, 10, 1])
      self._check([10, 10, 10], [-1, -1], [10, 10, 1])
      self._check([10, 10, 10], [-1, 0], [1, 10, 1])
      self._check([10, 10, 10], [-3], [1, 10, 10])


class BaseReductionTest(test.TestCase):

  def _tf_reduce(self, x, reduction_axes, keep_dims):
    raise NotImplementedError()

  def _np_reduce(self, x, reduction_axes, keep_dims):
    raise NotImplementedError()

  def _makeIncremental(self, shape, dtype):
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 2j * data
    return data

  def _makeRandom(self, shape, dtype):
    data = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 2j * data
    return data

  def _compare(self, x, reduction_axes, keep_dims, feed_dict=None):
    np_ans = self._np_reduce(x, reduction_axes, keep_dims)
    with self.test_session(use_gpu=True) as sess:
      tf_ans = self._tf_reduce(x, reduction_axes, keep_dims)
      out = sess.run(tf_ans, feed_dict)
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes, feed_dict=None):
    if reduction_axes is not None and np.shape(reduction_axes) == (1,):
      # Test scalar reduction_axes argument
      self._compareAll(x, reduction_axes[0])
    self._compare(x, reduction_axes, keep_dims=False, feed_dict=feed_dict)
    self._compare(x, reduction_axes, keep_dims=True, feed_dict=feed_dict)

  def _compareAllAxes(self, x, feed_dict=None):
    self._compareAll(x, None)
    for axes in _powerset(range(x.ndim)):
      self._compareAll(x, axes, feed_dict)

  def _compareGradient(self, x, reduction_axes, rtol=1e-8, atol=1e-8):
    if reduction_axes is not None and np.shape(reduction_axes) == (1,):
      # Test scalar reduction_axes argument
      self._compareGradient(x, reduction_axes[0], rtol=rtol, atol=atol)
    with self.test_session(use_gpu=True):
      t = ops.convert_to_tensor(x)
      su = self._tf_reduce(t, reduction_axes, False)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, x.shape, su, su.get_shape().as_list(), x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=rtol, atol=atol)

  def _compareGradientAxes(self, x, rtol=1e-8, atol=1e-8):
    self._compareGradient(x, None, rtol=rtol, atol=atol)
    self._compareGradient(x, [], rtol=rtol, atol=atol)
    self._compareGradient(x, 0, rtol=rtol, atol=atol)
    self._compareGradient(x, [1], rtol=rtol, atol=atol)
    self._compareGradient(x, [2], rtol=rtol, atol=atol)
    self._compareGradient(x, [1, 2], rtol=rtol, atol=atol)
    self._compareGradient(x, [0, 1, 2, 3], rtol=rtol, atol=atol)


class SumReductionTest(BaseReductionTest):

  def _tf_reduce(self, x, reduction_axes, keep_dims):
    return math_ops.reduce_sum(x, reduction_axes, keep_dims)

  def _np_reduce(self, x, reduction_axes, keep_dims):
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    return np.sum(x, axis=reduction_axes, keepdims=keep_dims)

  def testInfinity(self):
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  def testInt32(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.int32)
      self._compareAllAxes(np_arr)

  def testFloat32(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
      self._compareAllAxes(np_arr)

  def testFloat64(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float64)
      self._compareAllAxes(np_arr)

  def testComplex64(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex64)
      self._compareAllAxes(np_arr)

  def testComplex128(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex128)
      self._compareAllAxes(np_arr)

  def testInvalidIndex(self):
    np_arr = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    input_tensor = ops.convert_to_tensor(np_arr)
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Invalid reduction dimension" in str(e)):
      math_ops.reduce_sum(input_tensor, [-3])
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Invalid reduction dimension" in str(e)):
      math_ops.reduce_sum(input_tensor, [2])
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Invalid reduction dimension" in str(e)):
      math_ops.reduce_sum(input_tensor, [0, 2])

  def testPartialShapes(self):
    np.random.seed(1618)

    # Input shape is unknown.
    reduction_axes = [1, 2]
    c_unknown = array_ops.placeholder(dtypes.float32)
    s_unknown = math_ops.reduce_sum(c_unknown, reduction_axes)
    self.assertEqual(tensor_shape.unknown_shape(), s_unknown.get_shape())

    np_input = np.random.randn(3, 3, 3)
    self._compareAll(np_input, reduction_axes, {c_unknown: np_input})

    # Input shape only has known rank.
    c_known_rank = array_ops.placeholder(dtypes.float32)
    c_known_rank.set_shape(tensor_shape.unknown_shape(ndims=3))
    s_known_rank = math_ops.reduce_sum(
        c_known_rank, reduction_axes, keep_dims=True)
    self.assertEqual(3, s_known_rank.get_shape().ndims)

    np_input = np.random.randn(3, 3, 3)
    self._compareAll(np_input, reduction_axes, {c_known_rank: np_input})

    # Reduction indices are unknown.
    unknown_indices = array_ops.placeholder(dtypes.int32)
    c_unknown_indices = constant_op.constant([[10.0], [20.0]])
    s_unknown_indices = math_ops.reduce_sum(
        c_unknown_indices, unknown_indices, keep_dims=False)
    self.assertEqual(tensor_shape.unknown_shape(),
                     s_unknown_indices.get_shape())
    s_unknown_indices_keep = math_ops.reduce_sum(
        c_unknown_indices, unknown_indices, keep_dims=True)
    self.assertEqual(2, s_unknown_indices_keep.get_shape().ndims)

  def testWrongShapeForReductionIndices(self):
    reduction_axes = [[1], [2]]
    c_unknown = array_ops.placeholder(dtypes.float32)
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             ".*must be at most rank 1.*"):
      math_ops.reduce_sum(c_unknown, reduction_axes)

  # Int64??

  def testGradient(self):
    for dtype in [dtypes.float32, dtypes.float64, dtypes.complex64,
                  dtypes.complex128]:
      x = self._makeIncremental([2, 3, 4, 2], dtype)
      self._compareGradientAxes(x)

  def testHighRank(self):
    # Do a bunch of random high dimensional reductions
    np.random.seed(42)
    for _ in range(20):
      rank = np.random.randint(4, 10 + 1)
      axes, = np.nonzero(np.random.randint(2, size=rank))
      shape = tuple(np.random.randint(1, 3 + 1, size=rank))
      data = np.random.randint(1024, size=shape)
      self._compareAll(data, axes)
    # Check some particular axis patterns
    for rank in 4, 7, 10:
      shape = tuple(np.random.randint(1, 3 + 1, size=rank))
      data = np.random.randint(1024, size=shape)
      for axes in ([], np.arange(rank), np.arange(0, rank, 2),
                   np.arange(1, rank, 2)):
        self._compareAll(data, axes)

  def testExpand(self):
    # Reduce an empty tensor to a nonempty tensor
    x = np.zeros((5, 0))
    self._compareAll(x, [1])

  def testEmptyGradients(self):
    with self.test_session(use_gpu=True):
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_sum(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)

  def testDegenerate(self):
    with self.test_session(use_gpu=True):
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.complex64, dtypes.complex128):
        # A large number is needed to get Eigen to die
        x = array_ops.zeros((0, 9938), dtype=dtype)
        y = math_ops.reduce_sum(x, [0])
        self.assertAllEqual(y.eval(), np.zeros(9938))


class MeanReductionTest(BaseReductionTest):

  def _tf_reduce(self, x, reduction_axes, keep_dims):
    return math_ops.reduce_mean(x, reduction_axes, keep_dims)

  def _np_reduce(self, x, reduction_axes, keep_dims):
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    elif isinstance(reduction_axes, numbers.Integral):
      reduction_axes = (reduction_axes,)

    if reduction_axes is None:
      count = np.prod(x.shape)
    else:
      count = np.prod([x.shape[ax] for ax in reduction_axes])
    # np.mean automatically converts integer inputs to float, while TensorFlow's
    # reduce_mean does not. For integer inputs, we emulate TensorFlow's behavior
    # using np.sum and truncating division.
    np_sum = np.sum(x, axis=reduction_axes, keepdims=keep_dims)
    if np.issubdtype(x.dtype, np.integer):
      return np_sum // count
    return np_sum / count

  def testInfinity(self):
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  def testInt32(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.int32)
      self._compareAllAxes(np_arr)

  def testFloat32(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
      self._compareAllAxes(np_arr)

  def testFloat64(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float64)
      self._compareAllAxes(np_arr)

  def testComplex64(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex64)
      self._compareAllAxes(np_arr)

  def testComplex128(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex128)
      self._compareAllAxes(np_arr)

  def testGradient(self):
    s = [2, 3, 4, 2]
    for dtype in [dtypes.float32, dtypes.float64]:
      x = self._makeIncremental(s, dtype)
      self._compareGradientAxes(x, rtol=1e-3, atol=1e-3)

  def testEmptyGradients(self):
    with self.test_session(use_gpu=True):
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_mean(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)

  def testDegenerate(self):
    with self.test_session(use_gpu=True):
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64):
        # A large number is needed to get Eigen to die
        x = array_ops.zeros((0, 9938), dtype=dtype)
        y = math_ops.reduce_mean(x, [0]).eval()
        self.assertEqual(y.shape, (9938,))
        self.assertTrue(np.all(np.isnan(y)))


class ProdReductionTest(BaseReductionTest):

  def _tf_reduce(self, x, reduction_axes, keep_dims):
    return math_ops.reduce_prod(x, reduction_axes, keep_dims)

  def _np_reduce(self, x, reduction_axes, keep_dims):
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    return np.prod(x, axis=reduction_axes, keepdims=keep_dims)

  def testInfinity(self):
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  def testInt32(self):
    # Numpy automatically upgrades the type of np.prod from int32 to int64, so
    # Numpy does not overflow an int32 np.prod while TensorFlow does. To avoid
    # overflow, divide the incremental int32 array by 2.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.int32) / 2
      self._compareAllAxes(np_arr)

  def testFloat32(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
      self._compareAllAxes(np_arr)

  def testFloat64(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float64)
      self._compareAllAxes(np_arr)

  def testComplex64(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex64)
      self._compareAllAxes(np_arr)

  def testComplex128(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex128)
      self._compareAllAxes(np_arr)

  def testGradientWithZeros(self):
    s = [2, 3, 4, 2]
    x = self._makeIncremental(s, dtypes.float32) / 20.
    # No zeros in input
    self._compareGradientAxes(x, rtol=1e-3, atol=1e-3)
    # Zero at beginning
    x1 = x.copy()
    x1[:, :, 0, :] = 0
    self._compareGradientAxes(x1, rtol=1e-3, atol=1e-3)
    # Zero at end
    x2 = x.copy()
    x2[:, :, -1, :] = 0
    self._compareGradientAxes(x2, rtol=1e-3, atol=1e-3)
    # Zero in middle
    x3 = x.copy()
    x3[:, :, 2, :] = 0
    self._compareGradientAxes(x3, rtol=1e-3, atol=1e-3)
    # All zeros
    x4 = x.copy()
    x4[:, :, :, :] = 0
    self._compareGradientAxes(x4, rtol=1e-3, atol=1e-3)

  def testEmptyGradients(self):
    with self.test_session(use_gpu=True):
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_prod(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)

  def testDegenerate(self):
    with self.test_session(use_gpu=True):
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64):
        # A large number is needed to get Eigen to die
        x = array_ops.zeros((0, 9938), dtype=dtype)
        y = math_ops.reduce_prod(x, [0])
        self.assertAllEqual(y.eval(), np.ones(9938))


class MinReductionTest(test.TestCase):

  def _compare(self, x, reduction_axes, keep_dims, use_gpu=False):
    np_ans = x
    if reduction_axes is None:
      np_ans = np.amin(np_ans, keepdims=keep_dims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.amin(np_ans, axis=ra, keepdims=keep_dims)
    with self.test_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_min(x, reduction_axes, keep_dims)
      out = tf_ans.eval()
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, False, use_gpu=False)
    self._compare(x, reduction_axes, True, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=False)

  def testInfinity(self):
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  def testFloatReduce3D(self):
    # Create a 3D array of floats and reduce across all possible
    # dimensions
    np_arr = np.arange(0, 30).reshape([2, 3, 5]).astype(np.float32)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testDoubleReduce3D(self):
    # Create a 3D array of doubles and reduce across all possible
    # dimensions
    np_arr = np.arange(0, 30).reshape([2, 3, 5]).astype(np.float64)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testGradient(self):
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.test_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_min(t, [1, 2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testGradient2(self):
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.test_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_min(t, [1])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 4, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testGradient3(self):
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.test_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_min(t, [2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 3, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testGradient4(self):
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.test_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_min(t)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [1], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testEmptyGradients(self):
    with self.test_session():
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_min(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)


class MaxReductionTest(test.TestCase):

  def _compare(self, x, reduction_axes, keep_dims, use_gpu=False):
    np_ans = x
    if reduction_axes is None:
      np_ans = np.amax(np_ans, keepdims=keep_dims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.amax(np_ans, axis=ra, keepdims=keep_dims)
    with self.test_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_max(x, reduction_axes, keep_dims)
      out = tf_ans.eval()
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, False, use_gpu=False)
    self._compare(x, reduction_axes, True, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=False)

  def testInfinity(self):
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  def testFloatReduce3D(self):
    # Create a 3D array of floats and reduce across all possible
    # dimensions
    np_arr = np.arange(0, 30).reshape([2, 3, 5]).astype(np.float32)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testDoubleReduce3D(self):
    # Create a 3D array of doubles and reduce across all possible
    # dimensions
    np_arr = np.arange(0, 30).reshape([2, 3, 5]).astype(np.float64)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testGradient(self):
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.test_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [1, 2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testGradient2(self):
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.test_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [1])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 4, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testGradient3(self):
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.test_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 3, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testGradient4(self):
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.test_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [1], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testEmptyGradients(self):
    with self.test_session():
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_max(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)


class AllReductionTest(test.TestCase):

  def _compare(self, x, reduction_axes, keep_dims, use_gpu=False):
    np_ans = x
    if reduction_axes is None:
      np_ans = np.all(np_ans, keepdims=keep_dims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.all(np_ans, axis=ra, keepdims=keep_dims)
    with self.test_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_all(x, reduction_axes, keep_dims)
      out = tf_ans.eval()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, False, use_gpu=False)
    self._compare(x, reduction_axes, True, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=False)

  def testAll3D(self):
    # Create a 3D array of bools and reduce across all possible
    # dimensions
    np_arr = (np.random.uniform(0, 1, 30) > 0.1).reshape([2, 3, 5])
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testEmpty(self):
    self._compareAll([], [0])


class AnyReductionTest(test.TestCase):

  def _compare(self, x, reduction_axes, keep_dims, use_gpu=False):
    np_ans = x
    if reduction_axes is None:
      np_ans = np.any(np_ans, keepdims=keep_dims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.any(np_ans, axis=ra, keepdims=keep_dims)
    with self.test_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_any(x, reduction_axes, keep_dims)
      out = tf_ans.eval()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, False, use_gpu=False)
    self._compare(x, reduction_axes, True, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=False)

  def testAll3D(self):
    # Create a 3D array of bools and reduce across all possible
    # dimensions
    np_arr = (np.random.uniform(0, 1, 30) > 0.9).reshape([2, 3, 5])
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testEmpty(self):
    self._compareAll([], [0])


class CountNonzeroReductionTest(test.TestCase):

  def _compare(self,
               x,
               reduction_axes,
               keep_dims,
               use_gpu=False,
               feed_dict=None):
    np_ans = (x != 0).astype(np.int32)
    if reduction_axes is None:
      np_ans = np.sum(np_ans, keepdims=keep_dims)
    else:
      reduction_axes = np.array(reduction_axes).astype(np.int32)
      for ra in reduction_axes.ravel()[::-1]:
        np_ans = np.sum(np_ans, axis=ra, keepdims=keep_dims)
    with self.test_session(use_gpu=use_gpu) as sess:
      tf_ans = math_ops.count_nonzero(x, reduction_axes, keep_dims)
      out = sess.run(tf_ans, feed_dict)
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes, feed_dict=None):
    if reduction_axes is not None and np.shape(reduction_axes) == (1,):
      # Test scalar reduction_axes argument
      self._compareAll(x, reduction_axes[0])
    self._compare(x, reduction_axes, False, use_gpu=True, feed_dict=feed_dict)
    self._compare(x, reduction_axes, False, use_gpu=False, feed_dict=feed_dict)
    self._compare(x, reduction_axes, True, use_gpu=True, feed_dict=feed_dict)
    self._compare(x, reduction_axes, True, use_gpu=False, feed_dict=feed_dict)

  def testBoolReduce1D(self):
    # Create a 1D array of floats
    np_arr = np.asarray([False, False, True, False, False, True])
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])

  def testFloatReduce1D(self):
    # Create a 1D array of floats
    np_arr = np.asarray([0.0, 1.0, -1.0, 0.0, 0.0, 3.0]).astype(np.float32)
    self._compareAll(np_arr, [0])

  def testFloatReduce4D(self):
    # Create a 4D array of floats and reduce across some
    # dimensions
    np_arr = np.floor(np.arange(0.0, 210.0) / 100.0).reshape(
        [2, 3, 5, 7]).astype(np.float32)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    # Need specialization for reduce(4D, [0, 2])
    # self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])
    self._compareAll(np_arr, [1, 2, 3])
    self._compareAll(np_arr, [0, 1, 2, 3])

  def testExpand(self):
    # Reduce an empty tensor to a nonempty tensor
    x = np.zeros((5, 0))
    self._compareAll(x, [1])

  def testDegenerate(self):
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        for dtype in (dtypes.bool,):
          # A large number is needed to get Eigen to die
          x = array_ops.zeros((0, 9938), dtype=dtype)
          y = math_ops.count_nonzero(x, [0])
          self.assertAllEqual(y.eval(), np.zeros(9938))


if __name__ == "__main__":
  test.main()
