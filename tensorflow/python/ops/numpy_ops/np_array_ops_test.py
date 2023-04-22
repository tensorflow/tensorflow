# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf numpy array methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import operator
import sys
from absl.testing import parameterized
import numpy as np
from six.moves import range
from six.moves import zip

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


_virtual_devices_ready = False


def set_up_virtual_devices():
  global _virtual_devices_ready
  if _virtual_devices_ready:
    return
  physical_devices = config.list_physical_devices('CPU')
  config.set_logical_device_configuration(
      physical_devices[0], [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration()
      ])
  _virtual_devices_ready = True


class ArrayCreationTest(test.TestCase):

  def setUp(self):
    super(ArrayCreationTest, self).setUp()
    set_up_virtual_devices()
    python_shapes = [
        0, 1, 2, (), (1,), (2,), (1, 2, 3), [], [1], [2], [1, 2, 3]
    ]
    self.shape_transforms = [
        lambda x: x, lambda x: np.array(x, dtype=int),
        lambda x: np_array_ops.array(x, dtype=int), tensor_shape.TensorShape
    ]

    self.all_shapes = []
    for fn in self.shape_transforms:
      self.all_shapes.extend([fn(s) for s in python_shapes])

    if sys.version_info.major == 3:
      # There is a bug of np.empty (and alike) in Python 3 causing a crash when
      # the `shape` argument is an np_arrays.ndarray scalar (or tf.Tensor
      # scalar).
      def not_ndarray_scalar(s):
        return not (isinstance(s, np_arrays.ndarray) and s.ndim == 0)

      self.all_shapes = list(filter(not_ndarray_scalar, self.all_shapes))

    self.all_types = [
        int, float, np.int16, np.int32, np.int64, np.float16, np.float32,
        np.float64
    ]

    source_array_data = [
        1,
        5.5,
        7,
        (),
        (8, 10.),
        ((), ()),
        ((1, 4), (2, 8)),
        [],
        [7],
        [8, 10.],
        [[], []],
        [[1, 4], [2, 8]],
        ([], []),
        ([1, 4], [2, 8]),
        [(), ()],
        [(1, 4), (2, 8)],
    ]

    self.array_transforms = [
        lambda x: x,
        ops.convert_to_tensor,
        np.array,
        np_array_ops.array,
    ]
    self.all_arrays = []
    for fn in self.array_transforms:
      self.all_arrays.extend([fn(s) for s in source_array_data])

  def testEmpty(self):
    for s in self.all_shapes:
      actual = np_array_ops.empty(s)
      expected = np.empty(s)
      msg = 'shape: {}'.format(s)
      self.match_shape(actual, expected, msg)
      self.match_dtype(actual, expected, msg)

    for s, t in itertools.product(self.all_shapes, self.all_types):
      actual = np_array_ops.empty(s, t)
      expected = np.empty(s, t)
      msg = 'shape: {}, dtype: {}'.format(s, t)
      self.match_shape(actual, expected, msg)
      self.match_dtype(actual, expected, msg)

  def testEmptyLike(self):
    for a in self.all_arrays:
      actual = np_array_ops.empty_like(a)
      expected = np.empty_like(a)
      msg = 'array: {}'.format(a)
      self.match_shape(actual, expected, msg)
      self.match_dtype(actual, expected, msg)

    for a, t in itertools.product(self.all_arrays, self.all_types):
      actual = np_array_ops.empty_like(a, t)
      expected = np.empty_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match_shape(actual, expected, msg)
      self.match_dtype(actual, expected, msg)

  def testZeros(self):
    for s in self.all_shapes:
      actual = np_array_ops.zeros(s)
      expected = np.zeros(s)
      msg = 'shape: {}'.format(s)
      self.match(actual, expected, msg)

    for s, t in itertools.product(self.all_shapes, self.all_types):
      actual = np_array_ops.zeros(s, t)
      expected = np.zeros(s, t)
      msg = 'shape: {}, dtype: {}'.format(s, t)
      self.match(actual, expected, msg)

  def testZerosLike(self):
    for a in self.all_arrays:
      actual = np_array_ops.zeros_like(a)
      expected = np.zeros_like(a)
      msg = 'array: {}'.format(a)
      self.match(actual, expected, msg)

    for a, t in itertools.product(self.all_arrays, self.all_types):
      actual = np_array_ops.zeros_like(a, t)
      expected = np.zeros_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match(actual, expected, msg)

  def testOnes(self):
    for s in self.all_shapes:
      actual = np_array_ops.ones(s)
      expected = np.ones(s)
      msg = 'shape: {}'.format(s)
      self.match(actual, expected, msg)

    for s, t in itertools.product(self.all_shapes, self.all_types):
      actual = np_array_ops.ones(s, t)
      expected = np.ones(s, t)
      msg = 'shape: {}, dtype: {}'.format(s, t)
      self.match(actual, expected, msg)

  def testOnesLike(self):
    for a in self.all_arrays:
      actual = np_array_ops.ones_like(a)
      expected = np.ones_like(a)
      msg = 'array: {}'.format(a)
      self.match(actual, expected, msg)

    for a, t in itertools.product(self.all_arrays, self.all_types):
      actual = np_array_ops.ones_like(a, t)
      expected = np.ones_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match(actual, expected, msg)

  def testEye(self):
    n_max = 3
    m_max = 3

    for n in range(1, n_max + 1):
      self.match(np_array_ops.eye(n), np.eye(n))
      for k in range(-n, n + 1):
        self.match(np_array_ops.eye(n, k=k), np.eye(n, k=k))
      for m in range(1, m_max + 1):
        self.match(np_array_ops.eye(n, m), np.eye(n, m))
        for k in range(-n, m):
          self.match(np_array_ops.eye(n, k=k), np.eye(n, k=k))
          self.match(np_array_ops.eye(n, m, k), np.eye(n, m, k))

    for dtype in self.all_types:
      for n in range(1, n_max + 1):
        self.match(np_array_ops.eye(n, dtype=dtype), np.eye(n, dtype=dtype))
        for k in range(-n, n + 1):
          self.match(
              np_array_ops.eye(n, k=k, dtype=dtype),
              np.eye(n, k=k, dtype=dtype))
        for m in range(1, m_max + 1):
          self.match(
              np_array_ops.eye(n, m, dtype=dtype), np.eye(n, m, dtype=dtype))
          for k in range(-n, m):
            self.match(
                np_array_ops.eye(n, k=k, dtype=dtype),
                np.eye(n, k=k, dtype=dtype))
            self.match(
                np_array_ops.eye(n, m, k, dtype=dtype),
                np.eye(n, m, k, dtype=dtype))

  def testIdentity(self):
    n_max = 3

    for n in range(1, n_max + 1):
      self.match(np_array_ops.identity(n), np.identity(n))

    for dtype in self.all_types:
      for n in range(1, n_max + 1):
        self.match(
            np_array_ops.identity(n, dtype=dtype), np.identity(n, dtype=dtype))

  def testFull(self):
    # List of 2-tuples of fill value and shape.
    data = [
        (5, ()),
        (5, (7,)),
        (5., (7,)),
        ([5, 8], (2,)),
        ([5, 8], (3, 2)),
        ([[5], [8]], (2, 3)),
        ([[5], [8]], (3, 2, 5)),
        ([[5.], [8.]], (3, 2, 5)),
        ([[3, 4], [5, 6], [7, 8]], (3, 3, 2)),
    ]
    for f, s in data:
      for fn1, fn2 in itertools.product(self.array_transforms,
                                        self.shape_transforms):
        fill_value = fn1(f)
        shape = fn2(s)
        self.match(
            np_array_ops.full(shape, fill_value), np.full(shape, fill_value))
        for dtype in self.all_types:
          self.match(
              np_array_ops.full(shape, fill_value, dtype=dtype),
              np.full(shape, fill_value, dtype=dtype))

  def testFullLike(self):
    # List of 2-tuples of fill value and shape.
    data = [
        (5, ()),
        (5, (7,)),
        (5., (7,)),
        ([5, 8], (2,)),
        ([5, 8], (3, 2)),
        ([[5], [8]], (2, 3)),
        ([[5], [8]], (3, 2, 5)),
        ([[5.], [8.]], (3, 2, 5)),
    ]
    zeros_builders = [np_array_ops.zeros, np.zeros]
    for f, s in data:
      for fn1, fn2, arr_dtype in itertools.product(self.array_transforms,
                                                   zeros_builders,
                                                   self.all_types):
        fill_value = fn1(f)
        arr = fn2(s, arr_dtype)
        self.match(
            np_array_ops.full_like(arr, fill_value),
            np.full_like(arr, fill_value))
        for dtype in self.all_types:
          self.match(
              np_array_ops.full_like(arr, fill_value, dtype=dtype),
              np.full_like(arr, fill_value, dtype=dtype))

  def testArray(self):
    ndmins = [0, 1, 2, 5]
    for a, dtype, ndmin, copy in itertools.product(self.all_arrays,
                                                   self.all_types, ndmins,
                                                   [True, False]):
      self.match(
          np_array_ops.array(a, dtype=dtype, ndmin=ndmin, copy=copy),
          np.array(a, dtype=dtype, ndmin=ndmin, copy=copy))

    zeros_list = np_array_ops.zeros(5)

    def test_copy_equal_false():
      # Backing tensor is the same if copy=False, other attributes being None.
      self.assertIs(np_array_ops.array(zeros_list, copy=False), zeros_list)
      self.assertIs(np_array_ops.array(zeros_list, copy=False), zeros_list)

      # Backing tensor is different if ndmin is not satisfied.
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, ndmin=2),
          zeros_list)
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, ndmin=2),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, ndmin=1),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, ndmin=1),
          zeros_list)

      # Backing tensor is different if dtype is not satisfied.
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, dtype=int),
          zeros_list)
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, dtype=int),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, dtype=float),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, dtype=float),
          zeros_list)

    test_copy_equal_false()
    with ops.device('CPU:1'):
      test_copy_equal_false()

    self.assertNotIn('CPU:1', zeros_list.backing_device)
    with ops.device('CPU:1'):
      self.assertIn(
          'CPU:1', np_array_ops.array(zeros_list, copy=True).backing_device)
      self.assertIn(
          'CPU:1', np_array_ops.array(np.array(0), copy=True).backing_device)

  def testAsArray(self):
    for a, dtype in itertools.product(self.all_arrays, self.all_types):
      self.match(
          np_array_ops.asarray(a, dtype=dtype), np.asarray(a, dtype=dtype))

    zeros_list = np_array_ops.zeros(5)
    # Same instance is returned if no dtype is specified and input is ndarray.
    self.assertIs(np_array_ops.asarray(zeros_list), zeros_list)
    with ops.device('CPU:1'):
      self.assertIs(np_array_ops.asarray(zeros_list), zeros_list)
    # Different instance is returned if dtype is specified and input is ndarray.
    self.assertIsNot(np_array_ops.asarray(zeros_list, dtype=int), zeros_list)

  def testAsAnyArray(self):
    for a, dtype in itertools.product(self.all_arrays, self.all_types):
      self.match(
          np_array_ops.asanyarray(a, dtype=dtype),
          np.asanyarray(a, dtype=dtype))
    zeros_list = np_array_ops.zeros(5)
    # Same instance is returned if no dtype is specified and input is ndarray.
    self.assertIs(np_array_ops.asanyarray(zeros_list), zeros_list)
    with ops.device('CPU:1'):
      self.assertIs(np_array_ops.asanyarray(zeros_list), zeros_list)
    # Different instance is returned if dtype is specified and input is ndarray.
    self.assertIsNot(np_array_ops.asanyarray(zeros_list, dtype=int), zeros_list)

  def testAsContiguousArray(self):
    for a, dtype in itertools.product(self.all_arrays, self.all_types):
      self.match(
          np_array_ops.ascontiguousarray(a, dtype=dtype),
          np.ascontiguousarray(a, dtype=dtype))

  def testARange(self):
    int_values = np.arange(-3, 3).tolist()
    float_values = np.arange(-3.5, 3.5).tolist()
    all_values = int_values + float_values
    for dtype in self.all_types:
      for start in all_values:
        msg = 'dtype:{} start:{}'.format(dtype, start)
        self.match(np_array_ops.arange(start), np.arange(start), msg=msg)
        self.match(
            np_array_ops.arange(start, dtype=dtype),
            np.arange(start, dtype=dtype),
            msg=msg)
        for stop in all_values:
          msg = 'dtype:{} start:{} stop:{}'.format(dtype, start, stop)
          self.match(
              np_array_ops.arange(start, stop), np.arange(start, stop), msg=msg)
          # TODO(srbs): Investigate and remove check.
          # There are some bugs when start or stop is float and dtype is int.
          if not isinstance(start, float) and not isinstance(stop, float):
            self.match(
                np_array_ops.arange(start, stop, dtype=dtype),
                np.arange(start, stop, dtype=dtype),
                msg=msg)
          # Note: We intentionally do not test with float values for step
          # because numpy.arange itself returns inconsistent results. e.g.
          # np.arange(0.5, 3, step=0.5, dtype=int) returns
          # array([0, 1, 2, 3, 4])
          for step in int_values:
            msg = 'dtype:{} start:{} stop:{} step:{}'.format(
                dtype, start, stop, step)
            if not step:
              with self.assertRaises(ValueError):
                self.match(
                    np_array_ops.arange(start, stop, step),
                    np.arange(start, stop, step),
                    msg=msg)
                if not isinstance(start, float) and not isinstance(stop, float):
                  self.match(
                      np_array_ops.arange(start, stop, step, dtype=dtype),
                      np.arange(start, stop, step, dtype=dtype),
                      msg=msg)
            else:
              self.match(
                  np_array_ops.arange(start, stop, step),
                  np.arange(start, stop, step),
                  msg=msg)
              if not isinstance(start, float) and not isinstance(stop, float):
                self.match(
                    np_array_ops.arange(start, stop, step, dtype=dtype),
                    np.arange(start, stop, step, dtype=dtype),
                    msg=msg)

  def testDiag(self):
    array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64)
    ]

    def run_test(arr):
      for fn in array_transforms:
        arr = fn(arr)
        self.match(
            np_array_ops.diag(arr), np.diag(arr), msg='diag({})'.format(arr))
        for k in range(-3, 3):
          self.match(
              np_array_ops.diag(arr, k),
              np.diag(arr, k),
              msg='diag({}, k={})'.format(arr, k))

    # 2-d arrays.
    run_test(np.arange(9).reshape((3, 3)).tolist())
    run_test(np.arange(6).reshape((2, 3)).tolist())
    run_test(np.arange(6).reshape((3, 2)).tolist())
    run_test(np.arange(3).reshape((1, 3)).tolist())
    run_test(np.arange(3).reshape((3, 1)).tolist())
    run_test([[5]])
    run_test([[]])
    run_test([[], []])

    # 1-d arrays.
    run_test([])
    run_test([1])
    run_test([1, 2])

  def testDiagFlat(self):
    array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64)
    ]

    def run_test(arr):
      for fn in array_transforms:
        arr = fn(arr)
        self.match(
            np_array_ops.diagflat(arr),
            np.diagflat(arr),
            msg='diagflat({})'.format(arr))
        for k in range(-3, 3):
          self.match(
              np_array_ops.diagflat(arr, k),
              np.diagflat(arr, k),
              msg='diagflat({}, k={})'.format(arr, k))

    # 1-d arrays.
    run_test([])
    run_test([1])
    run_test([1, 2])
    # 2-d arrays.
    run_test([[]])
    run_test([[5]])
    run_test([[], []])
    run_test(np.arange(4).reshape((2, 2)).tolist())
    run_test(np.arange(2).reshape((2, 1)).tolist())
    run_test(np.arange(2).reshape((1, 2)).tolist())
    # 3-d arrays
    run_test(np.arange(8).reshape((2, 2, 2)).tolist())

  def match_shape(self, actual, expected, msg=None):
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype(self, actual, expected, msg=None):
    if msg:
      msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
          msg, expected.dtype, actual.dtype)
    self.assertEqual(actual.dtype, expected.dtype, msg=msg)

  def match(self, actual, expected, msg=None, almost=False, decimal=7):
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, np_arrays.ndarray)
    self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not almost:
      if not actual.shape.rank:
        self.assertEqual(actual.tolist(), expected.tolist())
      else:
        self.assertSequenceEqual(actual.tolist(), expected.tolist())
    else:
      np.testing.assert_almost_equal(
          actual.tolist(), expected.tolist(), decimal=decimal)

  def testIndexedSlices(self):
    dtype = dtypes.int64
    iss = indexed_slices.IndexedSlices(
        values=np_array_ops.ones([2, 3], dtype=dtype),
        indices=constant_op.constant([1, 9]),
        dense_shape=[10, 3])
    a = np_array_ops.array(iss, copy=False)
    expected = array_ops.scatter_nd([[1], [9]],
                                    array_ops.ones([2, 3], dtype=dtype),
                                    [10, 3])
    self.assertAllEqual(expected, a)


class ArrayMethodsTest(test.TestCase):

  def setUp(self):
    super(ArrayMethodsTest, self).setUp()
    set_up_virtual_devices()
    self.array_transforms = [
        lambda x: x,
        ops.convert_to_tensor,
        np.array,
        np_array_ops.array,
    ]

  def testAllAny(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arr = fn(arr)
        self.match(
            np_array_ops.all(arr, *args, **kwargs),
            np.all(arr, *args, **kwargs))
        self.match(
            np_array_ops.any(arr, *args, **kwargs),
            np.any(arr, *args, **kwargs))

    run_test(0)
    run_test(1)
    run_test([])
    run_test([[True, False], [True, True]])
    run_test([[True, False], [True, True]], axis=0)
    run_test([[True, False], [True, True]], axis=0, keepdims=True)
    run_test([[True, False], [True, True]], axis=1)
    run_test([[True, False], [True, True]], axis=1, keepdims=True)
    run_test([[True, False], [True, True]], axis=(0, 1))
    run_test([[True, False], [True, True]], axis=(0, 1), keepdims=True)
    run_test([5.2, 3.5], axis=0)
    run_test([1, 0], axis=0)

  def testCompress(self):

    def run_test(condition, arr, *args, **kwargs):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arg1 = fn1(condition)
          arg2 = fn2(arr)
          self.match(
              np_array_ops.compress(arg1, arg2, *args, **kwargs),
              np.compress(
                  np.asarray(arg1).astype(np.bool), arg2, *args, **kwargs))

    run_test([True], 5)
    run_test([False], 5)
    run_test([], 5)
    run_test([True, False, True], [1, 2, 3])
    run_test([True, False], [1, 2, 3])
    run_test([False, True], [[1, 2], [3, 4]])
    run_test([1, 0, 1], [1, 2, 3])
    run_test([1, 0], [1, 2, 3])
    run_test([0, 1], [[1, 2], [3, 4]])
    run_test([True], [[1, 2], [3, 4]])
    run_test([False, True], [[1, 2], [3, 4]], axis=1)
    run_test([False, True], [[1, 2], [3, 4]], axis=0)
    run_test([False, True], [[1, 2], [3, 4]], axis=-1)
    run_test([False, True], [[1, 2], [3, 4]], axis=-2)

  def testCopy(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.copy(arg, *args, **kwargs),
            np.copy(arg, *args, **kwargs))

    run_test([])
    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([True])
    run_test(np.arange(9).reshape((3, 3)).tolist())

    a = np_array_ops.asarray(0)
    self.assertNotIn('CPU:1', a.backing_device)
    with ops.device('CPU:1'):
      self.assertIn('CPU:1', np_array_ops.array(a, copy=True)
                    .backing_device)
      self.assertIn('CPU:1', np_array_ops.array(np.array(0), copy=True)
                    .backing_device)

  def testCumProdAndSum(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.cumprod(arg, *args, **kwargs),
            np.cumprod(arg, *args, **kwargs))
        self.match(
            np_array_ops.cumsum(arg, *args, **kwargs),
            np.cumsum(arg, *args, **kwargs))

    run_test([])
    run_test([1, 2, 3])
    run_test([1, 2, 3], dtype=float)
    run_test([1, 2, 3], dtype=np.float32)
    run_test([1, 2, 3], dtype=np.float64)
    run_test([1., 2., 3.])
    run_test([1., 2., 3.], dtype=int)
    run_test([1., 2., 3.], dtype=np.int32)
    run_test([1., 2., 3.], dtype=np.int64)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)

  def testImag(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.imag(arg, *args, **kwargs),
            # np.imag may return a scalar so we convert to a np.ndarray.
            np.array(np.imag(arg, *args, **kwargs)))

    run_test(1)
    run_test(5.5)
    run_test(5 + 3j)
    run_test(3j)
    run_test([])
    run_test([1, 2, 3])
    run_test([1 + 5j, 2 + 3j])
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]])

  def testAMaxAMin(self):

    def run_test(arr, *args, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              np_array_ops.amax(arr_arg, axis=axis_arg, *args, **kwargs),
              np.amax(arr_arg, axis=axis, *args, **kwargs))
          self.match(
              np_array_ops.amin(arr_arg, axis=axis_arg, *args, **kwargs),
              np.amin(arr_arg, axis=axis, *args, **kwargs))

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)
    self.assertRaises(ValueError, np_array_ops.amax, np.ones([2, 2]), out=[])
    self.assertRaises(ValueError, np_array_ops.amin, np.ones([2, 2]), out=[])

  def testMean(self):

    def run_test(arr, *args, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              np_array_ops.mean(arr_arg, axis=axis_arg, *args, **kwargs),
              np.mean(arr_arg, axis=axis, *args, **kwargs))

    run_test([1, 2, 1])
    run_test([1., 2., 1.])
    run_test([1., 2., 1.], dtype=int)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)
    self.assertRaises(ValueError, np_array_ops.mean, np.ones([2, 2]), out=[])

  def testProd(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.prod(arg, *args, **kwargs),
            np.prod(arg, *args, **kwargs))

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test(np.array([1, 2, 3], dtype=np.int16))
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)

  def _testReduce(self, math_fun, np_fun, name):
    axis_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64),
    ]

    def run_test(a, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in axis_transforms:
          arg1 = fn1(a)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              math_fun(arg1, axis=axis_arg, **kwargs),
              np_fun(arg1, axis=axis, **kwargs),
              msg='{}({}, axis={}, keepdims={})'.format(name, arg1, axis,
                                                        kwargs.get('keepdims')))

    run_test(5)
    run_test([2, 3])
    run_test([[2, -3], [-6, 7]])
    run_test([[2, -3], [-6, 7]], axis=0)
    run_test([[2, -3], [-6, 7]], axis=0, keepdims=True)
    run_test([[2, -3], [-6, 7]], axis=1)
    run_test([[2, -3], [-6, 7]], axis=1, keepdims=True)
    run_test([[2, -3], [-6, 7]], axis=(0, 1))
    run_test([[2, -3], [-6, 7]], axis=(1, 0))

  def testSum(self):
    self._testReduce(np_array_ops.sum, np.sum, 'sum')

  def testAmax(self):
    self._testReduce(np_array_ops.amax, np.amax, 'amax')

  def testSize(self):

    def run_test(arr, axis=None):
      onp_arr = np.array(arr)
      self.assertEqual(np_array_ops.size(arr, axis), np.size(onp_arr, axis))

    run_test(np_array_ops.array([1]))
    run_test(np_array_ops.array([1, 2, 3, 4, 5]))
    run_test(np_array_ops.ones((2, 3, 2)))
    run_test(np_array_ops.ones((3, 2)))
    run_test(np_array_ops.zeros((5, 6, 7)))
    run_test(1)
    run_test(np_array_ops.ones((3, 2, 1)))
    run_test(constant_op.constant(5))
    run_test(constant_op.constant([1, 1, 1]))
    self.assertRaises(NotImplementedError, np_array_ops.size, np.ones((2, 2)),
                      1)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(dtype=dtypes.float64, shape=None)])
    def f(arr):
      arr = np_array_ops.asarray(arr)
      return np_array_ops.size(arr)

    self.assertEqual(f(np_array_ops.ones((3, 2))).numpy(), 6)

  def testRavel(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.ravel(arg, *args, **kwargs),
            np.ravel(arg, *args, **kwargs))

    run_test(5)
    run_test(5.)
    run_test([])
    run_test([[]])
    run_test([[], []])
    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]])
    run_test(np.arange(8).reshape((2, 2, 2)).tolist())

  def testReal(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.real(arg, *args, **kwargs),
            np.array(np.real(arg, *args, **kwargs)))

    run_test(1)
    run_test(5.5)
    run_test(5 + 3j)
    run_test(3j)
    run_test([])
    run_test([1, 2, 3])
    run_test([1 + 5j, 2 + 3j])
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]])

  def testRepeat(self):

    def run_test(arr, repeats, *args, **kwargs):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          repeats_arg = fn2(repeats)
          self.match(
              np_array_ops.repeat(arr_arg, repeats_arg, *args, **kwargs),
              np.repeat(arr_arg, repeats_arg, *args, **kwargs))

    run_test(1, 2)
    run_test([1, 2], 2)
    run_test([1, 2], [2])
    run_test([1, 2], [1, 2])
    run_test([[1, 2], [3, 4]], 3, axis=0)
    run_test([[1, 2], [3, 4]], 3, axis=1)
    run_test([[1, 2], [3, 4]], [3], axis=0)
    run_test([[1, 2], [3, 4]], [3], axis=1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=0)
    run_test([[1, 2], [3, 4]], [3, 2], axis=1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=-1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=-2)

  def testAround(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.around(arg, *args, **kwargs),
            np.around(arg, *args, **kwargs))

    run_test(5.5)
    run_test(5.567, decimals=2)
    run_test([])
    run_test([1.27, 2.49, 2.75], decimals=1)
    run_test([23.6, 45.1], decimals=-1)

  def testReshape(self):

    def run_test(arr, newshape, *args, **kwargs):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          newshape_arg = fn2(newshape)
          self.match(
              np_array_ops.reshape(arr_arg, newshape_arg, *args, **kwargs),
              np.reshape(arr_arg, newshape, *args, **kwargs))

    run_test(5, [-1])
    run_test([], [-1])
    run_test([1, 2, 3], [1, 3])
    run_test([1, 2, 3], [3, 1])
    run_test([1, 2, 3, 4], [2, 2])
    run_test([1, 2, 3, 4], [2, 1, 2])

  def testExpandDims(self):

    def run_test(arr, axis):
      self.match(np_array_ops.expand_dims(arr, axis), np.expand_dims(arr, axis))

    run_test([1, 2, 3], 0)
    run_test([1, 2, 3], 1)

  def testSqueeze(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        # Note: np.squeeze ignores the axis arg for non-ndarray objects.
        # This looks like a bug: https://github.com/numpy/numpy/issues/8201
        # So we convert the arg to np.ndarray before passing to np.squeeze.
        self.match(
            np_array_ops.squeeze(arg, *args, **kwargs),
            np.squeeze(np.array(arg), *args, **kwargs))

    run_test(5)
    run_test([])
    run_test([5])
    run_test([[1, 2, 3]])
    run_test([[[1], [2], [3]]])
    run_test([[[1], [2], [3]]], axis=0)
    run_test([[[1], [2], [3]]], axis=2)
    run_test([[[1], [2], [3]]], axis=(0, 2))
    run_test([[[1], [2], [3]]], axis=-1)
    run_test([[[1], [2], [3]]], axis=-3)

  def testTranspose(self):

    def run_test(arr, axes=None):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axes_arg = fn2(axes) if axes is not None else None
          self.match(
              np_array_ops.transpose(arr_arg, axes_arg),
              np.transpose(arr_arg, axes))

    run_test(5)
    run_test([])
    run_test([5])
    run_test([5, 6, 7])
    run_test(np.arange(30).reshape(2, 3, 5).tolist())
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [0, 1, 2])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [0, 2, 1])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [1, 0, 2])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [1, 2, 0])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [2, 0, 1])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [2, 1, 0])

  def match_shape(self, actual, expected, msg=None):
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype(self, actual, expected, msg=None):
    if msg:
      msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
          msg, expected.dtype, actual.dtype)
    self.assertEqual(actual.dtype, expected.dtype, msg=msg)

  def match(self, actual, expected, msg=None, check_dtype=True):
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, np_arrays.ndarray)
    if check_dtype:
      self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape.rank:
      self.assertAllClose(actual.tolist(), expected.tolist())
    else:
      self.assertAllClose(actual.tolist(), expected.tolist())

  def testPad(self):
    t = [[1, 2, 3], [4, 5, 6]]
    paddings = [[
        1,
        1,
    ], [2, 2]]
    self.assertAllEqual(
        np_array_ops.pad(t, paddings, 'constant'),
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 0, 0], [0, 0, 4, 5, 6, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])

    self.assertAllEqual(
        np_array_ops.pad(t, paddings, 'reflect'),
        [[6, 5, 4, 5, 6, 5, 4], [3, 2, 1, 2, 3, 2, 1], [6, 5, 4, 5, 6, 5, 4],
         [3, 2, 1, 2, 3, 2, 1]])

    self.assertAllEqual(
        np_array_ops.pad(t, paddings, 'symmetric'),
        [[2, 1, 1, 2, 3, 3, 2], [2, 1, 1, 2, 3, 3, 2], [5, 4, 4, 5, 6, 6, 5],
         [5, 4, 4, 5, 6, 6, 5]])

  def testTake(self):
    a = [4, 3, 5, 7, 6, 8]
    indices = [0, 1, 4]
    self.assertAllEqual([4, 3, 6], np_array_ops.take(a, indices))
    indices = [[0, 1], [2, 3]]
    self.assertAllEqual([[4, 3], [5, 7]], np_array_ops.take(a, indices))
    a = [[4, 3, 5], [7, 6, 8]]
    self.assertAllEqual([[4, 3], [5, 7]], np_array_ops.take(a, indices))
    a = np.random.rand(2, 16, 3)
    axis = 1
    self.assertAllEqual(
        np.take(a, indices, axis=axis),
        np_array_ops.take(a, indices, axis=axis))

  def testWhere(self):
    self.assertAllEqual([[1.0, 1.0], [1.0, 1.0]],
                        np_array_ops.where([True], [1.0, 1.0],
                                           [[0, 0], [0, 0]]))

  def testShape(self):
    self.assertAllEqual((1, 2), np_array_ops.shape([[0, 0]]))

  def testSwapaxes(self):
    x = [[1, 2, 3]]
    self.assertAllEqual([[1], [2], [3]], np_array_ops.swapaxes(x, 0, 1))
    self.assertAllEqual([[1], [2], [3]], np_array_ops.swapaxes(x, -2, -1))
    x = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    self.assertAllEqual([[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
                        np_array_ops.swapaxes(x, 0, 2))
    self.assertAllEqual([[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
                        np_array_ops.swapaxes(x, -3, -1))

  def testMoveaxis(self):

    def _test(*args):
      # pylint: disable=no-value-for-parameter
      expected = np.moveaxis(*args)
      raw_ans = np_array_ops.moveaxis(*args)

      self.assertAllEqual(expected, raw_ans)

    a = np.random.rand(1, 2, 3, 4, 5, 6)

    # Basic
    _test(a, (0, 2), (3, 5))
    _test(a, (0, 2), (-1, -3))
    _test(a, (-6, -4), (3, 5))
    _test(a, (-6, -4), (-1, -3))
    _test(a, 0, 4)
    _test(a, -6, -2)
    _test(a, tuple(range(6)), tuple(range(6)))
    _test(a, tuple(range(6)), tuple(reversed(range(6))))
    _test(a, (), ())

  def testNdim(self):
    self.assertAllEqual(0, np_array_ops.ndim(0.5))
    self.assertAllEqual(1, np_array_ops.ndim([1, 2]))

  def testIsscalar(self):
    self.assertTrue(np_array_ops.isscalar(0.5))
    self.assertTrue(np_array_ops.isscalar(5))
    self.assertTrue(np_array_ops.isscalar(False))
    self.assertFalse(np_array_ops.isscalar([1, 2]))

  def assertListEqual(self, a, b):
    self.assertAllEqual(len(a), len(b))
    for x, y in zip(a, b):
      self.assertAllEqual(x, y)

  def testSplit(self):
    x = np_array_ops.arange(9)
    y = np_array_ops.split(x, 3)
    self.assertListEqual([([0, 1, 2]), ([3, 4, 5]), ([6, 7, 8])], y)

    x = np_array_ops.arange(8)
    y = np_array_ops.split(x, [3, 5, 6, 10])
    self.assertListEqual([([0, 1, 2]), ([3, 4]), ([5]), ([6, 7]), ([])], y)

  def testSign(self):
    state = np.random.RandomState(0)
    test_types = [np.float16, np.float32, np.float64, np.int32, np.int64,
                  np.complex64, np.complex128]
    test_shapes = [(), (1,), (2, 3, 4), (2, 3, 0, 4)]

    for dtype in test_types:
      for shape in test_shapes:
        if np.issubdtype(dtype, np.complex):
          arr = (np.asarray(state.randn(*shape) * 100, dtype=dtype) +
                 1j * np.asarray(state.randn(*shape) * 100, dtype=dtype))
        else:
          arr = np.asarray(state.randn(*shape) * 100, dtype=dtype)
        self.match(np_array_ops.sign(arr), np.sign(arr))


class ArrayManipulationTest(test.TestCase):

  def setUp(self):
    super(ArrayManipulationTest, self).setUp()
    self.array_transforms = [
        lambda x: x,
        ops.convert_to_tensor,
        np.array,
        np_array_ops.array,
    ]

  def testBroadcastTo(self):

    def run_test(arr, shape):
      for fn in self.array_transforms:
        arg1 = fn(arr)
        self.match(
            np_array_ops.broadcast_to(arg1, shape),
            np.broadcast_to(arg1, shape))

    run_test(1, 2)
    run_test(1, (2, 2))
    run_test([1, 2], (2, 2))
    run_test([[1], [2]], (2, 2))
    run_test([[1, 2]], (3, 2))
    run_test([[[1, 2]], [[3, 4]], [[5, 6]]], (3, 4, 2))

  def testIx_(self):
    possible_arys = [[True, True], [True, False], [False, False],
                     list(range(5)), np_array_ops.empty(0, dtype=np.int64)]
    for r in range(len(possible_arys)):
      for arys in itertools.combinations_with_replacement(possible_arys, r):
        tnp_ans = np_array_ops.ix_(*arys)
        onp_ans = np.ix_(*arys)
        for t, o in zip(tnp_ans, onp_ans):
          self.match(t, o)

  def match_shape(self, actual, expected, msg=None):
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype(self, actual, expected, msg=None):
    if msg:
      msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
          msg, expected.dtype, actual.dtype)
    self.assertEqual(actual.dtype, expected.dtype, msg=msg)

  def match(self, actual, expected, msg=None):
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, np_arrays.ndarray)
    self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape.rank:
      self.assertEqual(actual.tolist(), expected.tolist())
    else:
      self.assertSequenceEqual(actual.tolist(), expected.tolist())


class ArrayMathTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('complex_mul_1', 2j, [2], int, [4j], operator.mul),
      ('complex_mul_2', 2j, [0], int, [0], operator.mul),
      ('complex_mul_3', 2j, [-2.0], float, [-4j], operator.mul),
      ('complex_mul_4', 2j, [2j], complex, [-4], operator.mul),
      ('float_mul_1', 2.0, [2], int, [4], operator.mul),
      ('float_mul_2', 2.0, [0], int, [0], operator.mul),
      ('float_mul_3', 2.0, [-2.0], float, [-4], operator.mul),
      ('float_mul_4', 2.0, [2j], complex, [4j], operator.mul))
  def testConstantBinOp(self, a, b, b_type, expected_result, test_func):
    b = np_array_ops.array(b, dtype=b_type)
    result = test_func(a, b)
    if np.issubdtype(result.dtype.as_numpy_dtype, np.inexact):
      self.assertAllClose(result, expected_result)
    else:
      self.assertAllEqual(result, expected_result)

if __name__ == '__main__':
  ops.enable_eager_execution()
  ops.enable_numpy_style_type_promotion()
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
