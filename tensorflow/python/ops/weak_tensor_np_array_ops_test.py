# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf numpy array methods on WeakTensor."""

import itertools
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


_virtual_devices_ready = False
_get_weak_tensor = weak_tensor_test_util.get_weak_tensor

_all_types = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.complex128,
]

_NP_TO_TF = dtypes._NP_TO_TF

_NP_to_TF_result_inferred_types = {
    np.dtype(np.int32): dtypes.int32,
    np.dtype(np.int64): dtypes.int32,
    np.dtype(np.float32): dtypes.float32,
    np.dtype(np.float64): dtypes.float32,
    np.dtype(np.complex128): dtypes.complex128,
    dtypes.int32: dtypes.int32,
    dtypes.int64: dtypes.int32,
    dtypes.float32: dtypes.float32,
    dtypes.float64: dtypes.float32,
    dtypes.complex128: dtypes.complex128,
}


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
        np_array_ops.array,
        _get_weak_tensor,
    ]
    self.all_arrays = []
    for fn in self.array_transforms:
      self.all_arrays.extend([fn(s) for s in source_array_data])

  def testEmptyLikeOnWeakInputs(self):
    for a in self.all_arrays:
      expected = np.empty_like(a)
      actual = np_array_ops.empty_like(a)
      msg = 'array: {}'.format(a)
      self.match_shape(actual, expected, msg)
      self.match_dtype_and_type(
          actual,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
          msg,
      )

    for a, t in itertools.product(self.all_arrays, _all_types):
      actual = np_array_ops.empty_like(a, t)
      expected = np.empty_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match_shape(actual, expected, msg)
      # empty_like returns a Tensor if dtype is specified.
      self.match_dtype_and_type(actual, expected.dtype, tensor.Tensor, msg)

  def testZerosLikeOnWeakInputs(self):
    for a in self.all_arrays:
      actual = np_array_ops.zeros_like(a)
      expected = np.zeros_like(a)
      msg = 'array: {}'.format(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
          msg,
      )

    for a, t in itertools.product(self.all_arrays, _all_types):
      actual = np_array_ops.zeros_like(a, t)
      expected = np.zeros_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match_expected_attrs(
          actual, expected, expected.dtype, tensor.Tensor, msg
      )

  def testOnes(self):
    for s in self.all_shapes:
      actual = np_array_ops.ones(s)
      expected = np.ones(s)
      msg = 'shape: {}'.format(s)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          tensor.Tensor,
          msg,
      )

    for s, t in itertools.product(self.all_shapes, _all_types):
      actual = np_array_ops.ones(s, t)
      expected = np.ones(s, t)
      msg = 'shape: {}, dtype: {}'.format(s, t)
      self.match_expected_attrs(
          actual, expected, expected.dtype, tensor.Tensor, msg
      )

  def testOnesLike(self):
    for a in self.all_arrays:
      actual = np_array_ops.ones_like(a)
      expected = np.ones_like(a)
      msg = 'array: {}'.format(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
          msg,
      )

    for a, t in itertools.product(self.all_arrays, _all_types):
      actual = np_array_ops.ones_like(a, t)
      expected = np.ones_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match_expected_attrs(
          actual, expected, expected.dtype, tensor.Tensor, msg
      )

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
      for fn1, fn2, arr_dtype in itertools.product(
          self.array_transforms, zeros_builders, _all_types
      ):
        fill_value = fn1(f)
        arr = fn2(s, arr_dtype)
        wt_arr = _get_weak_tensor(arr)
        expected = np.full_like(arr, fill_value)
        self.match_expected_attrs(
            np_array_ops.full_like(wt_arr, fill_value),
            expected,
            expected.dtype,
            WeakTensor,
        )
        for dtype in _all_types:
          self.match_expected_attrs(
              np_array_ops.full_like(arr, fill_value, dtype=dtype),
              np.full_like(arr, fill_value, dtype=dtype),
              _NP_TO_TF[dtype],
              tensor.Tensor,
          )

  def testArray(self):
    ndmins = [0, 1, 2, 5]
    for a, dtype, ndmin, copy in itertools.product(
        self.all_arrays, _all_types, ndmins, [True, False]
    ):
      # Dtype specified.
      self.match_expected_attrs(
          np_array_ops.array(a, dtype=dtype, ndmin=ndmin, copy=copy),
          np.array(a, dtype=dtype, ndmin=ndmin, copy=copy),
          dtype,
          tensor.Tensor,
      )
      # No dtype specified.
      actual = np_array_ops.array(a, ndmin=ndmin, copy=copy)
      expected = np.array(a, ndmin=ndmin, copy=copy)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

  def testAsArray(self):
    for a, dtype in itertools.product(self.all_arrays, _all_types):
      # Dtype specified.
      self.match_expected_attrs(
          np_array_ops.asarray(a, dtype=dtype),
          np.asarray(a, dtype=dtype),
          _NP_TO_TF[dtype],
          tensor.Tensor,
      )
      # No dtype specified.
      actual = np_array_ops.asarray(a)
      expected = np.asarray(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

    zeros_list = np_array_ops.zeros(5)
    # Same instance is returned if no dtype is specified and input is ndarray.
    self.assertIs(np_array_ops.asarray(zeros_list), zeros_list)
    with ops.device('CPU:1'):
      self.assertIs(np_array_ops.asarray(zeros_list), zeros_list)
    # Different instance is returned if dtype is specified and input is ndarray.
    self.assertIsNot(np_array_ops.asarray(zeros_list, dtype=int), zeros_list)

  def testAsAnyArray(self):
    for a, dtype in itertools.product(self.all_arrays, _all_types):
      # Dtype specified.
      self.match_expected_attrs(
          np_array_ops.asanyarray(a, dtype=dtype),
          np.asanyarray(a, dtype=dtype),
          _NP_TO_TF[dtype],
          tensor.Tensor,
      )
      # No dtype specified.
      actual = np_array_ops.asanyarray(a)
      expected = np.asanyarray(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

    zeros_list = np_array_ops.zeros(5)
    # Same instance is returned if no dtype is specified and input is ndarray.
    self.assertIs(np_array_ops.asanyarray(zeros_list), zeros_list)
    with ops.device('CPU:1'):
      self.assertIs(np_array_ops.asanyarray(zeros_list), zeros_list)
    # Different instance is returned if dtype is specified and input is ndarray.
    self.assertIsNot(np_array_ops.asanyarray(zeros_list, dtype=int), zeros_list)

  def testAsContiguousArray(self):
    for a, dtype in itertools.product(self.all_arrays, _all_types):
      # Dtype specified.
      self.match_expected_attrs(
          np_array_ops.ascontiguousarray(a, dtype=dtype),
          np.ascontiguousarray(a, dtype=dtype),
          _NP_TO_TF[dtype],
          tensor.Tensor,
      )
      # No dtype specified.
      actual = np_array_ops.ascontiguousarray(a)
      expected = np.ascontiguousarray(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

  def testARange(self):
    int_values = np.arange(-3, 3).tolist()
    float_values = np.arange(-3.5, 3.5).tolist()
    all_values = int_values + float_values
    for dtype in _all_types:
      for start in all_values:
        msg = 'dtype:{} start:{}'.format(dtype, start)
        # Dtype specified.
        self.match_expected_attrs(
            np_array_ops.arange(start, dtype=dtype),
            np.arange(start, dtype=dtype),
            _NP_TO_TF[dtype],
            tensor.Tensor,
            msg=msg,
        )
        # No dtype specified.
        actual = np_array_ops.arange(start)
        expected = np.arange(start)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
            msg=msg,
        )

        for stop in all_values:
          msg = 'dtype:{} start:{} stop:{}'.format(dtype, start, stop)
          # TODO(srbs): Investigate and remove check.
          # There are some bugs when start or stop is float and dtype is int.
          if not isinstance(start, float) and not isinstance(stop, float):
            # Dtype specfieid.
            self.match_expected_attrs(
                np_array_ops.arange(start, stop, dtype=dtype),
                np.arange(start, stop, dtype=dtype),
                _NP_TO_TF[dtype],
                tensor.Tensor,
                msg=msg,
            )
            # No dtype specified.
            actual = np_array_ops.arange(start, stop)
            expected = np.arange(start, stop)
            self.match_expected_attrs(
                actual,
                expected,
                _NP_to_TF_result_inferred_types[expected.dtype],
                WeakTensor,
                msg=msg,
            )
          # Note: We intentionally do not test with float values for step
          # because numpy.arange itself returns inconsistent results. e.g.
          # np.arange(0.5, 3, step=0.5, dtype=int) returns
          # array([0, 1, 2, 3, 4])
          for step in int_values:
            msg = 'dtype:{} start:{} stop:{} step:{}'.format(
                dtype, start, stop, step)
            if not step:
              with self.assertRaises(ValueError):
                actual = np_array_ops.arange(start, stop, step)
                expected = np.arange(start, stop, step)
                self.match_expected_attrs(
                    actual,
                    expected,
                    _NP_to_TF_result_inferred_types[expected.dtype],
                    WeakTensor,
                    msg=msg,
                )
                if not isinstance(start, float) and not isinstance(stop, float):
                  self.match_expected_attrs(
                      np_array_ops.arange(start, stop, step, dtype=dtype),
                      np.arange(start, stop, step, dtype=dtype),
                      _NP_TO_TF[dtype],
                      tensor.Tensor,
                      msg=msg,
                  )
            else:
              if not isinstance(start, float) and not isinstance(stop, float):
                actual = np_array_ops.arange(start, stop, step)
                expected = np.arange(start, stop, step)
                self.match_expected_attrs(
                    actual,
                    expected,
                    _NP_to_TF_result_inferred_types[expected.dtype],
                    WeakTensor,
                    msg=msg,
                )
                self.match_expected_attrs(
                    np_array_ops.arange(start, stop, step, dtype=dtype),
                    np.arange(start, stop, step, dtype=dtype),
                    _NP_TO_TF[dtype],
                    tensor.Tensor,
                    msg=msg,
                )

  def testDiag(self):
    array_transforms = [
        lambda x: x,  # Identity,
        _get_weak_tensor,
        np_array_ops.array,
    ]

    def run_test(arr):
      for fn in array_transforms:
        arr = fn(arr)
        actual = np_array_ops.diag(arr)
        expected = np.diag(arr)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
            msg='diag({})'.format(arr),
        )
        for k in range(-3, 3):
          actual = np_array_ops.diag(arr, k)
          expected = np.diag(arr, k)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='diag({}, k={})'.format(arr, k),
          )

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
        _get_weak_tensor,
        np_array_ops.array,
    ]

    def run_test(arr):
      for fn in array_transforms:
        arr = fn(arr)
        actual = np_array_ops.diagflat(arr)
        expected = np.diagflat(arr)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
            msg='diagflat({})'.format(arr),
        )
        for k in range(-3, 3):
          actual = np_array_ops.diagflat(arr, k)
          expected = np.diagflat(arr, k)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='diagflat({})'.format(arr),
          )

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

  def match_dtype_and_type(self, actual, expected_dtype, res_type, msg=None):
    if msg:
      msg = (
          'Dtype and type match failed for: {}. Expected dtype: {} Actual'
          ' dtype: {}. Expected type: {} Actual type: {}.'.format(
              msg, expected_dtype, actual.dtype, res_type, type(actual)
          )
      )
    self.assertIsInstance(actual, res_type)
    self.assertEqual(actual.dtype, expected_dtype, msg=msg)

  def match_expected_attrs(
      self,
      actual,
      expected,
      expected_dtype,
      res_type,
      msg=None,
      almost=False,
      decimal=7,
  ):
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, res_type)
    self.match_dtype_and_type(actual, expected_dtype, res_type, msg)
    self.match_shape(actual, expected, msg)
    if not almost:
      if not actual.shape.rank:
        self.assertEqual(actual.tolist(), expected.tolist())
      else:
        self.assertSequenceEqual(actual.tolist(), expected.tolist())
    else:
      np.testing.assert_almost_equal(
          actual.tolist(), expected.tolist(), decimal=decimal)


class ArrayMethodsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ArrayMethodsTest, self).setUp()
    set_up_virtual_devices()
    self.array_transforms = [
        lambda x: x,
        _get_weak_tensor,
        np_array_ops.array,
    ]

  def testCopy(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.copy(arg, *args, **kwargs)
        expected = np.copy(arg, *args, **kwargs)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
            msg='copy({})'.format(arr),
        )

    run_test([])
    run_test([1, 2, 3])
    run_test([1., 2., 3.])
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
        # Cumprod Test
        actual = np_array_ops.cumprod(arg, *args, **kwargs)
        expected = np.cumprod(arg, *args, **kwargs)
        self.assertAllEqual(actual, expected)
        if kwargs.get('dtype', None) is None:
          self.match_dtype_and_type(
              actual,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )
        else:
          self.match_dtype_and_type(
              actual,
              flexible_dtypes.result_type(kwargs['dtype'])[0],
              tensor.Tensor,
          )

        # Cumsum Test
        actual = np_array_ops.cumsum(arg, *args, **kwargs)
        expected = np.cumsum(arg, *args, **kwargs)
        self.assertAllEqual(actual, expected)
        if kwargs.get('dtype', None) is None:
          self.match_dtype_and_type(
              actual,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )
        else:
          self.match_dtype_and_type(
              actual,
              flexible_dtypes.result_type(kwargs['dtype'])[0],
              tensor.Tensor,
          )

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

    def run_test(arr, dtype):
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.imag(arg)
        # np.imag may return a scalar so we convert to a np.ndarray.
        expected = np.array(np.imag(arg))
        self.match_expected_attrs(actual, expected, dtype, WeakTensor)

    # Weak complex128 input returns float64.
    run_test(1, dtypes.int32)
    run_test(5.5, dtypes.float32)
    run_test(5 + 3j, dtypes.float64)
    run_test(3j, dtypes.float64)
    run_test([], dtypes.float32)
    run_test([1, 2, 3], dtypes.int32)
    run_test([1 + 5j, 2 + 3j], dtypes.float64)
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]], dtypes.float64)

  def testAMaxAMin(self):

    def run_test(arr, *args, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          actual = np_array_ops.amax(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.amax(arr_arg, axis=axis, *args, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='amax({})'.format(arr),
          )
          actual = np_array_ops.amin(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.amin(arr_arg, axis=axis, *args, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='amin({})'.format(arr),
          )

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
          actual = np_array_ops.mean(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.mean(arr_arg, axis=axis, *args, **kwargs)
          dtype = kwargs.get('dtype', None)
          if dtype is None:
            self.match_expected_attrs(
                actual,
                expected,
                _NP_to_TF_result_inferred_types[expected.dtype],
                WeakTensor,
            )
          else:
            self.match_expected_attrs(
                actual,
                expected,
                flexible_dtypes.result_type(dtype)[0],
                tensor.Tensor,
            )
    run_test([1, 2, 1])
    run_test([1.0, 2.0, 1.0])
    run_test([1.0, 2.0, 1.0], dtype=int)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(0, 2),
        keepdims=True,
    )
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(2, 0),
        keepdims=True,
    )
    self.assertRaises(ValueError, np_array_ops.mean, np.ones([2, 2]), out=[])

  def testStd(self):
    def run_test(arr, res_dtype, *args, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          actual = np_array_ops.std(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.std(arr_arg, axis=axis, *args, **kwargs)
          res_dtype = (
              _NP_to_TF_result_inferred_types[expected.dtype]
              if res_dtype is None
              else res_dtype
          )
          self.match_expected_attrs(actual, expected, res_dtype, WeakTensor)

    run_test([1, 2, 1], res_dtype=None)
    run_test([1.0, 2.0, 1.0], res_dtype=None)
    run_test([1.0j, 2.0, 1.0j], res_dtype=dtypes.float64)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=1)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=0)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=-1)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=-2)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=(0, 1))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), res_dtype=None, axis=(0, 2)
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        res_dtype=None,
        axis=(0, 2),
        keepdims=True,
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), res_dtype=None, axis=(2, 0)
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        res_dtype=None,
        axis=(2, 0),
        keepdims=True,
    )

  def testVar(self):

    def run_test(arr, res_dtype, *args, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          actual = np_array_ops.var(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.var(arr_arg, axis=axis, *args, **kwargs)
          dtype = kwargs.get('dtype', None)
          res_type = tensor.Tensor if dtype is not None else WeakTensor
          res_dtype = (
              _NP_to_TF_result_inferred_types[expected.dtype]
              if res_dtype is None
              else res_dtype
          )
          self.match_expected_attrs(actual, expected, res_dtype, res_type)

    # Input of weak complex type (complex 128) always outputs float64.
    run_test([1, 2, 1], res_dtype=None)
    run_test([1.0, 2.0, 1.0], res_dtype=None)
    run_test([1.0j, 2.0, 1.0j], res_dtype=dtypes.float64)
    run_test([1.0, 2.0, 1.0], res_dtype=dtypes.int64, dtype=np.int64)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=1)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=0)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=-1)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=-2)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=(0, 1))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), res_dtype=None, axis=(0, 2)
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(0, 2),
        res_dtype=None,
        keepdims=True,
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), res_dtype=None, axis=(2, 0)
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(2, 0),
        res_dtype=None,
        keepdims=True,
    )
    self.assertRaises(ValueError, np_array_ops.var, np.ones([2, 2]), out=[])

  def testProd(self):
    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.prod(arg, *args, **kwargs)
        expected = np.prod(arg, *args, **kwargs)
        dtype = kwargs.get('dtype', None)
        if dtype is None:
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )
        else:
          self.match_expected_attrs(
              actual,
              expected,
              flexible_dtypes.result_type(dtype)[0],
              tensor.Tensor,
          )

    run_test([1, 2, 3])
    run_test([1.0, 2.0, 3.0])
    run_test(np.array([1, 2, 3]), dtype=np.int32)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(0, 2),
        keepdims=True,
    )
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(2, 0),
        keepdims=True,
    )

  def _testReduce(self, math_fun, np_fun, name):
    axis_transforms = [
        lambda x: x,  # Identity,
        np_array_ops.array,
        _get_weak_tensor,
    ]

    def run_test(a, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in axis_transforms:
          arg1 = fn1(a)
          axis_arg = fn2(axis) if axis is not None else None
          actual = math_fun(arg1, axis=axis_arg, **kwargs)
          expected = np_fun(arg1, axis=axis, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='{}({}, axis={}, keepdims={})'.format(
                  name, arg1, axis, kwargs.get('keepdims')
              ),
          )

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

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(dtype=dtypes.float32, shape=None)
        ]
    )
    def f(arr):
      arr = np_array_ops.asarray(arr)
      return np_array_ops.size(arr)

    self.assertEqual(f(np_array_ops.ones((3, 2))).numpy(), 6)

  def testRavel(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.ravel(arg, *args, **kwargs)
        expected = np.ravel(arg, *args, **kwargs)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
        )

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

    def run_test(arr, res_dtype, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.real(arg, *args, **kwargs)
        expected = np.array(np.real(arg, *args, **kwargs))
        res_dtype = (
            _NP_to_TF_result_inferred_types[expected.dtype]
            if res_dtype is None
            else res_dtype
        )
        self.match_expected_attrs(
            actual,
            expected,
            res_dtype,
            WeakTensor,
        )

    run_test(1, None)
    run_test(5.5, None)
    run_test(5 + 3j, dtypes.float64)
    run_test(3j, dtypes.float64)
    run_test([], None)
    run_test([1, 2, 3], None)
    run_test([1 + 5j, 2 + 3j], dtypes.float64)
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]], dtypes.float64)

  def testRepeat(self):

    def run_test(arr, repeats, *args, **kwargs):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          repeats_arg = fn2(repeats)
          actual = np_array_ops.repeat(arr_arg, repeats_arg, *args, **kwargs)
          expected = np.repeat(arr_arg, repeats_arg, *args, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )

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
        actual = np_array_ops.around(arg, *args, **kwargs)
        expected = np.around(arg, *args, **kwargs)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
        )

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
          actual = np_array_ops.reshape(arr_arg, newshape_arg, *args, **kwargs)
          expected = np.reshape(arr_arg, newshape, *args, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )

    run_test(5, [-1])
    run_test([], [-1])
    run_test([1, 2, 3], [1, 3])
    run_test([1, 2, 3], [3, 1])
    run_test([1, 2, 3, 4], [2, 2])
    run_test([1, 2, 3, 4], [2, 1, 2])

  def testExpandDims(self):

    def run_test(arr, axis):
      actual = np_array_ops.expand_dims(arr, axis)
      expected = np.expand_dims(arr, axis)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

    run_test([1, 2, 3], 0)
    run_test([1, 2, 3], 1)

  def testSqueeze(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        # Note: np.squeeze ignores the axis arg for non-ndarray objects.
        # This looks like a bug: https://github.com/numpy/numpy/issues/8201
        # So we convert the arg to np.ndarray before passing to np.squeeze.
        actual = np_array_ops.squeeze(arg, *args, **kwargs)
        expected = np.squeeze(np.array(arg), *args, **kwargs)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
        )

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
          actual = np_array_ops.transpose(arr_arg, axes_arg)
          expected = np.transpose(arr_arg, axes)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )

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

  def match_dtype_and_type(self, actual, expected_dtype, res_type, msg=None):
    if msg:
      msg = (
          'Dtype and type match failed for: {}. Expected dtype: {} Actual'
          ' dtype: {}. Expected type: {} Actual type: {}.'.format(
              msg, expected_dtype, actual.dtype, res_type, type(actual)
          )
      )
    self.assertIsInstance(actual, res_type, msg=msg)
    self.assertEqual(actual.dtype, expected_dtype, msg=msg)

  def match_expected_attrs(
      self, actual, expected, expected_dtype, res_type, msg=None
  ):
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.match_dtype_and_type(actual, expected_dtype, res_type, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape.rank:
      self.assertAllClose(actual.tolist(), expected.tolist())
    else:
      self.assertAllClose(actual.tolist(), expected.tolist())

  def testShape(self):
    self.assertAllEqual((1, 2), np_array_ops.shape([[0, 0]]))

  @parameterized.parameters(
      ([[1, 2, 3]], 0, 1, [[1], [2], [3]]),
      ([[1, 2, 3]], -2, -1, [[1], [2], [3]]),
      (
          [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
          0,
          2,
          [[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
      ),
      (
          [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
          -3,
          -1,
          [[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
      ),
  )
  def testSwapaxes(self, x, axis1, axis2, expected):
    actual = np_array_ops.swapaxes(x, axis1, axis2)
    self.assertIsInstance(actual, WeakTensor)
    self.assertAllEqual(actual, expected)

  def testMoveaxis(self):

    def _test(a, *args):
      # pylint: disable=no-value-for-parameter
      expected = np.moveaxis(a, *args)
      wt_a = _get_weak_tensor(a)
      raw_ans = np_array_ops.moveaxis(wt_a, *args)

      self.assertIsInstance(raw_ans, WeakTensor)
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


if __name__ == '__main__':
  ops.enable_eager_execution()
  ops.set_dtype_conversion_mode('all')
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
