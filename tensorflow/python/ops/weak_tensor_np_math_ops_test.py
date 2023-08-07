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
"""Tests for tf numpy mathematical methods on WeakTensor."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops.numpy_ops import np_random
from tensorflow.python.platform import googletest


_get_weak_tensor = weak_tensor_test_util.get_weak_tensor

_NP_to_TF_result_inferred_types = {
    np.dtype(np.int32): dtypes.int32,
    np.dtype(np.int64): dtypes.int32,
    np.dtype(np.float32): dtypes.float32,
    np.dtype(np.float64): dtypes.float32,
    np.dtype(np.complex128): dtypes.complex128,
}


class MathTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def setUp(self):
    super(MathTest, self).setUp()
    self.array_transforms = [
        lambda x: x,  # Identity,
        _get_weak_tensor,
        np_array_ops.array,
    ]
    self.types = [np.int32, np.int64, np.float32, np.float64]

  def _testUnaryOp(self, math_fun, np_fun, name, weak_result):
    def run_test(a):
      for fn in self.array_transforms:
        arg1 = fn(a)
        if weak_result and flexible_dtypes.result_type(arg1)[1]:
          self.assertIsInstance(math_fun(arg1), weak_tensor.WeakTensor)
        else:
          self.assertIsInstance(math_fun(arg1), np_arrays.ndarray)
        self.match(
            math_fun(arg1), np_fun(arg1), msg='{}({})'.format(name, arg1))

    run_test(5)
    run_test([2, 3])
    run_test([[2, -3], [-6, 7]])

  def testLog(self):
    self._testUnaryOp(np_math_ops.log, np.log, 'log', True)

  def testExp(self):
    self._testUnaryOp(np_math_ops.exp, np.exp, 'exp', True)

  def testTanh(self):
    self._testUnaryOp(np_math_ops.tanh, np.tanh, 'tanh', True)

  def testSqrt(self):
    self._testUnaryOp(np_math_ops.sqrt, np.sqrt, 'sqrt', True)

  def match(self, actual, expected, msg='', check_dtype=True):
    if check_dtype:
      self.assertEqual(
          actual.dtype,
          _NP_to_TF_result_inferred_types[expected.dtype],
          'Dtype mismatch.\nActual: {}\nExpected: {}\n{}'.format(
              actual.dtype.as_numpy_dtype,
              _NP_to_TF_result_inferred_types[expected.dtype],
              msg,
          ),
      )
    self.assertEqual(
        actual.shape, expected.shape,
        'Shape mismatch.\nActual: {}\nExpected: {}\n{}'.format(
            actual.shape, expected.shape, msg))
    np.testing.assert_allclose(actual.tolist(), expected.tolist(), rtol=1e-6)

  def testArgsort(self):
    self._testUnaryOp(np_math_ops.argsort, np.argsort, 'argsort', False)

    # Test stability
    r = np.arange(100)
    a = np.zeros(100)
    np.testing.assert_equal(np_math_ops.argsort(a, kind='stable'), r)

  def testArgMaxArgMin(self):
    data = [
        0,
        5,
        [1],
        [1, 2, 3],
        [[1, 2, 3]],
        [[4, 6], [7, 8]],
        [[[4, 6], [9, 10]], [[7, 8], [12, 34]]],
    ]
    for fn, d in itertools.product(self.array_transforms, data):
      arr = fn(d)
      # argmax and argmin returns indices (int64 type).
      self.match(np_math_ops.argmax(arr), np.argmax(arr), check_dtype=False)
      self.match(np_math_ops.argmin(arr), np.argmin(arr), check_dtype=False)
      if hasattr(arr, 'shape'):
        ndims = len(arr.shape)
      else:
        ndims = np_array_ops.array(arr, copy=False).ndim
      if ndims == 0:
        # Numpy flattens the scalar ndarray and treats it as a 1-d array of
        # size 1.
        ndims = 1
      for axis in range(-ndims, ndims):
        self.match(
            np_math_ops.argmax(arr, axis=axis),
            np.argmax(arr, axis=axis),
            check_dtype=False,
        )
        self.match(
            np_math_ops.argmin(arr, axis=axis),
            np.argmin(arr, axis=axis),
            check_dtype=False,
        )

  def testAverageWrongShape(self):
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), weights=np.ones([2, 4]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([2, 4]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([5]))

  def testPtp(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_math_ops.ptp(arg, *args, **kwargs),
            np.ptp(arg, *args, **kwargs),
            check_dtype=False,
        )

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)

  # Test that enable_numpy_methods() gets called when weak_tensor_ops is
  # imported.
  @parameterized.parameters([
      'T', 'ndim', 'size', 'data', '__pos__', '__round__', 'tolist', 'flatten',
      'transpose', 'reshape', 'ravel', 'clip', 'astype', 'max', 'mean', 'min'])
  def testNumpyMethodsOnTensor(self, np_method):
    a = ops.convert_to_tensor([1, 2])
    self.assertTrue(hasattr(a, np_method))

  def testFlatten(self):
    a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    a2 = _get_weak_tensor(a1)
    self.assertAllEqual(a1.flatten('C'), a2.flatten('C'))
    self.assertAllEqual(a1.flatten('F'), a2.flatten('F'))
    self.assertAllEqual(a1.flatten('C'), a2.flatten('A'))
    self.assertAllEqual(a1.flatten('C'), a2.flatten('K'))
    with self.assertRaises(ValueError):
      a2.flatten('invalid')

  def testIsInf(self):
    x1 = _get_weak_tensor(-2147483648)
    x2 = _get_weak_tensor(2147483647)
    self.assertFalse(np_math_ops.isinf(x1))
    self.assertFalse(np_math_ops.isinf(x2))
    self.assertFalse(np_math_ops.isposinf(x1))
    self.assertFalse(np_math_ops.isposinf(x2))
    self.assertFalse(np_math_ops.isneginf(x1))
    self.assertFalse(np_math_ops.isneginf(x2))

  def testRandomOpsReturnFloat32(self):
    x = np_random.rand(2, 50)
    np_x = np.random.rand(2, 50)
    self.assertEqual(x.dtype, dtypes.float32)
    self.assertEqual(np_x.shape, x.shape)

    x = np_random.standard_normal(50)
    np_x = np.random.standard_normal(50)
    self.assertEqual(x.dtype, dtypes.float32)
    self.assertEqual(np_x.shape, x.shape)

    x = np_random.uniform(low=-1, high=0, size=(50, 50))
    np_x = np.random.uniform(low=-1, high=0, size=(50, 50))
    self.assertEqual(x.dtype, dtypes.float32)
    self.assertEqual(np_x.shape, x.shape)


if __name__ == '__main__':
  tensor.enable_tensor_equality()
  np_math_ops.enable_numpy_methods_on_tensor()
  ops.enable_eager_execution()
  ops.set_dtype_conversion_mode('all')
  googletest.main()
