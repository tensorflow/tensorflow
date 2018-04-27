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
"""Tests for array_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import unittest

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as test_lib


class BatchMatrixTransposeTest(test_util.TensorFlowTestCase):

  def testNonBatchMatrix(self):
    matrix = [[1, 2, 3], [4, 5, 6]]  # Shape (2, 3)
    expected_transposed = [[1, 4], [2, 5], [3, 6]]  # Shape (3, 2)
    with self.test_session():
      transposed = array_ops.matrix_transpose(matrix)
      self.assertEqual((3, 2), transposed.get_shape())
      self.assertAllEqual(expected_transposed, transposed.eval())

  def testConjugate(self):
    m = [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]]
    expected_transposed = [[1 - 1j, 4 - 4j], [2 - 2j, 5 - 5j], [3 - 3j, 6 - 6j]]
    with self.test_session():
      matrix = ops.convert_to_tensor(m)
      transposed = array_ops.matrix_transpose(matrix, conjugate=True)
      self.assertEqual((3, 2), transposed.get_shape())
      self.assertAllEqual(expected_transposed, transposed.eval())

  def testBatchMatrix(self):
    matrix_0 = [[1, 2, 3], [4, 5, 6]]
    matrix_0_t = [[1, 4], [2, 5], [3, 6]]
    matrix_1 = [[11, 22, 33], [44, 55, 66]]
    matrix_1_t = [[11, 44], [22, 55], [33, 66]]
    batch_matrix = [matrix_0, matrix_1]  # Shape (2, 2, 3)
    expected_transposed = [matrix_0_t, matrix_1_t]  # Shape (2, 3, 2)
    with self.test_session():
      transposed = array_ops.matrix_transpose(batch_matrix)
      self.assertEqual((2, 3, 2), transposed.get_shape())
      self.assertAllEqual(expected_transposed, transposed.eval())

  def testNonBatchMatrixDynamicallyDefined(self):
    matrix = [[1, 2, 3], [4, 5, 6]]  # Shape (2, 3)
    expected_transposed = [[1, 4], [2, 5], [3, 6]]  # Shape (3, 2)
    with self.test_session():
      matrix_ph = array_ops.placeholder(dtypes.int32)
      transposed = array_ops.matrix_transpose(matrix_ph)
      self.assertAllEqual(
          expected_transposed, transposed.eval(feed_dict={
              matrix_ph: matrix
          }))

  def testBatchMatrixDynamicallyDefined(self):
    matrix_0 = [[1, 2, 3], [4, 5, 6]]
    matrix_0_t = [[1, 4], [2, 5], [3, 6]]
    matrix_1 = [[11, 22, 33], [44, 55, 66]]
    matrix_1_t = [[11, 44], [22, 55], [33, 66]]
    batch_matrix = [matrix_0, matrix_1]  # Shape (2, 2, 3)
    expected_transposed = [matrix_0_t, matrix_1_t]  # Shape (2, 3, 2)
    with self.test_session():
      batch_matrix_ph = array_ops.placeholder(dtypes.int32)
      transposed = array_ops.matrix_transpose(batch_matrix_ph)
      self.assertAllEqual(
          expected_transposed,
          transposed.eval(feed_dict={
              batch_matrix_ph: batch_matrix
          }))

  def testTensorWithStaticRankLessThanTwoRaisesBecauseNotAMatrix(self):
    vector = [1, 2, 3]
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "should be a "):
        array_ops.matrix_transpose(vector)


class BooleanMaskTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.rng = np.random.RandomState(42)

  def CheckVersusNumpy(self, ndims_mask, arr_shape, make_mask=None, axis=None):
    """Check equivalence between boolean_mask and numpy masking."""
    if make_mask is None:
      make_mask = lambda shape: self.rng.randint(0, 2, size=shape).astype(bool)
    arr = np.random.rand(*arr_shape)
    mask = make_mask(arr_shape[:ndims_mask])
    if axis is not None:
      mask = make_mask(arr_shape[axis:ndims_mask + axis])
    if axis is None or axis == 0:
      masked_arr = arr[mask]
    elif axis == 1:
      masked_arr = arr[:, mask]
    elif axis == 2:
      masked_arr = arr[:, :, mask]
    with self.test_session():
      masked_tensor = array_ops.boolean_mask(arr, mask, axis=axis)

      # Leading dimension size of masked_tensor is always unknown until runtime
      # since we don't how many elements will be kept.
      leading = 1 if axis is None else axis + 1
      self.assertAllEqual(masked_tensor.get_shape()[leading:],
                          masked_arr.shape[leading:])

      self.assertAllClose(masked_arr, masked_tensor.eval())

  def testMaskDim1ArrDim2Axis1(self):
    ndims_mask = 1
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      self.CheckVersusNumpy(ndims_mask, arr_shape, axis=1)

  def testMaskDim2ArrDim2Axis1(self):
    ndims_mask = 2
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      self.CheckVersusNumpy(ndims_mask, arr_shape, axis=1)

  def testMaskDim1ArrDim1(self):
    ndims_mask = 1
    for arr_shape in [(1,), (2,), (3,), (10,)]:
      self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testMaskDim1ArrDim2(self):
    ndims_mask = 1
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testMaskDim2ArrDim2(self):
    ndims_mask = 2
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testMaskDim2ArrDim3(self):
    ndims_mask = 2
    for arr_shape in [(1, 1, 1), (1, 2, 2), (2, 2, 1)]:
      self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testEmptyInput2D(self):
    mask = np.array([True, False])
    arr = np.array([[], []]).astype(np.float32)
    numpy_result = arr[mask]
    tf_result = array_ops.boolean_mask(arr, mask)
    self.assertAllEqual(numpy_result.shape[1:], tf_result.get_shape()[1:])
    with self.test_session():
      self.assertAllClose(numpy_result, tf_result.eval())

  def testEmptyInput1D(self):
    mask = np.array([]).astype(bool)
    arr = np.array([]).astype(np.float32)
    numpy_result = arr[mask]
    tf_result = array_ops.boolean_mask(arr, mask)
    self.assertAllEqual(numpy_result.shape[1:], tf_result.get_shape()[1:])
    with self.test_session():
      self.assertAllClose(numpy_result, tf_result.eval())

  def testEmptyOutput(self):
    make_mask = lambda shape: np.zeros(shape, dtype=bool)
    for ndims_mask in range(1, 4):
      for ndims_arr in range(ndims_mask, ndims_mask + 3):
        for _ in range(3):
          arr_shape = np.random.randint(1, 5, size=ndims_arr)
          self.CheckVersusNumpy(ndims_mask, arr_shape, make_mask=make_mask)

  def testWorksWithDimensionsEqualToNoneDuringGraphBuild(self):
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    with self.test_session() as sess:
      ph_tensor = array_ops.placeholder(dtypes.int32, shape=None)
      ph_mask = array_ops.placeholder(dtypes.bool, shape=[None])

      arr = np.array([[1, 2], [3, 4]])
      mask = np.array([False, True])

      masked_tensor = sess.run(
          array_ops.boolean_mask(ph_tensor, ph_mask),
          feed_dict={
              ph_tensor: arr,
              ph_mask: mask
          })
      np.testing.assert_allclose(masked_tensor, arr[mask])

  def testMaskDimensionsSetToNoneRaises(self):
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.int32, shape=[None, 2])
      mask = array_ops.placeholder(dtypes.bool, shape=None)
      with self.assertRaisesRegexp(ValueError, "dimensions must be specified"):
        array_ops.boolean_mask(tensor, mask)

  def testMaskHasMoreDimsThanTensorRaises(self):
    mask = [[True, True], [False, False]]
    tensor = [1, 2, 3, 4]
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "incompatible"):
        array_ops.boolean_mask(tensor, mask).eval()

  def testMaskIsScalarRaises(self):
    mask = True
    tensor = 1
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "mask.*scalar"):
        array_ops.boolean_mask(tensor, mask).eval()

  def testMaskShapeDifferentThanFirstPartOfTensorShapeRaises(self):
    mask = [True, True, True]
    tensor = [[1, 2], [3, 4]]
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "incompatible"):
        array_ops.boolean_mask(tensor, mask).eval()


class OperatorShapeTest(test_util.TensorFlowTestCase):

  def testExpandScalar(self):
    scalar = "hello"
    scalar_expanded = array_ops.expand_dims(scalar, [0])
    self.assertEqual(scalar_expanded.get_shape(), (1,))

  def testSqueezeScalar(self):
    scalar = "hello"
    scalar_squeezed = array_ops.squeeze(scalar, ())
    self.assertEqual(scalar_squeezed.get_shape(), ())

  def testSqueezeMatrix(self):
    matrix = [[1, 2, 3]]
    matrix_squeezed = array_ops.squeeze(matrix, [0])
    self.assertEqual(matrix_squeezed.get_shape(), (3))

    with self.assertRaises(ValueError):
      matrix_squeezed = array_ops.squeeze(matrix, [1])

  def testSqueezeScalarDim(self):
    matrix = [[1, 2, 3]]
    matrix_squeezed = array_ops.squeeze(matrix, 0)
    self.assertEqual(matrix_squeezed.get_shape(), (3))


class ReverseV2Test(test_util.TensorFlowTestCase):

  def testReverse0DimAuto(self):
    x_np = 4
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = array_ops.reverse_v2(x_np, []).eval()
        self.assertAllEqual(x_tf, x_np)

  def _reverse1DimAuto(self, np_dtype):
    x_np = np.array([1, 200, 3, 40, 5], dtype=np_dtype)

    for use_gpu in [False, True]:
      for axis_dtype in [dtypes.int32, dtypes.int64]:
        with self.test_session(use_gpu=use_gpu):
          x_tf = array_ops.reverse_v2(x_np,
                                      constant_op.constant(
                                          [0], dtype=axis_dtype)).eval()
          self.assertAllEqual(x_tf, np.asarray(x_np)[::-1])

  def _reverse2DimAuto(self, np_dtype):
    x_np = np.array([[1, 200, 3], [4, 5, 60]], dtype=np_dtype)

    for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
      for use_gpu in [False, True]:
        for axis_dtype in [dtypes.int32, dtypes.int64]:
          with self.test_session(use_gpu=use_gpu):
            x_tf_1 = reverse_f(x_np, constant_op.constant(
                [0], dtype=axis_dtype)).eval()
            x_tf_2 = reverse_f(x_np, constant_op.constant(
                [-2], dtype=axis_dtype)).eval()
            x_tf_3 = reverse_f(x_np, constant_op.constant(
                [1], dtype=axis_dtype)).eval()
            x_tf_4 = reverse_f(x_np, constant_op.constant(
                [-1], dtype=axis_dtype)).eval()
            x_tf_5 = reverse_f(x_np,
                               constant_op.constant([1, 0],
                                                    dtype=axis_dtype)).eval()
            self.assertAllEqual(x_tf_1, np.asarray(x_np)[::-1, :])
            self.assertAllEqual(x_tf_2, np.asarray(x_np)[::-1, :])
            self.assertAllEqual(x_tf_3, np.asarray(x_np)[:, ::-1])
            self.assertAllEqual(x_tf_4, np.asarray(x_np)[:, ::-1])
            self.assertAllEqual(x_tf_5, np.asarray(x_np)[::-1, ::-1])

  # This test covers the axis validation in the shape function
  # (no eval())
  def testInvalidAxis(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with self.assertRaisesRegexp(ValueError,
                                 "is out of valid range"):
      array_ops.reverse_v2(x_np, [-30])
    with self.assertRaisesRegexp(ValueError,
                                 "is out of valid range"):
      array_ops.reverse_v2(x_np, [2])
    with self.assertRaisesRegexp(ValueError,
                                 "axis 0 specified more than once"):
      array_ops.reverse_v2(x_np, [0, -2])

  # This is the version of reverse that uses axis indices rather than
  # bool tensors
  # TODO(b/32254538): Change this test to use array_ops.reverse
  #
  # Note: this test passes placeholder as constant axis is validated
  # in shape function (see testInvalidAxis)
  def testInvalid(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = array_ops.placeholder(dtypes.int32)
    with self.test_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "is out of valid range"):
        array_ops.reverse_v2(x_np, axis).eval(feed_dict={axis: [-30]})
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "is out of valid range"):
        array_ops.reverse_v2(x_np, axis).eval(feed_dict={axis: [2]})
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "axis 0 specified more than once"):
        array_ops.reverse_v2(x_np, axis).eval(feed_dict={axis: [0, -2]})

  def testReverse1DimAuto(self):
    for dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.bool,
        np.float16, np.float32, np.float64, np.complex64, np.complex128,
        np.array(b"").dtype.type
    ]:
      self._reverse1DimAuto(dtype)

  def testReverse2DimAuto(self):
    for dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.bool,
        np.float16, np.float32, np.float64, np.complex64, np.complex128,
        np.array(b"").dtype.type
    ]:
      self._reverse2DimAuto(dtype)

  def testUnknownDims(self):
    reverse_v2 = array_ops.reverse_v2
    data_t = array_ops.placeholder(dtypes.float32)
    axis_known_t = array_ops.placeholder(dtypes.int32, shape=[3])
    reverse_known_t = reverse_v2(data_t, axis_known_t)
    # Unlike V1 we cannot know this anymore
    self.assertEqual(None, reverse_known_t.get_shape().ndims)

    axis_unknown_t = array_ops.placeholder(dtypes.int32)
    reverse_unknown_t = reverse_v2(data_t, axis_unknown_t)
    self.assertIs(None, reverse_unknown_t.get_shape().ndims)

    data_2d_t = array_ops.placeholder(dtypes.float32, shape=[None, None])
    axis_2d_t = array_ops.placeholder(dtypes.int32, shape=[3])
    reverse_2d_t = reverse_v2(data_2d_t, axis_2d_t)
    self.assertEqual(2, reverse_2d_t.get_shape().ndims)

  def testReverseRowsOf3Channels(self):
    """Tests optimized code for reversing rows with last dim size = 3."""
    with self.test_session(use_gpu=True):
      for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
        for outer_size in (1, 2):
          for middle_size in list(range(50)) + [100000]:
            x_np = np.reshape(
                np.arange(outer_size * middle_size * 3, dtype=np.float32),
                newshape=(outer_size, middle_size, 3))
            x_tf = reverse_f(x_np, [1]).eval()
            np_answer = x_np[:, ::-1, :]
            self.assertAllEqual(x_tf, np_answer)

  def testReverseRowsOf4Channels(self):
    with self.test_session(use_gpu=True):
      for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
        for outer_size in (1, 2):
          for middle_size in list(range(50)) + [100000]:
            x_np = np.reshape(
                np.arange(outer_size * middle_size * 4, dtype=np.float32),
                newshape=(outer_size, middle_size, 4))
            x_tf = reverse_f(x_np, [1]).eval()
            np_answer = x_np[:, ::-1, :]
            self.assertAllEqual(x_tf, np_answer)

  def testReverseColumnsOf3Channels(self):
    with self.test_session(use_gpu=True):
      for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
        for outer_size in list(range(50)) + [100000]:
          for middle_size in (1, 2):
            x_np = np.reshape(
                np.arange(outer_size * middle_size * 3, dtype=np.float32),
                newshape=(outer_size, middle_size, 3))
            x_tf = reverse_f(x_np, [0]).eval()
            np_answer = x_np[::-1, :, :]
            self.assertAllEqual(x_tf, np_answer)


class MeshgridTest(test_util.TensorFlowTestCase):

  def _compareDiff(self, x, y, use_gpu):
    for index in ("ij", "xy"):
      numpy_out = np.meshgrid(x, y, indexing=index)
      tf_out = array_ops.meshgrid(x, y, indexing=index)
      with self.test_session(use_gpu=use_gpu):
        for xx, yy in zip(numpy_out, tf_out):
          self.assertAllEqual(xx, yy.eval())

  def _compareDiffType(self, n, np_dtype, use_gpu):
    inputs = []
    for index in ("ij", "xy"):
      for _ in range(n):
        x = np.linspace(-10, 10, 5).astype(np_dtype)
        if np_dtype in (np.complex64, np.complex128):
          x += 1j
        inputs.append(x)
      numpy_out = np.meshgrid(*inputs, indexing=index)
      with self.test_session(use_gpu=use_gpu):
        tf_out = array_ops.meshgrid(*inputs, indexing=index)
        for x_np, x_tf in zip(numpy_out, tf_out):
          self.assertAllEqual(x_np, x_tf.eval())

  def testCompare(self):
    for t in (np.float16, np.float32, np.float64, np.int32, np.int64,
              np.complex64, np.complex128):
      self._compareDiffType(2, t, False)
      self._compareDiffType(3, t, False)

      x = [1, 2, 3]
      y = [4, 5]

      a = [[1, 1], [1, 1]]

      self._compareDiff(x, y, False)
      self._compareDiff(x, a, False)


class StridedSliceChecker(object):
  """Check a given tensor against the numpy result."""

  REF_TENSOR = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
  REF_TENSOR_ALIGNED = np.arange(1, 97, dtype=np.float32).reshape(3, 4, 8)

  def __init__(self, test, x, tensor_type=dtypes.int32, check_type_infer=True):
    self.x_np = np.array(x).astype(tensor_type.as_numpy_dtype)
    # Give the value a non-zero imaginary component for complex types.
    if tensor_type.is_complex:
      self.x_np -= 1j * self.x_np
    self.test = test
    self.x = constant_op.constant(self.x_np, dtype=tensor_type)
    self.check_type_infer = check_type_infer

  def __getitem__(self, spec):
    op = self.x.__getitem__(spec)
    if not isinstance(spec, (list, tuple)):
      spec = [spec]

    tensor = op.eval()

    # Make a numpy spec that pre-evals the tensors
    np_specs = []

    def eval_if_tensor(x):
      try:
        return x.eval()
      except AttributeError:
        return x

    for s in spec:
      if isinstance(s, slice):
        start = eval_if_tensor(s.start)
        stop = eval_if_tensor(s.stop)
        step = eval_if_tensor(s.step)
        np_specs.append(slice(start, stop, step))
      else:
        np_specs.append(eval_if_tensor(s))

    self.test.assertAllEqual(self.x_np[tuple(np_specs)], tensor)
    if self.check_type_infer:
      self.test.assertAllEqual(tensor.shape, op.get_shape())
    return tensor


STRIDED_SLICE_TYPES = [
    dtypes.int32, dtypes.int64, dtypes.int16, dtypes.int8, dtypes.float32,
    dtypes.float64, dtypes.complex64, dtypes.complex128
]


class StridedSliceTest(test_util.TensorFlowTestCase):
  """Test the strided slice operation with variants of slices."""

  def test_basic_slice(self):
    for tensor_type in STRIDED_SLICE_TYPES:
      with self.test_session(use_gpu=not tensor_type.is_integer):
        checker = StridedSliceChecker(
            self, StridedSliceChecker.REF_TENSOR, tensor_type=tensor_type)
        _ = checker[:, :, :]
        # Various ways of representing identity slice
        _ = checker[:, :, :]
        _ = checker[::, ::, ::]
        _ = checker[::1, ::1, ::1]
        # Not zero slice
        _ = checker[::1, ::5, ::2]
        # Reverse in each dimension independently
        _ = checker[::-1, :, :]
        _ = checker[:, ::-1, :]
        _ = checker[:, :, ::-1]
        ## negative index tests i.e. n-2 in first component
        _ = checker[-2::-1, :, ::1]
        # negative index tests i.e. n-2 in first component, non-unit stride
        _ = checker[-2::-1, :, ::2]

        # Check rank-0 examples
        checker2 = StridedSliceChecker(self, 5, tensor_type=tensor_type)
        _ = checker2[None]
        _ = checker2[...]
        _ = checker2[tuple()]

  def testInt64GPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")
    with self.test_session(use_gpu=True, force_gpu=True):
      x = constant_op.constant([1., 2., 3.])
      begin = constant_op.constant([2], dtype=dtypes.int64)
      end = constant_op.constant([3], dtype=dtypes.int64)
      strides = constant_op.constant([1], dtype=dtypes.int64)
      s = array_ops.strided_slice(x, begin, end, strides)
      self.assertAllEqual([3.], self.evaluate(s))

  def testDegenerateSlices(self):
    with self.test_session(use_gpu=True):
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      # degenerate by offering a forward interval with a negative stride
      _ = checker[0:-1:-1, :, :]
      # degenerate with a reverse interval with a positive stride
      _ = checker[-1:0, :, :]
      # empty interval in every dimension
      _ = checker[-1:0, 2:2, 2:3:-1]

  def testEllipsis(self):
    with self.test_session(use_gpu=True):
      raw = [[[[[1, 2], [3, 4], [5, 6]]], [[[7, 8], [9, 10], [11, 12]]]]]
      checker = StridedSliceChecker(self, raw)

      _ = checker[0:]
      # implicit ellipsis
      _ = checker[0:, ...]
      # ellipsis alone
      _ = checker[...]
      # ellipsis at end
      _ = checker[0:1, ...]
      # ellipsis at begin
      _ = checker[..., 0:1]
      # ellipsis at middle
      _ = checker[0:1, ..., 0:1]
      # multiple ellipses not allowed
      with self.assertRaisesRegexp(ValueError, "Multiple ellipses"):
        _ = checker[..., :, ...].eval()

  def testShrink(self):
    with self.test_session(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw)
      _ = checker[:, :, :, :, 3]
      _ = checker[..., 3]
      _ = checker[:, 0]
      _ = checker[:, :, 0]

  def testTensorIndexing(self):
    with self.test_session(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw, check_type_infer=False)
      bar = constant_op.constant(2)
      bar2 = constant_op.constant(3)
      _ = checker[..., bar:bar2]
      _ = checker[..., bar]
      with self.assertRaisesRegexp(
          TypeError,
          "Value passed to parameter 'begin' has DataType float32 not in "
          "list of allowed values"):
        _ = checker[..., 3.0]
      _ = checker[..., 3]

  def testExpand(self):
    with self.test_session(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw)
      # new axis (followed by implicit ellipsis)
      _ = checker[np.newaxis]
      # newaxis after ellipsis
      _ = checker[..., np.newaxis]
      # newaxis in between ellipsis and explicit range
      _ = checker[..., np.newaxis, :]
      _ = checker[:, ..., np.newaxis, :, :]
      # Reverse final dimension with new axis
      _ = checker[:, :, np.newaxis, :, 2::-1]
      # Ellipsis in middle of two newaxis
      _ = checker[np.newaxis, ..., np.newaxis]

  def testExpandVariable(self):
    with self.test_session(use_gpu=True):
      x = variables.Variable(7, dtype=dtypes.int32)
      x.initializer.run()
      y = x[None].eval()
      self.assertEqual(y.shape, (1,))
      self.assertAllEqual(y, (7,))

  def testOptimizedCases(self):
    with self.test_session(use_gpu=True):
      checker = StridedSliceChecker(self,
                                    StridedSliceChecker.REF_TENSOR_ALIGNED)
      # Identity
      _ = checker[:]
      # Identity
      _ = checker[...]
      # Identity
      _ = checker[np.newaxis, ..., np.newaxis]
      # First axis slice
      _ = checker[1:]
      # First axis slice
      _ = checker[np.newaxis, 1:]


class StridedSliceShapeChecker(object):

  def __init__(self, x):
    self.x = x

  def __getitem__(self, spec):
    op = self.x.__getitem__(spec)
    return op.get_shape()


class StridedSliceShapeTest(test_util.TensorFlowTestCase):
  """Test the shape inference of StridedSliceShapes."""

  def testUnknown(self):
    with self.test_session(use_gpu=True):
      uncertain_tensor = array_ops.placeholder(dtypes.float32)
      a = StridedSliceShapeChecker(uncertain_tensor)
      a_slice_shape = a[...]
      self.assertAllEqual(a_slice_shape.ndims, None)

  def tensorShapeEqual(self, x, y):
    self.assertTrue(x is not None and y is not None or x is None and y is None)
    self.assertEqual(x.as_list(), y.as_list())

  def testTensorShapeUncertain(self):
    with self.test_session(use_gpu=True):
      uncertain_tensor = array_ops.placeholder(
          dtypes.float32, shape=(5, None, 7))
      a = StridedSliceShapeChecker(uncertain_tensor)
      self.tensorShapeEqual(a[3:5], tensor_shape.TensorShape([2, None, 7]))
      self.tensorShapeEqual(a[3:5, :, 4], tensor_shape.TensorShape([2, None]))
      self.tensorShapeEqual(a[3:5, 3:4, 4], tensor_shape.TensorShape([2, None]))
      self.tensorShapeEqual(a[3:5, :, 5:10],
                            tensor_shape.TensorShape([2, None, 2]))
      self.tensorShapeEqual(a[3:5, :, 50:3],
                            tensor_shape.TensorShape([2, None, 0]))
      self.tensorShapeEqual(a[3:5, :, array_ops.newaxis, 50:3,],
                            tensor_shape.TensorShape([2, None, 1, 0]))
      self.tensorShapeEqual(a[1:5:2, :, array_ops.newaxis, 50:3,],
                            tensor_shape.TensorShape([2, None, 1, 0]))
      self.tensorShapeEqual(a[:5:3, :, array_ops.newaxis, 50:3,],
                            tensor_shape.TensorShape([2, None, 1, 0]))
      self.tensorShapeEqual(a[:2:3, :, array_ops.newaxis, 50:3,],
                            tensor_shape.TensorShape([1, None, 1, 0]))
      self.tensorShapeEqual(a[::-1, :, array_ops.newaxis, ::-2],
                            tensor_shape.TensorShape([5, None, 1, 4]))

  def testTensorValuedIndexShape(self):
    with self.test_session(use_gpu=True):
      defined_shape_tensor = array_ops.placeholder(
          dtypes.float32, shape=(5, 3, 7))
      index_value = array_ops.placeholder(dtypes.int32, shape=())
      a = StridedSliceShapeChecker(defined_shape_tensor)
      self.tensorShapeEqual(a[index_value], tensor_shape.TensorShape([3, 7]))
      self.tensorShapeEqual(a[index_value, ::-1],
                            tensor_shape.TensorShape([3, 7]))
      self.tensorShapeEqual(a[index_value, ::-2],
                            tensor_shape.TensorShape([2, 7]))
      other_scalar = array_ops.placeholder(dtypes.int32, shape=())
      self.tensorShapeEqual(a[index_value, other_scalar:2],
                            tensor_shape.TensorShape([None, 7]))


class GradSliceChecker(object):
  """Tests that we can compute a gradient for var^2."""

  def __init__(self, test, sess, var, varnp):
    self.test = test
    self.sess = sess
    self.val = var * var
    self.var = var
    self.varnp = varnp

  def __getitem__(self, spec):
    slice_var = self.var[spec]
    slice_val = self.val[spec]

    # compute analytic 2nd derivative
    analytic_grad2 = 2 * slice_val

    dy = variables.Variable(
        array_ops.ones(shape=slice_var.get_shape(), dtype=dtypes.float32))
    assign = dy.assign(slice_var)
    slice_val_grad, = gradients_impl.gradients(slice_val, self.var, grad_ys=dy)
    slice_val_grad2, = gradients_impl.gradients(
        slice_val_grad, dy, grad_ys=self.var)
    self.sess.run(assign)
    slice_val_grad_evaled, slice_val_grad2_evaled = (
        self.sess.run([slice_val_grad, slice_val_grad2]))
    analytic_grad2_evaled = analytic_grad2.eval()
    self.test.assertAllEqual(slice_val_grad2_evaled, analytic_grad2_evaled)

    # compute analytic gradient for slice
    np_val_grad = (2 * self.varnp * self.varnp)
    np_sliceval_grad = np.zeros(self.var.get_shape())
    np_sliceval_grad[spec] = np_val_grad[spec]
    # verify gradient
    self.test.assertAllEqual(slice_val_grad_evaled, np_sliceval_grad)


class StridedSliceGradTest(test_util.TensorFlowTestCase):
  """Test that strided slice's custom gradient produces correct gradients."""

  def testGradient(self):
    with self.test_session(use_gpu=True) as sess:
      var = variables.Variable(
          array_ops.reshape(
              math_ops.range(1, 97, 1, dtype=dtypes.float32), shape=(6, 4, 4)))
      init = variables.global_variables_initializer()
      sess.run(init)

      grad = GradSliceChecker(self, sess, var,
                              np.array(range(1, 97, 1)).reshape((6, 4, 4)))
      _ = grad[2:6:2, 1:3, 1:3]
      _ = grad[3:0:-2, 1:3, 1:3]
      _ = grad[3:0:-2, array_ops.newaxis, 1:3, 2, array_ops.newaxis]
      _ = grad[3:0:-2, 1:3, 2]
      _ = grad[:, -1, :]
      _ = grad[:, -2, :]
      with self.assertRaisesRegexp(ValueError, "out of bounds"):
        _ = grad[:, -200, :]
      with self.assertRaisesRegexp(ValueError, "out of bounds"):
        _ = grad[:, 200, :]

  def testGradientZero(self):
    with self.test_session(use_gpu=True) as sess:
      var = variables.Variable(8.)
      init = variables.global_variables_initializer()
      sess.run(init)
      grad = GradSliceChecker(self, sess, var, np.array(8))
      _ = grad[tuple()]

  def testInt64Indices(self):
    with self.test_session(use_gpu=True) as sess:
      a = math_ops.range(3, dtype=dtypes.float32)
      index = constant_op.constant(1, dtype=dtypes.int64)
      b = 2. * a[index]
      grad, = gradients_impl.gradients(b, a)
      self.assertAllEqual(sess.run(grad), [0., 2., 0.])


class StridedSliceGradTypeTest(test_util.TensorFlowTestCase):
  """Test varied index types and host located memory."""

  def testHostVsDevice(self):
    with self.test_session(use_gpu=True) as sess:
      var2 = variables.Variable(
          array_ops.reshape(
              math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
              shape=(4, 1, 1)))
      varshape = variables.Variable([6, 4, 4], dtype=dtypes.int32)
      sess.run(variables.global_variables_initializer())
      begin = constant_op.constant([0, 0, 0])
      end = constant_op.constant([4, 1, 1])
      strides = constant_op.constant([1, 1, 1])
      foo = array_ops.strided_slice_grad(varshape, begin, end, strides, var2)
      sess.run(foo)

  def testInt64Shape(self):
    with self.test_session(use_gpu=True) as sess:
      original_dy = array_ops.reshape(
          math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
          shape=(4, 1, 1))
      original_shape = constant_op.constant([6, 4, 4], dtype=dtypes.int64)
      sess.run(variables.global_variables_initializer())
      begin = constant_op.constant([0, 0, 0], dtype=dtypes.int64)
      end = constant_op.constant([4, 1, 1], dtype=dtypes.int64)
      strides = constant_op.constant([1, 1, 1], dtype=dtypes.int64)
      dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                        original_dy)
      sess.run(dx)

  def testMixedIndexTypes(self):
    with self.test_session(use_gpu=True) as sess:
      original_dy = array_ops.reshape(
          math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
          shape=(4, 1, 1))
      original_shape = constant_op.constant([6, 4, 4], dtype=dtypes.int64)
      sess.run(variables.global_variables_initializer())
      begin = constant_op.constant([0, 0, 0], dtype=dtypes.int32)
      end = constant_op.constant([4, 1, 1], dtype=dtypes.int64)
      strides = constant_op.constant([1, 1, 1], dtype=dtypes.int64)
      with self.assertRaisesRegexp(
          TypeError, "Input 'begin' of 'StridedSliceGrad' Op has type int32"
          " that does not match type int64 of argument 'shape'"):
        dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                          original_dy)
        sess.run(dx)


class BenchmarkSlice(object):

  def __init__(self, tensor):
    self.tensor = tensor

  def __getitem__(self, x):
    return self.tensor[x]


class StridedSliceBenchmark(test_lib.Benchmark):
  """Benchmark new strided slice operation on non-trivial case."""

  def run_and_time(self, slice_op):
    variables.global_variables_initializer().run()
    for _ in range(10):
      _ = slice_op.eval()
    iters = 1000
    t0 = time.time()
    for _ in range(iters):
      slice_op.eval()
    t1 = time.time()
    self.report_benchmark(iters=iters, wall_time=(t1 - t0) / 1000.0)

  def make_variable(self):
    n = 256
    shape = (n, n, n)
    items = n**3
    var = variables.Variable(
        array_ops.reshape(math_ops.linspace(1., float(items), items), shape),
        dtype=dtypes.float32)
    return var

  def benchmark_strided_slice_skip(self):
    with session.Session():
      var = self.make_variable()
      helper = BenchmarkSlice(var)
      slice_op = helper[::2, ::1, ::2]
      self.run_and_time(slice_op)

  def benchmark_strided_slice_easy(self):
    with session.Session():
      var = self.make_variable()
      helper = BenchmarkSlice(var)
      slice_op = helper[3::1, 3::1, 3::1]
      self.run_and_time(slice_op)

  def benchmark_slice_easy(self):
    with session.Session():
      var = self.make_variable()
      slice_op = var[3::1, 3::1, 3::1]
      self.run_and_time(slice_op)


class StridedSliceAssignChecker(object):

  def __init__(self, test, x, tensor_type=dtypes.float32, use_resource=False):
    self.tensor_type = tensor_type
    self.test = test
    self._use_resource = use_resource

    self.x_np = np.array(x).astype(tensor_type.as_numpy_dtype)
    # Give the value a non-zero imaginary component for complex types.
    if tensor_type.is_complex:
      self.x_np -= 1j * self.x_np
    self.x = constant_op.constant(self.x_np, dtype=tensor_type)

  def __setitem__(self, index, value):
    value = np.array(value).astype(self.tensor_type.as_numpy_dtype)
    # Give the value a non-zero imaginary component for complex types.
    if self.tensor_type.is_complex:
      value -= 1j * value

    with self.test.test_session(
        use_gpu=not self.tensor_type.is_integer) as sess:
      if self._use_resource:
        var = resource_variable_ops.ResourceVariable(self.x)
      else:
        var = variables.Variable(self.x)
      sess.run(variables.variables_initializer([var]))
      val = sess.run(var[index].assign(value))
      # val_copy is used to check that tf.assign works equivalently to the
      # assign method above.
      val_copy = sess.run(state_ops.assign(var[index], value))
      valnp = np.copy(self.x_np)
      valnp[index] = np.array(value)
      self.test.assertAllEqual(val, valnp)
      self.test.assertAllEqual(val_copy, valnp)


class SliceAssignTest(test_util.TensorFlowTestCase):

  def testInvalidSlice(self):
    with self.test_session() as sess:
      foo = constant_op.constant([1, 2, 3])
      with self.assertRaisesRegexp(ValueError, "Sliced assignment"
                                   " is only supported for variables"):
        bar = foo[:2].assign(constant_op.constant([1, 2]))
        sess.run(bar)

  def doTestSliceAssign(self, use_resource):
    for dtype in STRIDED_SLICE_TYPES:
      checker = StridedSliceAssignChecker(
          self, [[1, 2, 3], [4, 5, 6]],
          use_resource=use_resource,
          tensor_type=dtype)
      # Check if equal
      checker[:] = [[10, 20, 30], [40, 50, 60]]
      # Check trivial (1,1) shape tensor
      checker[1:2, 1:2] = [[66]]
      # shrinks shape changes
      checker[1:2, 1] = [66]
      checker[1, 1:2] = [66]
      checker[1, 1] = 66
      # newaxis shape changes
      checker[:, None, :] = [[[10, 20, 30]], [[40, 50, 50]]]
      # shrink and newaxis
      checker[None, None, 0, 0:1] = [[[99]]]
      # Non unit strides
      checker[::1, ::-2] = [[3, 33], [4, 44]]
      # degenerate interval
      checker[8:10, 0] = []
      checker[8:10, 8:10] = [[]]
    # Assign vector to scalar (rank-0) using newaxis
    checker2 = StridedSliceAssignChecker(self, 222)
    checker2[()] = 6  # no indices
    checker2[...] = 6  # ellipsis
    checker2[None] = [6]  # new axis

  def testSliceAssign(self):
    self.doTestSliceAssign(use_resource=False)

  def testSliceAssignResource(self):
    self.doTestSliceAssign(use_resource=True)

  def testUninitialized(self):
    with self.assertRaisesRegexp(
        errors.FailedPreconditionError,
        "Attempting to use uninitialized value Variable"):
      with self.test_session() as sess:
        v = variables.Variable([1, 2])
        sess.run(v[:].assign([1, 2]))

  def testTypeError(self):
    init_val = constant_op.constant([1, 2], dtype=dtypes.int32)
    too_small_val = constant_op.constant([3, 4], dtype=dtypes.int8)
    too_large_val = constant_op.constant([3, 4], dtype=dtypes.int64)
    v = variables.Variable(init_val)
    with self.assertRaises(TypeError):
      v[:].assign(too_small_val)
    with self.assertRaises(TypeError):
      v[:].assign(too_large_val)

  def testTypeErrorResource(self):
    init_val = constant_op.constant([1, 2], dtype=dtypes.int32)
    too_small_val = constant_op.constant([3, 4], dtype=dtypes.int8)
    too_large_val = constant_op.constant([3, 4], dtype=dtypes.int64)
    v = resource_variable_ops.ResourceVariable(init_val)
    with self.test_session() as sess:
      sess.run(v.initializer)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "l-value dtype int32 does not match r-value dtype int64"):
        sess.run(v[:].assign(too_large_val))
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(v[:].assign(too_small_val))


class ShapeSizeRankTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testDenseShape(self):
    t_value = [[0, 42], [24, 0]]
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(t_value)))
    self.assertEqual(4, self.evaluate(array_ops.size(t_value)))
    self.assertEqual(2, self.evaluate(array_ops.rank(t_value)))

    t = constant_op.constant(t_value)
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(t)))
    self.assertEqual(4, self.evaluate(array_ops.size(t)))
    self.assertEqual(2, self.evaluate(array_ops.rank(t)))

  @test_util.run_in_graph_and_eager_modes()
  def testSparseShape(self):
    sp_value = sparse_tensor.SparseTensorValue(
        indices=((0, 1), (1, 0)), values=(42, 24), dense_shape=(2, 2))
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(sp_value)))
    self.assertEqual(4, self.evaluate(array_ops.size(sp_value)))
    self.assertEqual(2, self.evaluate(array_ops.rank(sp_value)))

    sp = sparse_tensor.SparseTensor.from_value(sp_value)
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(sp)))
    self.assertEqual(4, self.evaluate(array_ops.size(sp)))
    self.assertEqual(2, self.evaluate(array_ops.rank(sp)))

  @test_util.run_in_graph_and_eager_modes()
  def testSizeDtype(self):
    tensor = [1]
    self.assertEqual(dtypes.int32, self.evaluate(array_ops.size(tensor)).dtype)
    self.assertEqual(
        dtypes.int64,
        self.evaluate(array_ops.size(tensor, out_type=dtypes.int64)).dtype)


@test_util.with_c_api
class SequenceMaskTest(test_util.TensorFlowTestCase):

  def testExceptions(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "maxlen must be scalar"):
        array_ops.sequence_mask([10, 20], [10, 20])

  def testOneDimensionalWithMaxlen(self):
    with self.test_session():
      res = array_ops.sequence_mask(constant_op.constant([1, 3, 2]), 5)
      self.assertAllEqual(res.get_shape(), [3, 5])
      self.assertAllEqual(
          res.eval(),
          [[True, False, False, False, False], [True, True, True, False, False],
           [True, True, False, False, False]])

  @test_util.enable_c_shapes
  def testOneDimensionalDtypeWithoutMaxlen(self):
    with self.test_session():
      # test dtype and default maxlen:
      res = array_ops.sequence_mask(constant_op.constant([0, 1, 4]),
                                    dtype=dtypes.float32)
      if ops._USE_C_API:
        self.assertAllEqual(res.get_shape().as_list(), [3, 4])
      else:
        self.assertAllEqual(res.get_shape().as_list(), [3, None])
      self.assertAllEqual(
          res.eval(),
          [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

  @test_util.enable_c_shapes
  def testOneDimensionalWithoutMaxlen(self):
    with self.test_session():
      res = array_ops.sequence_mask(
          constant_op.constant([0, 1, 4]))
      if ops._USE_C_API:
        self.assertAllEqual(res.get_shape().as_list(), [3, 4])
      else:
        self.assertAllEqual(res.get_shape().as_list(), [3, None])
      self.assertAllEqual(
          res.eval(),
          [[False, False, False, False],
           [True, False, False, False],
           [True, True, True, True]])

  @test_util.enable_c_shapes
  def testTwoDimensional(self):
    with self.test_session():
      res = array_ops.sequence_mask(constant_op.constant([[1, 3, 2]]), 5)
      self.assertAllEqual(res.get_shape(), [1, 3, 5])
      self.assertAllEqual(res.eval(), [[[True, False, False, False, False], [
          True, True, True, False, False
      ], [True, True, False, False, False]]])

      # test dtype and default maxlen:
      res = array_ops.sequence_mask(
          constant_op.constant([[0, 1, 4], [1, 2, 3]]), dtype=dtypes.float32)
      if ops._USE_C_API:
        self.assertAllEqual(res.get_shape().as_list(), [2, 3, 4])
      else:
        self.assertAllEqual(res.get_shape().as_list(), [2, 3, None])
      self.assertAllEqual(
          res.eval(),
          [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
           [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]])

  def testUnknownShape(self):
    lengths = array_ops.placeholder(dtype=dtypes.int32)
    res = array_ops.sequence_mask(lengths)
    self.assertEqual(res.shape, None)

  def testDtypes(self):

    def check_dtypes(lengths_dtype, maxlen_dtype):
      res = array_ops.sequence_mask(
          constant_op.constant([1, 3, 2], dtype=lengths_dtype),
          constant_op.constant(5, dtype=maxlen_dtype))
      self.assertAllEqual(res.get_shape(), [3, 5])
      self.assertAllEqual(
          res.eval(),
          [[True, False, False, False, False], [True, True, True, False, False],
           [True, True, False, False, False]])

    with self.test_session():
      check_dtypes(dtypes.int32, dtypes.int32)
      check_dtypes(dtypes.int32, dtypes.int64)
      check_dtypes(dtypes.int64, dtypes.int32)
      check_dtypes(dtypes.int64, dtypes.int64)


class ConcatSliceResourceTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testConcatSlice(self):
    r1 = test_ops.stub_resource_handle_op(container="a", shared_name="b")
    r2 = test_ops.stub_resource_handle_op(container="a", shared_name="c")
    c = array_ops.stack([r1, r2])
    s = array_ops.strided_slice(c, [1], [2])
    self.evaluate(test_ops.resource_create_op(s))
    with self.assertRaises(errors.AlreadyExistsError):
      self.evaluate(test_ops.resource_create_op(r2))


class IdentityTest(test_util.TensorFlowTestCase):

  def testEagerIdentity(self):
    with context.eager_mode():
      ctx = context.get_default_context()
      if not ctx.num_gpus():
        self.skipTest("No GPUs found")

      def _test(x, y, device):
        self.assertAllEqual(x.numpy(), y.numpy())
        self.assertTrue(device in y.device.lower())

      with ops.device("gpu:0"):
        a = constant_op.constant([[2], [3]], dtype=dtypes.float32)
      with ops.device("gpu:0"):
        b = array_ops.identity(a)
        _test(a, b, "gpu")
      with ops.device("cpu:0"):
        c = array_ops.identity(b)
        _test(b, c, "cpu")
      with ops.device("cpu:0"):
        d = array_ops.identity(c)
        _test(c, d, "cpu")
      with ops.device("gpu:0"):
        e = array_ops.identity(d)
        _test(d, e, "gpu")


class PadTest(test_util.TensorFlowTestCase):

  def testEager(self):
    with context.eager_mode():
      t = constant_op.constant([[1, 2, 3], [4, 5, 6]])
      paddings = constant_op.constant([[
          1,
          1,
      ], [2, 2]])
      padded = array_ops.pad(t, paddings, "CONSTANT")
      self.assertAllEqual(padded.numpy(),
                          [[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 0, 0],
                           [0, 0, 4, 5, 6, 0, 0], [0, 0, 0, 0, 0, 0, 0]])


class InvertPermutationTest(test_util.TensorFlowTestCase):

  def testInvertPermutation(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.test_session(use_gpu=True):
        x = constant_op.constant([3, 4, 0, 2, 1], dtype=dtype)
        y = array_ops.invert_permutation(x)
        self.assertAllEqual(y.get_shape(), [5])
        self.assertAllEqual(y.eval(), [2, 4, 3, 0, 1])


class UnravelIndexTest(test_util.TensorFlowTestCase):

  # TODO(b/73086570): Reenable test.
  @unittest.skip("Test does not pass internally.")
  def testUnravelIndex(self):
    with self.test_session():
      for dtype in [dtypes.int32, dtypes.int64]:
        indices_1 = constant_op.constant(1621, dtype=dtype)
        dims_1 = constant_op.constant([6, 7, 8, 9], dtype=dtype)
        out_1 = array_ops.unravel_index(indices_1, dims_1)
        self.assertAllEqual(out_1.eval(), [3, 1, 4, 1])

        indices_2 = constant_op.constant([1621], dtype=dtype)
        dims_2 = constant_op.constant([6, 7, 8, 9], dtype=dtype)
        out_2 = array_ops.unravel_index(indices_2, dims_2)
        self.assertAllEqual(out_2.eval(), [[3], [1], [4], [1]])

        indices_3 = constant_op.constant([22, 41, 37], dtype=dtype)
        dims_3 = constant_op.constant([7, 6], dtype=dtype)
        out_3 = array_ops.unravel_index(indices_3, dims_3)
        self.assertAllEqual(out_3.eval(), [[3, 6, 6], [4, 5, 1]])


class GuaranteeConstOpTest(test_util.TensorFlowTestCase):

  def testSimple(self):
    with self.test_session():
      a = array_ops.constant(10)
      guarantee_a = array_ops.guarantee_const(a)
      self.assertEqual(10, guarantee_a.eval())

  def testVariables(self):
    with self.test_session() as sess:
      for use_resource in [False, True]:
        a = variable_scope.get_variable(
            "var_{}".format(use_resource), [],
            initializer=init_ops.constant_initializer(10.0),
            use_resource=use_resource)
        guarantee_a = array_ops.guarantee_const(a)
        sess.run(variables.global_variables_initializer())
        self.assertEqual(10.0, guarantee_a.eval())

  def testResourceRejection(self):
    with self.test_session() as sess:
      a = variable_scope.get_variable(
          "resource_var", [],
          initializer=init_ops.constant_initializer(10.0),
          use_resource=True)
      guarantee_a = array_ops.guarantee_const(a.handle)
      sess.run(variables.global_variables_initializer())
      with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                               "cannot be a resource variable"):
        guarantee_a.eval()


class SnapshotOpTest(test_util.TensorFlowTestCase):

  def testInvertPermutation(self):
    for dtype in [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]:
      with self.test_session(use_gpu=True):
        x = constant_op.constant([0, 1, 2, 3], dtype=dtype)
        y = gen_array_ops.snapshot(x)
        self.assertAllEqual(y.eval(), [0, 1, 2, 3])


if __name__ == "__main__":
  test_lib.main()
