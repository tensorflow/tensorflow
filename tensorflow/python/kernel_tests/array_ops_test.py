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

import re
import time
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as test_lib


@test_util.run_all_in_graph_and_eager_modes
class BatchMatrixTransposeTest(test_util.TensorFlowTestCase):

  def testNonBatchMatrix(self):
    matrix = [[1, 2, 3], [4, 5, 6]]  # Shape (2, 3)
    expected_transposed = [[1, 4], [2, 5], [3, 6]]  # Shape (3, 2)
    transposed = array_ops.matrix_transpose(matrix)
    self.assertEqual((3, 2), transposed.get_shape())
    self.assertAllEqual(expected_transposed, transposed)

  def testConjugate(self):
    m = [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]]
    expected_transposed = [[1 - 1j, 4 - 4j], [2 - 2j, 5 - 5j], [3 - 3j, 6 - 6j]]
    matrix = ops.convert_to_tensor(m)
    transposed = array_ops.matrix_transpose(matrix, conjugate=True)
    self.assertEqual((3, 2), transposed.get_shape())
    self.assertAllEqual(expected_transposed, transposed)

  def testBatchMatrix(self):
    matrix_0 = [[1, 2, 3], [4, 5, 6]]
    matrix_0_t = [[1, 4], [2, 5], [3, 6]]
    matrix_1 = [[11, 22, 33], [44, 55, 66]]
    matrix_1_t = [[11, 44], [22, 55], [33, 66]]
    batch_matrix = [matrix_0, matrix_1]  # Shape (2, 2, 3)
    expected_transposed = [matrix_0_t, matrix_1_t]  # Shape (2, 3, 2)
    transposed = array_ops.matrix_transpose(batch_matrix)
    self.assertEqual((2, 3, 2), transposed.get_shape())
    self.assertAllEqual(expected_transposed, transposed)

  def testNonBatchMatrixDynamicallyDefined(self):
    # needs explicit `constant` because lists are not automatically
    # converted to sensors when applying `transpose` below
    matrix = constant_op.constant([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
    expected_transposed = [[1, 4], [2, 5], [3, 6]]  # Shape (3, 2)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
    ])
    def transpose(matrix):
      self.assertIs(matrix.shape.ndims, None)
      return array_ops.matrix_transpose(matrix)

    self.assertAllEqual(expected_transposed, transpose(matrix))

  def testBatchMatrixDynamicallyDefined(self):
    matrix_0 = [[1, 2, 3], [4, 5, 6]]
    matrix_0_t = [[1, 4], [2, 5], [3, 6]]
    matrix_1 = [[11, 22, 33], [44, 55, 66]]
    matrix_1_t = [[11, 44], [22, 55], [33, 66]]
    # needs explicit `constant` because lists are not automatically
    # converted to sensors when applying `transpose` below
    batch_matrix = constant_op.constant([matrix_0, matrix_1])  # Shape (2, 2, 3)
    expected_transposed = [matrix_0_t, matrix_1_t]  # Shape (2, 3, 2)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
    ])
    def transpose(matrix):
      self.assertIs(matrix.shape.ndims, None)
      return array_ops.matrix_transpose(matrix)

    self.assertAllEqual(expected_transposed, transpose(batch_matrix))

  def testTensorWithStaticRankLessThanTwoRaisesBecauseNotAMatrix(self):
    vector = [1, 2, 3]
    with self.assertRaisesRegex(ValueError, "should be a "):
      array_ops.matrix_transpose(vector)

  def testNarrowMatrixConjugateTranspose(self):
    for dtype in (dtypes.float32, dtypes.float64):
      for conjugate in (True, False):
        with self.subTest(complex_type=dtype, conjugate=conjugate):
          vector = math_ops.complex(
              constant_op.constant(0, dtype=dtype),
              math_ops.range(96, dtype=dtype))
          column_vector = array_ops.expand_dims(vector, axis=-1)
          row_vector = array_ops.expand_dims(vector, axis=0)
          narrow_matrix = array_ops.tile(column_vector, [1, 2])  # [96, 2]
          expected_transposed = array_ops.tile(row_vector, [2, 1])  # [2, 96]
          if conjugate:
            expected_transposed = -expected_transposed

          transposed = array_ops.matrix_transpose(
              narrow_matrix, conjugate=conjugate)

          self.assertEqual((2, 96), transposed.get_shape())
          self.assertAllEqual(expected_transposed, transposed)


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
    with self.cached_session():
      masked_tensor = array_ops.boolean_mask(arr, mask, axis=axis)

      # Leading dimension size of masked_tensor is always unknown until runtime
      # since we don't how many elements will be kept.
      leading = 1 if axis is None else axis + 1
      self.assertAllEqual(masked_tensor.get_shape()[leading:],
                          masked_arr.shape[leading:])

      self.assertAllClose(masked_arr, masked_tensor)

  @test_util.run_deprecated_v1
  def testMaskDim1ArrDim2Axis1(self):
    ndims_mask = 1
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape, axis=1)

  @test_util.run_deprecated_v1
  def testMaskDim2ArrDim2Axis1(self):
    ndims_mask = 2
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape, axis=1)

  @test_util.run_deprecated_v1
  def testMaskDim1ArrDim1(self):
    ndims_mask = 1
    for arr_shape in [(1,), (2,), (3,), (10,)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  @test_util.run_deprecated_v1
  def testMaskDim1ArrDim2(self):
    ndims_mask = 1
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  @test_util.run_deprecated_v1
  def testMaskDim2ArrDim2(self):
    ndims_mask = 2
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  @test_util.run_deprecated_v1
  def testMaskDim2ArrDim3(self):
    ndims_mask = 2
    for arr_shape in [(1, 1, 1), (1, 2, 2), (2, 2, 1)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  @test_util.run_deprecated_v1
  def testEmptyInput2D(self):
    mask = np.array([True, False])
    arr = np.array([[], []]).astype(np.float32)
    numpy_result = arr[mask]
    tf_result = array_ops.boolean_mask(arr, mask)
    self.assertAllEqual(numpy_result.shape[1:], tf_result.get_shape()[1:])
    with self.cached_session():
      self.assertAllClose(numpy_result, tf_result)

  @test_util.run_deprecated_v1
  def testEmptyInput1D(self):
    mask = np.array([]).astype(bool)
    arr = np.array([]).astype(np.float32)
    numpy_result = arr[mask]
    tf_result = array_ops.boolean_mask(arr, mask)
    self.assertAllEqual(numpy_result.shape[1:], tf_result.get_shape()[1:])
    with self.cached_session():
      self.assertAllClose(numpy_result, tf_result)

  @test_util.run_deprecated_v1
  def testEmptyOutput(self):
    make_mask = lambda shape: np.zeros(shape, dtype=bool)
    for ndims_mask in range(1, 4):
      for ndims_arr in range(ndims_mask, ndims_mask + 3):
        for _ in range(3):
          with self.subTest(ndims_mask=ndims_mask, ndims_arr=ndims_arr, _=_):
            arr_shape = np.random.randint(1, 5, size=ndims_arr)
            self.CheckVersusNumpy(ndims_mask, arr_shape, make_mask=make_mask)

  @test_util.run_deprecated_v1
  def testWorksWithDimensionsEqualToNoneDuringGraphBuild(self):
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    with self.cached_session() as sess:
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

  @test_util.run_deprecated_v1
  def testMaskDimensionsSetToNoneRaises(self):
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.int32, shape=[None, 2])
      mask = array_ops.placeholder(dtypes.bool, shape=None)
      with self.assertRaisesRegex(ValueError, "dimensions must be specified"):
        array_ops.boolean_mask(tensor, mask)

  def testMaskHasMoreDimsThanTensorRaises(self):
    mask = [[True, True], [False, False]]
    tensor = [1, 2, 3, 4]
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "incompatible"):
        array_ops.boolean_mask(tensor, mask).eval()

  def testMaskIsScalarRaises(self):
    mask = True
    tensor = 1
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "mask.*scalar"):
        array_ops.boolean_mask(tensor, mask).eval()

  def testMaskShapeDifferentThanFirstPartOfTensorShapeRaises(self):
    mask = [True, True, True]
    tensor = [[1, 2], [3, 4]]
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "incompatible"):
        array_ops.boolean_mask(tensor, mask).eval()

  @test_util.run_deprecated_v1
  def testStringMask(self):
    # Reproduces b/111171330, where the optimized boolean_mask graph would
    # be incorrectly placed on GPU.
    with ops.Graph().as_default():
      tile_placeholder = array_ops.placeholder(dtypes.int32, [2])
      string_tensor = array_ops.tile([["hello"]], tile_placeholder)
      bool_tensor = array_ops.tile([[True]], tile_placeholder)
      masked_tensor = array_ops.boolean_mask(string_tensor, bool_tensor)
      config = config_pb2.ConfigProto()
      config.graph_options.rewrite_options.shape_optimization = 1
      config.gpu_options.per_process_gpu_memory_fraction = 0.3
      with session.Session(config=config) as sess:
        result = sess.run(masked_tensor, feed_dict={tile_placeholder: [2, 2]})
        self.assertAllEqual([b"hello", b"hello", b"hello", b"hello"], result)

  def testMaskWithAxisTensor(self):

    @def_function.function(autograph=False)
    def f():
      return array_ops.boolean_mask([1, 2, 3], [True, False, True],
                                    axis=constant_op.constant(
                                        0, dtype=dtypes.int32))

    self.assertAllEqual(self.evaluate(f()), [1, 3])

  def testMaskWithAxisNonConstTensor(self):

    @def_function.function(
        autograph=False,
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
        ])
    def f(axis):
      return array_ops.boolean_mask([1, 2, 3], [True, False, True], axis=axis)

    self.assertAllEqual(
        self.evaluate(f(constant_op.constant(0, dtype=dtypes.int32))), [1, 3])


@test_util.run_all_in_graph_and_eager_modes
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

    with self.assertRaisesRegex(
        Exception, "Can not squeeze dim.1., expected a dimension of 1, got 3"):
      matrix_squeezed = array_ops.squeeze(matrix, [1])

  def testSqueezeScalarDim(self):
    matrix = [[1, 2, 3]]
    matrix_squeezed = array_ops.squeeze(matrix, 0)
    self.assertEqual(matrix_squeezed.get_shape(), (3))

  def testExpandDimsWithNonScalarDim(self):
    with self.assertRaisesRegex(Exception,
                                "must be a tensor with a single value"):
      array_ops.expand_dims(1, axis=[0, 1])


class ReverseV2Test(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testReverse0DimAuto(self):
    x_np = 4
    for use_gpu in [False, True]:
      with self.subTest(use_gpu=use_gpu):
        with self.cached_session(use_gpu=use_gpu):
          x_tf = array_ops.reverse_v2(x_np, []).eval()
          self.assertAllEqual(x_tf, x_np)

  def _reverse1DimAuto(self, np_dtype):
    x_np = np.array([1, 200, 3, 40, 5], dtype=np_dtype)

    for use_gpu in [False, True]:
      for axis_dtype in [dtypes.int32, dtypes.int64]:
        with self.subTest(use_gpu=use_gpu, axis_dtype=axis_dtype):
          with self.cached_session(use_gpu=use_gpu):
            x_tf = array_ops.reverse_v2(
                x_np, constant_op.constant([0], dtype=axis_dtype)).eval()
            self.assertAllEqual(x_tf, np.asarray(x_np)[::-1])

  def _reverse2DimAuto(self, np_dtype):
    x_np = np.array([[1, 200, 3], [4, 5, 60]], dtype=np_dtype)

    for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
      for use_gpu in [False, True]:
        for axis_dtype in [dtypes.int32, dtypes.int64]:
          with self.subTest(
              reverse_f=reverse_f, use_gpu=use_gpu, axis_dtype=axis_dtype):
            with self.cached_session(use_gpu=use_gpu):
              x_tf_1 = reverse_f(x_np,
                                 constant_op.constant([0],
                                                      dtype=axis_dtype)).eval()
              x_tf_2 = reverse_f(x_np,
                                 constant_op.constant([-2],
                                                      dtype=axis_dtype)).eval()
              x_tf_3 = reverse_f(x_np,
                                 constant_op.constant([1],
                                                      dtype=axis_dtype)).eval()
              x_tf_4 = reverse_f(x_np,
                                 constant_op.constant([-1],
                                                      dtype=axis_dtype)).eval()
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
  @test_util.run_deprecated_v1
  def testInvalidAxis(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "is out of valid range"):
      array_ops.reverse_v2(x_np, [-30])
    with self.assertRaisesRegex(ValueError, "is out of valid range"):
      array_ops.reverse_v2(x_np, [2])
    with self.assertRaisesRegex(ValueError, "axis 0 specified more than once"):
      array_ops.reverse_v2(x_np, [0, -2])

  # This is the version of reverse that uses axis indices rather than
  # bool tensors
  # TODO(b/32254538): Change this test to use array_ops.reverse
  #
  # Note: this test passes placeholder as constant axis is validated
  # in shape function (see testInvalidAxis)
  @test_util.run_deprecated_v1
  def testInvalid(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = array_ops.placeholder(dtypes.int32)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is out of.*range"):
        array_ops.reverse_v2(x_np, axis).eval(feed_dict={axis: [-30]})
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is out of.*range"):
        array_ops.reverse_v2(x_np, axis).eval(feed_dict={axis: [2]})
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "(axis 0 specified more than once|canonicalized axis 0 was repeated.)"
      ):
        array_ops.reverse_v2(x_np, axis).eval(feed_dict={axis: [0, -2]})

  @test_util.run_deprecated_v1
  def testReverse1DimAuto(self):
    for dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.bool,
        np.float16, np.float32, np.float64, np.complex64, np.complex128,
        np.array(b"").dtype.type
    ]:
      self._reverse1DimAuto(dtype)

  @test_util.run_deprecated_v1
  def testReverse2DimAuto(self):
    for dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.bool,
        np.float16, np.float32, np.float64, np.complex64, np.complex128,
        np.array(b"").dtype.type
    ]:
      self._reverse2DimAuto(dtype)

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def testReverseRowsOf3Channels(self):
    """Tests optimized code for reversing rows with last dim size = 3."""
    with self.session():
      for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
        for outer_size in (1, 2):
          for middle_size in list(range(50)) + [100000]:
            with self.subTest(
                reverse_f=reverse_f,
                outer_size=outer_size,
                middle_size=middle_size):
              x_np = np.reshape(
                  np.arange(outer_size * middle_size * 3, dtype=np.float32),
                  newshape=(outer_size, middle_size, 3))
              x_tf = reverse_f(x_np, [1]).eval()
              np_answer = x_np[:, ::-1, :]
              self.assertAllEqual(x_tf, np_answer)

  @test_util.run_deprecated_v1
  def testReverseRowsOf4Channels(self):
    with self.session():
      for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
        for outer_size in (1, 2):
          for middle_size in list(range(50)) + [100000]:
            with self.subTest(
                reverse_f=reverse_f,
                outer_size=outer_size,
                middle_size=middle_size):
              x_np = np.reshape(
                  np.arange(outer_size * middle_size * 4, dtype=np.float32),
                  newshape=(outer_size, middle_size, 4))
              x_tf = reverse_f(x_np, [1]).eval()
              np_answer = x_np[:, ::-1, :]
              self.assertAllEqual(x_tf, np_answer)

  @test_util.run_deprecated_v1
  def testReverseColumnsOf3Channels(self):
    with self.session():
      for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
        for outer_size in list(range(50)) + [100000]:
          for middle_size in (1, 2):
            with self.subTest(
                reverse_f=reverse_f,
                outer_size=outer_size,
                middle_size=middle_size):
              x_np = np.reshape(
                  np.arange(outer_size * middle_size * 3, dtype=np.float32),
                  newshape=(outer_size, middle_size, 3))
              x_tf = reverse_f(x_np, [0]).eval()
              np_answer = x_np[::-1, :, :]
              self.assertAllEqual(x_tf, np_answer)

  def testReverseInvalidShape(self):
    x = np.ndarray(shape=[0, 1, 1])
    v = array_ops.reverse_v2(x, axis=[1])
    self.assertAllEqual(self.evaluate(v), v)


class MeshgridTest(test_util.TensorFlowTestCase):

  def _compareDiff(self, x, y, use_gpu):
    for index in ("ij", "xy"):
      numpy_out = np.meshgrid(x, y, indexing=index)
      tf_out = array_ops.meshgrid(x, y, indexing=index)
      with self.cached_session(use_gpu=use_gpu):
        for xx, yy in zip(numpy_out, tf_out):
          self.assertAllEqual(xx, yy)

  def _compareDiffType(self, n, np_dtype, use_gpu):
    inputs = []
    for index in ("ij", "xy"):
      for _ in range(n):
        x = np.linspace(-10, 10, 5).astype(np_dtype)
        if np_dtype in (np.complex64, np.complex128):
          x += 1j
        inputs.append(x)
      numpy_out = np.meshgrid(*inputs, indexing=index)
      with self.cached_session(use_gpu=use_gpu):
        tf_out = array_ops.meshgrid(*inputs, indexing=index)
        for x_np, x_tf in zip(numpy_out, tf_out):
          self.assertAllEqual(x_np, x_tf)

  @test_util.run_deprecated_v1
  def testCompare(self):
    for t in (np.float16, np.float32, np.float64, np.int32, np.int64,
              np.complex64, np.complex128):
      with self.subTest(t=t):
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
    if tensor_type.is_bool:
      self.x_np = np.array(x % 3).astype(np.bool)
    # Give the value a non-zero imaginary component for complex types.
    if tensor_type.is_complex:
      self.x_np -= 1j * self.x_np
    self.test = test
    self.x = constant_op.constant(self.x_np, dtype=tensor_type)
    self.check_type_infer = check_type_infer

  def __getitem__(self, spec):
    op = self.x.__getitem__(spec)

    def eval_if_tensor(x):
      try:
        return x.eval()
      except AttributeError:
        return x

    if isinstance(spec, bool) or \
      (isinstance(spec, ops.Tensor) and spec.dtype == dtypes.bool) or \
      (isinstance(spec, np.ndarray) and spec.dtype == bool) or \
      (isinstance(spec, (list, tuple)) and np.asarray(spec).dtype == bool):
      tensor = op.eval()
      np_spec = eval_if_tensor(spec)
      self.test.assertAllEqual(self.x_np[np_spec], tensor)
      return tensor

    if not isinstance(spec, (list, tuple)):
      spec = [spec]

    tensor = op.eval()

    # Make a numpy spec that pre-evals the tensors
    np_specs = []

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
    dtypes.float64, dtypes.complex64, dtypes.complex128, dtypes.bool
]


class StridedSliceTest(test_util.TensorFlowTestCase):
  """Test the strided slice operation with variants of slices."""

  @test_util.run_deprecated_v1
  def test_basic_slice(self):
    for tensor_type in STRIDED_SLICE_TYPES:
      with self.subTest(tensor_type=tensor_type):
        with self.cached_session():
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

    with test_util.force_gpu():
      x = constant_op.constant([1., 2., 3.])
      begin = constant_op.constant([2], dtype=dtypes.int64)
      end = constant_op.constant([3], dtype=dtypes.int64)
      strides = constant_op.constant([1], dtype=dtypes.int64)
      s = array_ops.strided_slice(x, begin, end, strides)
      self.assertAllEqual([3.], self.evaluate(s))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  @test_util.assert_no_garbage_created
  def testTensorSliceEagerMemory(self):
    with context.eager_mode():
      inputs = constant_op.constant([[[1], [2], [3], [4]]],
                                    dtype=dtypes.float32)
      # Tests that slicing an EagerTensor doesn't leak memory
      inputs[0]  # pylint: disable=pointless-statement

  @test_util.assert_no_new_pyobjects_executing_eagerly
  @test_util.assert_no_garbage_created
  def testVariableSliceEagerMemory(self):
    with context.eager_mode():
      v = variables.Variable([1., 2.])
      v[0]  # pylint: disable=pointless-statement

  @test_util.run_deprecated_v1
  def testDegenerateSlices(self):
    with self.session():
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      # degenerate by offering a forward interval with a negative stride
      _ = checker[0:-1:-1, :, :]
      # degenerate with a reverse interval with a positive stride
      _ = checker[-1:0, :, :]
      # empty interval in every dimension
      _ = checker[-1:0, 2:2, 2:3:-1]
      # empty first dimension only (used to break for aligned tensors).
      checker = StridedSliceChecker(self,
                                    StridedSliceChecker.REF_TENSOR_ALIGNED)
      _ = checker[1:0]

  @test_util.run_deprecated_v1
  def testSliceWithUndefinedDimension(self):
    t = constant_op.constant([1, 2, 3])
    d = tensor_shape.Dimension(None)
    self.assertAllEqual(t[d:d:d], t)

  @test_util.run_deprecated_v1
  def testEllipsis(self):
    with self.session():
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
      with self.assertRaisesRegex(ValueError, "Multiple ellipses"):
        _ = checker[..., :, ...].eval()

  @test_util.run_deprecated_v1
  def testShrink(self):
    with self.session():
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw)
      _ = checker[:, :, :, :, 3]
      _ = checker[..., 3]
      _ = checker[:, 0]
      _ = checker[:, :, 0]

  @test_util.run_deprecated_v1
  def testBothNewAxisAndShrink(self):
    with self.session():
      ones = array_ops.placeholder(shape=[2, 2], dtype=dtypes.int16)
      self.assertAllEqual(
          ones[array_ops.newaxis, :,
               0].eval(feed_dict={ones: [[1, 1], [1, 1]]}), [[1, 1]])

  @test_util.run_deprecated_v1
  def testTensorIndexing(self):
    with self.session():
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw, check_type_infer=False)
      bar = constant_op.constant(2)
      bar2 = constant_op.constant(3)
      _ = checker[..., bar:bar2]
      _ = checker[..., bar]
      _ = checker[..., 3]
      _ = checker[..., 2**64 // 2**63]  # Test longs in Python 2

  def testTensorIndexingTypeError(self):
    with self.session():
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      expected = re.escape(array_ops._SLICE_TYPE_ERROR)
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker["foo"]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant("foo")]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[0.0]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant(0.0)]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant([1, 2, 3])]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[[2.1, -0.7, 1.5]]

  @test_util.run_deprecated_v1
  def testExpand(self):
    with self.session():
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

  @test_util.run_deprecated_v1
  def testExpandVariable(self):
    with self.session():
      x = variables.Variable(7, dtype=dtypes.int32)
      self.evaluate(x.initializer)
      y = x[None].eval()
      self.assertEqual(y.shape, (1,))
      self.assertAllEqual(y, (7,))

  @test_util.run_deprecated_v1
  def testOptimizedCases(self):
    with self.session():
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

  @test_util.run_v1_only("currently failing on v2")
  def testMasks(self):
    with self.session():
      scalar = np.array(0)
      # Test tensor type mask
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      _ = checker[checker.x > 2]
      _ = checker[checker.x <= 5]
      _ = checker[ops.convert_to_tensor(scalar)]

      # Test numpy array type mask
      raw = np.array([[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
                       [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23,
                                                              24]]]]])
      checker1 = StridedSliceChecker(self, raw)
      _ = checker1[raw >= 4]
      _ = checker1[raw < 19]
      _ = checker1[scalar]

      # Test boolean and non boolean cases
      mask = np.array([True, False, True])
      raw1 = np.array([[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]])
      checker2 = StridedSliceChecker(self, raw1)
      _ = checker2[mask]
      _ = checker2[ops.convert_to_tensor(mask)]


class StridedSliceShapeChecker(object):

  def __init__(self, x):
    self.x = x

  def __getitem__(self, spec):
    op = self.x.__getitem__(spec)
    return op.get_shape()


class StridedSliceShapeTest(test_util.TensorFlowTestCase):
  """Test the shape inference of StridedSliceShapes."""

  @test_util.run_deprecated_v1
  def testUnknown(self):
    with self.session():
      uncertain_tensor = array_ops.placeholder(dtypes.float32)
      a = StridedSliceShapeChecker(uncertain_tensor)
      a_slice_shape = a[...]
      self.assertAllEqual(a_slice_shape.ndims, None)

  def tensorShapeEqual(self, x, y):
    self.assertTrue(x is not None and y is not None or x is None and y is None)
    self.assertEqual(x.as_list(), y.as_list())

  @test_util.run_deprecated_v1
  def testTensorShapeUncertain(self):
    with self.session():
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

  @test_util.run_deprecated_v1
  def testTensorValuedIndexShape(self):
    with self.session():
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
        array_ops.ones_like(slice_var, dtype=dtypes.float32))
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
    if isinstance(spec, ops.Tensor):
      spec = self.sess.run([spec])
    np_sliceval_grad[spec] = np_val_grad[spec]
    # verify gradient
    self.test.assertAllEqual(slice_val_grad_evaled, np_sliceval_grad)


class StridedSliceGradTest(test_util.TensorFlowTestCase):
  """Test that strided slice's custom gradient produces correct gradients."""

  @test_util.run_v1_only("b/120545219")
  def testGradient(self):
    with self.session() as sess:
      var = variables.Variable(
          array_ops.reshape(
              math_ops.range(1, 97, 1, dtype=dtypes.float32), shape=(6, 4, 4)))
      init = variables.global_variables_initializer()
      sess.run(init)

      raw = np.array(range(1, 97, 1)).reshape((6, 4, 4))
      grad = GradSliceChecker(self, sess, var, raw)
      _ = grad[2:6:2, 1:3, 1:3]
      _ = grad[3:0:-2, 1:3, 1:3]
      _ = grad[3:0:-2, array_ops.newaxis, 1:3, 2, array_ops.newaxis]
      _ = grad[3:0:-2, 1:3, 2]
      _ = grad[:, -1, :]
      _ = grad[:, -2, :]
      with self.assertRaisesRegex(ValueError, "out of bounds"):
        _ = grad[:, -200, :]
      with self.assertRaisesRegex(ValueError, "out of bounds"):
        _ = grad[:, 200, :]

      # Test numpy array type mask
      _ = grad[raw > 51]
      # Test tensor type mask
      _ = grad[ops.convert_to_tensor(raw) <= 76]

  @test_util.run_v1_only("b/120545219")
  def testGradientZero(self):
    with self.session() as sess:
      var = variables.Variable(8.)
      init = variables.global_variables_initializer()
      sess.run(init)
      grad = GradSliceChecker(self, sess, var, np.array(8))
      _ = grad[tuple()]

  @test_util.run_deprecated_v1
  def testInt64Indices(self):
    with self.session():
      a = math_ops.range(3, dtype=dtypes.float32)
      index = constant_op.constant(1, dtype=dtypes.int64)
      b = 2. * a[index]
      grad, = gradients_impl.gradients(b, a)
      self.assertAllEqual(self.evaluate(grad), [0., 2., 0.])


class StridedSliceGradTypeTest(test_util.TensorFlowTestCase):
  """Test varied index types and host located memory."""

  @test_util.run_deprecated_v1
  def testHostVsDevice(self):
    with self.session() as sess:
      var2 = variables.Variable(
          array_ops.reshape(
              math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
              shape=(4, 1, 1)))
      varshape = variables.Variable([6, 4, 4], dtype=dtypes.int32)
      self.evaluate(variables.global_variables_initializer())
      begin = constant_op.constant([0, 0, 0])
      end = constant_op.constant([4, 1, 1])
      strides = constant_op.constant([1, 1, 1])
      foo = array_ops.strided_slice_grad(varshape, begin, end, strides, var2)
      sess.run(foo)

  @test_util.run_deprecated_v1
  def testInt64Shape(self):
    with self.session() as sess:
      original_dy = array_ops.reshape(
          math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
          shape=(4, 1, 1))
      original_shape = constant_op.constant([6, 4, 4], dtype=dtypes.int64)
      self.evaluate(variables.global_variables_initializer())
      begin = constant_op.constant([0, 0, 0], dtype=dtypes.int64)
      end = constant_op.constant([4, 1, 1], dtype=dtypes.int64)
      strides = constant_op.constant([1, 1, 1], dtype=dtypes.int64)
      dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                        original_dy)
      sess.run(dx)

  @test_util.run_deprecated_v1
  def testMixedIndexTypes(self):
    with self.session() as sess:
      original_dy = array_ops.reshape(
          math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
          shape=(4, 1, 1))
      original_shape = constant_op.constant([6, 4, 4], dtype=dtypes.int64)
      self.evaluate(variables.global_variables_initializer())
      begin = constant_op.constant([0, 0, 0], dtype=dtypes.int32)
      end = constant_op.constant([4, 1, 1], dtype=dtypes.int64)
      strides = constant_op.constant([1, 1, 1], dtype=dtypes.int64)
      with self.assertRaisesRegex(
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
    self.evaluate(variables.global_variables_initializer())
    for _ in range(10):
      _ = self.evaluate(slice_op)
    iters = 1000
    t0 = time.time()
    for _ in range(iters):
      self.evaluate(slice_op)
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

    with self.test.test_session() as sess:
      if self._use_resource:
        var = resource_variable_ops.ResourceVariable(self.x)
      else:
        var = variables.Variable(self.x)
      sess.run(variables.variables_initializer([var]))
      val = sess.run(var[index].assign(value))
      # val_copy is used to check that tf.compat.v1.assign works equivalently
      # to the assign method above.
      val_copy = sess.run(state_ops.assign(var[index], value))
      valnp = np.copy(self.x_np)
      valnp[index] = np.array(value)
      self.test.assertAllEqual(val, valnp)
      self.test.assertAllEqual(val_copy, valnp)


class SliceAssignTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testInvalidSlice(self):
    foo = constant_op.constant([1, 2, 3])
    with self.assertRaisesRegex(AttributeError, "no attribute 'assign'"):
      bar = foo[:2].assign(constant_op.constant([1, 2]))
      self.evaluate(bar)

  def doTestSliceAssign(self, use_resource):
    for dtype in STRIDED_SLICE_TYPES:
      with self.subTest(dtype=dtype):
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

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123559667")
  def testSliceAssign(self):
    self.doTestSliceAssign(use_resource=False)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123559667")
  def testSliceAssignResource(self):
    self.doTestSliceAssign(use_resource=True)

  @test_util.run_v1_only("b/120545219")
  def testUninitialized(self):
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "Attempting to use uninitialized value Variable"):
      with self.cached_session() as sess:
        v = variables.VariableV1([1, 2])
        sess.run(v[:].assign([1, 2]))

  @test_util.run_v1_only("b/120545219")
  def testTypeError(self):
    init_val = constant_op.constant([1, 2], dtype=dtypes.int32)
    too_small_val = constant_op.constant([3, 4], dtype=dtypes.int8)
    too_large_val = constant_op.constant([3, 4], dtype=dtypes.int64)
    v = variables.VariableV1(init_val)
    with self.assertRaises(TypeError):
      v[:].assign(too_small_val)
    with self.assertRaises(TypeError):
      v[:].assign(too_large_val)

  @test_util.run_deprecated_v1
  def testTypeErrorResource(self):
    init_val = constant_op.constant([1, 2], dtype=dtypes.int32)
    too_small_val = constant_op.constant([3, 4], dtype=dtypes.int8)
    too_large_val = constant_op.constant([3, 4], dtype=dtypes.int64)
    v = resource_variable_ops.ResourceVariable(init_val)
    with self.cached_session() as sess:
      self.evaluate(v.initializer)
      with self.assertRaises(ValueError):
        sess.run(v[:].assign(too_large_val))
      with self.assertRaises(ValueError):
        sess.run(v[:].assign(too_small_val))

  @test_util.disable_xla("b/123559667")
  @test_util.run_in_graph_and_eager_modes
  def testTensorStridedSliceUpdateWithInputForward(self):
    """Tests tensor_strided_slice_update with input-forwarding taking effect."""
    @def_function.function
    def assign(x):
      y = x + 1
      return gen_array_ops.tensor_strided_slice_update(y, [0], [1], [1], [0])
    self.assertAllEqual([0, 1], self.evaluate(assign(array_ops.zeros([2]))))

  @test_util.disable_xla("b/123559667")
  @test_util.run_in_graph_and_eager_modes
  def testTensorStridedSliceUpdateNoInputForward(self):
    """Tests tensor_strided_slice_update with no input-forwarding."""
    x = constant_op.constant([0.2, 0.3])
    y = x + 1
    # y's buffer won't be forwarded to z because y and z will be alive at the
    # same time later.
    z = gen_array_ops.tensor_strided_slice_update(y, [0], [1], [1], [0.4])
    ans = y + z
    self.assertAllClose([1.6, 2.6], self.evaluate(ans))

  @test_util.disable_xla("b/123559667")
  def testTensorStridedSliceUpdateGradSimple(self):
    original = constant_op.constant([0.2, 0.3])
    updates = constant_op.constant([0.4])
    with backprop.GradientTape() as tape:
      tape.watch([original, updates])
      updated = gen_array_ops.tensor_strided_slice_update(
          original, [0], [1], [1], updates)
    d1, d2 = tape.gradient(updated, [original, updates],
                           output_gradients=constant_op.constant([2.0, 3.0]))
    self.assertAllClose([0.0, 3.0], d1)
    self.assertAllClose([2.0], d2)

  @parameterized.named_parameters(
      ("_%s" % i, *args) for i, args in enumerate([  # pylint:disable=g-complex-comprehension
          ([2, 5], [0, 1], [1, 0], [1, 2], [2], 0, 2, 0, 0, 1),
          ([4], [5], [3], [1], [3], 1, 0, 0, 0, 0),
          ([2, 2, 3, 2], [0, 0, 1], [1, 0, 2], [1, 0, 1], [2, 3], 0, 0, 2, 0, 5)
      ]))
  @test_util.disable_xla("b/123559667")
  def testTensorStridedSliceUpdateGrad(
      self, shape, begin, end, strides, updates_shape, *args):
    with self.cached_session():
      def f(a, b):
        return gen_array_ops.tensor_strided_slice_update(
            a, begin, end, strides, b, *args)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [array_ops.zeros(shape), array_ops.ones(updates_shape)], delta=1.0)
      self.assertAllClose(theoretical, numerical)


class ShapeSizeRankTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testDenseShape(self):
    t_value = [[0, 42], [24, 0]]
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(t_value)))
    self.assertEqual(4, self.evaluate(array_ops.size(t_value)))
    self.assertEqual(2, self.evaluate(array_ops.rank(t_value)))

    t = constant_op.constant(t_value)
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(t)))
    self.assertEqual(4, self.evaluate(array_ops.size(t)))
    self.assertEqual(2, self.evaluate(array_ops.rank(t)))

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
  def testSizeDtype(self):
    tensor = [1]
    self.assertEqual(dtypes.int32, self.evaluate(array_ops.size(tensor)).dtype)
    self.assertEqual(
        dtypes.int64,
        self.evaluate(array_ops.size(tensor, out_type=dtypes.int64)).dtype)


class SequenceMaskTest(test_util.TensorFlowTestCase):

  def testExceptions(self):
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "maxlen must be scalar"):
        array_ops.sequence_mask([10, 20], [10, 20])

  @test_util.run_deprecated_v1
  def testOneDimensionalWithMaxlen(self):
    with self.cached_session():
      res = array_ops.sequence_mask(constant_op.constant([1, 3, 2]), 5)
      self.assertAllEqual(res.get_shape(), [3, 5])
      self.assertAllEqual(
          res,
          [[True, False, False, False, False], [True, True, True, False, False],
           [True, True, False, False, False]])

  @test_util.run_deprecated_v1
  def testOneDimensionalDtypeWithoutMaxlen(self):
    with self.cached_session():
      # test dtype and default maxlen:
      res = array_ops.sequence_mask(
          constant_op.constant([0, 1, 4]), dtype=dtypes.float32)
      self.assertAllEqual(res.get_shape().as_list(), [3, 4])
      self.assertAllEqual(
          res,
          [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

  @test_util.run_deprecated_v1
  def testOneDimensionalWithoutMaxlen(self):
    with self.cached_session():
      res = array_ops.sequence_mask(constant_op.constant([0, 1, 4]))
      self.assertAllEqual(res.get_shape().as_list(), [3, 4])
      self.assertAllEqual(
          res, [[False, False, False, False], [True, False, False, False],
                [True, True, True, True]])

  @test_util.run_deprecated_v1
  def testTwoDimensional(self):
    with self.cached_session():
      res = array_ops.sequence_mask(constant_op.constant([[1, 3, 2]]), 5)
      self.assertAllEqual(res.get_shape(), [1, 3, 5])
      self.assertAllEqual(res, [[[True, False, False, False, False],
                                 [True, True, True, False, False],
                                 [True, True, False, False, False]]])

      # test dtype and default maxlen:
      res = array_ops.sequence_mask(
          constant_op.constant([[0, 1, 4], [1, 2, 3]]), dtype=dtypes.float32)
      self.assertAllEqual(res.get_shape().as_list(), [2, 3, 4])
      self.assertAllEqual(
          res,
          [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
           [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]])

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    lengths = array_ops.placeholder(dtype=dtypes.int32)
    res = array_ops.sequence_mask(lengths)
    self.assertEqual(res.shape, None)

  @test_util.run_deprecated_v1
  def testDtypes(self):

    def check_dtypes(lengths_dtype, maxlen_dtype):
      res = array_ops.sequence_mask(
          constant_op.constant([1, 3, 2], dtype=lengths_dtype),
          constant_op.constant(5, dtype=maxlen_dtype))
      self.assertAllEqual(res.get_shape(), [3, 5])
      self.assertAllEqual(
          res,
          [[True, False, False, False, False], [True, True, True, False, False],
           [True, True, False, False, False]])

    with self.cached_session():
      check_dtypes(dtypes.int32, dtypes.int32)
      check_dtypes(dtypes.int32, dtypes.int64)
      check_dtypes(dtypes.int64, dtypes.int32)
      check_dtypes(dtypes.int64, dtypes.int64)

  def testOutputDtype(self):

    def check_output_dtype(output_dtype):
      res = self.evaluate(
          array_ops.sequence_mask(
              constant_op.constant([1, 3, 2], dtype=dtypes.int32),
              constant_op.constant(5, dtype=dtypes.int32),
              dtype=output_dtype))
      self.assertAllEqual(
          res,
          self.evaluate(
              math_ops.cast([[True, False, False, False, False],
                             [True, True, True, False, False],
                             [True, True, False, False, False]], output_dtype)))

    check_output_dtype(dtypes.bool)
    check_output_dtype("bool")
    check_output_dtype(np.bool)
    check_output_dtype(dtypes.int32)
    check_output_dtype("int32")
    check_output_dtype(np.int32)
    check_output_dtype(dtypes.float32)
    check_output_dtype("float32")
    check_output_dtype(np.float32)
    check_output_dtype(dtypes.int64)
    check_output_dtype("float64")
    check_output_dtype(np.float64)


class ConcatSliceResourceTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def testConcatSlice(self):
    r1 = test_ops.stub_resource_handle_op(container="a", shared_name="b")
    r2 = test_ops.stub_resource_handle_op(container="a", shared_name="c")
    c = array_ops.stack([r1, r2])
    s = array_ops.strided_slice(c, [1], [2])
    self.evaluate(test_ops.resource_create_op(s))
    with self.assertRaises(errors.AlreadyExistsError):
      self.evaluate(test_ops.resource_create_op(r2))


class IdentityTest(test_util.TensorFlowTestCase):

  @test_util.run_gpu_only
  def testEagerIdentity(self):
    with context.eager_mode():

      def _test(x, y, device):
        self.assertAllEqual(x.numpy(), y.numpy())
        self.assertTrue(device in y.device.lower())

      with test_util.force_gpu():
        a = constant_op.constant([[2], [3]], dtype=dtypes.float32)
      with test_util.force_gpu():
        b = array_ops.identity(a)
        _test(a, b, "gpu")
      with test_util.force_cpu():
        c = array_ops.identity(b)
        _test(b, c, "cpu")
      with test_util.force_cpu():
        d = array_ops.identity(c)
        _test(c, d, "cpu")
      with test_util.force_gpu():
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

  def testSymmetricMirrorPadGrad(self):
    t = np.broadcast_to(np.arange(0, 7), (3, 2, 1, 7))
    paddings = constant_op.constant([
        [1, 1],
        [0, 0],
        [0, 0],
        [2, 2],
    ])
    expected = np.broadcast_to(np.array([9, 27, 27]), (1, 2, 1, 3))
    result = gen_array_ops.mirror_pad_grad(t, paddings, "SYMMETRIC")
    self.assertAllEqual(result, expected)

  def testReflectMirrorPadGrad(self):
    t = np.broadcast_to(np.reshape(np.arange(0, 7), (7, 1)), (1, 4, 7, 1))
    paddings = constant_op.constant([
        [0, 0],
        [1, 1],
        [2, 2],
        [0, 0],
    ])
    expected = np.broadcast_to(
        np.reshape(np.array([16, 18, 8]), (3, 1)), (1, 2, 3, 1))
    result = gen_array_ops.mirror_pad_grad(t, paddings, "REFLECT")
    self.assertAllEqual(result, expected)


class InvertPermutationTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testInvertPermutation(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.subTest(dtype=dtype):
        with self.cached_session():
          x = constant_op.constant([3, 4, 0, 2, 1], dtype=dtype)
          y = array_ops.invert_permutation(x)
          self.assertAllEqual(y.get_shape(), [5])
          self.assertAllEqual(y, [2, 4, 3, 0, 1])


class UnravelIndexTest(test_util.TensorFlowTestCase):

  # TODO(b/73086570): Reenable test.
  @unittest.skip("Test does not pass internally.")
  def testUnravelIndex(self):
    with self.cached_session():
      for dtype in [dtypes.int32, dtypes.int64]:
        with self.subTest(dtype=dtype):
          indices_1 = constant_op.constant(1621, dtype=dtype)
          dims_1 = constant_op.constant([6, 7, 8, 9], dtype=dtype)
          out_1 = array_ops.unravel_index(indices_1, dims_1)
          self.assertAllEqual(out_1, [3, 1, 4, 1])

          indices_2 = constant_op.constant([1621], dtype=dtype)
          dims_2 = constant_op.constant([6, 7, 8, 9], dtype=dtype)
          out_2 = array_ops.unravel_index(indices_2, dims_2)
          self.assertAllEqual(out_2, [[3], [1], [4], [1]])

          indices_3 = constant_op.constant([22, 41, 37], dtype=dtype)
          dims_3 = constant_op.constant([7, 6], dtype=dtype)
          out_3 = array_ops.unravel_index(indices_3, dims_3)
          self.assertAllEqual(out_3, [[3, 6, 6], [4, 5, 1]])

  # Test case for GitHub issue 40204.
  def testUnravelIndexZeroDim(self):
    with self.cached_session():
      for dtype in [dtypes.int32, dtypes.int64]:
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    "index is out of bound as with dims"):
          indices = constant_op.constant([2, 5, 7], dtype=dtype)
          dims = constant_op.constant([3, 0], dtype=dtype)
          self.evaluate(array_ops.unravel_index(indices=indices, dims=dims))


class GuaranteeConstOpTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testSimple(self):
    with self.cached_session():
      a = array_ops.constant(10)
      guarantee_a = array_ops.guarantee_const(a)
      self.assertEqual(10, self.evaluate(guarantee_a))

  @test_util.run_deprecated_v1
  def testVariables(self):
    with self.cached_session() as sess:
      for use_resource in [False, True]:
        with self.subTest(use_resource=use_resource):
          a = variable_scope.get_variable(
              "var_{}".format(use_resource), [],
              initializer=init_ops.constant_initializer(10.0),
              use_resource=use_resource)
          guarantee_a = array_ops.guarantee_const(a)
          self.evaluate(variables.global_variables_initializer())
          self.assertEqual(10.0, self.evaluate(guarantee_a))

  @test_util.run_deprecated_v1
  def testResourceRejection(self):
    with self.cached_session() as sess:
      a = variable_scope.get_variable(
          "resource_var", [],
          initializer=init_ops.constant_initializer(10.0),
          use_resource=True)
      guarantee_a = array_ops.guarantee_const(a.handle)
      self.evaluate(variables.global_variables_initializer())
      with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                               "cannot be a resource variable"):
        self.evaluate(guarantee_a)


class SnapshotOpTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testInvertPermutation(self):
    for dtype in [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]:
      with self.subTest(dtype=dtype):
        with self.cached_session():
          x = constant_op.constant([0, 1, 2, 3], dtype=dtype)
          y = gen_array_ops.snapshot(x)
          self.assertAllEqual(y, [0, 1, 2, 3])


@test_util.run_all_in_graph_and_eager_modes
class QuantizeAndDequantizeTest(test_util.TensorFlowTestCase):

  # Generates a tensor of the specified `shape` using values from `values`
  # scaled by (slice_idx + 1) along `axis` dimension.
  def _scale_per_slice(self, shape, axis, values):
    # Note: repeats the values if the shape is larger than values.
    out = np.take(values, np.remainder(np.arange(np.prod(shape)),
                                       len(values))).reshape(shape)
    if axis is not None:
      scale_shape = [1] * len(shape)
      scale_shape[axis] = shape[axis]
      out *= np.arange(1, shape[axis] + 1).reshape(scale_shape)
    return out

  def testAxis(self):
    shape = np.array([2, 3, 4, 5])
    values = np.array([-1, -0.5, 0, 0.3, 0.8, 0.555, 0.5], dtype=np.float32)
    quant_values = np.array(
        [-1, -0.5, 0, 38.0 / 128, 102.0 / 128, 71.0 / 128, 0.5],
        dtype=np.float32)
    for axis in [None, 0, 1, 2, 3]:
      with self.subTest(axis=axis):
        inputs = constant_op.constant(
            self._scale_per_slice(shape, axis, values))
        expected = self._scale_per_slice(shape, axis, quant_values)
        unused_minmax_value = 0 if axis is None else [0] * shape[axis]
        fake_quantized = self.evaluate(
            array_ops.quantize_and_dequantize_v2(
                inputs,
                unused_minmax_value,
                unused_minmax_value,
                range_given=False,
                round_mode="HALF_UP",
                axis=axis))
        self.assertAllEqual(fake_quantized, expected)
        if axis is not None:
          fake_quantized = self.evaluate(
              array_ops.quantize_and_dequantize_v2(
                  inputs,
                  unused_minmax_value,
                  unused_minmax_value,
                  range_given=False,
                  axis=(axis - 4)))
          self.assertAllClose(fake_quantized, expected)

  def testBadAxis(self):
    input_tensor = [2.5, 2.5]
    input_min = [0, 0]
    input_max = [1, 1]
    error_message_pattern = "Shape must be at least rank 11 but is rank 1"
    # TODO(b/171260356): Eager mode and graph mode throw different error types
    error = errors.InvalidArgumentError if context.executing_eagerly(
    ) else ValueError
    with self.assertRaisesRegex(error, error_message_pattern):
      self.evaluate(
          array_ops.quantize_and_dequantize_v2(
              input=input_tensor,
              input_min=input_min,
              input_max=input_max,
              axis=10))

  def testQuantizeDequantizeGrad(self):
    shape = (2, 2)
    max_threshold = 0
    min_threshold = -10
    input_value = np.random.rand(2, 2) * 40.0 - 20.0
    input_tensor = constant_op.constant(input_value, shape=shape,
                                        name="input_tensor")
    with self.cached_session():
      def f(a):
        return array_ops.quantize_and_dequantize_v2(
            a,
            input_min=min_threshold,
            input_max=max_threshold,
            range_given=True)
      output_grad = gradient_checker_v2.compute_gradient(f, [input_tensor])
      self.assertAllClose(output_grad[0], np.zeros([1, 4, 4]))


@test_util.run_all_in_graph_and_eager_modes
class SortedSearchTest(test_util.TensorFlowTestCase):

  def testUpperBoundFloatHandCoded(self):
    cdf = np.array([0, .2, .5, .6, .8, 1.], dtype=np.float32)
    arr = np.array([.04, .99, .53, .58, .31, .01, .79, .8, .21],
                   dtype=np.float32)
    result = np.searchsorted(cdf, arr, side="right")
    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))
    self.assertAllEqual(result, tf_result)

  def testUpperBoundFloatRandomNd(self):
    dim_size = 7
    for d in range(1, 5):
      shape = [dim_size] * d
      cdf = np.cumsum(
          np.random.uniform(size=shape).astype(np.float32), axis=(d - 1))
      arr = np.random.uniform(size=shape).astype(np.float32) * dim_size

      tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))

      cdf = cdf.reshape([-1, dim_size])
      arr = arr.reshape([-1, dim_size])
      result = np.zeros(arr.shape, dtype=np.int32)
      for i in range(dim_size**(d - 1)):
        result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="right")

      result = result.reshape(shape)

      self.assertAllEqual(result, tf_result)

  def testUpperBoundFloatUneven(self):
    batch_size = 7
    size_search_array = 1000
    size_values = 47
    cdf = np.cumsum(
        np.random.uniform(size=[batch_size, size_search_array]).astype(
            np.float32),
        axis=1)
    arr = np.random.uniform(size=[batch_size, size_values]).astype(
        np.float32) * size_search_array

    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))

    result = np.zeros(arr.shape, dtype=np.int32)
    for i in range(batch_size):
      result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="right")

    self.assertAllEqual(result, tf_result)

  def testLowerBoundFloatHandCoded(self):
    cdf = np.array([0, .2, .5, .6, .8, 1.], dtype=np.float32)
    arr = np.array([.04, .99, .53, .58, .31, .01, .79, .8, .21],
                   dtype=np.float32)
    result = np.searchsorted(cdf, arr, side="left")
    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))
    self.assertAllEqual(result, tf_result)

  def testLowerBoundFloatRandomNd(self):
    dim_size = 7
    for d in range(1, 5):
      shape = [dim_size] * d
      cdf = np.cumsum(
          np.random.uniform(size=shape).astype(np.float32), axis=(d - 1))
      arr = np.random.uniform(size=shape).astype(np.float32) * dim_size

      tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))

      cdf = cdf.reshape([-1, dim_size])
      arr = arr.reshape([-1, dim_size])
      result = np.zeros(arr.shape, dtype=np.int32)
      for i in range(dim_size**(d - 1)):
        result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="left")

      result = result.reshape(shape)

      self.assertAllEqual(result, tf_result)

  def testLowerBoundFloatUneven(self):
    batch_size = 7
    size_search_array = 1000
    size_values = 47
    cdf = np.cumsum(
        np.random.uniform(size=[batch_size, size_search_array]).astype(
            np.float32),
        axis=1)
    arr = np.random.uniform(size=[batch_size, size_values]).astype(
        np.float32) * size_search_array

    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))

    result = np.zeros(arr.shape, dtype=np.int32)
    for i in range(batch_size):
      result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="left")

    self.assertAllEqual(result, tf_result)

  def testUpperBoundIntHandCoded(self):
    cdf = np.array([0, 20, 50, 60, 80, 100], dtype=np.int64)
    arr = np.array([4, 99, 53, 58, 31, 1, 79, 8, 21], dtype=np.int64)
    result = np.searchsorted(cdf, arr, side="right")
    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))
    self.assertAllEqual(result, tf_result)

  def testUpperBoundIntRandomNd(self):
    dim_size = 7
    for d in range(1, 5):
      shape = [dim_size] * d
      cdf = np.cumsum(
          np.random.randint(low=0, high=10, size=shape).astype(np.int64),
          axis=(d - 1))
      arr = np.random.randint(
          low=0, high=10 * dim_size, size=shape).astype(np.int64)

      tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))

      cdf = cdf.reshape([-1, dim_size])
      arr = arr.reshape([-1, dim_size])
      result = np.zeros(arr.shape, dtype=np.int32)
      for i in range(dim_size**(d - 1)):
        result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="right")

      result = result.reshape(shape)

      self.assertAllEqual(result, tf_result)

  def testUpperBoundIntUneven(self):
    batch_size = 7
    size_search_array = 1000
    size_values = 47
    cdf = np.cumsum(
        np.random.randint(low=0, high=10,
                          size=[batch_size,
                                size_search_array]).astype(np.int64),
        axis=1)
    arr = np.random.randint(
        low=0, high=10 * size_search_array, size=[batch_size,
                                                  size_values]).astype(np.int64)

    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))

    result = np.zeros(arr.shape, dtype=np.int32)
    for i in range(batch_size):
      result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="right")

    self.assertAllEqual(result, tf_result)

  def testLowerBoundIntHandCoded(self):
    cdf = np.array([0, 20, 50, 60, 80, 100], dtype=np.int64)
    arr = np.array([4, 99, 53, 58, 31, 1, 79, 8, 21], dtype=np.int64)
    result = np.searchsorted(cdf, arr, side="left")
    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))
    self.assertAllEqual(result, tf_result)

  def testLowerBoundIntRandomNd(self):
    dim_size = 7
    for d in range(1, 5):
      shape = [dim_size] * d
      cdf = np.cumsum(
          np.random.randint(low=0, high=10, size=shape).astype(np.int64),
          axis=(d - 1))
      arr = np.random.randint(
          low=0, high=10 * dim_size, size=shape).astype(np.int64)

      tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))

      cdf = cdf.reshape([-1, dim_size])
      arr = arr.reshape([-1, dim_size])
      result = np.zeros(arr.shape, dtype=np.int32)
      for i in range(dim_size**(d - 1)):
        result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="left")

      result = result.reshape(shape)

      self.assertAllEqual(result, tf_result)

  def testLowerBoundIntUneven(self):
    batch_size = 7
    size_search_array = 1000
    size_values = 47
    cdf = np.cumsum(
        np.random.randint(low=0, high=10,
                          size=[batch_size,
                                size_search_array]).astype(np.int64),
        axis=1)
    arr = np.random.randint(
        low=0, high=10 * size_search_array, size=[batch_size,
                                                  size_values]).astype(np.int64)

    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))

    result = np.zeros(arr.shape, dtype=np.int32)
    for i in range(batch_size):
      result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="left")

    self.assertAllEqual(result, tf_result)

  def testZeroSequenceSize(self):
    dtype = dtypes.int32
    for side in ("left", "right"):
      with self.subTest(side=side):
        self.assertAllEqual(
            array_ops.searchsorted(
                array_ops.ones([2, 0]),
                array_ops.ones([2, 3]),
                side=side,
                out_type=dtype), array_ops.zeros([2, 3], dtype))

  def testZeroValueSize(self):
    dtype = dtypes.int32
    for side in ("left", "right"):
      with self.subTest(side=side):
        self.assertAllEqual(
            array_ops.searchsorted(
                array_ops.ones([2, 3]),
                array_ops.ones([2, 0]),
                side=side,
                out_type=dtype), array_ops.zeros([2, 0], dtype))


class BatchGatherNdTest(test_util.TensorFlowTestCase):

  def testShapesMatch(self):
    """Tests for various different shape combinations."""
    shapes = []
    # params_shape, indices_shape, batch_dims
    shapes.append(((2, 2, 2), (2, 1), 1),)
    shapes.append(((2, 2, 2), (2, 2), 1),)
    shapes.append(((2, 2, 2), (2, 3), 0),)
    shapes.append(((2, 2, 2), (3,), 0),)
    shapes.append(((2, 2, 2), (1,), 0),)
    shapes.append(((2, 2, 3, 2), (2, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3, 1), 1),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 2),)

    for params_shape, indices_shape, batch_dims in shapes:
      with self.subTest(
          params_shape=params_shape,
          indices_shape=indices_shape,
          batch_dims=batch_dims):
        params = constant_op.constant(1.0, shape=(params_shape))
        indices = constant_op.constant(
            1, shape=(indices_shape), dtype=dtypes.int32)
        out = array_ops.batch_gather_nd(
            params=params, indices=indices, batch_dims=batch_dims)
        ndims_params = len(params_shape) - batch_dims
        ndims_rows = ndims_params - indices_shape[-1]
        expected_out_shape = indices_shape[:-1]
        if ndims_rows > 0:
          expected_out_shape += params_shape[-ndims_rows:]
        self.assertSequenceEqual(out.shape, expected_out_shape)

  def testReducesToGatherNDWhenBatchDimIsZero(self):
    """Confirms setting batch_dims to zero reduces to tf.gather_nd."""
    params = constant_op.constant(np.random.uniform(0.0, 1.0, size=(7, 8, 9)))
    indices_shapes = []
    indices_shapes.append((1,))
    indices_shapes.append((3, 1))
    indices_shapes.append((3, 3, 1))
    indices_shapes.append((2,))
    indices_shapes.append((3, 2))
    indices_shapes.append((3, 3, 2))
    indices_shapes.append((3,))
    indices_shapes.append((3, 3))
    indices_shapes.append((3, 3, 3))

    for indices_shape in indices_shapes:
      with self.subTest(indices_shape=indices_shape):
        indices = np.random.randint(0, 7, size=indices_shape)
        gather_nd_result = gen_array_ops.gather_nd(params, indices)
        batch_gather_nd_result = array_ops.batch_gather_nd(
            params=params, indices=indices, batch_dims=0)
        self.assertAllEqual(gather_nd_result, batch_gather_nd_result)

  def testSameResultAsMapFn(self):
    """Compares results with gather_nd called on every element with map_fn."""
    shapes = []
    # params_shape, indices_shape, batch_dims
    shapes.append(((2, 2, 2), (2, 1), 1),)
    shapes.append(((2, 2, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3, 1), 1),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 2),)

    for params_shape, indices_shape, batch_dims in shapes:
      with self.subTest(
          params_shape=params_shape,
          indices_shape=indices_shape,
          batch_dims=batch_dims):
        params = constant_op.constant(
            np.random.uniform(0.0, 1.0, size=(params_shape)))
        indices = np.random.randint(0, 2, size=indices_shape)
        batch_gather_nd_result = array_ops.batch_gather_nd(
            params=params, indices=indices, batch_dims=batch_dims)

        if batch_dims > 1:
          params = array_ops.reshape(
              params, shape=[-1] + list(params_shape[batch_dims:]))
          indices = array_ops.reshape(
              indices, shape=[-1] + list(indices_shape[batch_dims:]))

        map_fn_gather_nd_result = map_fn.map_fn(
            fn=self._map_fn_body, elems=(params, indices), dtype=dtypes.float64)

        if batch_dims > 1:
          out_shape = map_fn_gather_nd_result.shape.as_list()
          out_shape = list(params_shape[:batch_dims]) + out_shape[1:]
          map_fn_gather_nd_result = array_ops.reshape(
              map_fn_gather_nd_result, shape=out_shape)

        self.assertAllEqual(map_fn_gather_nd_result, batch_gather_nd_result)

  def _map_fn_body(self, elems):
    return gen_array_ops.gather_nd(elems[0], elems[1])

  def testBatchDimsAsTensor(self):
    """Tests Tensor batch_dims as input works as intended."""
    shapes = []
    # params_shape, indices_shape, batch_dims
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 0),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 1),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 2),)

    for params_shape, indices_shape, batch_dims in shapes:
      with self.subTest(
          params_shape=params_shape,
          indices_shape=indices_shape,
          batch_dims=batch_dims):
        params = constant_op.constant(
            np.random.uniform(0.0, 1.0, size=(params_shape)))
        indices = np.random.randint(0, 2, size=indices_shape)
        batch_gather_nd_result = array_ops.gather_nd(
            params=params, indices=indices, batch_dims=batch_dims)
        batch_dims_tensor = constant_op.constant([batch_dims])
        batch_gather_nd_tensor_batch_dims_result = array_ops.gather_nd(
            params=params, indices=indices, batch_dims=batch_dims_tensor)

        self.assertAllEqual(batch_gather_nd_tensor_batch_dims_result,
                            batch_gather_nd_result)

  def testInvalidBatchDimsRaisesException(self):
    """Tests whether invalid batch_dims raise expected exceptions."""
    params = constant_op.constant(
        np.random.uniform(0.0, 1.0, size=(3, 2, 2, 3, 4)))
    indices = np.random.randint(0, 2, size=(3, 2, 3))

    with self.assertRaises(TypeError):
      array_ops.batch_gather_nd(
          params=params,
          indices=indices,
          batch_dims=constant_op.constant((0, 1)))

    with self.assertRaises(ValueError):
      array_ops.batch_gather_nd(params=params, indices=indices, batch_dims=-1)

    with self.assertRaises(ValueError):
      array_ops.batch_gather_nd(params=params, indices=indices, batch_dims=4)

  @test_util.run_deprecated_v1
  def testNoneBatchDimensions(self):
    """Tests gather_nd works with None dimensions."""
    shapes = []
    # params_shape, indices_shape, batch_dims
    shapes.append(((2, 2, 2), (2, 1), 1),)
    shapes.append(((2, 2, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3, 1), 1),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 2),)

    for params_shape, indices_shape, batch_dims in shapes:
      params_ph_shape = list(params_shape)
      indices_ph_shape = list(indices_shape)
      for i in range(batch_dims):
        params_ph_shape[i] = None
        indices_ph_shape[i] = None

      params = array_ops.placeholder(dtypes.float32, shape=params_ph_shape)
      indices = array_ops.placeholder(dtypes.int32, shape=indices_ph_shape)
      out = array_ops.batch_gather_nd(
          params=params, indices=indices, batch_dims=batch_dims)

      with self.cached_session() as sess:
        params_val = np.ones(dtype=np.float32, shape=params_shape)
        indices_val = np.ones(dtype=np.int32, shape=indices_shape)
        res = sess.run(
            out, feed_dict={
                params: params_val,
                indices: indices_val
            })
      row_ndims = len(params_shape) - batch_dims - indices_shape[-1]
      expected_out_shape = indices_shape[:-1]
      if row_ndims > 0:
        expected_out_shape += params_shape[-row_ndims:]

      self.assertSequenceEqual(res.shape, expected_out_shape)

  @test_util.run_deprecated_v1
  def testUnknownIndices(self):
    """Tests whether indices with unknown rank works correctly."""
    params = constant_op.constant(((0, 1, 2),))
    indices = array_ops.placeholder(dtypes.int32)
    gather_nd_t = array_ops.gather_nd(params, indices, batch_dims=1)
    shape = gather_nd_t.get_shape()
    self.assertEqual(None, shape.ndims)
    self.assertEqual(None, tensor_shape.dimension_value(shape[0]))


@test_util.run_all_in_graph_and_eager_modes
class RepeatTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters(
      (3, 4, None),
      ([[1, 2], [3, 4]], 2, None),
      ([[1, 2], [3, 4]], [1, 2], 0),
      ([[1, 2], [3, 4]], [1, 2], 1),
      ([[1, 2], [3, 4]], 3, 1),
      ([[1, 2], [3, 4]], [1, 2, 3, 4], None),
      (np.ones([0, 4]), 0, 1),
      (np.ones([1, 2]), [2], None),
  )
  def testRepeat(self, array, repeats, axis):
    array = np.array(array)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)] * 2)
    def repeat_fn(array, repeats):
      return array_ops.repeat(array, repeats, axis)

    v_tf = array_ops.repeat(constant_op.constant(array), repeats, axis)
    v_tf_fn = repeat_fn(
        constant_op.constant(array, dtype=dtypes.int32), repeats)
    v_np = np.repeat(array, repeats, axis)
    self.assertAllEqual(v_tf, v_np)
    self.assertAllEqual(v_tf_fn, v_np)


@test_util.run_all_in_graph_and_eager_modes
class TileVariantTest(test_util.TensorFlowTestCase):

  def test_tile_tensor_list(self):
    t = constant_op.constant(np.random.uniform(size=[2, 3, 4]))
    handle = list_ops.tensor_list_from_tensor(t, element_shape=None)
    with ops.device("CPU:0"):
      tiled_handles = array_ops.tile(array_ops.reshape(handle, [1]), [2])
    tiled_tensor_0 = list_ops.tensor_list_stack(tiled_handles[0], t.dtype, 2,
                                                [3, 4])
    tiled_tensor_1 = list_ops.tensor_list_stack(tiled_handles[1], t.dtype, 2,
                                                [3, 4])
    self.assertAllEqual(t, tiled_tensor_0)
    self.assertAllEqual(t, tiled_tensor_1)
    # Now mutate some of the lists and make sure the changes are not reflected
    # in the tiled handles.
    with ops.control_dependencies([
        list_ops.tensor_list_scatter([t[0] + 1], [0], input_handle=handle),
        list_ops.tensor_list_set_item(tiled_handles[0], 0, t[0] + 2)]):
      tiled_tensor_0 = list_ops.tensor_list_stack(tiled_handles[0], t.dtype, 2,
                                                  [3, 4])
      tiled_tensor_1 = list_ops.tensor_list_stack(tiled_handles[1], t.dtype, 2,
                                                  [3, 4])
    self.assertAllEqual(t, tiled_tensor_0)
    self.assertAllEqual(t, tiled_tensor_1)


if __name__ == "__main__":
  test_lib.main()
