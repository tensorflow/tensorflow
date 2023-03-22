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
"""Tests for tensorflow.ops.tf.gather."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

_TEST_TYPES = (dtypes.int64, dtypes.bfloat16, dtypes.float32, dtypes.complex64,
               dtypes.complex128)
_INDEX_TYPES = (dtypes.int16, dtypes.int32, dtypes.int64)

# TODO(virimia): Add a benchmark for gather_v2, with batch_dims and axis set.


def _to_str_elements(values):
  """Converts the inner list elements to strings."""
  if isinstance(values, list):
    return [_to_str_elements(value) for value in values]
  else:
    return str(values).encode("utf-8")


class GatherTest(test.TestCase, parameterized.TestCase):

  def _buildParams(self, data, dtype):
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testScalar1D(self):
    with self.cached_session():
      data = np.array([0, 1, 2, 3, 7, 5])
      for dtype in _TEST_TYPES:
        for itype in _INDEX_TYPES:
          for indices in 4, [1, 2, 2, 4, 5]:
            with self.subTest(dtype=dtype, itype=itype, indices=indices):
              params_np = self._buildParams(data, dtype)
              params = constant_op.constant(params_np)
              indices_tf = constant_op.constant(indices, dtype=itype)
              gather_t = array_ops.gather(params, indices_tf)
              gather_val = self.evaluate(gather_t)
              np_val = params_np[indices]
              self.assertAllEqual(np_val, gather_val)
              self.assertEqual(np_val.shape, gather_t.get_shape())

  def testScalar2D(self):
    with self.session():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for itype in _INDEX_TYPES:
          for axis in range(data.ndim):
            with self.subTest(dtype=dtype, itype=itype, axis=axis):
              params_np = self._buildParams(data, dtype)
              params = constant_op.constant(params_np)
              indices = constant_op.constant(2, dtype=itype)
              gather_t = array_ops.gather(params, indices, axis=axis)
              gather_val = self.evaluate(gather_t)
              self.assertAllEqual(np.take(params_np, 2, axis=axis), gather_val)
              expected_shape = data.shape[:axis] + data.shape[axis + 1 :]
              self.assertEqual(expected_shape, gather_t.get_shape())

  def testSimpleTwoD32(self):
    with self.session():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for itype in _INDEX_TYPES:
          for axis in range(data.ndim):
            with self.subTest(dtype=dtype, itype=itype, axis=axis):
              params_np = self._buildParams(data, dtype)
              params = constant_op.constant(params_np)
              # The indices must be in bounds for any axis.
              indices = constant_op.constant([0, 1, 0, 2], dtype=itype)
              gather_t = array_ops.gather(params, indices, axis=axis)
              gather_val = self.evaluate(gather_t)
              self.assertAllEqual(
                  np.take(params_np, [0, 1, 0, 2], axis=axis), gather_val
              )
              expected_shape = data.shape[:axis] + (4,) + data.shape[axis + 1 :]
              self.assertEqual(expected_shape, gather_t.get_shape())

  def testHigherRank(self):
    with ops.Graph().as_default():
      # We check that scalar and empty indices shapes work as well
      shape = (2, 1, 3, 2)
      for indices_shape in (), (0,), (2, 0), (2, 3):
        for dtype in _TEST_TYPES:
          for axis in range(len(shape)):
            params = self._buildParams(np.random.randn(*shape), dtype)
            indices = np.random.randint(shape[axis], size=indices_shape)
            with self.subTest(
                indices_shape=indices_shape,
                dtype=dtype,
                axis=axis,
                indices=indices,
            ):
              tf_params = constant_op.constant(params)
              tf_indices = constant_op.constant(indices)
              # Check that both positive and negative indices for axis work.
              tf_axis = constant_op.constant(axis)
              tf_negative_axis = constant_op.constant(-len(shape) + axis)
              gather = array_ops.gather(tf_params, tf_indices, axis=tf_axis)
              gather_negative_axis = array_ops.gather(
                  tf_params, tf_indices, axis=tf_negative_axis
              )
              gather_value, gather_negative_axis_value = self.evaluate(
                  [gather, gather_negative_axis]
              )
              gather_np = np.take(params, indices, axis)
              self.assertAllEqual(gather_np, gather_value)
              self.assertAllEqual(gather_np, gather_negative_axis_value)
              expected_shape = (
                  params.shape[:axis] + indices.shape + params.shape[axis + 1 :]
              )
              self.assertEqual(expected_shape, gather.shape)
              self.assertEqual(expected_shape, gather_negative_axis.shape)

              # Test gradients
              gather_grad = np.random.randn(
                  *gather.get_shape().as_list()
              ).astype(dtype.as_numpy_dtype)
              if dtype.is_complex:
                gather_grad -= 1j * gather_grad
              params_grad, indices_grad, axis_grad = gradients_impl.gradients(
                  gather, [tf_params, tf_indices, tf_axis], gather_grad
              )
              self.assertIsNone(indices_grad)
              self.assertIsNone(axis_grad)
              if dtype.is_integer:
                self.assertIsNone(params_grad)
                continue
              # For axis 0, we are able to create an efficient IndexedSlices
              # for the gradient.
              if axis == 0:
                self.assertEqual(
                    type(params_grad), indexed_slices.IndexedSlices
                )
                params_grad = ops.convert_to_tensor(params_grad)
              correct_params_grad = np.zeros(shape).astype(dtype.as_numpy_dtype)
              outer_dims = axis
              inner_dims = len(shape) - axis - 1
              gather_grad = gather_grad.reshape(
                  shape[:axis] + (indices.size,) + shape[axis + 1 :]
              )
              for source_index, dest_index in enumerate(indices.flat):
                dest_slice = (
                    (slice(None),) * outer_dims
                    + (dest_index,)
                    + (slice(None),) * inner_dims
                )
                source_slice = (
                    (slice(None),) * outer_dims
                    + (source_index,)
                    + (slice(None),) * inner_dims
                )
                correct_params_grad[dest_slice] += gather_grad[source_slice]
              self.assertAllCloseAccordingToType(
                  correct_params_grad,
                  self.evaluate(params_grad),
                  atol=2e-6,
                  rtol=2e-6,
              )

  def testHigherRankGradientTape(self):
    # We check that scalar and empty indices shapes work as well
    shape = (2, 1, 3, 2)
    for indices_shape in (), (0,), (2, 0), (2, 3):
      for dtype in _TEST_TYPES:
        for axis in range(len(shape)):
          params = self._buildParams(np.random.randn(*shape), dtype)
          indices = np.random.randint(shape[axis], size=indices_shape)
          with self.subTest(
              indices_shape=indices_shape,
              dtype=dtype,
              axis=axis,
              indices=indices,
          ):
            with backprop.GradientTape() as tape:
              tf_params = constant_op.constant(params)
              tf_indices = constant_op.constant(indices)
              # Check that both positive and negative indices for axis work.
              tf_axis = constant_op.constant(axis)
              tape.watch(tf_params)
              tape.watch(tf_indices)
              tape.watch(tf_axis)
              tf_negative_axis = constant_op.constant(-len(shape) + axis)
              gather = array_ops.gather(tf_params, tf_indices, axis=tf_axis)
              gather_negative_axis = array_ops.gather(
                  tf_params, tf_indices, axis=tf_negative_axis
              )
              gather_value, gather_negative_axis_value = self.evaluate(
                  [gather, gather_negative_axis]
              )
              gather_np = np.take(params, indices, axis)
              self.assertAllEqual(gather_np, gather_value)
              self.assertAllEqual(gather_np, gather_negative_axis_value)
              expected_shape = (
                  params.shape[:axis] + indices.shape + params.shape[axis + 1 :]
              )
              self.assertEqual(expected_shape, gather.shape)
              self.assertEqual(expected_shape, gather_negative_axis.shape)

              # Test gradients
              gather_grad = np.random.randn(
                  *gather.get_shape().as_list()
              ).astype(dtype.as_numpy_dtype)
              if dtype.is_complex:
                gather_grad -= 1j * gather_grad
            params_grad, indices_grad, axis_grad = tape.gradient(
                gather, [tf_params, tf_indices, tf_axis], gather_grad
            )
            self.assertIsNone(indices_grad)
            self.assertIsNone(axis_grad)
            if dtype.is_integer:
              self.assertIsNone(params_grad)
              continue
            # For axis 0, we are able to create an efficient IndexedSlices for
            # the gradient.
            if axis == 0:
              self.assertEqual(type(params_grad), indexed_slices.IndexedSlices)
              params_grad = ops.convert_to_tensor(params_grad)
            correct_params_grad = np.zeros(shape).astype(dtype.as_numpy_dtype)
            outer_dims = axis
            inner_dims = len(shape) - axis - 1
            gather_grad = gather_grad.reshape(
                shape[:axis] + (indices.size,) + shape[axis + 1 :]
            )
            for source_index, dest_index in enumerate(indices.flat):
              dest_slice = ((slice(None),) * outer_dims + (dest_index,) +
                            (slice(None),) * inner_dims)
              source_slice = ((slice(None),) * outer_dims + (source_index,) +
                              (slice(None),) * inner_dims)
              correct_params_grad[dest_slice] += gather_grad[source_slice]
            self.assertAllCloseAccordingToType(
                correct_params_grad,
                self.evaluate(params_grad),
                atol=2e-6,
                rtol=2e-6,
            )

  def testString(self):
    params = np.array([[b"asdf", b"zxcv"], [b"qwer", b"uiop"]])
    self.assertAllEqual([b"qwer", b"uiop"], array_ops.gather(params, 1, axis=0))
    self.assertAllEqual([b"asdf", b"qwer"], array_ops.gather(params, 0, axis=1))

  def testUInt32AndUInt64(self):
    for unsigned_type in (dtypes.uint32, dtypes.uint64):
      with self.subTest(unsigned_type=unsigned_type):
        params = self._buildParams(
            np.array([[1, 2, 3], [7, 8, 9]]), unsigned_type)
        with self.cached_session():
          self.assertAllEqual([7, 8, 9], array_ops.gather(params, 1, axis=0))
          self.assertAllEqual([1, 7], array_ops.gather(params, 0, axis=1))

  def testUnknownIndices(self):
    # This test is purely a test for placeholder inputs which is only applicable
    # in graph mode.
    with ops.Graph().as_default():
      params = constant_op.constant([[0, 1, 2]])
      indices = array_ops.placeholder(dtypes.int32)
      gather_t = array_ops.gather(params, indices)
      self.assertEqual(None, gather_t.get_shape())

  def testUnknownAxis(self):
    # This test is purely a test for placeholder inputs which is only applicable
    # in graph mode.
    with ops.Graph().as_default():
      params = constant_op.constant([[0, 1, 2]])
      indices = constant_op.constant([[0, 0], [0, 0]])
      axis = array_ops.placeholder(dtypes.int32)
      gather_t = array_ops.gather(params, indices, axis=axis)
      # Rank 2 params with rank 2 indices results in a rank 3 shape.
      self.assertEqual([None, None, None], gather_t.shape.as_list())

      # If indices is also unknown the result rank is unknown.
      indices = array_ops.placeholder(dtypes.int32)
      gather_t = array_ops.gather(params, indices, axis=axis)
      self.assertEqual(None, gather_t.shape)

  def testBadIndicesType(self):
    with self.assertRaisesRegex(
        (TypeError, errors.InvalidArgumentError),
        "float.* not in.* list of allowed values: int16, int32, int64"):
      self.evaluate(array_ops.gather([0], 0.))

  @test_util.disable_xla(
      "Assertion inside an op is not supported in XLA. Instead XLA clamps the "
      "index to be in bounds and returns the indexed value there (Don't rely "
      "on this behavior).")
  def testBadIndicesCPU(self):
    with test_util.force_cpu():
      params = [[0, 1, 2], [3, 4, 5]]
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 2\)"):
        self.evaluate(array_ops.gather(params, [[7]], axis=0))
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 3\)"):
        self.evaluate(array_ops.gather(params, [[7]], axis=1))

  def _disabledTestBadIndicesGPU(self):
    # TODO disabled due to different behavior on GPU and CPU
    # On GPU the bad indices do not raise error but fetch 0 values
    if not test.is_gpu_available():
      return
    with self.session():
      params = [[0, 1, 2], [3, 4, 5]]
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 2\)"):
        array_ops.gather(params, [[7]], axis=0).eval()
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 3\)"):
        array_ops.gather(params, [[7]], axis=1).eval()

  def testBadAxis(self):

    @def_function.function(autograph=False, jit_compile=False)
    def gather(x, indices, axis):
      return array_ops.gather(x, indices, axis=axis)

    @def_function.function(
        autograph=False,
        jit_compile=False,
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
        ] * 3)
    def gather_shape_inf_disabled(x, indices, axis):
      return array_ops.gather(x, indices, axis=axis)

    @def_function.function(
        autograph=False,
        jit_compile=True,
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
        ] * 3)
    def xla_gather(x, indices, axis):
      return array_ops.gather(x, indices, axis=axis)

    params = [0, 1, 2]
    indices = 0
    functions = [("array_ops.gather", array_ops.gather), ("gather", gather),
                 ("gather_shape_inf_disabled", gather_shape_inf_disabled),
                 ("xla_gather", xla_gather)]
    for bad_axis in (1, 2, -2):
      for fn_name, fn in functions:
        # Shape inference can validate axis for known params rank.
        with self.subTest(bad_axis=bad_axis, msg=fn_name, fn=fn):
          with self.assertRaisesRegex(
              (ValueError, errors.InvalidArgumentError),
              "Shape must be at least rank .* but is rank 1"):
            fn(params, indices, axis=bad_axis)

  def testEmptySlices(self):
    for dtype in _TEST_TYPES:
      for itype in _INDEX_TYPES:
        # Leading axis gather.
        with self.subTest(dtype=dtype, itype=itype):
          params = np.zeros((7, 0, 0), dtype=dtype.as_numpy_dtype)
          indices = np.array([3, 4], dtype=itype.as_numpy_dtype)
          gather = array_ops.gather(params, indices, axis=0)
          self.assertAllEqual(gather, np.zeros((2, 0, 0)))

          # Middle axis gather.
          params = np.zeros((0, 7, 0), dtype=dtype.as_numpy_dtype)
          gather = array_ops.gather(params, indices, axis=1)
          self.assertAllEqual(gather, np.zeros((0, 2, 0)))

          # Trailing axis gather.
          params = np.zeros((0, 0, 7), dtype=dtype.as_numpy_dtype)
          gather = array_ops.gather(params, indices, axis=2)
          self.assertAllEqual(gather, np.zeros((0, 0, 2)))

  @parameterized.parameters([
      # batch_dims=0 (equivalent to tf.gather)
      dict(  # 2D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[2, 1], [0, 3]],
          expected=[[8, 7], [6, 9]]),
      dict(  # 3D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[9, 7], [8, 6]], [[6, 9], [8, 8]]]),
      dict(  # 4D indices
          batch_dims=0,
          params=[8, 9],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[8, 9], [9, 8]], [[8, 8], [9, 9]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),

      # batch_dims=indices.shape.ndims - 1
      # (equivalent to tf.compat.v1.batch_gather)
      dict(  # 2D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=2,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),
      dict(  # 2D indices (1 batch dim)
          batch_dims=-1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=-1,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),

      # batch_dims=indices.shape.ndims
      dict(  # 1D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[2, 1],
          expected=[12, 21]),
      dict(  # 2D indices (2 batch dim)
          batch_dims=2,
          params=[[[100, 101, 102, 103], [110, 111, 112, 113]],
                  [[200, 201, 202, 203], [210, 211, 212, 213]]],
          indices=[[2, 1], [0, 3]],
          expected=[[102, 111], [200, 213]]),

      # 0 < batch_dims < indices.shape.ndims - 1
      dict(  # 3D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[13, 11], [12, 10]], [[20, 23], [22, 22]]]),
      dict(  # 4D indices (1 batch dim)
          batch_dims=1,
          params=[[6, 7], [8, 9]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[6, 7], [7, 6]], [[6, 6], [7, 7]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),
      dict(  # 4D indices (2 batch dims)
          batch_dims=2,
          params=[[[2, 3], [4, 5]], [[6, 7], [8, 9]]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[2, 3], [3, 2]], [[4, 4], [5, 5]]],
                    [[[7, 7], [6, 6]], [[8, 9], [9, 8]]]]),

      # axis > 0
      dict(  # 3D indices, batch_dims=1, axis=2
          # params.shape  = [I1, J1, J2] = [2, 2, 3]
          # indices.shape = [I1, K1, K2] = [2, 1, 5]
          # result.shape  = [I1, J1, K1, K2] = [2, 2, 1, 5]
          batch_dims=1,
          axis=2,
          params=[[[10, 11, 12], [13, 14, 15]], [[20, 21, 22], [23, 24, 25]]],
          indices=[[[0, 1, 2, 1, 0]], [[0, 1, 2, 1, 0]]],
          expected=[[[[10, 11, 12, 11, 10]], [[13, 14, 15, 14, 13]]],
                    [[[20, 21, 22, 21, 20]], [[23, 24, 25, 24, 23]]]]),
      dict(  # 3D indices, batch_dims=None, axis=1
          batch_dims=None,
          axis=1,
          params=[[10, 11, 12], [13, 14, 15]],
          indices=[1, 0],
          expected=[[11, 10], [14, 13]]),
      dict(  # 3D indices, batch_dims=-3, axis=1
          batch_dims=-3,
          axis=1,
          params=[[0, 1, 2], [3, 4, 5]],
          indices=[[[0, 1], [1, 0]]],
          expected=[[[[0, 1], [1, 0]]], [[[3, 4], [4, 3]]]]),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testBatchDims(self, params, indices, batch_dims, expected=None,
                    axis=None):
    result = array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    self.assertAllEqual(expected, result)

    # Test gradients
    f64_params = math_ops.cast(params, dtypes.float64)
    def gather(params):
      return array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        gather, [f64_params])
    self.assertAllClose(theoretical, numerical)

    # Test gradients when input shapes are unknown
    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.float64),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
    ])
    def gather_unknown_shapes(params, indices):
      return array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    if batch_dims is None or batch_dims >= 0:
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda p: gather_unknown_shapes(p, indices), [f64_params])
      self.assertAllClose(theoretical, numerical)
    else:
      with self.assertRaisesRegex(
          ValueError,
          "Currently, it is unsupported to take the gradient of tf.gather"):
        gradient_checker_v2.compute_gradient(
            lambda p: gather_unknown_shapes(p, indices), [f64_params])

    # Test the gradients shape.
    with backprop.GradientTape() as tape:
      zeros = array_ops.zeros_like(params, dtype=dtypes.float32)
      tape.watch(zeros)
      values = zeros * 2 + zeros
      result = array_ops.gather(
          values, indices, axis=axis, batch_dims=batch_dims)
    gradients = tape.gradient(result, zeros)

    self.assertAllEqual(array_ops.shape(params), array_ops.shape(gradients))

    # Run the same test for strings.
    params = _to_str_elements(params)
    expected = _to_str_elements(expected)
    result = array_ops.gather(
        params, indices, axis=axis, batch_dims=batch_dims)

    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=2,
          output_shape=[2, 3, 8, 9, 10, 5, 6, 7]
          # = params.shape[:2] + indices.shape[2:] + params.shape[3:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=3,
          output_shape=[2, 3, 4, 8, 9, 10, 6, 7]
          # = params.shape[:3] + indices.shape[2:] + params.shape[4:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=4,
          output_shape=[2, 3, 4, 5, 8, 9, 10, 7]
          # = params.shape[:4] + indices.shape[2:] + params.shape[5:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=5,
          output_shape=[2, 3, 4, 5, 6, 8, 9, 10]
          # = params.shape[:5] + indices.shape[2:] + params.shape[6:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=-4,
          output_shape=[2, 3, 8, 9, 10, 5, 6, 7]
          # = params.shape[:2] + indices.shape[2:] + params.shape[3:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=-3,
          output_shape=[2, 3, 4, 8, 9, 10, 6, 7]
          # = params.shape[:3] + indices.shape[2:] + params.shape[4:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=-2,
          output_shape=[2, 3, 4, 5, 8, 9, 10, 7]
          # = params.shape[:4] + indices.shape[2:] + params.shape[5:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=-1,
          output_shape=[2, 3, 4, 5, 6, 8, 9, 10]
          # = params.shape[:5] + indices.shape[2:] + params.shape[6:]
          ),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testBatchDimsMatchesPythonBatching(self, params_shape, indices_shape,
                                         batch_dims, axis, output_shape):
    """Checks that batch_dims matches multiple calls to tf.gather()."""
    # Generate a `params` tensor with the indicated shape.
    params_size = np.prod(params_shape)
    params = np.reshape(np.arange(params_size), params_shape)

    # Generate an `indices` tensor with the indicated shape, where each index
    # is within the appropriate range.
    indices_size = np.prod(indices_shape)
    indices = np.reshape(np.arange(indices_size), indices_shape)
    indices = indices % params_shape[axis]

    # Perform repeated (batched) gather operations with numpy, to find the
    # expected result.
    expected = self._batchNumpyGather(params, indices, axis, batch_dims)

    # On Windows, we get an exception if we pass in the transformed numpy
    # arrays ("Failed to convert numpy ndarray to a Tensor (Unsupported
    # feed type)."); so convert them back to lists before calling tf.gather.
    params = params.tolist()
    indices = indices.tolist()

    result = array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    self.assertAllEqual(output_shape, result.shape.as_list())
    self.assertAllEqual(expected, result)

    # Run the same test for strings.
    params = _to_str_elements(params)
    expected = _to_str_elements(expected.tolist())
    result = array_ops.gather(
        params, indices, axis=axis, batch_dims=batch_dims)

    self.assertAllEqual(output_shape, result.shape.as_list())
    self.assertAllEqual(expected, result)

  def _batchNumpyGather(self, params, indices, axis, batch_dims):
    """Performs a batch gather by making recursive calls to np.take().

    This is used by testBatchDims() to construct the expected value.

    Args:
      params: A numpy array
      indices: A numpy array
      axis: An integer
      batch_dims: An integer
    Returns:
      A numpy array
    """
    if batch_dims == 0:
      return np.take(params, indices, axis=axis)
    self.assertEqual(params.shape[0], indices.shape[0])
    if axis > 0:
      axis -= 1
    return np.stack([
        self._batchNumpyGather(params[i], indices[i], axis, batch_dims - 1)
        for i in range(params.shape[0])
    ])

  @test_util.run_v1_only("RefVariable is not supported in v2")
  def testGatherRefVariable(self):
    with self.cached_session():
      v = variables.RefVariable(constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather(v, [0, 2])
      if not context.executing_eagerly():  # .op doesn't make sense in Eager
        self.assertEqual("GatherV2", gather.op.name)
      self.assertAllEqual([[1, 2], [5, 6]], gather)

  @test_util.run_in_graph_and_eager_modes
  def testGatherResourceVariable(self):
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather(v, [0, 2])
      if not context.executing_eagerly():  # .op doesn't make sense in Eager
        self.assertEqual("ResourceGather", gather.op.inputs[0].op.type)
      self.assertAllEqual([[1, 2], [5, 6]], gather)

if __name__ == "__main__":
  test.main()
