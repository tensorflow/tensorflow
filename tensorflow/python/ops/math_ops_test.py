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
"""Tests for tensorflow.ops.math_ops."""
from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_util.TensorFlowTestCase):

  def testReduceAllDims(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with test_util.device(use_gpu=True):
      y_tf = self.evaluate(math_ops.reduce_sum(x))
      self.assertEqual(y_tf, 21)

  def testReduceExtendType(self):
    in_f32 = np.random.randn(1000, 1000).astype(np.float32)
    in_bf16 = math_ops.cast(in_f32, dtypes.bfloat16)

    out_f32 = self.evaluate(math_ops.reduce_sum(in_f32))
    out_bf16 = self.evaluate(math_ops.reduce_sum(in_bf16))
    expected = math_ops.cast(out_f32, dtypes.bfloat16)

    self.assertAllClose(out_bf16, expected, 1e-3)

  def testReduceExplicitAxes(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with test_util.device(use_gpu=True):
      for axis in (0, -2):
        self.assertAllEqual(
            self.evaluate(math_ops.reduce_sum(x, axis=axis)), [5, 7, 9])
      for axis in (1, -1):
        self.assertAllEqual(
            self.evaluate(math_ops.reduce_sum(x, axis=axis)), [6, 15])
      for axis in (None, (0, 1), (1, 0), (-1, 0), (0, -1), (-2, 1), (1, -2),
                   (-1, -2), (-2, -1)):
        self.assertEqual(self.evaluate(math_ops.reduce_sum(x, axis=axis)), 21)

  def testReduceInvalidAxis(self):
    if context.executing_eagerly():
      # The shape check is in run a graph construction time. In eager mode,
      # it misses the check, magically return result given wrong shape.
      return
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([[0], [1]])
    with self.assertRaisesRegex(ValueError, "must be at most rank 1"):
      math_ops.reduce_sum(x, axis)

  def testReduceVar(self):
    x = np.array([[0, 0, 0], [0, 0, 0]], "float32")
    self.assertAllClose(self.evaluate(math_ops.reduce_variance(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_variance(x, axis=0)), [0, 0, 0])

    x = [[1, 2, 1, 1], [1, 1, 0, 1]]
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      math_ops.reduce_variance(x)

    x = [[1., 2., 1., 1.], [1., 1., 0., 1.]]
    self.assertEqual(self.evaluate(math_ops.reduce_variance(x)), 0.25)
    x_np = np.array(x)
    self.assertEqual(np.var(x_np), 0.25)
    self.assertEqual(self.evaluate(math_ops.reduce_variance(x_np)), 0.25)

    x = ragged_factory_ops.constant([[5., 1., 4., 1.], [], [5., 9., 2.], [5.],
                                     []])
    self.assertAllClose(math_ops.reduce_variance(x, axis=0), [0., 16., 1., 0.])

  def testReduceVarComplex(self):
    # Ensure that complex values are handled to be consistent with numpy
    complex_ys = [([0 - 1j, 0 + 1j], dtypes.float64),
                  (np.array([0 - 1j, 0 + 1j], "complex64"), dtypes.float32),
                  (np.array([0 - 1j, 0 + 1j], "complex128"), dtypes.float64)]
    for y, dtype in complex_ys:
      y_result = math_ops.reduce_variance(y)
      self.assertEqual(np.var(y), 1.0)
      self.assertEqual(self.evaluate(y_result), 1.0)
      self.assertEqual(y_result.dtype, dtype)

  def testReduceStd(self):
    x = np.array([[0, 0, 0], [0, 0, 0]], "float32")
    self.assertAllClose(self.evaluate(math_ops.reduce_std(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_std(x, axis=0)), [0, 0, 0])

    x = [[1, 2, 1, 1], [1, 1, 0, 1]]
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      math_ops.reduce_std(x)

    x = [[1., 2., 1., 1.], [1., 1., 0., 1.]]
    self.assertEqual(self.evaluate(math_ops.reduce_std(x)), 0.5)
    x_np = np.array(x)
    self.assertEqual(np.std(x_np), 0.5)
    self.assertEqual(self.evaluate(math_ops.reduce_std(x_np)), 0.5)

    x = ragged_factory_ops.constant([[5., 1., 4., 1.], [], [5., 9., 2.], [5.],
                                     []])
    self.assertAllClose(math_ops.reduce_std(x, axis=0), [0., 4., 1., 0.])

  def testReduceStdComplex(self):
    # Ensure that complex values are handled to be consistent with numpy
    complex_ys = [([0 - 1j, 0 + 1j], dtypes.float64),
                  (np.array([0 - 1j, 0 + 1j], "complex64"), dtypes.float32),
                  (np.array([0 - 1j, 0 + 1j], "complex128"), dtypes.float64)]
    for y, dtype in complex_ys:
      y_result = math_ops.reduce_std(y)
      self.assertEqual(np.std(y), 1.0)
      self.assertEqual(self.evaluate(y_result), 1.0)
      self.assertEqual(y_result.dtype, dtype)


@test_util.run_all_in_graph_and_eager_modes
class LogSumExpTest(test_util.TensorFlowTestCase):

  def testReduceLogSumExp(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf_np = math_ops.reduce_logsumexp(x_np)
        y_np = np.log(np.sum(np.exp(x_np)))
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf = math_ops.reduce_logsumexp(x_np, axis=[0])
        y_np = np.log(np.sum(np.exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = self.evaluate(y_tf)
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices2(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf = math_ops.reduce_logsumexp(x_np, axis=0)
        y_np = np.log(np.sum(np.exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = self.evaluate(y_tf)
        self.assertAllClose(y_tf_np, y_np)

  def testKeepDims(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf_np = math_ops.reduce_logsumexp(x_np, keepdims=True)
        self.assertEqual(y_tf_np.shape.rank, x_np.ndim)
        y_np = np.log(np.sum(np.exp(x_np), keepdims=True))
        self.assertAllClose(y_tf_np, y_np)

  def testOverflow(self):
    x = [1000, 1001, 1002, 1003]
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegex(RuntimeWarning,
                                  "overflow encountered in exp"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == np.inf:
          raise RuntimeWarning("overflow encountered in exp")

      with test_util.use_gpu():
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testUnderflow(self):
    x = [-1000, -1001, -1002, -1003]
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegex(RuntimeWarning,
                                  "divide by zero encountered in log"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == -np.inf:
          raise RuntimeWarning("divide by zero encountered in log")

      with test_util.use_gpu():
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testInfinity(self):
    with test_util.use_gpu():
      res = math_ops.reduce_logsumexp(-np.inf)
      self.assertEqual(-np.inf, self.evaluate(res))

  def testRaggedTensor(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.double]:
      x_rt = ragged_factory_ops.constant([[1, 2], [], [3, 4, 5]], dtype=dtype)
      x_np = np.array(self.evaluate(x_rt.flat_values))
      with test_util.use_gpu():
        y_rt = math_ops.reduce_logsumexp(x_rt)
        y_np = np.log(np.sum(np.exp(x_np - np.max(x_np)))) + np.max(x_np)
        self.assertAllClose(y_rt, y_np)


@test_util.run_all_in_graph_and_eager_modes
class RoundTest(test_util.TensorFlowTestCase):

  def testRounding(self):
    x = np.arange(-5.0, 5.0, .25)
    for dtype in [np.float32, np.double, np.int32]:
      x_np = np.array(x, dtype=dtype)
      with test_util.device(use_gpu=True):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf = math_ops.round(x_tf)
        y_tf_np = self.evaluate(y_tf)
        y_np = np.round(x_np)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class MatMulTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  """Test for matmul."""

  SUPPORTED_DTYPES = [
      dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
      dtypes.int64, dtypes.bfloat16, dtypes.complex64, dtypes.complex128
  ]

  def testMatMul2D(self):
    for dtype in self.SUPPORTED_DTYPES:
      a = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
      b = constant_op.constant([7, 8, 9, 10, 11, 12], shape=[3, 2], dtype=dtype)
      c = math_ops.matmul(a, b)
      c_np = constant_op.constant([[58, 64], [139, 154]],
                                  shape=(2, 2),
                                  dtype=dtype)
      self.assertAllClose(c, c_np, atol=1e-2)

  def testBatchMatMul(self):
    for dtype in self.SUPPORTED_DTYPES:
      a = constant_op.constant(np.arange(1, 13), shape=[2, 2, 3], dtype=dtype)
      b = constant_op.constant(np.arange(13, 25), shape=[2, 3, 2], dtype=dtype)
      c = math_ops.matmul(a, b)
      c_np = constant_op.constant(
          [[[94, 100], [229, 244]], [[508, 532], [697, 730]]],
          shape=[2, 2, 2],
          dtype=dtype)
      self.assertAllClose(c, c_np, atol=1e-2)

  def testUnsupportedtypeMatmul(self):
    a = constant_op.constant(
        np.arange(1, 13), shape=[2, 2, 3], dtype=dtypes.int8)
    b = constant_op.constant(
        np.arange(13, 25), shape=[2, 3, 2], dtype=dtypes.int8)
    with self.assertRaisesRegex((TypeError, errors.InvalidArgumentError),
                                "list of allowed values:"):
      math_ops.matmul(a, b)

  @parameterized.parameters((dtypes.int8, dtypes.int8),
                            (dtypes.int8, dtypes.uint8),
                            (dtypes.uint8, dtypes.int8))
  # TODO(shivaniagrawal): matmul (dtypes.uint8, dtypes.uint8) fails in xla_gpu.
  def testInt8MatMul2D(self, a_dtype, b_dtype):
    a = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=a_dtype)
    b = constant_op.constant([7, 8, 9, 10, 11, 12], shape=[3, 2], dtype=b_dtype)
    c = math_ops.matmul(a, b, output_type=dtypes.int32)
    c_np = constant_op.constant([[58, 64], [139, 154]],
                                shape=(2, 2),
                                dtype=dtypes.int32)
    self.assertAllClose(c, c_np)

  @parameterized.parameters((dtypes.int8), (dtypes.uint8))
  def testMixPrecMatMul2D(self, b_dtype):
    a = constant_op.constant([1, 2, 3, 4, 5, 6],
                             shape=[2, 3],
                             dtype=dtypes.bfloat16)
    b = constant_op.constant([7, 8, 9, 10, 11, 12], shape=[3, 2], dtype=b_dtype)
    c = math_ops.matmul(a, b, output_type=dtypes.bfloat16)
    c_np = constant_op.constant([[58, 64], [139, 154]],
                                shape=(2, 2),
                                dtype=dtypes.bfloat16)
    self.assertAllClose(c, c_np, atol=1e-2)

  @parameterized.parameters((dtypes.int8, dtypes.int8),
                            (dtypes.int8, dtypes.uint8),
                            (dtypes.uint8, dtypes.int8))
  # TODO(shivaniagrawal): matmul (dtypes.uint8, dtypes.uint8) fails in xla_gpu.
  def testInt8BatchMatmul(self, a_dtype, b_dtype):
    a = constant_op.constant(np.arange(1, 13), shape=[2, 2, 3], dtype=a_dtype)
    b = constant_op.constant(np.arange(13, 25), shape=[2, 3, 2], dtype=b_dtype)
    c_np = constant_op.constant(
        [[[94, 100], [229, 244]], [[508, 532], [697, 730]]],
        shape=[2, 2, 2],
        dtype=dtypes.int32)
    c = math_ops.matmul(a, b, output_type=dtypes.int32)
    self.assertAllEqual(c, c_np)

  @parameterized.parameters((dtypes.int8), (dtypes.uint8))
  def testMixPrecBatchMatmul(self, b_dtype):
    a = constant_op.constant(
        np.arange(1, 13), shape=[2, 2, 3], dtype=dtypes.bfloat16)
    b = constant_op.constant(np.arange(13, 25), shape=[2, 3, 2], dtype=b_dtype)
    c_np = constant_op.constant(
        [[[94, 100], [229, 244]], [[508, 532], [697, 730]]],
        shape=[2, 2, 2],
        dtype=dtypes.bfloat16)
    c = math_ops.matmul(a, b, output_type=dtypes.bfloat16)
    self.assertAllClose(c, c_np, atol=1e-2)

  def testInvalidOutputTypeMatmul(self):
    for dtype in [dtypes.int8, dtypes.bfloat16]:
      a = constant_op.constant(np.arange(1, 13), shape=[2, 2, 3], dtype=dtype)
      b = constant_op.constant(
          np.arange(13, 25), shape=[2, 3, 2], dtype=dtypes.int8)
      if context.executing_eagerly():
        if context.is_tfrt_enabled():
          with self.assertRaisesRegex(errors.InvalidArgumentError,
                                      "NodeDef expected inputs"):
            math_ops.matmul(a, b, output_type=dtypes.float32)
        else:
          with self.assertRaisesRegex(errors.NotFoundError,
                                      "Could not find device for node:"):
            math_ops.matmul(a, b, output_type=dtypes.float32)
      else:
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    "No OpKernel was registered to support Op"):
          self.evaluate(math_ops.matmul(a, b, output_type=dtypes.float32))


@test_util.run_all_in_graph_and_eager_modes
class ModTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    x = [0.5, 0.7, 0.3]
    for dtype in [np.float32, np.double]:
      # Test scalar and vector versions.
      for denom in [x[0], [x[0]] * 3]:
        x_np = np.array(x, dtype=dtype)
        with test_util.use_gpu():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = self.evaluate(y_tf)
          y_np = np.fmod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)

  def testFixed(self):
    x = [5, 10, 23]
    for dtype in [np.int32, np.int64]:
      # Test scalar and vector versions.
      for denom in [x[0], x]:
        x_np = np.array(x, dtype=dtype)
        with test_util.use_gpu():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = self.evaluate(y_tf)
          y_np = np.mod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np)


@test_util.run_all_in_graph_and_eager_modes
class SquaredDifferenceTest(test_util.TensorFlowTestCase):

  def testSquaredDifference(self):
    for dtype in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype,
        np.int32, np.int64
    ]:
      x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
      y = np.array([-3, -2, -1], dtype=dtype)
      z = (x - y) * (x - y)
      with test_util.device(use_gpu=True):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)

  def testComplexSquaredDifference(self):
    for dtype in [np.complex64, np.complex128]:
      x = np.array([[1 + 3j, 2 + 2j, 3 + 1j], [4 - 1j, 5 - 2j, 6 - 3j]],
                   dtype=dtype)
      y = np.array([-3 + 1j, -2 + 2j, -1 + 3j], dtype=dtype)
      z = np.conj(x - y) * (x - y)
      with test_util.device(use_gpu=False):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class ApproximateEqualTest(test_util.TensorFlowTestCase):

  def testApproximateEqual(self):
    for dtype in [np.float32, np.double]:
      x = dtype(1)
      y = dtype(1.00009)
      z = False
      with test_util.device(use_gpu=True):
        # Default tolerance is 0.00001
        z_tf = self.evaluate(math_ops.approximate_equal(x, y))
        self.assertAllEqual(z, z_tf)

    for dtype in [np.float32, np.double]:
      x = dtype(1)
      y = dtype(1.000009)
      z = True
      with test_util.device(use_gpu=True):
        # Default tolerance is 0.00001
        z_tf = self.evaluate(math_ops.approximate_equal(x, y))
        self.assertAllEqual(z, z_tf)

    for dtype in [np.float32, np.double]:
      x = np.array([[[[-1, 2.00009999], [-3, 4.01]]]], dtype=dtype)
      y = np.array([[[[-1.001, 2], [-3.00009, 4]]]], dtype=dtype)
      z = np.array([[[[False, True], [True, False]]]], dtype=np.bool_)
      with test_util.device(use_gpu=True):
        z_tf = self.evaluate(math_ops.approximate_equal(x, y, tolerance=0.0001))
        self.assertAllEqual(z, z_tf)

  def testApproximateEqualShape(self):
    for dtype in [np.float32, np.double]:
      x = np.array([1, 2], dtype=dtype)
      y = np.array([[1, 2]], dtype=dtype)
      # The inputs 'x' and 'y' must have the same shape.
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError),
          "Shapes must be equal rank|must be of the same shape"):
        math_ops.approximate_equal(x, y)

  def testApproximateEqualShapeXla(self):

    @def_function.function(jit_compile=True)
    def approximate_equal(x, y):
      return math_ops.approximate_equal(x, y)

    for dtype in [np.float32, np.double]:
      x = np.array([1, 2], dtype=dtype)
      y = np.array([[1, 2]], dtype=dtype)
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError),
          "Shapes must be equal rank|must be of the same shape"):
        approximate_equal(x, y)


@test_util.run_all_in_graph_and_eager_modes
class ScalarMulTest(test_util.TensorFlowTestCase):

  def testAcceptsRefs(self):
    if context.executing_eagerly():
      var = resource_variable_ops.ResourceVariable(10, name="var")
    else:
      var = variables.Variable(10)
    result = math_ops.scalar_mul(3, var)
    init = variables.global_variables_initializer()
    with test_util.device(use_gpu=True):
      self.evaluate(init)
      self.assertEqual(30, self.evaluate(result))

  def testAcceptsConstant(self):
    const = constant_op.constant(10)
    result = math_ops.scalar_mul(3, const)
    with test_util.device(use_gpu=True):
      self.assertEqual(30, self.evaluate(result))

  def testAcceptsTensor(self):
    tensor = array_ops.ones([10, 10])
    result = math_ops.scalar_mul(3, tensor)
    expected = array_ops.ones([10, 10]) * 3

    with test_util.device(use_gpu=True):
      self.assertAllEqual(self.evaluate(expected), self.evaluate(result))

  def testAcceptsIndexedSlices(self):
    values = constant_op.constant([2, 3, 5, 7, 0, -1], shape=[3, 2])
    indices = constant_op.constant([0, 2, 5])
    x = math_ops.scalar_mul(-3, indexed_slices.IndexedSlices(values, indices))
    with test_util.device(use_gpu=True):
      self.assertAllEqual(
          self.evaluate(x.values), [[-6, -9], [-15, -21], [0, 3]])
      self.assertAllEqual(self.evaluate(x.indices), [0, 2, 5])


@test_util.run_all_in_graph_and_eager_modes
class AddNTest(test_util.TensorFlowTestCase):

  def testPartials(self):
    """Test that previously revealed a bug in buffer forwarding for AddN."""
    partials = []
    for _ in range(98):
      partials.append(math_ops.add_n([constant_op.constant(1)]))
    partials.append(
        math_ops.add_n([constant_op.constant(1),
                        constant_op.constant(1)]))

    res = math_ops.add_n(partials) + constant_op.constant(0)
    with test_util.use_gpu():
      self.assertAllEqual(res, 100)

  def testFloat(self):
    np.random.seed(12345)
    for num_inputs in range(1, 10):
      x = [np.random.random((1, 2, 3, 4, 5)) - 0.5 for _ in range(num_inputs)]
      tf_x = ops.convert_n_to_tensor(x)
      with test_util.use_gpu():
        self.assertAllClose(sum(x), math_ops.add_n(tf_x))
        self.assertAllClose(x[0] * num_inputs,
                            math_ops.add_n([tf_x[0]] * num_inputs))

  def testInt(self):
    np.random.seed(54321)
    for num_inputs in range(1, 10):
      x = [
          np.random.randint(-128, 128, (5, 4, 3, 2, 1))
          for _ in range(num_inputs)
      ]
      tf_x = ops.convert_n_to_tensor(x)
      with test_util.use_gpu():
        self.assertAllEqual(sum(x), math_ops.add_n(tf_x))
        self.assertAllEqual(x[0] * num_inputs,
                            math_ops.add_n([tf_x[0]] * num_inputs))

  def testGrad(self):
    np.random.seed(42)
    for num_inputs in range(1, 10):
      with test_util.use_gpu():
        input_vars = [
            variables.Variable(10.0 * np.random.random())
            for _ in range(0, num_inputs)
        ]
        self.evaluate(variables.global_variables_initializer())
        if context.executing_eagerly():
          with backprop.GradientTape() as tape:
            tape.watch(input_vars)
            addn = math_ops.add_n(input_vars)
            add_n_grad = tape.gradient(addn, input_vars)
        else:
          addn = math_ops.add_n(input_vars)
          add_n_grad = gradients.gradients(addn, input_vars)

        self.assertAllEqual(
            np.repeat(1.0, num_inputs),  # d/dx (x + y + ...) = 1
            [self.evaluate(g) for g in add_n_grad])

  def testIndexedSlices(self):
    slc = indexed_slices.IndexedSlices(
        array_ops.constant([1, 2], shape=[1, 2]), array_ops.constant([1]),
        array_ops.constant([2, 2]))
    slc_as_dense = np.array([[0, 0], [1, 2]])
    with test_util.use_gpu():
      # add_n currently always converts IndexedSlices to dense
      self.assertAllEqual(slc_as_dense, math_ops.add_n([slc]))
      self.assertAllEqual(2 * slc_as_dense, math_ops.add_n([slc, slc]))

  def test_iterable(self):
    """Test that add_n supports iterables (e.g. generators and dict values)."""

    def fn():
      yield 1
      yield 2

    values_dict = {"a": 1, "b": 2}
    with test_util.use_gpu():
      self.assertAllEqual(3, math_ops.add_n(fn()))
      self.assertAllEqual(3, math_ops.add_n(values_dict.values()))


@test_util.run_all_in_graph_and_eager_modes
class DivAndModTest(test_util.TensorFlowTestCase):
  # TODO(aselle): Test more types before exposing new division operators.

  def intTestData(self):
    nums = np.arange(-10, 10, 1).reshape(20, 1)
    divs = np.arange(-3, 4, 2).reshape(1, 4)
    return nums, divs

  def floatTestData(self):
    nums = np.arange(-10, 10, .25).reshape(80, 1)
    divs = np.arange(-3, 0, .25).reshape(1, 12)
    return nums, divs

  def numpySafeFloorDivInt(self, x, y):
    z = x // y
    # Numpy produces 0 for INT_MIN/-1, but we expect an overflow to INT_MIN
    # so that (INT_MIN/-1) + (INT_MIN % -1) = INT_MIN + 0 = INT_MIN.
    z[(x == np.iinfo(x.dtype).min) & (y == -1)] = np.iinfo(x.dtype).min
    return z

  def numpySafeFloorModInt(self, x, y):
    # Numpy crashes with a FPE for INT_MIN % -1.
    z = self.numpySafeFloorDivInt(x, y)
    return x - z * y

  def numpySafeTruncateDivInt(self, x, y):
    z = self.numpySafeFloorDivInt(x, y)
    # Round up if non-zero remainder and inputs have opposite signs.
    z[(x != z * y) & ((x < 0) != (y < 0))] += 1
    return z

  def numpySafeTruncateModInt(self, x, y):
    # Numpy crashes with a FPE for INT_MIN % -1.
    z = self.numpySafeTruncateDivInt(x, y)
    return x - z * y

  def testFloorModInt(self):
    nums, divs = self.intTestData()
    for dtype in [np.int32, np.int64]:
      x = nums.astype(dtype)
      y = divs.astype(dtype)
      tf_result = math_ops.floormod(x, y)
      np_result = self.numpySafeFloorModInt(x, y)
      self.assertAllEqual(tf_result, np_result)
      tf2_result = (array_ops.constant(x) % array_ops.constant(y))
      self.assertAllEqual(tf2_result, tf_result)

  def testFloorModFloat(self):
    nums, divs = self.floatTestData()
    for dtype in [np.float16, np.float32, np.float64]:
      x = nums.astype(dtype)
      y = divs.astype(dtype)
      tf_result = math_ops.floormod(x, y)
      np_result = x % y
      self.assertAllEqual(tf_result, np_result)
      tf2_result = (array_ops.constant(x) % array_ops.constant(y))
      self.assertAllEqual(tf2_result, tf_result)

  def testFloorModBfloat16(self):
    nums, divs = self.floatTestData()
    tf_result = math_ops.floormod(
        math_ops.cast(nums, dtypes.bfloat16),
        math_ops.cast(divs, dtypes.bfloat16))
    np_result = nums % divs
    self.assertAllEqual(tf_result, np_result)

  def testTruncateModInt(self):
    nums, divs = self.intTestData()
    tf_result = math_ops.truncatemod(nums, divs)
    np_result = np.fmod(nums, divs)
    self.assertAllEqual(tf_result, np_result)

  def testTruncateModFloat(self):
    nums, divs = self.floatTestData()
    tf_result = math_ops.truncatemod(nums, divs)
    np_result = np.fmod(nums, divs)
    self.assertAllEqual(tf_result, np_result)

  def testFloorDivideInt(self):
    nums, divs = self.intTestData()
    tf_result = math_ops.floor_div(nums, divs)
    np_result = self.numpySafeFloorDivInt(nums, divs)
    self.assertAllEqual(tf_result, np_result)
    tf2_result = (array_ops.constant(nums) // array_ops.constant(divs))
    self.assertAllEqual(tf2_result, tf_result)

  def testTruncateDivideInt(self):
    nums, divs = self.intTestData()
    tf_result = math_ops.truncatediv(nums, divs)
    np_result = self.numpySafeTruncateDivInt(nums, divs)
    self.assertAllEqual(tf_result, np_result)

  def testTruncateDivideFloat(self):
    nums, divs = self.floatTestData()
    tf_result = math_ops.truncatediv(nums, divs)
    np_result = np.trunc(nums / divs)
    self.assertAllEqual(tf_result, np_result)

  @test_util.deprecated_graph_mode_only
  def testDivideName(self):
    op = math_ops.divide(
        array_ops.constant(3), array_ops.constant(4), name="my_cool_divide")
    self.assertEqual(op.name, "my_cool_divide:0")

  def testRealDiv(self):
    nums, divs = self.floatTestData()
    tf_result = math_ops.realdiv(nums, divs)
    np_result = np.divide(nums, divs)
    self.assertAllClose(tf_result, np_result)

  def testDivideType(self):
    a = array_ops.constant([2], dtype=dtypes.int32)
    # Since __future__.division is effect, we should always upgrade to float64
    b = math_ops.divide(a, 1)
    self.assertEqual(b.dtype, dtypes.float64)
    self.assertEqual(2.0, self.evaluate(b))
    c = math_ops.divide(a, 4)
    self.assertEqual(c.dtype, dtypes.float64)
    self.assertEqual(0.5, self.evaluate(c))

  def testComplexDiv(self):
    foo = array_ops.constant([1. + 3.j])
    _ = math_ops.divide(foo, 1.)
    _ = math_ops.div(foo, 2.)

  def testFloorDivGrad(self):
    a = variables.Variable(2.)
    b = variables.Variable(4.)
    input_vars = [a, b]
    self.evaluate(variables.global_variables_initializer())
    if context.executing_eagerly():
      # TDOO(rmlarsen): Is there a more compact way of
      # writing this for multiple expressions?
      with backprop.GradientTape() as tape:
        tape.watch(input_vars)
        c_grad0 = tape.gradient(math_ops.divide(a, b), input_vars)
      with backprop.GradientTape() as tape:
        tape.watch(input_vars)
        c_grad1 = tape.gradient(math_ops.div(a, b), input_vars)
      with backprop.GradientTape() as tape:
        tape.watch(input_vars)
        c_grad2 = tape.gradient(math_ops.floordiv(a, b), input_vars)
    else:
      c_grad0 = gradients.gradients(math_ops.divide(a, b), input_vars)
      c_grad1 = gradients.gradients(math_ops.div(a, b), input_vars)
      c_grad2 = gradients.gradients(math_ops.floordiv(a, b), input_vars)
    self.assertAllEqual([self.evaluate(x) for x in c_grad0], [.25, -.125])
    self.assertAllEqual([self.evaluate(x) for x in c_grad1], [.25, -.125])
    self.assertAllEqual(
        [None if x is None else self.evaluate(x) for x in c_grad2],
        [None, None])

  def testConsistent(self):
    nums, divs = self.intTestData()
    tf_result = (
        math_ops.floor_div(nums, divs) * divs + math_ops.floormod(nums, divs))
    tf_nums = array_ops.constant(nums)
    tf_divs = array_ops.constant(divs)
    tf2_result = (tf_nums // tf_divs * tf_divs + tf_nums % tf_divs)
    np_result = (nums // divs) * divs + (nums % divs)
    # Consistent with numpy
    self.assertAllEqual(tf_result, np_result)
    # Consistent with two forms of divide
    self.assertAllEqual(tf_result, tf2_result)
    # consistency for truncation form
    tf3_result = (
        math_ops.truncatediv(nums, divs) * divs +
        math_ops.truncatemod(nums, divs))
    expanded_nums = np.reshape(
        np.tile(nums, divs.shape[1]), (nums.shape[0], divs.shape[1]))
    # Consistent with desire to get numerator
    self.assertAllEqual(tf3_result, expanded_nums)
    # Consistent with desire to get numerator
    self.assertAllEqual(tf_result, expanded_nums)

  def testWithPythonValue(self):
    # Test case for https://github.com/tensorflow/tensorflow/issues/39475
    x = math_ops.divide(5, 2)
    self.assertIsInstance(x, ops.Tensor)
    x = math_ops.divide(5, array_ops.constant(2.0))
    self.assertIsInstance(x, ops.Tensor)

  def intEdgeTestData(self, dtype):
    """Edge-case test data for integer types."""
    # INT_MIN/-1 expected to produce signed-integer overflow,
    # INT_MIN/INT_MAX expected to work.
    nums = np.array([np.iinfo(dtype).min, -1, 1,
                     np.iinfo(dtype).max],
                    dtype=dtype).reshape([4, 1])
    divs = nums.reshape([1, 4])
    return nums, divs

  @test_util.disable_asan("Expected signed integer overflow.")
  @test_util.disable_ubsan("Expected signed integer overflow.")
  def testFloorDivModIntEdges(self):
    for dtype in [np.int32, np.int64]:
      x, y = self.intEdgeTestData(dtype)
      tf_floor_div = math_ops.floor_div(x, y)
      np_floor_div = self.numpySafeFloorDivInt(x, y)
      self.assertAllEqual(tf_floor_div, np_floor_div)
      tf_floor_mod = math_ops.floormod(x, y)
      np_floor_mod = self.numpySafeFloorModInt(x, y)
      self.assertAllEqual(tf_floor_mod, np_floor_mod)
      z = math_ops.add(math_ops.multiply(tf_floor_div, y), tf_floor_mod)
      # x = floor_div(x, y) * y + floor_mod(x, y)
      self.assertAllEqual(z, np.broadcast_to(x, z.shape))

  @test_util.disable_asan("Expected signed integer overflow.")
  @test_util.disable_ubsan("Expected signed integer overflow.")
  def testTruncateDivModIntEdges(self):
    for dtype in [np.int32, np.int64]:
      x, y = self.intEdgeTestData(dtype)
      tf_truncate_div = math_ops.truncatediv(x, y)
      np_truncate_div = self.numpySafeTruncateDivInt(x, y)
      self.assertAllEqual(tf_truncate_div, np_truncate_div)
      tf_truncate_mod = math_ops.truncatemod(x, y)
      np_truncate_mod = self.numpySafeTruncateModInt(x, y)
      self.assertAllEqual(tf_truncate_mod, np_truncate_mod)
      z = math_ops.add(math_ops.multiply(tf_truncate_div, y), tf_truncate_mod)
      # x = truncatediv(x, y) * y + truncatemod(x, y)
      self.assertAllEqual(z, np.broadcast_to(x, z.shape))


@test_util.run_all_in_graph_and_eager_modes
class DivNoNanTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  _SUPPORTED_DTYPES = [dtypes.int8, dtypes.uint8,
                       dtypes.int16, dtypes.uint16,
                       dtypes.int32, dtypes.uint32,
                       dtypes.int64, dtypes.uint64,
                       dtypes.bfloat16, dtypes.float16,
                       dtypes.float32, dtypes.float64,
                       dtypes.complex64, dtypes.complex128]

  @parameterized.parameters(*_SUPPORTED_DTYPES)
  def testBasic(self, dtype):
    if dtype.is_unsigned:
      nums = np.arange(0, 120, 3).reshape(40, 1)
      divs = np.arange(0, 48, 4).reshape(1, 12)
    elif dtype.is_integer:
      nums = np.arange(-120, 120, 3).reshape(80, 1)
      divs = np.arange(-48, 48, 4).reshape(1, 24)
    else:
      nums = np.arange(-10, 10, .25).reshape(80, 1)
      divs = np.arange(-3, 3, .25).reshape(1, 24)
    assert 0 in divs, "Bad test set-up"

    tf_nums = constant_op.constant(nums, dtype=dtype)
    tf_divs = constant_op.constant(divs, dtype=dtype)

    # Use tf versions for expected value to ensure inputs are identical
    # (e.g. in the case of bfloat16).
    np_nums = self.evaluate(tf_nums)
    np_divs = self.evaluate(tf_divs)
    np_result = np.true_divide(np_nums, np_divs)
    np_result[:, np_divs[0] == 0] = 0

    with test_util.use_gpu():
      tf_result = math_ops.div_no_nan(tf_nums, tf_divs)
      self.assertAllCloseAccordingToType(tf_result, np_result)

  @parameterized.product(
      type_x=_SUPPORTED_DTYPES + [float, int],
      type_y=_SUPPORTED_DTYPES + [float, int])
  def testSameSupportedTypesAsDivide(self, type_x, type_y):
    def one(type_):
      if type_ is int:
        return 1
      elif type_ is float:
        return 1.0
      else:
        return constant_op.constant(1, dtype=type_)

    x = one(type_x)
    y = one(type_y)

    divide_raises = False
    try:
      divide_result = math_ops.divide(x, y)
    except TypeError:
      divide_raises = True

    if divide_raises:
      with self.assertRaises(TypeError):
        _ = math_ops.div_no_nan(x, y)
    else:
      divide_no_nan_result = math_ops.div_no_nan(x, y)
      self.assertEqual(divide_no_nan_result.dtype, divide_result.dtype)
      self.assertAllEqual(divide_no_nan_result, divide_result)

  @parameterized.parameters((dtypes.bfloat16), (dtypes.float16),
                            (dtypes.float32), (dtypes.float64),
                            (dtypes.complex64), (dtypes.complex128))
  def testSmall(self, dtype):
    # Choose values whose squared magnitude underflows to zero/subnormal.
    zero = constant_op.constant([0, 0, 0, 0], dtype=dtype)
    divs = constant_op.constant([1e-25, -1e-20, 1e-165, -1e-160], dtype=dtype)
    tf_result = math_ops.div_no_nan(zero, divs)

    # Results should always be exactly zero.
    self.assertAllEqual(tf_result, zero)

  @parameterized.parameters((dtypes.bfloat16), (dtypes.float16),
                            (dtypes.float32), (dtypes.float64),
                            (dtypes.complex64), (dtypes.complex128))
  def testNonFiniteInNumerator(self, dtype):
    nums = constant_op.constant([np.nan, np.inf, np.NINF], dtype=dtype)
    zeros = constant_op.constant([0, 0, 0], dtype=dtype)
    ones = constant_op.constant([1, 1, 1], dtype=dtype)
    with test_util.use_gpu():
      tf_result_zeros = math_ops.div_no_nan(nums, zeros)
      self.assertAllEqual([0, 0, 0], tf_result_zeros)
      tf_result_ones = math_ops.div_no_nan(nums, ones)
      self.assertAllEqual(nums / ones, tf_result_ones)


@test_util.run_all_in_graph_and_eager_modes
class MultiplyNoNanTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    for dtype in [np.float32, np.float64]:
      values = [0, 1, np.nan, np.inf, np.NINF]
      x = constant_op.constant(values, dtype=dtype)
      zeros = constant_op.constant(np.zeros((5,)), dtype=dtype)
      ones = constant_op.constant(np.ones((5,)), dtype=dtype)
      with test_util.use_gpu():
        tf_result_zeros = math_ops.multiply_no_nan(x, zeros)
        self.assertAllEqual(tf_result_zeros, zeros)
        tf_result_ones = math_ops.multiply_no_nan(x, ones)
        self.assertAllEqual(tf_result_ones, x)
        # Normal floating point arithmetic if nonfinite values are in the
        # second argument.
        tf_result_reverseargs = math_ops.multiply_no_nan(zeros, x)
        self.assertAllEqual(zeros * x, tf_result_reverseargs)


@test_util.run_all_in_graph_and_eager_modes
class XlogyTest(test_util.TensorFlowTestCase):

  def testXlogyNoZero(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [3.1, 4., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy = self.evaluate(math_ops.xlogy(x, y))
        xtimeslogy = self.evaluate(x * math_ops.log(y))
        self.assertAllClose(xlogy, xtimeslogy)

  def testXlogyWithZero(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy_tf_np = self.evaluate(math_ops.xlogy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xlogy_tf_np, zeros_np)

  def testXlogyWithZeroBroadcast(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy_tf_np = self.evaluate(math_ops.xlogy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        xtimes_logy = self.evaluate(math_ops.log(y[1]))
        self.assertAllClose(zeros_np, xlogy_tf_np[0])
        self.assertAllClose(xtimes_logy, xlogy_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class Xlog1pyTest(test_util.TensorFlowTestCase):

  def testXlog1pyNoNeg1(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[-0.1, -0.2, 3.5], [3.1, -0.9, 2.]],
                               dtype=dtype)
      with test_util.use_gpu():
        xlog1py = self.evaluate(math_ops.xlog1py(x, y))
        xtimeslog1py = self.evaluate(x * math_ops.log1p(y))
        self.assertAllClose(xlog1py, xtimeslog1py)

  def testXlog1pyWithNegOne(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [-1., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlog1py_tf_np = self.evaluate(math_ops.xlog1py(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xlog1py_tf_np, zeros_np)

  def testXlog1pyWithZeroBroadcast(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[-0.1, -0.2, -1.], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlog1py_tf_np = self.evaluate(math_ops.xlog1py(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        xtimes_log1py = self.evaluate(math_ops.log1p(y[1]))
        self.assertAllClose(zeros_np, xlog1py_tf_np[0])
        self.assertAllClose(xtimes_log1py, xlog1py_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class XdivyTest(test_util.TensorFlowTestCase):

  def testXdivyNoZero(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [3.1, 4., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy = self.evaluate(math_ops.xdivy(x, y))
        x_over_y = self.evaluate(x / y)
        self.assertAllClose(xdivy, x_over_y)

  def testXdivyWithZero(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy_tf_np = self.evaluate(math_ops.xdivy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xdivy_tf_np, zeros_np)

  def testXdivyWithZeroBroadcast(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy_tf_np = self.evaluate(math_ops.xdivy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        x_over_y = self.evaluate(1 / y[1])
        self.assertAllClose(zeros_np, xdivy_tf_np[0])
        self.assertAllClose(x_over_y, xdivy_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class NextAfterTest(test_util.TensorFlowTestCase):

  # Basic NextAfter tests that replicate numpy nextafter tests.
  def testBasic(self):

    for dtype in [dtypes.float32, dtypes.float64]:
      one = constant_op.constant([1], dtype=dtype)
      two = constant_op.constant([2], dtype=dtype)
      zero = constant_op.constant([0], dtype=dtype)
      nan = constant_op.constant([np.nan], dtype=dtype)

      eps = constant_op.constant([np.finfo(dtype.as_numpy_dtype).eps],
                                 dtype=dtype)

      self.assertAllEqual(math_ops.nextafter(one, two) - one, eps)
      self.assertAllLess(math_ops.nextafter(one, zero) - one, 0)
      self.assertAllEqual(math_ops.is_nan(math_ops.nextafter(nan, one)), [True])
      self.assertAllEqual(math_ops.is_nan(math_ops.nextafter(one, nan)), [True])
      self.assertAllEqual(math_ops.nextafter(one, one), one)

  def testBroadcasting(self):

    for dtype in [dtypes.float32, dtypes.float64]:
      one = constant_op.constant([1, 1], dtype=dtype)
      two = constant_op.constant([2], dtype=dtype)

      eps = np.finfo(dtype.as_numpy_dtype).eps

      eps_const = constant_op.constant([eps, eps], dtype=dtype)

      self.assertAllEqual(math_ops.nextafter(one, two) - one, eps_const)


@test_util.run_all_in_graph_and_eager_modes
class BinaryOpsTest(test_util.TensorFlowTestCase):

  def testErrorReceivedIfDtypeMismatchFromOp(self):
    if context.executing_eagerly():
      error = errors_impl.InvalidArgumentError
      error_message = (
          r"cannot compute Add(V2)? as input #1\(zero-based\) was expected to "
          r"be a int32 tensor but is a float tensor \[Op:Add(V2)?\]")
    else:
      error = TypeError
      error_message = (
          "Input 'y' of 'Add(V2)?' Op has type float32 that does not "
          "match type int32 of argument 'x'.")
    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + 1.0
      self.evaluate(a)

  def testRHSDispatchingAndErrorRaising(self):
    if context.executing_eagerly():
      error = ValueError
      error_message = (
          r"Attempt to convert a value .* with an unsupported type")
    else:
      error = TypeError
      error_message = (r"Failed to convert elements of .* to Tensor")

    class RHSReturnsTrue:

      def __radd__(self, other):
        return True

    a = array_ops.ones([1], dtype=dtypes.int32) + RHSReturnsTrue()
    self.assertEqual(a, True)

    class RHSRaisesError:

      def __radd__(self, other):
        raise TypeError("RHS not implemented")

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSRaisesError()
      self.evaluate(a)

    class RHSReturnsNotImplemented:

      def __radd__(self, other):
        return NotImplemented

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSReturnsNotImplemented()
      self.evaluate(a)

    class RHSNotImplemented:
      pass

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSNotImplemented()
      self.evaluate(a)


class SignTest(test_util.TensorFlowTestCase):

  def test_complex_sign_gradient(self):
    with context.eager_mode():
      x = math_ops.complex(1., 1.)
      with backprop.GradientTape() as t:
        t.watch(x)
        y = math_ops.sign(x)
      self.assertAllClose(
          t.gradient(y, x), math_ops.complex(0.353553, -0.353553))


@test_util.run_all_in_graph_and_eager_modes
class ReciprocalNoNanTest(test_util.TensorFlowTestCase):

  allowed_dtypes = [
      dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
      dtypes.complex128
  ]

  def testBasic(self):
    for dtype in self.allowed_dtypes:
      x = constant_op.constant([1.0, 2.0, 0.0, 4.0], dtype=dtype)

      y = math_ops.reciprocal_no_nan(x)

      target = constant_op.constant([1.0, 0.5, 0.0, 0.25], dtype=dtype)

      self.assertAllEqual(y, target)
      self.assertEqual(y.dtype.base_dtype, target.dtype.base_dtype)

  def testInverse(self):
    for dtype in self.allowed_dtypes:
      x = np.random.choice([0, 1, 2, 4, 5], size=(5, 5, 5))
      x = constant_op.constant(x, dtype=dtype)

      y = math_ops.reciprocal_no_nan(math_ops.reciprocal_no_nan(x))

      self.assertAllClose(y, x)
      self.assertEqual(y.dtype.base_dtype, x.dtype.base_dtype)


class EqualityTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @test_util.run_all_in_graph_and_eager_modes
  def testEqualityNone(self):
    x = constant_op.constant([1.0, 2.0, 0.0, 4.0], dtype=dtypes.float32)
    self.assertNotEqual(x, None)
    self.assertNotEqual(None, x)
    self.assertFalse(math_ops.tensor_equals(x, None))
    self.assertTrue(math_ops.tensor_not_equals(x, None))

  @parameterized.named_parameters(
      (f"-is_equals={is_equals}-float_literal_type={type(float_literal)}"  # pylint: disable=g-complex-comprehension
       f"-float_literal={float_literal}", is_equals, float_literal)
      for float_literal in [4.6, np.float32(4.6), 4.4, np.float32(4.4)]
      for is_equals in [True, False])
  def testEqualityNoDowncast(self, is_equals, float_literal):
    if (tf2.enabled() and isinstance(float_literal, np.float32) or
        not tf2.enabled() and isinstance(float_literal, float)):
      # TODO(b/199262800): Remove this skip
      self.skipTest("There is a bug in type promotion.")
    if is_equals:
      op = math_ops.tensor_equals
    else:
      op = math_ops.tensor_not_equals
    x = constant_op.constant(4)
    try:
      result = op(x, float_literal)
      if isinstance(result, ops.Tensor):
        result = self.evaluate(result)
    except TypeError:
      # Throwing a TypeError is OK
      return
    self.assertEqual(result, not is_equals)


@test_util.run_all_in_graph_and_eager_modes
class RangeTest(test_util.TensorFlowTestCase):

  def testConvertToTensorRange(self):
    values = range(5)
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((5,), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))

  def testInputsNearInt64Max(self):
    int64_t_max = 2**63 - 1
    x = math_ops.range(0, 201, int64_t_max - 200, dtype=dtypes.int64)
    self.assertAllEqual((0,), self.evaluate(x))  # just below potential overflow
    x = math_ops.range(0, 202, int64_t_max - 200, dtype=dtypes.int64)
    self.assertAllEqual(
        (0,), self.evaluate(x))  # smallest input with potential overflow


@test_util.run_all_in_graph_and_eager_modes
class ErfcinvTest(test_util.TensorFlowTestCase):

  def testErfcinv(self):
    values = np.random.uniform(0.1, 1.9, size=int(1e4)).astype(np.float32)
    approx_id = math_ops.erfc(math_ops.erfcinv(values))
    self.assertAllClose(values, self.evaluate(approx_id))


@test_util.run_all_in_graph_and_eager_modes
class ArgMaxMinTest(test_util.TensorFlowTestCase):

  def _generateRandomTensor(self, dtype, shape):
    if dtype.is_integer:
      array = np.random.default_rng().integers(
          low=dtype.min, high=dtype.max, size=shape, endpoint=True)
      return constant_op.constant(array, dtype=dtype)
    else:
      array = np.random.default_rng().uniform(low=-1.0, high=1.0, size=shape)
      return constant_op.constant(array, dtype=dtype)

  def _getValidDtypes(self):
    return (dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64,
            dtypes.int32, dtypes.int64)

  def testArgMax(self):
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_max = np.argmax(np_values, axis=axis)
        tf_max = math_ops.argmax(tf_values, axis=axis)
        self.assertAllEqual(tf_max, np_max)

  def testArgMaxReturnsFirstOccurence(self):
    for dtype in self._getValidDtypes():
      values = constant_op.constant(
          [[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype)
      self.assertAllEqual(
          math_ops.argmax(values, axis=1),
          np.argmax(self.evaluate(values), axis=1))

      # Long tensor to ensure works with multithreading/GPU
      values = array_ops.zeros(shape=(193681,), dtype=dtype)
      self.assertAllEqual(math_ops.argmax(values), 0)

  def testArgMaxUint16(self):
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_max = np.argmax(np_values, axis=axis)
        tf_max = math_ops.argmax(
            tf_values, axis=axis, output_type=dtypes.uint16)
        self.assertAllEqual(tf_max, np_max)

  def testArgMin(self):
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_min = np.argmin(np_values, axis=axis)
        tf_min = math_ops.argmin(tf_values, axis=axis)
        self.assertAllEqual(tf_min, np_min)

  def testArgMinReturnsFirstOccurence(self):
    for dtype in self._getValidDtypes():
      values = constant_op.constant(
          [[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype)
      self.assertAllEqual(
          math_ops.argmin(values, axis=1),
          np.argmin(self.evaluate(values), axis=1))

      # Long tensor to ensure works with multithreading/GPU
      values = array_ops.zeros(shape=(193681,), dtype=dtype)
      self.assertAllEqual(math_ops.argmin(values), 0)


class CastTest(test_util.TensorFlowTestCase):

  def testCastWithFullType(self):

    @def_function.function
    def test_fn():
      ta = tensor_array_ops.TensorArray(dtypes.int32, size=1)
      h = math_ops.cast(ta.flow, dtypes.variant)

      t = full_type_pb2.FullTypeDef(
          type_id=full_type_pb2.TFT_PRODUCT,
          args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_ARRAY)])
      h.op.experimental_set_type(t)

      ta = tensor_array_ops.TensorArray(dtypes.int32, flow=h)
      ta = ta.write(0, constant_op.constant(1))
      return ta.stack()

    self.assertAllEqual(self.evaluate(test_fn()), [1])

if __name__ == "__main__":
  googletest.main()
