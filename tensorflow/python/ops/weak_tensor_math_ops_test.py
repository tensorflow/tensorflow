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
"""Tests for tensorflow.ops.math_ops on WeakTensor."""
from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest

_convert_to_input_type = weak_tensor_test_util.convert_to_input_type
_get_weak_tensor = weak_tensor_test_util.get_weak_tensor


@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # Test unary ops with optional dtype arg.

  @parameterized.parameters(
      ("WeakTensor", WeakTensor),
      ("Python", WeakTensor),
      ("NumPy", tensor.Tensor),
      ("Tensor", tensor.Tensor),
  )
  def testReduceAllDims(self, input_type, result_type):
    test_input = _convert_to_input_type(
        [[1, 2, 3], [4, 5, 6]], input_type, np.int32
    )
    with test_util.device(use_gpu=True):
      res = math_ops.reduce_sum(test_input)
      self.assertIsInstance(res, result_type)
      self.assertEqual(self.evaluate(res), 21)

  def testReduceExtendType(self):
    test_in = np.random.randn(1000, 1000).astype(np.float32)
    in_f32 = _get_weak_tensor(test_in, dtypes.float32)
    in_bfl6 = math_ops.cast(test_in, dtypes.bfloat16)

    out_f32 = self.evaluate(math_ops.reduce_sum(in_f32))
    out_bf16 = self.evaluate(math_ops.reduce_sum(in_bfl6))
    expected = math_ops.cast(out_f32, dtypes.bfloat16)

    self.assertAllClose(out_bf16, expected, 1e-3)

  def testCountNonzero(self):
    # simple case
    x = _get_weak_tensor([[0, -2, 0], [4, 0, 0]], dtypes.int32)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 2)

    # boolean input
    x = math_ops.not_equal(x, 0)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 2)

    # would overflow if int8 would be used for internal calculations
    x = 2 * np.ones(512, dtype=np.int8)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 512)

  @parameterized.parameters(
      ("WeakTensor", WeakTensor),
      ("Python", WeakTensor),
      ("NumPy", tensor.Tensor),
      ("Tensor", tensor.Tensor),
  )
  def testReduceExplicitAxes(self, input_type, result_type):
    x = _convert_to_input_type([[1, 2, 3], [4, 5, 6]], input_type, np.int32)
    with test_util.device(use_gpu=True):
      for axis in (0, -2):
        res = math_ops.reduce_sum(x, axis=axis)
        self.assertIsInstance(res, result_type)
        self.assertAllEqual(res, [5, 7, 9])
      for axis in (1, -1):
        res = math_ops.reduce_sum(x, axis=axis)
        self.assertIsInstance(res, result_type)
        self.assertAllEqual(res, [6, 15])
      for axis in (None, (0, 1), (1, 0), (-1, 0), (0, -1), (-2, 1), (1, -2),
                   (-1, -2), (-2, -1)):
        res = math_ops.reduce_sum(x, axis=axis)
        self.assertIsInstance(res, result_type)
        self.assertEqual(self.evaluate(res), 21)

  def testReduceInvalidAxis(self):
    if context.executing_eagerly():
      # The shape check is in run a graph construction time. In eager mode,
      # it misses the check, magically return result given wrong shape.
      return
    x = _get_weak_tensor([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([[0], [1]])
    with self.assertRaisesRegex(ValueError, "must be at most rank 1"):
      math_ops.reduce_sum(x, axis)

  def testReduceVar(self):
    x = _get_weak_tensor([[0, 0, 0], [0, 0, 0]], dtype=dtypes.float32)
    self.assertAllClose(self.evaluate(math_ops.reduce_variance(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_variance(x, axis=0)), [0, 0, 0])

    x = _get_weak_tensor([[1, 2, 1, 1], [1, 1, 0, 1]])
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      math_ops.reduce_variance(x)

    x = _get_weak_tensor([[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    self.assertEqual(self.evaluate(math_ops.reduce_variance(x)), 0.25)
    x_np = np.array([[1, 2, 1, 1], [1, 1, 0, 1]], "float32")
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
    x = _get_weak_tensor([[0, 0, 0], [0, 0, 0]], dtypes.float32)
    self.assertAllClose(self.evaluate(math_ops.reduce_std(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_std(x, axis=0)), [0, 0, 0])

    x = _get_weak_tensor([[1, 2, 1, 1], [1, 1, 0, 1]])
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      math_ops.reduce_std(x)

    x = [[1., 2., 1., 1.], [1., 1., 0., 1.]]
    res = math_ops.reduce_std(x)
    self.assertEqual(self.evaluate(res), 0.5)
    self.assertIsInstance(res, WeakTensor)
    x_np = np.array(x)
    self.assertEqual(np.std(x_np), 0.5)
    self.assertEqual(self.evaluate(math_ops.reduce_std(x_np)), 0.5)
    self.assertIsInstance(math_ops.reduce_std(x_np), tensor.Tensor)

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
    for dtype in [np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegex(RuntimeWarning,
                                  "overflow encountered in exp"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == np.inf:
          raise RuntimeWarning("overflow encountered in exp")

      with test_util.use_gpu():
        x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testUnderflow(self):
    x = [-1000, -1001, -1002, -1003]
    for dtype in [np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegex(RuntimeWarning,
                                  "divide by zero encountered in log"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == -np.inf:
          raise RuntimeWarning("divide by zero encountered in log")

      with test_util.use_gpu():
        x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testInfinity(self):
    with test_util.use_gpu():
      res = math_ops.reduce_logsumexp(-np.inf)
      self.assertEqual(-np.inf, self.evaluate(res))


@test_util.run_all_in_graph_and_eager_modes
class RoundTest(test_util.TensorFlowTestCase):

  def testRounding(self):
    x = np.arange(-5.0, 5.0, .25)
    for dtype in [np.float32, np.double, np.int32]:
      x_np = np.array(x, dtype=dtype)
      with test_util.device(use_gpu=True):
        x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
        y_tf = math_ops.round(x_tf)
        y_tf_np = self.evaluate(y_tf)
        y_np = np.round(x_np)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)


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

  allowed_dtypes = [dtypes.float32, dtypes.float64, dtypes.complex128]

  def testBasic(self):
    for dtype in self.allowed_dtypes:
      x = _get_weak_tensor([1.0, 2.0, 0.0, 4.0], dtype=dtype)

      y = math_ops.reciprocal_no_nan(x)

      target = _get_weak_tensor([1.0, 0.5, 0.0, 0.25], dtype=dtype)

      self.assertAllEqual(y, target)
      self.assertEqual(y.dtype.base_dtype, target.dtype.base_dtype)

  def testInverse(self):
    for dtype in self.allowed_dtypes:
      x = np.random.choice([0, 1, 2, 4, 5], size=(5, 5, 5))
      x = _get_weak_tensor(x, dtype=dtype)

      y = math_ops.reciprocal_no_nan(math_ops.reciprocal_no_nan(x))

      self.assertAllClose(y, x)
      self.assertEqual(y.dtype.base_dtype, x.dtype.base_dtype)


class EqualityTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @test_util.run_all_in_graph_and_eager_modes
  def testEqualityNone(self):
    x = _get_weak_tensor([1.0, 2.0, 0.0, 4.0], dtype=dtypes.float32)
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
    x = _get_weak_tensor(4)
    try:
      result = op(x, float_literal)
      if isinstance(result, tensor.Tensor):
        result = self.evaluate(result)
    except TypeError:
      # Throwing a TypeError is OK
      return
    self.assertEqual(result, not is_equals)


@test_util.run_all_in_graph_and_eager_modes
class ErfcinvTest(test_util.TensorFlowTestCase):

  def testErfcinv(self):
    values = _get_weak_tensor(
        np.random.uniform(0.1, 1.9, size=int(1e4)).astype(np.float32)
    )
    approx_id = math_ops.erfc(math_ops.erfcinv(values))
    self.assertAllClose(values, self.evaluate(approx_id))


@test_util.run_all_in_graph_and_eager_modes
class ArgMaxMinTest(test_util.TensorFlowTestCase):

  def _generateRandomWeakTensor(self, dtype, shape):
    if dtype.is_integer:
      array = np.random.default_rng().integers(
          low=dtype.min, high=dtype.max, size=shape, endpoint=True)
      return _get_weak_tensor(array, dtype=dtype)
    else:
      array = np.random.default_rng().uniform(low=-1.0, high=1.0, size=shape)
      return _get_weak_tensor(array, dtype=dtype)

  def _getValidDtypes(self):
    return (dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64)

  def testArgMax(self):
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomWeakTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_max = np.argmax(np_values, axis=axis)
        tf_max = math_ops.argmax(tf_values, axis=axis)
        self.assertAllEqual(tf_max, np_max)

  def testArgMaxReturnsFirstOccurence(self):
    for dtype in self._getValidDtypes():
      values = _get_weak_tensor(
          [[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype
      )
      self.assertAllEqual(
          math_ops.argmax(values, axis=1),
          np.argmax(self.evaluate(values), axis=1))

      # Long tensor to ensure works with multithreading/GPU
      values = array_ops.zeros(shape=(193681,), dtype=dtype)
      self.assertAllEqual(math_ops.argmax(values), 0)

  def testArgMaxUint16(self):
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomWeakTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_max = np.argmax(np_values, axis=axis)
        tf_max = math_ops.argmax(
            tf_values, axis=axis, output_type=dtypes.uint16)
        self.assertAllEqual(tf_max, np_max)

  def testArgMin(self):
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomWeakTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_min = np.argmin(np_values, axis=axis)
        tf_min = math_ops.argmin(tf_values, axis=axis)
        self.assertAllEqual(tf_min, np_min)

  def testArgMinReturnsFirstOccurence(self):
    for dtype in self._getValidDtypes():
      values = _get_weak_tensor(
          [[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype
      )
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
      ta = ta.write(0, _get_weak_tensor(1))
      return ta.stack()

    self.assertAllEqual(self.evaluate(test_fn()), [1])

if __name__ == "__main__":
  ops.set_dtype_conversion_mode("all")
  googletest.main()
