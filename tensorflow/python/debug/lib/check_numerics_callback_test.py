# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class LimitStringLengthTest(test_util.TensorFlowTestCase):

  def testLimitStringLengthWithExplicitLimit(self):
    self.assertEqual(
        check_numerics_callback.limit_string_length("", max_len=2), "")
    self.assertEqual(
        check_numerics_callback.limit_string_length("e", max_len=2), "e")
    self.assertEqual(
        check_numerics_callback.limit_string_length("de", max_len=2), "de")
    self.assertEqual(
        check_numerics_callback.limit_string_length("abcde", max_len=2),
        "...de")

  def testLimitStringLengthWithNoLimit(self):
    self.assertEqual(check_numerics_callback.limit_string_length(
        "A" * 100 + "B", max_len=None), "A" * 100 + "B")
    self.assertEqual(
        check_numerics_callback.limit_string_length("", max_len=None), "")

  def testLimitStringLengthWithDefaultLimit(self):
    self.assertEqual(
        check_numerics_callback.limit_string_length("A" * 50 + "B"),
        "..." + "A" * 49 + "B")


class CheckNumericsCallbackTest(test_util.TensorFlowTestCase):

  def tearDown(self):
    check_numerics_callback.disable_check_numerics()
    super(CheckNumericsCallbackTest, self).tearDown()

  def testCallingDisableCheckNumericsWithoutEnablingFirstIsTolerated(self):
    check_numerics_callback.disable_check_numerics()

  def testNoCatchEagerOpExecution(self):
    """Test running multiple steps of eager execution without Inf/NaN."""
    check_numerics_callback.enable_check_numerics()
    x = constant_op.constant([2.0, 3.0])
    y = constant_op.constant([1.0, 0.0])
    self.assertAllClose((x + y) * (x - y), [3.0, 9.0])

  @test_util.run_in_graph_and_eager_modes
  def testDatasetMapHealthyResults(self):
    check_numerics_callback.enable_check_numerics()

    tensor = constant_op.constant(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    def map_fn(x):
      return math_ops.log(math_ops.square(x) + 1)

    dataset = dataset_ops.Dataset.from_tensor_slices(tensor).batch(2).map(
        map_fn)

    @def_function.function
    def get_batches():
      iterator = iter(dataset)
      return [next(iterator), next(iterator)]

    batches = self.evaluate(get_batches())
    self.assertLen(batches, 2)
    self.assertAllClose(batches[0], np.log([1.25, 2]))
    self.assertAllClose(batches[1], np.log([3.25, 5]))

  @test_util.run_in_graph_and_eager_modes
  def testGraphModeUsesCorrectPathLengthAndStackHeightLimits(self):
    check_numerics_callback.enable_check_numerics(
        stack_height_limit=123, path_length_limit=1200)

    @def_function.function
    def add_fn(x, y):
      return x + y

    fake_get_check_numerics_error_message = test.mock.MagicMock(
        return_value="dummy_message")
    with test.mock.patch.object(check_numerics_callback,
                                "get_check_numerics_error_message",
                                fake_get_check_numerics_error_message):
      x = constant_op.constant(2.0)
      y = constant_op.constant(3.0)
      self.assertAllClose(self.evaluate(add_fn(x, y)), 5.0)
      (_, call_kwargs) = fake_get_check_numerics_error_message.call_args
      self.assertEqual(call_kwargs["stack_height_limit"], 123)
      self.assertEqual(call_kwargs["path_length_limit"], 1200)


class CheckNumericsCallbackUnhealthyTest(test_util.TensorFlowTestCase):
  """Test for cases in which enable_check_numerics() catches infs or nans."""

  def tearDown(self):
    check_numerics_callback.disable_check_numerics()
    super(CheckNumericsCallbackUnhealthyTest, self).tearDown()

  def _assertRaisesInvalidArgumentErrorAndGetMessage(self, func):
    caught = None
    try:
      func()
    except errors.InvalidArgumentError as error:
      caught = error
    self.assertTrue(caught, "Failed to catch expected InvalidArgumentError")
    return caught.message

  def testCatchEagerOpFloat32Inf(self):
    """Test catching Infinity in eager op execution: float32."""
    check_numerics_callback.enable_check_numerics()

    x = constant_op.constant([2.0, 3.0])
    y = constant_op.constant([1.0, 0.0])
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: x / y)

    # Check the content of the error message.
    self.assertTrue(re.search(r"eagerly-executing op.*\"RealDiv\"", message))
    self.assertTrue(re.search(r"dtype.*float32", message))
    self.assertIn("shape: (2,)\n", message)
    self.assertIn("# of +Inf elements: 1\n", message)
    self.assertIn("0: %s" % x, message)
    self.assertIn("1: %s" % y, message)

  def testEnableCheckNumericsIsIdempotent(self):
    """Two calls to enable_check_numerics() have same effect as one call."""
    check_numerics_callback.enable_check_numerics()
    check_numerics_callback.enable_check_numerics()

    x = constant_op.constant([2.0, 3.0])
    y = constant_op.constant([1.0, 0.0])
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: x / y)

    # Check the content of the error message.
    self.assertTrue(re.search(r"eagerly-executing op.*\"RealDiv\"", message))
    self.assertTrue(re.search(r"dtype.*float32", message))
    self.assertIn("shape: (2,)\n", message)
    self.assertIn("# of +Inf elements: 1\n", message)
    self.assertIn("0: %s" % x, message)
    self.assertIn("1: %s" % y, message)

  def testCatchEagerOpFloat16NaN(self):
    """Test catching Infinity in eager op execution: float16."""
    check_numerics_callback.enable_check_numerics()
    def log1p(x):
      y = 1.0 + x
      return math_ops.log(y)
    x = constant_op.constant([[-1.0]], dtype=dtypes.float16)
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: log1p(x))

    # Check the content of the error message.
    self.assertTrue(re.search(r"eagerly-executing op.*\"Log\"", message))
    self.assertTrue(re.search(r"dtype.*float16", message))
    self.assertIn("shape: (1, 1)\n", message)
    self.assertIn("# of -Inf elements: 1\n", message)
    self.assertTrue(re.search(r"Input tensor.*0\.", message))

  @test_util.run_in_graph_and_eager_modes
  def testCatchFunctionOpInfFloat64(self):
    """Test catching infinites generated in a FuncGraph."""

    check_numerics_callback.enable_check_numerics()
    @def_function.function
    def divide_sum_with_diff(x, y):
      w1 = x + y
      w2 = x - y
      u = w1 / w2
      return u * 2.0
    x = constant_op.constant(2.0, dtype=dtypes.float64)
    y = constant_op.constant(2.0, dtype=dtypes.float64)
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: self.evaluate(divide_sum_with_diff(x, y)))

    # Check the content of the error message.
    self.assertTrue(re.search(r"graph op.*\"RealDiv\"", message))
    self.assertTrue(re.search(r"dtype.*float64", message))
    self.assertIn("shape: ()\n", message)
    self.assertIn("Input tensors (2):", message)
    # Check that the correct input ops are printed.
    self.assertTrue(re.search(r"0:.*Tensor.*add:0", message))
    self.assertTrue(re.search(r"1:.*Tensor.*sub:0", message))
    # Check that the correct line for op creation is printed.
    self.assertTrue(re.search(r"Stack trace of op's creation", message))
    self.assertIn("u = w1 / w2", message)

  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_xla(
      "TODO(b/141100809): XLA has no way to assert inside of a kernel.")
  def testControlFlowGraphWithNaNBFloat16(self):
    """Test catching bfloat16 NaNs in a control-flow-v2 FuncGraph."""
    check_numerics_callback.enable_check_numerics()

    @def_function.function
    def my_conditional(x):
      if math_ops.less(math_ops.reduce_sum(x), 0.0):
        return math_ops.log(x)
      else:
        return math_ops.log(-x)

    x = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.bfloat16)
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: self.evaluate(my_conditional(x)))
    # Check the content of the error message.
    self.assertTrue(re.search(r"graph op.*\"Log\"", message))
    self.assertTrue(re.search(r"dtype.*bfloat16", message))
    self.assertIn("shape: (3,)\n", message)
    # Check that the correct input op is printed.
    self.assertTrue(re.search(r"Input tensor.*Tensor.*Neg", message))
    # Check that the correct line for op creation is printed.
    self.assertTrue(re.search(r"Stack trace of op's creation", message))
    self.assertIn("return math_ops.log(-x)", message)

  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_xla(
      "There is a small inconsistency in the step at which overflow happens: "
      "128 (without XLA) and 127 (with XLA).")
  @test_util.disable_tfrt("b/177261532: TFRT cannot detect overflow yet.")
  def testOverflowInTfFunction(self):
    """Test catching Infinity caused by overflow in a tf.function with while."""
    check_numerics_callback.enable_check_numerics()

    @def_function.function
    def accumulation_function(counter, lim, accum):
      while math_ops.less(counter, lim):
        accum.assign(accum * 2.0)
        counter.assign_add(1)

    counter = variables.Variable(0, dtype=dtypes.int32)
    # Repeated `* 2.0` overflows a float32 tensor in 128 steps. So the
    # 1000-step limit is sufficient.
    lim = constant_op.constant(1000, dtype=dtypes.int32)
    accum = variables.Variable(1.0)

    if not context.executing_eagerly():
      self.evaluate([counter.initializer, accum.initializer])

    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: self.evaluate(accumulation_function(counter, lim, accum)))

    self.assertAllClose(self.evaluate(counter), 128)
    # Check the content of the error message.
    # The overflow to +Infinity happens during the `* 2.0` operation.
    self.assertTrue(re.search(r"graph op.*\"Mul\"", message))
    self.assertTrue(re.search(r"dtype.*float32", message))
    self.assertIn("shape: ()\n", message)
    # Check that the correct input op is printed.
    self.assertIn("Input tensors (2):", message)
    # Check that the correct input ops are printed.
    self.assertTrue(re.search(r"0:.*Tensor.*ReadVariableOp:0", message))
    self.assertTrue(re.search(r"1:.*Tensor.*mul/y:0", message))
    # Check that the correct line for op creation is printed.
    self.assertTrue(re.search(r"Stack trace of op's creation", message))
    self.assertIn("accum.assign(accum * 2.0)", message)

  @test_util.run_in_graph_and_eager_modes
  def testNanInConstIsCaptured(self):
    check_numerics_callback.enable_check_numerics()
    v = variables.Variable(3.0, dtype=dtypes.float32)
    @def_function.function
    def add_a_bad_constant(x):
      c = constant_op.constant(np.nan)
      return x + c
    if not context.executing_eagerly():
      self.evaluate(v.initializer)
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: self.evaluate(add_a_bad_constant(v)))
    self.assertTrue(re.search(r"graph op.*\"Const\"", message))
    self.assertTrue(re.search(r"dtype:.*float32", message))
    self.assertTrue(re.search(r"shape:.*\(\)", message))
    self.assertTrue(re.search(r"Graph name:.*add_a_bad_constant", message))

  @test_util.run_in_graph_and_eager_modes
  def testCatchInfinityInDatasetMapFunction(self):
    """Test that callback catches NaN in a tf.dataset map function."""
    check_numerics_callback.enable_check_numerics()

    def generate_nan(x):
      """Intentionally generates NaNs by taking log of negative number."""
      casted_x = math_ops.cast(x, dtypes.float32)
      return math_ops.log([[-1.0, 1.0], [3.0, 5.0]]) + casted_x

    dataset = dataset_ops.Dataset.range(10).map(generate_nan)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: self.evaluate(iterator.get_next()))

    # Check the content of the error message.
    self.assertTrue(re.search(r"graph op.*\"Log\"", message))
    self.assertTrue(re.search(r"dtype.*float32", message))
    self.assertIn("shape: (2, 2)\n", message)
    self.assertTrue(re.search(r"Input tensor.*Tensor.*Log/x:0", message))
    self.assertIn(
        "-> |   return math_ops.log([[-1.0, 1.0], [3.0, 5.0]]) + casted_x",
        message)

  @test_util.run_in_graph_and_eager_modes
  def testCustomGradientWithNaNWithTfFunction(self):
    """Test that callback catches NaN in a gradient function during backprop."""
    check_numerics_callback.enable_check_numerics()

    @custom_gradient.custom_gradient
    def func_with_bad_grad(x):
      output = math_ops.sin(x)
      @def_function.function
      def grad(dy):
        # `dy` will come in as 1.0. Taking log of -1.0 leads to NaN.
        return math_ops.log(-dy)
      return output, grad

    x = constant_op.constant(-2.0, dtype=dtypes.float16)
    def f(x):
      return func_with_bad_grad(x)

    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: gradient_checker_v2.compute_gradient(f, [x]))

    # Check the content of the error message.
    self.assertTrue(re.search(r"graph op.*\"Log\"", message))
    self.assertTrue(re.search(r"dtype.*float16", message))
    if context.executing_eagerly():
      self.assertIn("shape: ()\n", message)
    self.assertTrue(re.search(r"Input tensor.*Tensor.*Neg:0", message))
    self.assertIn("-> |   return math_ops.log(-dy)", message)

  @test_util.run_in_graph_and_eager_modes
  def testNestedFunctionGradientCall(self):
    """Catching inf in the inner nested tf.function during backprop."""
    check_numerics_callback.enable_check_numerics()

    x = constant_op.constant(1.0 - 1e-8, dtype=dtypes.float32)

    @def_function.function
    def asinp1(x):
      # asin()'s gradient overflows at the value close to 1.0.
      return math_ops.asin(x) + 1.0

    @def_function.function
    def loss(x):
      return math_ops.square(asinp1(x))

    with backprop.GradientTape() as tape:
      tape.watch(x)
      y = loss(x)
      message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
          lambda: self.evaluate(tape.gradient(y, x)))
      # Check the content of the error message.
      # Assume the op Reciprocal or Xdivy is used in the gradient function for
      # asin().
      self.assertTrue((re.search(r"graph op.*\"Reciprocal\"", message) or
                       re.search(r"graph op.*\"Xdivy\"", message)))
      self.assertTrue(re.search(r"dtype.*float32", message))

  def testEagerModeUsesCorrectPathLengthAndStackHeightLimits(self):
    check_numerics_callback.enable_check_numerics(
        stack_height_limit=123, path_length_limit=1200)
    fake_get_check_numerics_error_message = test.mock.MagicMock(
        return_value="dummy_message")
    with test.mock.patch.object(check_numerics_callback,
                                "get_check_numerics_error_message",
                                fake_get_check_numerics_error_message):
      x = constant_op.constant(2.0)
      y = constant_op.constant(0.0)
      self._assertRaisesInvalidArgumentErrorAndGetMessage(
          lambda: x / y)  # Expected to generate an inf.
      (_, call_kwargs) = fake_get_check_numerics_error_message.call_args
      self.assertEqual(call_kwargs["stack_height_limit"], 123)
      self.assertEqual(call_kwargs["path_length_limit"], 1200)

  @test_util.run_in_graph_and_eager_modes
  def testExpectedNaNOpOutputs(self):
    """Test calling operations with benign NaN output."""
    check_numerics_callback.enable_check_numerics()

    # Empty input tensor
    x = constant_op.constant(1, dtype=dtypes.float32, shape=[0, 1, 1, 1])
    scale = constant_op.constant([1], dtype=dtypes.float32)
    offset = constant_op.constant([1], dtype=dtypes.float32)

    # Calling fused_batch_norm with an empty input should output a NaN in the
    # latter four outputs without triggering the check_numerics callback
    batch_norm_res = gen_nn_ops._fused_batch_norm(
        x=x, scale=scale, offset=offset, mean=[], variance=[])

    _, batch_mean, batch_variance, _, _ = self.evaluate(batch_norm_res)

    self.assertTrue(np.isnan(batch_mean.squeeze()))
    self.assertTrue(np.isnan(batch_variance.squeeze()))


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
