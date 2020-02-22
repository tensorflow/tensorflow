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
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


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
  def testKerasModelHealthyPredictAndFitCalls(self):
    """Test a simple healthy keras model runs fine under the callback."""
    check_numerics_callback.enable_check_numerics()

    model = models.Sequential()
    model.add(layers.Dense(
        units=100,
        input_shape=(5,),
        use_bias=False,
        activation="relu",
        kernel_initializer="ones"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(
        units=1,
        activation="linear",
        kernel_initializer="ones"))

    model.compile(
        loss="mse", optimizer=optimizer_v2.gradient_descent.SGD(1e-3))

    batch_size = 16
    xs = np.zeros([batch_size, 5])
    ys = np.ones([batch_size, 1])

    outputs = model.predict(xs)
    self.assertEqual(outputs.shape, (batch_size, 1))

    epochs = 100
    history = model.fit(xs, ys, epochs=epochs, verbose=0)
    self.assertEqual(len(history.history["loss"]), epochs)

  @test_util.run_in_graph_and_eager_modes
  def testKerasModelWithRNNHealthyPredictAndFitCalls(self):
    """Test a simple healthy keras recurrent model works under the callback."""
    check_numerics_callback.enable_check_numerics()

    model = models.Sequential()
    model.add(layers.LSTM(1, input_shape=(2, 4)))
    model.compile(loss="mse", optimizer="rmsprop")

    xs = np.zeros([8, 2, 4], dtype=np.float32)
    ys = np.zeros([8, 1], dtype=np.float32)

    model.predict(xs)

    epochs = 3
    history = model.fit(xs, ys, epochs=epochs, verbose=0)
    self.assertEqual(len(history.history["loss"]), epochs)

  @test_util.run_in_graph_and_eager_modes
  def testDatasetMapHealthyResults(self):
    check_numerics_callback.enable_check_numerics()

    tensor = constant_op.constant(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    def map_fn(x):
      return math_ops.log(math_ops.square(x) + 1)

    dataset = dataset_ops.Dataset.from_tensor_slices(tensor).batch(2).map(
        map_fn)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    self.assertAllClose(self.evaluate(iterator.get_next()), np.log([1.25, 2]))
    self.assertAllClose(self.evaluate(iterator.get_next()), np.log([3.25, 5]))

  @test_util.run_in_graph_and_eager_modes
  def testMobileNetV2Fit(self):
    """Test training Keras MobileNetV2 application works w/ check numerics."""
    check_numerics_callback.enable_check_numerics()
    model = mobilenet_v2.MobileNetV2(alpha=0.1, weights=None)

    xs = np.zeros([2] + list(model.input_shape[1:]))
    ys = np.zeros([2] + list(model.output_shape[1:]))
    model.compile(optimizer="sgd", loss="categorical_crossentropy")
    epochs = 1
    history = model.fit(xs, ys, epochs=epochs, verbose=0)
    self.assertEqual(len(history.history["loss"]), epochs)


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
  def testKerasModelUnhealthyPredictAndFitCallsWithLargeLearningRate(self):
    """Test keras model training crashes with Infinity is caught by callback."""
    check_numerics_callback.enable_check_numerics()

    model = models.Sequential()
    # Use weight initializers for deterministic behavior during test.
    model.add(layers.Dense(
        units=100,
        input_shape=(5,),
        activation="relu",
        kernel_initializer="ones"))
    model.add(layers.Dense(
        units=1,
        activation="linear",
        kernel_initializer="ones"))

    lr = 1e3    # Intentionally huge learning rate.
    model.compile(loss="mse", optimizer=optimizer_v2.gradient_descent.SGD(lr))

    batch_size = 16
    xs = np.zeros([batch_size, 5])
    ys = np.ones([batch_size, 1])

    outputs = model.predict(xs)
    self.assertEqual(outputs.shape, (batch_size, 1))

    epochs = 100
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: model.fit(xs, ys, epochs=epochs, verbose=0))

    # Check the content of the error message.
    # Let's not hardcode the op name for future-proof.
    self.assertTrue(re.search(r"graph op.*\".*\"", message))
    self.assertTrue(re.search(r"dtype:.*float32", message))
    self.assertTrue(re.search(r"shape:.*\(.*\)", message))
    # Check that the correct input op is printed.
    self.assertTrue(re.search(r"Input tensor.*", message))
    # Check that the correct line for op creation is printed.
    self.assertTrue(re.search(r"Stack trace of op's creation", message))
    # The stacks are different between when eager execution is enabled and
    # when it's not (i.e., v1 graph). TODO(cais): Investigate if we can improve
    # this.
    if context.executing_eagerly():
      self.assertIn("lambda: model.fit(xs, ys,", message)
    else:
      self.assertIn("model.compile(", message)

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
    if context.executing_eagerly():
      # The code path for raising error is slightly different under graph mode.
      self.assertTrue(message.endswith("\n"))

  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_xla(
      "There is a small inconsistency in the step at which overflow happens: "
      "128 (without XLA) and 127 (with XLA).")
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
  def testInfInCustomKerasLayerWithTfFunctionPredictCall(self):
    """Test catching Infinity in a custom layer, w/ tf.function."""
    check_numerics_callback.enable_check_numerics()

    class DivByXLayer(layers.Layer):

      @def_function.function
      def call(self, x):
        """The computation performed by the for-test custom layer.

        Generates Infinity by intention.

        Args:
          x: Input tensor of scalar shape.

        Returns:
          A scalar tensor.
        """
        one_over_x = 1.0 / x
        return one_over_x

    model = models.Sequential()
    model.add(DivByXLayer(input_shape=[5]))

    # TODO(b/140245224): Currently the model must be compiled prior to
    # predict() being called(). Or keras will fall back to V1 behavior.
    # Remove this after the bug is fixed.
    model.compile(loss="mse", optimizer="sgd")

    xs = np.ones([1, 5])
    # Calling the model with non-zero inputs should be fine.
    self.assertAllClose(model.predict(xs), [[1.0, 1.0, 1.0, 1.0, 1.0]])

    xs = np.zeros([1, 5])
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: model.predict(xs))

    # Check the content of the error message.
    self.assertTrue(re.search(r"graph op.*\"RealDiv\"", message))
    self.assertTrue(re.search(r"dtype.*float32", message))
    self.assertTrue(re.search(r"shape: \(.*, 5\)", message))
    # # Check that the correct input op is printed.
    self.assertIn("Input tensors (2):", message)
    # # # Check that the correct line for op creation is printed.
    self.assertTrue(re.search(r"Stack trace of op's creation", message))
    self.assertIn("one_over_x = 1.0 / x", message)

  @test_util.run_in_graph_and_eager_modes
  def testInfInCustomKerasLayerWithoutTfFunctionPredictCall(self):
    """Test catching Infinity in a custom layer, w/o tf.function."""
    check_numerics_callback.enable_check_numerics()

    class DivByXLayer(layers.Layer):

      # Not using the tf.function decorator here.
      def call(self, x):
        """The computation performed by the for-test custom layer.

        Generates Infinity by intention.

        Args:
          x: Input tensor of scalar shape.

        Returns:
          A scalar tensor.
        """
        one_over_x = 1.0 / x
        return one_over_x

    model = models.Sequential()
    model.add(DivByXLayer(input_shape=[5]))

    # TODO(b/140245224): Currently the model must be compiled prior to
    # predict() being called(). Or keras will fall back to V1 behavior.
    # Remove this after the bug is fixed.
    model.compile(loss="mse", optimizer="sgd")

    xs = np.ones([1, 5])
    # Calling the model with non-zero inputs should be fine.
    self.assertAllClose(model.predict(xs), [[1.0, 1.0, 1.0, 1.0, 1.0]])

    xs = np.zeros([1, 5])
    message = self._assertRaisesInvalidArgumentErrorAndGetMessage(
        lambda: model.predict(xs))

    # Check the content of the error message.
    self.assertTrue(re.search(r"graph op.*\"RealDiv\"", message))
    self.assertTrue(re.search(r"dtype.*float32", message))
    self.assertTrue(re.search(r"shape: \(.*, 5\)", message))
    # Check that the correct input op is printed.
    self.assertIn("Input tensors (2):", message)
    # Check that the correct line for op creation is printed.
    self.assertTrue(re.search(r"Stack trace of op's creation", message))
    self.assertIn("one_over_x = 1.0 / x", message)

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
