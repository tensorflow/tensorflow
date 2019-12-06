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
"""Tests for tfdbg op callbacks running with various `DistributionStrategy`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import gradient_descent


def filter_by_device_name(items, device_names, target_device_name):
  """Filter a list of items by device name.

  Args:
    items: A list of items to be filtered according to their corresponding
      device names.
    device_names: A list of the device names. Must have the same legnth
      as `items`.
    target_device_name: A `str` representing the desired device name.

  Returns:
    Filtered items from `items`.
  """
  assert len(items) == len(device_names)
  assert all(device_names), "device_names are not all non-empty strings"
  # Note: we use `endswith` instead of `==` for device-name filtering because
  # in some cases, the device names from kernel/op execution can have slightly
  # different values than the device names from
  # `distribution.extended.worker_devices`.
  return [items[i] for i, device_name in enumerate(device_names)
          if device_name.endswith(target_device_name)]


def filter_by_device_name_and_op_type(
    items, device_names, op_types, target_device_name, target_op_type):
  assert len(items) == len(device_names)
  assert len(items) == len(op_types)
  assert all(device_names), "device_names are not all non-empty strings"
  assert all(op_types), "op_types are not all non-empty strings"
  return [items[i] for i, device_name in enumerate(device_names)
          if device_name.endswith(target_device_name)
          and op_types[i] == target_op_type]


class MiniModel(keras.Model):
  """Minimal subclassed Keras model."""

  def __init__(self, generate_infinity=False):
    super(MiniModel, self).__init__(name="")
    self._generate_infinity = generate_infinity
    self.fc = keras.layers.Dense(
        1, kernel_initializer="ones", bias_initializer="ones",
        activation="linear")

  @def_function.function
  def call(self, inputs, training=True):
    y = self.fc(inputs)
    if self._generate_infinity:
      y = math_ops.divide(y, array_ops.zeros_like(y))
    return y


class DistributedDumpingCallbackTest(
    dumping_callback_test_lib.DumpingCallbackTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.one_device_strategy_gpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
          ],
          inside_scope=[False, True],
          # TODO(cais): Investigate that under V1 graph mode (mode="graph"),
          # occasionally (~1-2% of time) the test runs into the following error:
          # CancelledError: [_Derived_] Function was cancelled before it was
          # started.
          mode=["eager"],
      ))
  def testCheckingInfinityInMiniModelOnOneOrTwoDevices(
      self, distribution, inside_scope):
    if not inside_scope:
      check_numerics_callback.enable_check_numerics()
    with distribution.scope():
      if inside_scope:
        check_numerics_callback.enable_check_numerics()

      mini_model = MiniModel(generate_infinity=True)
      def train_step():
        with backprop.GradientTape() as tape:
          loss = mini_model(array_ops.ones([1, 10]))
          return tape.gradient(loss, mini_model.weights)

      caught_error = None
      try:
        distribution.experimental_run_v2(train_step)
      except errors.InvalidArgumentError as error:
        caught_error = error
      self.assertTrue(caught_error)
      self.assertTrue(re.search(
          r"Detected Infinity or NaN.*\"RealDiv\"", caught_error.message))
      self.assertIn(
          "-> |   y = math_ops.divide(y, array_ops.zeros_like(y))",
          caught_error.message)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.one_device_strategy_gpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
          ],
          mode=["eager"],
          tensor_debug_mode=["NO_TENSOR", "FULL_TENSOR"],
      ))
  def testDumpingMiniModel(self, distribution, tensor_debug_mode):
    with distribution.scope():
      writer = dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_debug_mode=tensor_debug_mode)

      mini_model = MiniModel()
      optimizer = gradient_descent.GradientDescentOptimizer(0.25)

      def train_step():
        with backprop.GradientTape() as tape:
          loss = mini_model(array_ops.ones([1, 10]))
          grads = tape.gradient(loss, mini_model.weights)
          grads_and_vars = zip(grads, mini_model.weights)
          optimizer.apply_gradients(grads_and_vars)

      distribution.experimental_run_v2(train_step)

      updated_var_values = self.evaluate(mini_model.variables)
      num_devices = len(distribution.extended.worker_devices)
      assert num_devices in (1, 2)
      if num_devices == 1:
        self.assertAllEqual(0.75 * np.ones([10, 1]), updated_var_values[0])
        self.assertAllEqual([0.75], updated_var_values[1])
      else:
        self.assertAllEqual(0.5 * np.ones([10, 1]), updated_var_values[0])
        self.assertAllEqual([0.5], updated_var_values[1])

      writer.FlushNonExecutionFiles()
      writer.FlushExecutionFiles()

    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()
    (context_ids, _,
     op_name_to_op_type, _) = self._readAndCheckGraphsFile(stack_frame_by_id)
    (op_names, device_names, _,
     tensor_values) = self._readAndCheckGraphExecutionTracesFile(context_ids)
    executed_op_types = [op_name_to_op_type[op_name] for op_name in op_names]

    device_name_0 = distribution.extended.worker_devices[0]
    logging.info("device_name_0 = %s", device_name_0)
    if num_devices > 1:
      device_name_1 = distribution.extended.worker_devices[1]
      logging.info("device_name_1 = %s", device_name_1)

    device_0_executed_op_types = filter_by_device_name(
        executed_op_types, device_names, device_name_0)
    if num_devices > 1:
      device_1_executed_op_types = filter_by_device_name(
          executed_op_types, device_names, device_name_1)
    # Verify graph-execution traces are available for both devices.
    # We don't assert MatMul occurs exactly once because the gradient of MatMul
    # involves MatMul.
    self.assertIn("MatMul", device_0_executed_op_types)
    self.assertEqual(device_0_executed_op_types.count("BiasAdd"), 1)
    if num_devices > 1:
      self.assertIn("MatMul", device_1_executed_op_types)
      self.assertEqual(device_1_executed_op_types.count("BiasAdd"), 1)

    if tensor_debug_mode == "NO_TENSOR":
      for value_list in tensor_values:
        for tensor_value in value_list:
          self.assertEqual(tensor_value.dtype, np.float32)
          self.assertEqual(tensor_value.shape, [])
    elif tensor_debug_mode == "FULL_TENSOR":
      device_0_matmul_values = filter_by_device_name_and_op_type(
          tensor_values, device_names, executed_op_types, device_name_0,
          "MatMul")
      device_0_bias_add_values = filter_by_device_name_and_op_type(
          tensor_values, device_names, executed_op_types, device_name_0,
          "BiasAdd")
      self.assertAllClose(device_0_matmul_values[0], [[10.0]])
      self.assertAllClose(device_0_bias_add_values[0], [[11.0]])
      if num_devices > 1:
        device_1_matmul_values = filter_by_device_name_and_op_type(
            tensor_values, device_names, executed_op_types, device_name_1,
            "MatMul")
        device_1_bias_add_values = filter_by_device_name_and_op_type(
            tensor_values, device_names, executed_op_types, device_name_1,
            "BiasAdd")
        self.assertAllClose(device_1_matmul_values[0], [[10.0]])
        self.assertAllClose(device_1_bias_add_values[0], [[11.0]])

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.one_device_strategy_gpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
          ],
          mode=["eager"],
          tensor_debug_mode=["NO_TENSOR", "FULL_TENSOR"],
      ))
  def testKerasModelFitOnOneOrTwoDevices(self, distribution, tensor_debug_mode):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    with distribution.scope():
      model = keras.Sequential()
      model.add(keras.layers.Dense(
          units=10, input_shape=[5], activation="relu"))
      model.add(keras.layers.Dense(units=1))
      model.compile(loss="mse", optimizer="sgd")

      batch_size = 20
      x = np.ones([batch_size, 5])
      y = np.ones([batch_size, 1])
      epochs = 1
      history = model.fit(x, y, epochs=epochs, verbose=0)
      self.assertLen(history.history["loss"], epochs)

      writer.FlushNonExecutionFiles()
      writer.FlushExecutionFiles()

    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()
    (context_ids, _,
     op_name_to_op_type, _) = self._readAndCheckGraphsFile(stack_frame_by_id)
    (op_names, device_names, _,
     tensor_values) = self._readAndCheckGraphExecutionTracesFile(context_ids)

    # Eager execution of tf.function should be recorded.
    executed_op_types, _, _, _, _ = self._readAndCheckExecutionFile()
    fit_functions = [op_type for op_type in executed_op_types
                     if "_distributed_function" in op_type]
    self.assertLen(fit_functions, epochs)

    num_devices = len(distribution.extended.worker_devices)

    device_name_0 = distribution.extended.worker_devices[0]
    logging.info("device_name_0 = %s", device_name_0)
    if num_devices > 1:
      device_name_1 = distribution.extended.worker_devices[1]
      logging.info("device_name_1 = %s", device_name_1)

    executed_op_types = [op_name_to_op_type[op_name] for op_name in op_names]
    device_0_executed_op_types = filter_by_device_name(
        executed_op_types, device_names, device_name_0)
    if num_devices > 1:
      device_1_executed_op_types = filter_by_device_name(
          executed_op_types, device_names, device_name_1)

    self.assertIn("MatMul", device_0_executed_op_types)
    self.assertIn("BiasAdd", device_0_executed_op_types)
    self.assertIn("Relu", device_0_executed_op_types)
    self.assertIn("ReluGrad", device_0_executed_op_types)
    if num_devices > 1:
      # If there are two devices involved, assert the ops inside tf.functions
      # are executed and recorded for the equal numbers of times by the
      # dumping op-callback.
      self.assertEqual(device_0_executed_op_types.count("MatMul"),
                       device_1_executed_op_types.count("MatMul"))
      self.assertEqual(device_0_executed_op_types.count("BiasAdd"),
                       device_1_executed_op_types.count("BiasAdd"))
      self.assertEqual(device_0_executed_op_types.count("Relu"),
                       device_1_executed_op_types.count("Relu"))
      self.assertEqual(device_0_executed_op_types.count("ReluGrad"),
                       device_1_executed_op_types.count("ReluGrad"))

    if tensor_debug_mode == "NO_TENSOR":
      for value_list in tensor_values:
        for tensor_value in value_list:
          self.assertEqual(tensor_value.dtype, np.float32)
          self.assertEqual(tensor_value.shape, [])
    elif tensor_debug_mode == "FULL_TENSOR":
      gpu_0_relu_values = filter_by_device_name_and_op_type(
          tensor_values, device_names, executed_op_types, device_name_0, "Relu")
      self.assertTrue(gpu_0_relu_values)
      gpu_0_relu_grad_values = filter_by_device_name_and_op_type(
          tensor_values, device_names, executed_op_types, device_name_0,
          "ReluGrad")
      self.assertTrue(gpu_0_relu_grad_values)
      if num_devices > 1:
        gpu_1_relu_values = filter_by_device_name_and_op_type(
            tensor_values, device_names, executed_op_types, device_name_1,
            "Relu")
        self.assertTrue(gpu_1_relu_values)
        for i in range(len(gpu_0_relu_values)):
          self.assertEqual(gpu_0_relu_values[i].shape,
                           gpu_1_relu_values[i].shape)
        gpu_1_relu_grad_values = filter_by_device_name_and_op_type(
            tensor_values, device_names, executed_op_types, device_name_1,
            "ReluGrad")
        self.assertTrue(gpu_1_relu_grad_values)
        for i in range(len(gpu_0_relu_grad_values)):
          self.assertEqual(
              gpu_0_relu_grad_values[i].shape, gpu_1_relu_grad_values[i].shape)


if __name__ == "__main__":
  googletest.main()
