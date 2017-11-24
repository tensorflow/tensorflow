# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.contrib.kfac.estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kfac.python.ops import estimator
from tensorflow.contrib.kfac.python.ops import layer_collection as lc
from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test

_ALL_ESTIMATION_MODES = ["gradients", "empirical", "curvature_prop", "exact"]


class DeviceContextGeneratorTest(test.TestCase):

  def testNoDevice(self):
    device_context_generator = estimator._DeviceContextGenerator(None)
    with ops.device("/device:CPU:0"):  # This is what will be used
      with device_context_generator():  # Does nothing
        a = constant_op.constant([2.0], name="a")
    self.assertEqual("/device:CPU:0", a.op.device)

  def testTwoDevices(self):
    device_context_generator = estimator._DeviceContextGenerator(
        ["/device:GPU:0", "/device:GPU:1"])
    with ops.device("/device:CPU:0"):  # Will be over-ridden by the inner scopes
      with device_context_generator():
        a = constant_op.constant([2.0], name="a")
      with device_context_generator():
        b = constant_op.constant([2.0], name="b")
      with device_context_generator():
        c = constant_op.constant([2.0], name="c")
    self.assertEqual("/device:GPU:0", a.op.device)
    self.assertEqual("/device:GPU:1", b.op.device)
    self.assertEqual("/device:GPU:0", c.op.device)


class EstimatorTest(test.TestCase):

  def setUp(self):
    self._graph = ops.Graph()
    with self._graph.as_default():
      self.layer_collection = lc.LayerCollection()

      self.inputs = random_ops.random_normal((2, 2), dtype=dtypes.float32)
      self.weights = variable_scope.get_variable(
          "w", shape=(2, 2), dtype=dtypes.float32)
      self.bias = variable_scope.get_variable(
          "b", initializer=init_ops.zeros_initializer(), shape=(2, 1))
      self.output = math_ops.matmul(self.inputs, self.weights) + self.bias

      # Only register the weights.
      self.layer_collection.register_fully_connected(
          params=(self.weights,), inputs=self.inputs, outputs=self.output)

      self.outputs = math_ops.tanh(self.output)
      self.targets = array_ops.zeros_like(self.outputs)
      self.layer_collection.register_categorical_predictive_distribution(
          logits=self.outputs, targets=self.targets)

  def testEstimatorInitManualRegistration(self):
    with self._graph.as_default():
      # We should be able to build an estimator for only the registered vars.
      estimator.FisherEstimator([self.weights], 0.1, 0.2, self.layer_collection)

      # Check that we throw an error if we try to build an estimator for vars
      # that were not manually registered.
      with self.assertRaises(ValueError):
        estimator.FisherEstimator([self.weights, self.bias], 0.1, 0.2,
                                  self.layer_collection)

      # Check that we throw an error if we don't include registered variables,
      # i.e. self.weights
      with self.assertRaises(ValueError):
        estimator.FisherEstimator([], 0.1, 0.2, self.layer_collection)

  @test.mock.patch.object(utils.SubGraph, "variable_uses", return_value=42)
  def testVariableWrongNumberOfUses(self, mock_uses):
    with self.assertRaises(ValueError):
      estimator.FisherEstimator([self.weights], 0.1, 0.2, self.layer_collection)

  def testInvalidEstimationMode(self):
    with self.assertRaises(ValueError):
      estimator.FisherEstimator([self.weights], 0.1, 0.2, self.layer_collection,
                                "not_a_real_mode")

  def testModeListCorrect(self):
    with self._graph.as_default():
      est = estimator.FisherEstimator([self.weights], 0.1, 0.2,
                                      self.layer_collection)
    self.assertItemsEqual(_ALL_ESTIMATION_MODES, est._gradient_fns.keys())

  def testAllModesBuild(self):
    for mode in _ALL_ESTIMATION_MODES:
      with self._graph.as_default():
        estimator.FisherEstimator([self.weights], 0.1, 0.2,
                                  self.layer_collection, mode)


if __name__ == "__main__":
  test.main()
