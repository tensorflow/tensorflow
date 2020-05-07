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
"""Tests for dynamic control flow behavior with Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class ControlFlowLayer1(base_layer.Layer):
  """Layer with an `if` condition in call."""

  def call(self, inputs):
    if math_ops.reduce_sum(inputs) > 0:
      return math_ops.sqrt(inputs)
    else:
      return math_ops.square(inputs)


class ControlFlowLayer2(base_layer.Layer):
  """Layer with a `for` loop in call."""

  def call(self, inputs):
    samples = tensor_array_ops.TensorArray(
        dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
    i = 0
    for sample in inputs:
      samples = samples.write(i, math_ops.square(sample))
      i += 1
    return samples.stack()


class NestedControlFlowLayer(base_layer.Layer):
  """Layer nested with a control flow layer."""

  def __init__(self, **kwargs):
    super(NestedControlFlowLayer, self).__init__(**kwargs)
    self.layer = ControlFlowLayer1()

  def call(self, inputs):
    return self.layer(inputs)


class ControlFlowModel(keras.Model):
  """Model with an `if` condition in call."""

  def call(self, inputs):
    if math_ops.reduce_sum(inputs) > 0:
      return math_ops.sqrt(inputs)
    else:
      return math_ops.square(inputs)


class NestedControlFlowModel(keras.Model):
  """Model with an `if` condition in call using a control flow layer."""

  def __init__(self, **kwargs):
    super(NestedControlFlowModel, self).__init__(**kwargs)
    self.layer = NestedControlFlowLayer()

  def call(self, inputs):
    inputs = self.layer(inputs)
    if math_ops.reduce_sum(inputs) > 0:
      return math_ops.sqrt(inputs)
    else:
      return math_ops.square(inputs)


class FunctionControlFlowModel(keras.Model):
  """Model with control flow where `call` is wrapped in function already."""

  @def_function.function
  def call(self, inputs):
    if math_ops.reduce_sum(inputs) > 0:
      return math_ops.sqrt(inputs)
    else:
      return math_ops.square(inputs)


@keras_parameterized.run_all_keras_modes
class AutographWrapperTest(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @parameterized.named_parameters(('with_if', ControlFlowLayer1),
                                  ('with_for', ControlFlowLayer2),
                                  ('nested', NestedControlFlowLayer))
  def test_control_flow_layer(self, layer_class):
    model = testing_utils.get_model_from_layers([layer_class()],
                                                input_shape=(3,))
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  @parameterized.named_parameters(
      ('with_if', ControlFlowModel), ('nested', NestedControlFlowModel),
      ('wrapped_in_function', FunctionControlFlowModel))
  def test_control_flow_model(self, model_class):
    model = model_class()
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_control_flow_in_deferred_sequential_model(self):
    model = keras.Sequential(
        [ControlFlowLayer1(),
         keras.layers.Dense(3),
         ControlFlowLayer2()])
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))


if __name__ == '__main__':
  test.main()
