# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TensorFlow 2.0 layer behavior."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer


class DynamicLayer1(base_layer.Layer):

  def call(self, inputs):
    if math_ops.reduce_sum(inputs) > 0:
      return math_ops.sqrt(inputs)
    else:
      return math_ops.square(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape


class DynamicLayer2(base_layer.Layer):

  def call(self, inputs):
    samples = []
    for sample in inputs:
      samples.append(math_ops.square(sample))
    return array_ops.stack(samples, axis=0)

  def compute_output_shape(self, input_shape):
    return input_shape


class InvalidLayer(base_layer.Layer):

  def call(self, inputs):
    raise ValueError('You did something wrong!')

  def compute_output_shape(self, input_shape):
    return input_shape


class BaseLayerTest(test.TestCase):

  def test_dynamic_layer_in_functional_model_in_graph_mode(self):
    with context.graph_mode():
      inputs = keras.Input((3,))
      with self.assertRaisesRegexp(
          TypeError, 'Using a `tf.Tensor` as a Python `bool` is not allowed'):
        _ = DynamicLayer1()(inputs)

      inputs = keras.Input((3,))
      with self.assertRaisesRegexp(
          TypeError, 'Tensor objects are only iterable when eager'):
        _ = DynamicLayer2()(inputs)

  def test_dynamic_layer_in_functional_model_in_eager_mode(self):
    inputs = keras.Input((3,))
    outputs = DynamicLayer1()(inputs)
    model = keras.Model(inputs, outputs)
    self.assertEqual(model._is_static_graph_friendly, False)
    model.compile(RMSPropOptimizer(0.001), loss='mse')
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

    inputs = keras.Input((3,))
    outputs = DynamicLayer2()(inputs)
    model = keras.Model(inputs, outputs)
    self.assertEqual(model._is_static_graph_friendly, False)
    model.compile(RMSPropOptimizer(0.001), loss='mse')
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def nested_dynamic_layers_in_eager_mode(self):
    inputs = keras.Input((3,))
    outputs = DynamicLayer1()(inputs)
    inner_model = keras.Model(inputs, outputs)

    inputs = keras.Input((3,))
    x = DynamicLayer2()(inputs)
    outputs = inner_model(x)

    model = keras.Model(inputs, outputs)
    self.assertEqual(model._is_static_graph_friendly, False)
    model.compile(RMSPropOptimizer(0.001), loss='mse')
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_invalid_forward_pass_in_graph_mode(self):
    with context.graph_mode():
      inputs = keras.Input((3,))
      with self.assertRaisesRegexp(ValueError, 'You did something wrong!'):
        _ = InvalidLayer()(inputs)

  def test_invalid_forward_pass_in_eager_mode(self):
    inputs = keras.Input((3,))
    outputs = InvalidLayer()(inputs)
    model = keras.Model(inputs, outputs)
    self.assertEqual(model._is_static_graph_friendly, False)
    model.compile(RMSPropOptimizer(0.001), loss='mse')
    with self.assertRaisesRegexp(ValueError, 'You did something wrong!'):
      model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
