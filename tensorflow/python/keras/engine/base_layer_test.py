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
    self.assertEqual(model._static_graph_friendly, False)
    model.compile(RMSPropOptimizer(0.001), loss='mse')
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

    inputs = keras.Input((3,))
    outputs = DynamicLayer2()(inputs)
    model = keras.Model(inputs, outputs)
    self.assertEqual(model._static_graph_friendly, False)
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
    self.assertEqual(model._static_graph_friendly, False)
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
    self.assertEqual(model._static_graph_friendly, False)
    model.compile(RMSPropOptimizer(0.001), loss='mse')
    with self.assertRaisesRegexp(ValueError, 'You did something wrong!'):
      model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_using_symbolic_tensors_with_tf_ops(self):
    # Single-input.
    x = keras.Input((3,))
    y = math_ops.square(x)
    self.assertEqual(y.graph, keras.backend.get_graph())

    # Multi-inputs.
    x1, x2 = keras.Input((3,)), keras.Input((3,))
    y = array_ops.concat([x1, x2], axis=1)
    self.assertEqual(y.graph, keras.backend.get_graph())

    # Mixing Keras symbolic tensors and graph tensors from the same graph works.
    with keras.backend.get_graph().as_default():
      x1 = keras.Input((3,))
    x2 = keras.Input((3,))
    y = math_ops.matmul(x1, x2)
    self.assertEqual(y.graph, keras.backend.get_graph())

    # Creating same op type (matmul) multiple times in the Keras graph works.
    x1 = keras.Input((3,))
    x2 = keras.Input((3,))
    y = math_ops.matmul(x1, x2)
    self.assertEqual(y.graph, keras.backend.get_graph())

  def test_mixing_eager_and_graph_tensors(self):
    with ops.Graph().as_default():
      x1 = array_ops.ones((3, 3))
    x2 = array_ops.ones((3, 3))
    self.assertTrue(isinstance(x2, ops.EagerTensor))
    with self.assertRaisesRegexp(TypeError,
                                 'provided list of inputs contains '
                                 'objects other than \'EagerTensor\''):
      math_ops.matmul(x1, x2)

  def test_mixing_numpy_arrays_and_graph_tensors(self):
    with ops.Graph().as_default():
      x1 = array_ops.ones((3, 3))
    x2 = np.ones((3, 3), dtype='float32')
    with self.assertRaisesRegexp(TypeError,
                                 'provided list of inputs contains '
                                 'objects other than \'EagerTensor\''):
      math_ops.matmul(x1, x2)

  def test_mixing_keras_symbolic_tensors_and_eager_tensors(self):
    x1 = keras.Input((3,))
    x2 = array_ops.ones((3, 3))
    y = math_ops.matmul(x1, x2)
    self.assertEqual(y.graph, keras.backend.get_graph())
    fn = keras.backend.function(inputs=[x1], outputs=[y])
    x_val = np.random.random((3, 3))
    y_val = np.ones((3, 3))
    self.assertAllClose(fn([x_val])[0],
                        np.matmul(x_val, y_val),
                        atol=1e-5)

  def test_mixing_keras_symbolic_tensors_and_numpy_arrays(self):
    x1 = keras.Input((3,))
    x2 = np.ones((3, 3), dtype='float32')
    y = math_ops.matmul(x1, x2)
    self.assertEqual(y.graph, keras.backend.get_graph())
    fn = keras.backend.function(inputs=[x1], outputs=[y])
    x_val = np.random.random((3, 3))
    y_val = np.ones((3, 3))
    self.assertAllClose(fn([x_val])[0],
                        np.matmul(x_val, y_val),
                        atol=1e-5)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
