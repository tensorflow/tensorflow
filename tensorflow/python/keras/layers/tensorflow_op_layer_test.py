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
"""Test for allowing TF ops to work with Keras Functional API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


def _single_op_at_end():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  outputs = gen_nn_ops.relu(x)
  return inputs, outputs


def _single_identity_op_at_end():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  outputs = array_ops.identity(x)
  assert 'Identity' in outputs.name
  return inputs, outputs


def _multiple_ops_at_end():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  x = gen_nn_ops.relu(x)
  outputs = gen_nn_ops.relu(x)
  return inputs, outputs


def _single_op_in_middle():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  x = gen_nn_ops.relu(x)
  outputs = keras.layers.Dense(10)(x)
  return inputs, outputs


def _multiple_ops_in_middle():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  x = gen_nn_ops.relu(x)
  x = gen_nn_ops.relu(x)
  outputs = keras.layers.Dense(10)(x)
  return inputs, outputs


def _single_standalone_branch():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  outputs = x * 2
  return inputs, outputs


def _single_op_with_attrs():
  inputs = keras.Input(shape=(10,))
  x = math_ops.reduce_mean(inputs, axis=1, keepdims=True)
  outputs = keras.layers.Dense(10)(x)
  return inputs, outputs


def _multiple_uses():
  inputs = keras.Input(shape=(10,))
  x = math_ops.reduce_mean(inputs, axis=1, keepdims=True)
  x1 = keras.layers.Dense(10)(x)
  x2 = keras.layers.Dense(10)(x)
  outputs = x1 + x2
  return inputs, outputs


def _op_with_tensor_list():
  inputs = keras.Input(shape=(10,))
  x = array_ops.concat([inputs, inputs], axis=1)
  outputs = keras.layers.Dense(10)(x)
  return inputs, outputs


def _add_n():
  inputs = keras.Input(shape=(10,))
  outputs = math_ops.add_n([inputs, inputs, inputs])
  return inputs, outputs


def _reuse_op():
  inputs = keras.Input(shape=(10,))
  # This op needs to be checked multiple times.
  x = gen_nn_ops.relu(inputs)
  y = keras.layers.Dense(10)(x)
  x2 = x * 2
  y2 = keras.layers.Dense(10)(x2)
  outputs = y + y2
  return inputs, outputs


class LayerWithLayer(keras.layers.Layer):

  def build(self, input_shape):
    self.bias = self.add_weight(name='bias', dtype='float32')
    self.layer = keras.layers.Dense(10)

  def call(self, inputs):
    inputs = inputs * self.bias
    # Would throw an error if Keras History was created here.
    return self.layer(inputs)


def _inner_layer():
  inputs = keras.Input(shape=(10,))
  outputs = LayerWithLayer()(inputs)
  return inputs, outputs


@keras_parameterized.run_all_keras_modes
class AutoLambdaTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_op_at_end', _single_op_at_end),
      ('single_identity_op_at_end', _single_identity_op_at_end),
      ('multiple_ops_at_end', _multiple_ops_at_end),
      ('single_op_in_middle', _single_op_in_middle),
      ('multiple_ops_in_middle', _multiple_ops_in_middle),
      ('single_standalone_branch', _single_standalone_branch),
      ('single_op_with_attrs', _single_op_with_attrs),
      ('multiple_uses', _multiple_uses),
      ('op_with_tensor_list', _op_with_tensor_list), ('add_n', _add_n),
      ('_reuse_op', _reuse_op), ('_inner_layer', _inner_layer))
  def test_autolambda(self, model_fn):
    inputs, outputs = model_fn()
    model = keras.Model(inputs, outputs)
    model.compile(
        adam.Adam(0.001), 'mse', run_eagerly=testing_utils.should_run_eagerly())

    np_inputs = nest.map_structure(lambda x: np.ones((10, 10), 'float32'),
                                   inputs)
    np_outputs = nest.map_structure(lambda x: np.ones((10, 10), 'float32'),
                                    outputs)
    model.fit(np_inputs, np_outputs, batch_size=2)
    model(np_inputs)  # Test calling the model directly on inputs.

    new_model = keras.Model.from_config(
        model.get_config(), custom_objects={'LayerWithLayer': LayerWithLayer})
    new_model.compile(
        adam.Adam(0.001), 'mse', run_eagerly=testing_utils.should_run_eagerly())
    new_model.fit(np_inputs, np_outputs, batch_size=2)
    new_model(np_inputs)  # Test calling the new model directly on inputs.

  def test_numerical_correctness_simple(self):
    x = ops.convert_to_tensor([[-1., 0., -2., 1.]])
    inputs = keras.Input(shape=(4,))
    outputs = gen_nn_ops.relu(inputs)
    model = keras.Model(inputs, outputs)
    y = self.evaluate(model(x))
    self.assertAllClose(y, [[0., 0., 0., 1.]])

  def test_numerical_correctness_with_attrs(self):
    x = ops.convert_to_tensor([[1.5, 1.5], [2.5, 3.5]])
    inputs = keras.Input(shape=(10,))
    outputs = math_ops.reduce_mean(inputs, axis=1)
    model = keras.Model(inputs, outputs)
    y = self.evaluate(model(x))
    self.assertAllClose(y, [1.5, 3.])

  def test_numerical_correctness_serialization(self):
    x = ops.convert_to_tensor([-1., 0., -2., 1.])
    inputs = keras.Input(shape=(4,))
    outputs = gen_nn_ops.relu(inputs)
    model1 = keras.Model(inputs, outputs)
    y1 = self.evaluate(model1(x))
    model2 = model1.from_config(model1.get_config())
    y2 = self.evaluate(model2(x))
    self.assertAllClose(y1, y2)

  def test_gradient_tape_in_function(self):
    z = keras.Input((1,))
    x = math_ops.matmul(z, constant_op.constant(2.0, shape=(1, 1)))
    x = math_ops.reduce_mean(x, axis=0, keepdims=True)
    h = gen_nn_ops.relu(x)
    m = keras.Model(z, h)

    @def_function.function()
    def f(x):
      with backprop.GradientTape() as t:
        t.watch(x)
        z = m(x ** 2)
      grads = t.gradient(z, x)
      return grads

    self.assertAllEqual(f(constant_op.constant(10.0, shape=(1, 1))),
                        constant_op.constant(40.0, shape=(1, 1)))

    f = def_function.function(f)

    self.assertAllEqual(f(constant_op.constant(10.0, shape=(1, 1))),
                        constant_op.constant(40.0, shape=(1, 1)))

  def test_no_tracking(self):
    x = keras.backend.placeholder((10, 10))
    keras.layers.Dense(1)(x)
    self.assertTrue(x._keras_history_checked)

  def test_timing_scales_linearly(self):

    def _construct_graph_of_size(size):
      start = time.time()
      x = keras.backend.placeholder(shape=(10, 4))

      for _ in range(size):
        x = keras.layers.Dense(4)(x)
        x = gen_nn_ops.relu(x)

      end = time.time()
      return end - start

    size_50 = _construct_graph_of_size(50)
    size_500 = _construct_graph_of_size(500)

    # Check construction time grows approx. linearly with size.
    e = 3  # Fudge factor to prevent flakiness.
    self.assertLess(size_500, (10 * e) * size_50)

  def test_no_mask_tracking(self):
    x = keras.backend.placeholder((10, 10))
    y = keras.layers.Masking(0.)(x)
    self.assertTrue(y._keras_mask._keras_history_checked)

  def test_built(self):
    inputs = keras.Input(shape=(10,))
    outputs = gen_nn_ops.relu(inputs)
    model = keras.Model(inputs, outputs)
    model.compile('sgd', 'mse')
    for layer in model.layers:
      self.assertTrue(layer.built)
    # Test something that requires Layers to be built.
    model.summary()


if __name__ == '__main__':
  test.main()
