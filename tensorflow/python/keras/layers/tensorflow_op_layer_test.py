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
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


def _single_op_at_end():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  outputs = gen_nn_ops.relu(x)
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


@keras_parameterized.run_all_keras_modes
class AutoLambdaTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_op_at_end', _single_op_at_end),
      ('multiple_ops_at_end', _multiple_ops_at_end),
      ('single_op_in_middle', _single_op_in_middle),
      ('multiple_ops_in_middle', _multiple_ops_in_middle),
      ('single_standalone_branch', _single_standalone_branch),
      ('single_op_with_attrs', _single_op_with_attrs),
      ('multiple_uses', _multiple_uses))
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

  def test_serialization(self):
    x = ops.convert_to_tensor([-1., 0., -2., 1.])
    inputs = keras.Input(shape=(4,))
    outputs = gen_nn_ops.relu(inputs)
    model1 = keras.Model(inputs, outputs)
    y1 = self.evaluate(model1(x))
    model2 = model1.from_config(model1.get_config())
    y2 = self.evaluate(model2(x))
    self.assertAllClose(y1, y2)

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

    # Check reasonable graph construction time.
    self.assertLess(size_50, 5)
    # Check construction time grows approx. linearly with size.
    e = 1.5  # Fudge factor to prevent flakiness.
    self.assertLess(size_500, (10 * e) * size_50)


if __name__ == '__main__':
  test.main()
