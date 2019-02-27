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
"""Tests for numerical correctness."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class Bias(keras.layers.Layer):
  """Layer that add a bias to its inputs."""

  def build(self, input_shape):
    self.bias = self.add_variable('bias', (1,), initializer='zeros')

  def call(self, inputs):
    return inputs + self.bias


class MultiInputSubclassed(keras.Model):
  """Subclassed Model that adds its inputs and then adds a bias."""

  def __init__(self):
    super(MultiInputSubclassed, self).__init__()
    self.add = keras.layers.Add()
    self.bias = Bias()

  def call(self, inputs):
    added = self.add(inputs)
    return self.bias(added)


def multi_input_functional():
  """Functional Model that adds its inputs and then adds a bias."""
  input_1 = keras.Input(shape=(1,))
  input_2 = keras.Input(shape=(1,))
  input_3 = keras.Input(shape=(1,))
  added = keras.layers.Add()([input_1, input_2, input_3])
  output = Bias()(added)
  return keras.Model([input_1, input_2, input_3], output)


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class SimpleBiasTest(keras_parameterized.TestCase):

  def _get_simple_bias_model(self):
    model = testing_utils.get_model_from_layers([Bias()], input_shape=(1,))
    model.compile(
        keras.optimizer_v2.gradient_descent.SGD(0.1),
        'mae',
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  def test_simple_bias_fit(self):
    x = np.array([[0.], [1.], [2.]])
    y = np.array([[0.5], [2.], [3.5]])
    model = self._get_simple_bias_model()

    history = model.fit(x, y, batch_size=3, epochs=5)
    self.assertAllClose(history.history['loss'], [1., 0.9, 0.8, 0.7, 0.6])

  def test_simple_bias_evaluate(self):
    x = np.array([[0.], [1.], [2.]])
    y = np.array([[1.], [3.], [5.]])
    model = self._get_simple_bias_model()

    loss = model.evaluate(x, y, batch_size=1)
    self.assertAlmostEqual(loss, 2.)

  def test_simple_bias_predict(self):
    x = np.array([[0.], [1.], [2.]])
    model = self._get_simple_bias_model()

    pred = model.predict(x, batch_size=1)
    self.assertAllClose(x, pred)


@keras_parameterized.run_all_keras_modes
class MultipleInputTest(keras_parameterized.TestCase):

  def _get_multiple_input_model(self, subclassed=True):
    if subclassed:
      model = MultiInputSubclassed()
    else:
      model = multi_input_functional()
    model.compile(
        keras.optimizer_v2.gradient_descent.SGD(0.1),
        'mae',
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  @parameterized.named_parameters(('subclassed', True), ('functional', False))
  def test_multiple_input_fit(self, subclassed):
    x = [
        np.array([[1.], [2.], [3.]]),
        np.array([[4.], [5.], [6.]]),
        np.array([[7.], [8.], [9.]])
    ]
    y = np.array([[12.5], [16.], [19.5]])

    model = self._get_multiple_input_model(subclassed)
    history = model.fit(x, y, batch_size=3, epochs=5)
    self.assertAllClose(history.history['loss'], [1., 0.9, 0.8, 0.7, 0.6])

  @parameterized.named_parameters(('subclassed', True), ('functional', False))
  def test_multiple_input_evaluate(self, subclassed):
    x = [
        np.array([[1.], [2.], [3.]]),
        np.array([[4.], [5.], [6.]]),
        np.array([[7.], [8.], [9.]])
    ]
    y = np.array([[13.], [17.], [21.]])

    model = self._get_multiple_input_model(subclassed)
    loss = model.evaluate(x, y, batch_size=3)
    self.assertAlmostEqual(loss, 2.)

  @parameterized.named_parameters(('subclassed', True), ('functional', False))
  def test_multiple_input_predict(self, subclassed):
    x = [
        np.array([[1.], [2.], [3.]]),
        np.array([[4.], [5.], [6.]]),
        np.array([[7.], [8.], [9.]])
    ]

    model = self._get_multiple_input_model(subclassed)
    pred = model.predict(x, batch_size=1)
    self.assertAllClose(pred, [[12.], [15.], [18.]])


if __name__ == '__main__':
  test.main()
