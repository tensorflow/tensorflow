# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for advanced activation layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class AdvancedActivationsTest(keras_parameterized.TestCase):

  def test_leaky_relu(self):
    for alpha in [0., .5, -1.]:
      testing_utils.layer_test(keras.layers.LeakyReLU,
                               kwargs={'alpha': alpha},
                               input_shape=(2, 3, 4))

  def test_prelu(self):
    testing_utils.layer_test(keras.layers.PReLU, kwargs={},
                             input_shape=(2, 3, 4))

  def test_prelu_share(self):
    testing_utils.layer_test(keras.layers.PReLU,
                             kwargs={'shared_axes': 1},
                             input_shape=(2, 3, 4))

  def test_elu(self):
    for alpha in [0., .5, -1.]:
      testing_utils.layer_test(keras.layers.ELU,
                               kwargs={'alpha': alpha},
                               input_shape=(2, 3, 4))

  def test_thresholded_relu(self):
    testing_utils.layer_test(keras.layers.ThresholdedReLU,
                             kwargs={'theta': 0.5},
                             input_shape=(2, 3, 4))

  def test_softmax(self):
    testing_utils.layer_test(keras.layers.Softmax,
                             kwargs={'axis': 1},
                             input_shape=(2, 3, 4))

  def test_relu(self):
    testing_utils.layer_test(keras.layers.ReLU,
                             kwargs={'max_value': 10},
                             input_shape=(2, 3, 4))
    x = keras.backend.ones((3, 4))
    if not context.executing_eagerly():
      # Test that we use `leaky_relu` when appropriate in graph mode.
      self.assertTrue(
          'LeakyRelu' in keras.layers.ReLU(negative_slope=0.2)(x).name)
      # Test that we use `relu` when appropriate in graph mode.
      self.assertTrue('Relu' in keras.layers.ReLU()(x).name)
      # Test that we use `relu6` when appropriate in graph mode.
      self.assertTrue('Relu6' in keras.layers.ReLU(max_value=6)(x).name)

  def test_relu_with_invalid_arg(self):
    with self.assertRaisesRegexp(
        ValueError, 'max_value of Relu layer cannot be negative value: -10'):
      testing_utils.layer_test(keras.layers.ReLU,
                               kwargs={'max_value': -10},
                               input_shape=(2, 3, 4))
    with self.assertRaisesRegexp(
        ValueError,
        'negative_slope of Relu layer cannot be negative value: -2'):
      with self.cached_session():
        testing_utils.layer_test(
            keras.layers.ReLU,
            kwargs={'negative_slope': -2},
            input_shape=(2, 3, 4))

  @keras_parameterized.run_with_all_model_types
  def test_layer_as_activation(self):
    layer = keras.layers.Dense(1, activation=keras.layers.ReLU())
    model = testing_utils.get_model_from_layers([layer], input_shape=(10,))
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    model.fit(np.ones((10, 10)), np.ones((10, 1)), batch_size=2)


if __name__ == '__main__':
  test.main()
