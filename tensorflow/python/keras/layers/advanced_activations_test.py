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

from tensorflow.python import keras
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class AdvancedActivationsTest(test.TestCase):

  def test_leaky_relu(self):
    with self.test_session():
      for alpha in [0., .5, -1.]:
        testing_utils.layer_test(keras.layers.LeakyReLU,
                                 kwargs={'alpha': alpha},
                                 input_shape=(2, 3, 4))

  def test_prelu(self):
    with self.test_session():
      testing_utils.layer_test(keras.layers.PReLU, kwargs={},
                               input_shape=(2, 3, 4))

  def test_prelu_share(self):
    with self.test_session():
      testing_utils.layer_test(keras.layers.PReLU,
                               kwargs={'shared_axes': 1},
                               input_shape=(2, 3, 4))

  def test_elu(self):
    with self.test_session():
      for alpha in [0., .5, -1.]:
        testing_utils.layer_test(keras.layers.ELU,
                                 kwargs={'alpha': alpha},
                                 input_shape=(2, 3, 4))

  def test_thresholded_relu(self):
    with self.test_session():
      testing_utils.layer_test(keras.layers.ThresholdedReLU,
                               kwargs={'theta': 0.5},
                               input_shape=(2, 3, 4))

  def test_softmax(self):
    with self.test_session():
      testing_utils.layer_test(keras.layers.Softmax,
                               kwargs={'axis': 1},
                               input_shape=(2, 3, 4))

  def test_relu(self):
    with self.test_session():
      testing_utils.layer_test(keras.layers.ReLU,
                               kwargs={'max_value': 10},
                               input_shape=(2, 3, 4))

  def test_relu_with_invalid_arg(self):
    with self.assertRaisesRegexp(
        ValueError, 'max_value of Relu layer cannot be negative value: -10'):
      with self.test_session():
        testing_utils.layer_test(keras.layers.ReLU,
                                 kwargs={'max_value': -10},
                                 input_shape=(2, 3, 4))
    with self.assertRaisesRegexp(
        ValueError,
        'negative_slope of Relu layer cannot be negative value: -2'):
      with self.test_session():
        testing_utils.layer_test(
            keras.layers.ReLU,
            kwargs={'negative_slope': -2},
            input_shape=(2, 3, 4))


if __name__ == '__main__':
  test.main()
