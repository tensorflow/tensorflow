# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for differentiating layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras.backend import constant, get_value
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class DifferentiatingTest(test.TestCase):
  """Testing code for differentiating layer by layer_test"""
  @tf_test_util.run_in_graph_and_eager_modes
  def test_differentiating_1d(self):
    testing_utils.layer_test(keras.layers.differentiating.Diff1D,
                             input_shape=(3, 4, 5))
    testing_utils.layer_test(keras.layers.differentiating.Diff1D,
                             kwargs={'data_format': 'channels_first'},
                             input_shape=(3, 4, 5))

  @tf_test_util.run_in_graph_and_eager_modes
  def test_differentiating_1d_additional(self):
    """Testing code by manually defined values."""
    x = np.array([1., 8., 5., 6., 2.])
    d_x_valid = np.array([7., -3., 1., -4.])
    d_x_same = np.array([7., -3., 1., -4., -4.])

    last = constant(x, shape=(1, 5, 1))
    first = constant(x, shape=(1, 1, 5))

    d_last_valid = keras.layers.differentiating.Diff1D()(last)
    d_last_same = keras.layers.differentiating.Diff1D(padding="same")(last)
    d_first_valid = keras.layers.differentiating.Diff1D(
        data_format='channels_first')(first)
    d_first_same = keras.layers.differentiating.Diff1D(
        data_format='channels_first', padding="same")(first)

    print(get_value(d_last_valid), d_x_valid.reshape((1, 4, 1)))
    self.assertAllEqual(get_value(d_last_valid), d_x_valid.reshape((1, 4, 1)))
    self.assertAllEqual(get_value(d_last_same), d_x_same.reshape((1, 5, 1)))
    self.assertAllEqual(get_value(d_first_valid), d_x_valid.reshape((1, 1, 4)))
    self.assertAllEqual(get_value(d_first_same), d_x_same.reshape((1, 1, 5)))


if __name__ == '__main__':
  test.main()
