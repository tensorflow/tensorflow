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
"""Tests for noise layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class NoiseLayersTest(keras_parameterized.TestCase):

  def test_GaussianNoise(self):
    testing_utils.layer_test(
        keras.layers.GaussianNoise,
        kwargs={'stddev': 1.},
        input_shape=(3, 2, 3))

  def test_GaussianDropout(self):
    testing_utils.layer_test(
        keras.layers.GaussianDropout,
        kwargs={'rate': 0.5},
        input_shape=(3, 2, 3))

  def test_AlphaDropout(self):
    testing_utils.layer_test(
        keras.layers.AlphaDropout, kwargs={'rate': 0.2}, input_shape=(3, 2, 3))

  @staticmethod
  def _make_model(dtype, class_type):
    assert dtype in (dtypes.float32, dtypes.float64)
    assert class_type in ('gaussian_noise', 'gaussian_dropout', 'alpha_noise')
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_shape=(32,), dtype=dtype))
    if class_type == 'gaussian_noise':
      layer = keras.layers.GaussianNoise(0.0003, dtype=dtype)
    elif class_type == 'gaussian_dropout':
      layer = keras.layers.GaussianDropout(0.1, dtype=dtype)
    else:
      layer = keras.layers.AlphaDropout(0.5, dtype=dtype)
    model.add(layer)
    return model

  def _train_model(self, dtype, gtype):
    model = self._make_model(dtype, gtype)
    model.compile(
        optimizer='sgd',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.train_on_batch(np.zeros((8, 32)), np.zeros((8, 8)))

  def test_noise_float32(self):
    self._train_model(dtypes.float32, 'gaussian_noise')

  def test_noise_float64(self):
    self._train_model(dtypes.float64, 'gaussian_noise')

  def test_dropout_float32(self):
    self._train_model(dtypes.float32, 'gaussian_dropout')

  def test_dropout_float64(self):
    self._train_model(dtypes.float64, 'gaussian_dropout')

  def test_alpha_dropout_float32(self):
    self._train_model(dtypes.float32, 'alpha_noise')

  def test_alpha_dropout_float64(self):
    self._train_model(dtypes.float64, 'alpha_noise')


if __name__ == '__main__':
  test.main()
