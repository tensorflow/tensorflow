# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras high level APIs, e.g. fit, evaluate and predict."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.distribute.strategy_combinations import all_strategies
from tensorflow.python.platform import test


class KerasModelsTest(test.TestCase, parameterized.TestCase):

  @ds_combinations.generate(
      combinations.combine(
          distribution=all_strategies, mode=["eager"]))
  def test_lstm_model_with_dynamic_batch(self, distribution):
    input_data = np.random.random([1, 32, 64, 64, 3])
    input_shape = tuple(input_data.shape[1:])

    def build_model():
      model = keras.models.Sequential()
      model.add(
          keras.layers.ConvLSTM2D(
              4,
              kernel_size=(4, 4),
              activation="sigmoid",
              padding="same",
              input_shape=input_shape))
      model.add(keras.layers.GlobalMaxPooling2D())
      model.add(keras.layers.Dense(2, activation="sigmoid"))
      return model

    with distribution.scope():
      model = build_model()
      model.compile(loss="binary_crossentropy", optimizer="adam")
      result = model.predict(input_data)
      self.assertEqual(result.shape, (1, 2))


if __name__ == "__main__":
  test.main()
