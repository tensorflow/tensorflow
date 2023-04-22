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
#,============================================================================
"""Tests for `get_config` backwards compatibility."""

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.tests import get_config_samples
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class TestGetConfigBackwardsCompatible(keras_parameterized.TestCase):

  def test_functional_dnn(self):
    model = training.Model.from_config(get_config_samples.FUNCTIONAL_DNN)
    self.assertLen(model.layers, 3)

  def test_functional_cnn(self):
    model = training.Model.from_config(get_config_samples.FUNCTIONAL_CNN)
    self.assertLen(model.layers, 4)

  def test_functional_lstm(self):
    model = training.Model.from_config(get_config_samples.FUNCTIONAL_LSTM)
    self.assertLen(model.layers, 3)

  def test_sequential_dnn(self):
    model = sequential.Sequential.from_config(get_config_samples.SEQUENTIAL_DNN)
    self.assertLen(model.layers, 2)

  def test_sequential_cnn(self):
    model = sequential.Sequential.from_config(get_config_samples.SEQUENTIAL_CNN)
    self.assertLen(model.layers, 3)

  def test_sequential_lstm(self):
    model = sequential.Sequential.from_config(
        get_config_samples.SEQUENTIAL_LSTM)
    self.assertLen(model.layers, 2)


if __name__ == '__main__':
  test.main()
