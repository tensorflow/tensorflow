# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Distribution tests for keras.layers.preprocessing.text_vectorization."""

import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.distribute import strategy_combinations
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.layers.preprocessing import text_vectorization


@ds_combinations.generate(
    combinations.combine(
        strategy=strategy_combinations.all_strategies +
        strategy_combinations.multi_worker_mirrored_strategies,
        mode=["eager"]))
class TextVectorizationDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution_strategy_output(self, strategy):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_array).batch(
        2, drop_remainder=True)

    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    config.set_soft_device_placement(True)

    with strategy.scope():
      input_data = keras.Input(shape=(None,), dtype=dtypes.string)
      layer = text_vectorization.TextVectorization(
          max_tokens=None,
          standardize=None,
          split=None,
          output_mode=text_vectorization.INT)
      layer.set_vocabulary(vocab_data)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)

    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)

  def test_distribution_strategy_output_with_adapt(self, strategy):
    vocab_data = [[
        "earth", "earth", "earth", "earth", "wind", "wind", "wind", "and",
        "and", "fire"
    ]]
    vocab_dataset = dataset_ops.Dataset.from_tensors(vocab_data)
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_array).batch(
        2, drop_remainder=True)

    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    config.set_soft_device_placement(True)

    with strategy.scope():
      input_data = keras.Input(shape=(None,), dtype=dtypes.string)
      layer = text_vectorization.TextVectorization(
          max_tokens=None,
          standardize=None,
          split=None,
          output_mode=text_vectorization.INT)
      layer.adapt(vocab_dataset)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)

    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)

if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
