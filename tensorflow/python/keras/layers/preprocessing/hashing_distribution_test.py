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
"""Tests for keras.layers.preprocessing.hashing."""

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.distribute.strategy_combinations import all_strategies
from tensorflow.python.keras.layers.preprocessing import hashing
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.platform import test


@ds_combinations.generate(
    combinations.combine(
        distribution=all_strategies,
        mode=["eager", "graph"]))
class HashingDistributionTest(keras_parameterized.TestCase,
                              preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution(self, distribution):
    input_data = np.asarray([["omar"], ["stringer"], ["marlo"], ["wire"]])
    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_data).batch(
        2, drop_remainder=True)
    expected_output = [[0], [0], [1], [0]]

    config.set_soft_device_placement(True)

    with distribution.scope():
      input_data = keras.Input(shape=(None,), dtype=dtypes.string)
      layer = hashing.Hashing(num_bins=2)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
  test.main()
