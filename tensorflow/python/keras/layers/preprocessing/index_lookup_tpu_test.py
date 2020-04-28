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
"""Tests for keras.layers.preprocessing.normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.distribute import tpu_strategy_test_utils
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(
    always_skip_v1=True, always_skip_eager=True)
class IndexLookupDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_tpu_distribution(self):
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
    strategy = tpu_strategy_test_utils.get_tpu_strategy()

    with strategy.scope():
      input_data = keras.Input(shape=(None,), dtype=dtypes.string)
      layer = index_lookup.IndexLookup()
      layer.adapt(vocab_dataset)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
  test.main()
