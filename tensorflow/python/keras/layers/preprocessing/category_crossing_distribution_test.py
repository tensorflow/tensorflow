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
"""Distribution tests for keras.layers.preprocessing.category_crossing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.distribute.strategy_combinations import all_strategies
from tensorflow.python.keras.layers.preprocessing import category_crossing
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.platform import test


def batch_wrapper(dataset, batch_size, distribution, repeat=None):
  if repeat:
    dataset = dataset.repeat(repeat)
  # TPUs currently require fully defined input shapes, drop_remainder ensures
  # the input will have fully defined shapes.
  if isinstance(distribution,
                (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
    return dataset.batch(batch_size, drop_remainder=True)
  else:
    return dataset.batch(batch_size)


@ds_combinations.generate(
    combinations.combine(
        # Investigate why crossing is not supported with TPU.
        distribution=all_strategies,
        mode=['eager', 'graph']))
class CategoryCrossingDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution(self, distribution):
    input_array_1 = np.array([['a', 'b'], ['c', 'd']])
    input_array_2 = np.array([['e', 'f'], ['g', 'h']])
    inp_dataset = dataset_ops.DatasetV2.from_tensor_slices(
        {'input_1': input_array_1, 'input_2': input_array_2})
    inp_dataset = batch_wrapper(inp_dataset, 2, distribution)

    # pyformat: disable
    expected_output = [[b'a_X_e', b'a_X_f', b'b_X_e', b'b_X_f'],
                       [b'c_X_g', b'c_X_h', b'd_X_g', b'd_X_h']]
    config.set_soft_device_placement(True)

    with distribution.scope():
      input_data_1 = keras.Input(shape=(2,), dtype=dtypes.string,
                                 name='input_1')
      input_data_2 = keras.Input(shape=(2,), dtype=dtypes.string,
                                 name='input_2')
      input_data = [input_data_1, input_data_2]
      layer = category_crossing.CategoryCrossing()
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(inp_dataset)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == '__main__':
  test.main()
