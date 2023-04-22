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
"""Distribution tests for keras.layers.preprocessing.category_encoding."""

import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.distribute import strategy_combinations
from tensorflow.python.keras.layers.preprocessing import category_encoding
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils


def batch_wrapper(dataset, batch_size, strategy, repeat=None):
  if repeat:
    dataset = dataset.repeat(repeat)
  # TPUs currently require fully defined input shapes, drop_remainder ensures
  # the input will have fully defined shapes.
  if isinstance(strategy,
                (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
    return dataset.batch(batch_size, drop_remainder=True)
  else:
    return dataset.batch(batch_size)


@ds_combinations.generate(
    combinations.combine(
        # (b/156783625): Outside compilation failed for eager mode only.
        strategy=strategy_combinations.strategies_minus_tpu +
        strategy_combinations.multi_worker_mirrored_strategies,
        mode=["eager", "graph"]))
class CategoryEncodingDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_strategy(self, strategy):
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])
    inp_dataset = dataset_ops.DatasetV2.from_tensor_slices(input_array)
    inp_dataset = batch_wrapper(inp_dataset, 2, strategy)

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0, 0],
                       [1, 1, 0, 1, 0, 0]]
    # pyformat: enable
    num_tokens = 6
    config.set_soft_device_placement(True)

    with strategy.scope():
      input_data = keras.Input(shape=(4,), dtype=dtypes.int32)
      layer = category_encoding.CategoryEncoding(
          num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(inp_dataset)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
