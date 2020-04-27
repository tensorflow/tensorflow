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
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.distribute import tpu_strategy_test_utils
from tensorflow.python.keras.layers.preprocessing import discretization
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(
    always_skip_v1=True, always_skip_eager=True)
class DiscretizationDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_tpu_distribution(self):
    input_array = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])

    expected_output = [[0, 2, 3, 1], [1, 3, 2, 1]]
    expected_output_shape = [None, None]

    strategy = tpu_strategy_test_utils.get_tpu_strategy()
    with strategy.scope():
      input_data = keras.Input(shape=(None,))
      layer = discretization.Discretization(
          bins=[0., 1., 2.], output_mode=discretization.INTEGER)
      bucket_data = layer(input_data)
      self.assertAllEqual(expected_output_shape, bucket_data.shape.as_list())

      model = keras.Model(inputs=input_data, outputs=bucket_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
  test.main()
