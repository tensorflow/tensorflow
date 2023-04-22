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
"""Preprocessing stage tests."""
# pylint: disable=g-classes-have-attributes

import time
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.layers.preprocessing import preprocessing_stage
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class PreprocessingStageTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_adapt(self):

    class PL(base_preprocessing_layer.PreprocessingLayer):

      def __init__(self, **kwargs):
        self.adapt_time = None
        self.adapt_count = 0
        super(PL, self).__init__(**kwargs)

      def adapt(self, data, reset_state=True):
        self.adapt_time = time.time()
        self.adapt_count += 1

      def call(self, inputs):
        return inputs + 1.

    # Test with NumPy array
    stage = preprocessing_stage.PreprocessingStage([
        PL(),
        PL(),
        PL(),
    ])
    stage.adapt(np.ones((3, 4)))
    self.assertEqual(stage.layers[0].adapt_count, 1)
    self.assertEqual(stage.layers[1].adapt_count, 1)
    self.assertEqual(stage.layers[2].adapt_count, 1)
    self.assertLessEqual(stage.layers[0].adapt_time, stage.layers[1].adapt_time)
    self.assertLessEqual(stage.layers[1].adapt_time, stage.layers[2].adapt_time)

    # Check call
    y = stage(array_ops.ones((3, 4)))
    self.assertAllClose(y, np.ones((3, 4)) + 3.)

    # Test with dataset
    adapt_data = dataset_ops.Dataset.from_tensor_slices(np.ones((3, 10)))
    adapt_data = adapt_data.batch(2)  # 5 batches of 2 samples

    stage.adapt(adapt_data)
    self.assertEqual(stage.layers[0].adapt_count, 2)
    self.assertEqual(stage.layers[1].adapt_count, 2)
    self.assertEqual(stage.layers[2].adapt_count, 2)
    self.assertLess(stage.layers[0].adapt_time, stage.layers[1].adapt_time)
    self.assertLess(stage.layers[1].adapt_time, stage.layers[2].adapt_time)

    # Test error with bad data
    with self.assertRaisesRegex(ValueError, 'requires a '):
      stage.adapt(None)


if __name__ == '__main__':
  test.main()
