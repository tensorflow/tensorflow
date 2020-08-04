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
"""Functional preprocessing stage tests."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers import core

from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.layers.preprocessing import normalization
from tensorflow.python.keras.layers.preprocessing import preprocessing_stage_functional
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


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

      def call(self, data, training=True):
        return data + 1


    class PLMerge(PL):
      def call(self, data, training=True):
        return data[0] + data[1]

    class PLSplit(PL):
      def call(self, data, training=True):
        return data + 1, data - 1

    # Test with NumPy array
    x0 = Input(shape=(3,))
    x1 = Input(shape=(3,))
    x2 = Input(shape=(3,))

    l0 = PLMerge()
    y = l0([x0, x1])

    l1 = PLMerge()
    y = l1([y, x2])

    l2 = PLSplit()
    z, y = l2(y)

    stage = preprocessing_stage_functional.PreprocessingStageFunctional([x0, x1, x2], [y, z])
    stage.compile()
    one_array = np.ones((4, 3), dtype='float32')
    stage.adapt([one_array, one_array, one_array])
    self.assertEqual(l0.adapt_count, 1)
    self.assertEqual(l1.adapt_count, 1)
    self.assertEqual(l2.adapt_count, 1)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Check call
    y, z = stage([array_ops.ones((4, 3), dtype='float32'),
                    array_ops.ones((4, 3), dtype='float32'),
                    array_ops.ones((4, 3), dtype='float32')])
    self.assertAllClose(y, np.ones((4, 3), dtype='float32') + 1.)
    self.assertAllClose(z, np.ones((4, 3), dtype='float32') + 3.)

    # Test with dataset
    ds0 = dataset_ops.Dataset.from_tensor_slices(np.ones((10, 3),
                                                         dtype='float32'))
    adapt_data = dataset_ops.Dataset.zip((ds0, ds0, ds0))
    adapt_data = adapt_data.batch(2)  # 5 batches of 2 samples

    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 2)
    self.assertEqual(l1.adapt_count, 2)
    self.assertEqual(l2.adapt_count, 2)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Test error with bad data
    with self.assertRaisesRegex(ValueError, 'requires a '):
      stage.adapt(None)

  def test_mixing_preprocessing_and_regular_layers(self):
    x0 = Input(shape=(10, 10, 3))
    x1 = Input(shape=(10, 10, 3))
    x2 = Input(shape=(10, 10, 3))

    y0 = merge.Add()([x0, x1])
    y1 = image_preprocessing.CenterCrop(8, 8)(x2)
    y1 = convolutional.ZeroPadding2D(padding=1)(y1)

    z = merge.Add()([y0, y1])
    z = normalization.Normalization()(z)
    z = convolutional.Conv2D(4, 3)(z)

    stage = preprocessing_stage_functional.PreprocessingStageFunctional([x0, x1, x2], z)

    data = [np.ones((12, 10, 10, 3), dtype='float32'),
            np.ones((12, 10, 10, 3), dtype='float32'),
            np.ones((12, 10, 10, 3), dtype='float32')]

    stage.adapt(data)
    _ = stage(data)
    stage.compile('rmsprop', 'mse')
    stage.fit(data, np.ones((12, 8, 8, 4)))

    dataset_x0 = dataset_ops.Dataset.from_tensor_slices(np.ones((12, 10, 10, 3)))
    dataset_x1 = dataset_ops.Dataset.from_tensor_slices(np.ones((12, 10, 10, 3)))
    dataset_x2 = dataset_ops.Dataset.from_tensor_slices(np.ones((12, 10, 10, 3)))
    dataset_x = dataset_ops.Dataset.zip((dataset_x0, dataset_x1, dataset_x2))
    dataset_y = dataset_ops.Dataset.from_tensor_slices(np.ones((12, 8, 8, 4)))
    dataset = dataset_ops.Dataset.zip((dataset_x, dataset_y)).batch(4)
    stage.fit(dataset)
    _ = stage.evaluate(data, np.ones((12, 8, 8, 4)))
    _ = stage.predict(data)


if __name__ == '__main__':
  test.main()
