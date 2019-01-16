# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras Layer utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.platform import test


class ConvertDataFormatTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(['channels_first', 'channels_last'])
  def test_convert_format(self, data_format):
    def get_model(shape, data_format):
      model = keras.Sequential()
      model.add(
          keras.layers.Conv2D(filters=2, kernel_size=(4, 3),
                              input_shape=shape, data_format=data_format))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(5))
      return model

    if data_format == 'channels_first':
      shape = (3, 5, 5)
      target_shape = (5, 5, 3)
      prev_shape = (2, 3, 2)
      flip = lambda x: np.flip(np.flip(x, axis=2), axis=3)
      transpose = lambda x: np.transpose(x, (0, 2, 3, 1))
      target_data_format = 'channels_last'
    elif data_format == 'channels_last':
      shape = (5, 5, 3)
      target_shape = (3, 5, 5)
      prev_shape = (2, 2, 3)
      flip = lambda x: np.flip(np.flip(x, axis=1), axis=2)
      transpose = lambda x: np.transpose(x, (0, 3, 1, 2))
      target_data_format = 'channels_first'

      model1 = get_model(shape, data_format)
      model2 = get_model(target_shape, target_data_format)
      conv = keras.backend.function([model1.input], [model1.layers[0].output])

      x = np.random.random((1,) + shape)

      convout1 = conv([x])[0]
      layer_utils.convert_all_kernels_in_model(model1)
      convout2 = flip(conv([flip(x)])[0])

      self.assertAllClose(convout1, convout2)

      out1 = model1.predict(x)
      layer_utils.convert_dense_weights_data_format(
          model1.layers[2], prev_shape, target_data_format)
      for (src, dst) in zip(model1.layers, model2.layers):
        dst.set_weights(src.get_weights())
      out2 = model2.predict(transpose(x))

      self.assertAllClose(out1, out2)


if __name__ == '__main__':
  test.main()
