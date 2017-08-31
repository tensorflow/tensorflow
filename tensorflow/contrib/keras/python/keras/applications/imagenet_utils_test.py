# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Inception V3 application."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.python.platform import test


class ImageNetUtilsTest(test.TestCase):

  def test_preprocess_input(self):
    x = np.random.uniform(0, 255, (2, 3, 2, 3))
    self.assertEqual(
        keras.applications.imagenet_utils.preprocess_input(x).shape, x.shape)

    out1 = keras.applications.imagenet_utils.preprocess_input(
        x, 'channels_last')
    out2 = keras.applications.imagenet_utils.preprocess_input(
        np.transpose(x, (0, 3, 1, 2)), 'channels_first')
    self.assertAllClose(out1, out2.transpose(0, 2, 3, 1))

  def test_obtain_input_shape(self):
    # input_shape and default_size are not identical.
    with self.assertRaises(ValueError):
      keras.applications.imagenet_utils._obtain_input_shape(
          input_shape=(224, 224, 3),
          default_size=299,
          min_size=139,
          data_format='channels_last',
          include_top=True)

    # Test invalid use cases
    for data_format in ['channels_last', 'channels_first']:
      # input_shape is smaller than min_size.
      shape = (100, 100)
      if data_format == 'channels_last':
        input_shape = shape + (3,)
      else:
        input_shape = (3,) + shape
      with self.assertRaises(ValueError):
        keras.applications.imagenet_utils._obtain_input_shape(
            input_shape=input_shape,
            default_size=None,
            min_size=139,
            data_format=data_format,
            include_top=False)

      # shape is 1D.
      shape = (100,)
      if data_format == 'channels_last':
        input_shape = shape + (3,)
      else:
        input_shape = (3,) + shape
      with self.assertRaises(ValueError):
        keras.applications.imagenet_utils._obtain_input_shape(
            input_shape=input_shape,
            default_size=None,
            min_size=139,
            data_format=data_format,
            include_top=False)

      # the number of channels is 5 not 3.
      shape = (100, 100)
      if data_format == 'channels_last':
        input_shape = shape + (5,)
      else:
        input_shape = (5,) + shape
      with self.assertRaises(ValueError):
        keras.applications.imagenet_utils._obtain_input_shape(
            input_shape=input_shape,
            default_size=None,
            min_size=139,
            data_format=data_format,
            include_top=False)

    assert keras.applications.imagenet_utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_last',
        include_top=False) == (None, None, 3)

    assert keras.applications.imagenet_utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_first',
        include_top=False) == (3, None, None)

    assert keras.applications.imagenet_utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_last',
        include_top=False) == (None, None, 3)

    assert keras.applications.imagenet_utils._obtain_input_shape(
        input_shape=(150, 150, 3),
        default_size=None,
        min_size=139,
        data_format='channels_last',
        include_top=False) == (150, 150, 3)

    assert keras.applications.imagenet_utils._obtain_input_shape(
        input_shape=(3, None, None),
        default_size=None,
        min_size=139,
        data_format='channels_first',
        include_top=False) == (3, None, None)


if __name__ == '__main__':
  test.main()
