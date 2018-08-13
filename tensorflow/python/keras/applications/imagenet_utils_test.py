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

from tensorflow.python import keras
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.platform import test


class ImageNetUtilsTest(test.TestCase):

  def test_preprocess_input(self):
    # Test batch of images
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    self.assertEqual(preprocess_input(x).shape, x.shape)
    out1 = preprocess_input(x, 'channels_last')
    out2 = preprocess_input(np.transpose(x, (0, 3, 1, 2)), 'channels_first')
    self.assertAllClose(out1, out2.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    self.assertEqual(preprocess_input(x).shape, x.shape)
    out1 = preprocess_input(x, 'channels_last')
    out2 = preprocess_input(np.transpose(x, (2, 0, 1)), 'channels_first')
    self.assertAllClose(out1, out2.transpose(1, 2, 0))

  def test_preprocess_input_symbolic(self):
    # Test image batch
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    inputs = keras.layers.Input(shape=x.shape[1:])
    outputs = keras.layers.Lambda(
        preprocess_input, output_shape=x.shape[1:])(inputs)
    model = keras.models.Model(inputs, outputs)
    assert model.predict(x).shape == x.shape
    # pylint: disable=g-long-lambda
    outputs1 = keras.layers.Lambda(lambda x:
                                   preprocess_input(x, 'channels_last'),
                                   output_shape=x.shape[1:])(inputs)
    model1 = keras.models.Model(inputs, outputs1)
    out1 = model1.predict(x)
    x2 = np.transpose(x, (0, 3, 1, 2))
    inputs2 = keras.layers.Input(shape=x2.shape[1:])
    # pylint: disable=g-long-lambda
    outputs2 = keras.layers.Lambda(lambda x:
                                   preprocess_input(x, 'channels_first'),
                                   output_shape=x2.shape[1:])(inputs2)
    model2 = keras.models.Model(inputs2, outputs2)
    out2 = model2.predict(x2)
    self.assertAllClose(out1, out2.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    inputs = keras.layers.Input(shape=x.shape)
    outputs = keras.layers.Lambda(preprocess_input,
                                  output_shape=x.shape)(inputs)
    model = keras.models.Model(inputs, outputs)
    assert model.predict(x[np.newaxis])[0].shape == x.shape
    # pylint: disable=g-long-lambda
    outputs1 = keras.layers.Lambda(lambda x:
                                   preprocess_input(x, 'channels_last'),
                                   output_shape=x.shape)(inputs)
    model1 = keras.models.Model(inputs, outputs1)
    out1 = model1.predict(x[np.newaxis])[0]
    x2 = np.transpose(x, (2, 0, 1))
    inputs2 = keras.layers.Input(shape=x2.shape)
    outputs2 = keras.layers.Lambda(lambda x:
                                   preprocess_input(x, 'channels_first'),
                                   output_shape=x2.shape)(inputs2)  # pylint: disable=g-long-lambda
    model2 = keras.models.Model(inputs2, outputs2)
    out2 = model2.predict(x2[np.newaxis])[0]
    self.assertAllClose(out1, out2.transpose(1, 2, 0))


if __name__ == '__main__':
  test.main()
