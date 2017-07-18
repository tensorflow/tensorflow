# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for noise layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import testing_utils
from tensorflow.python.platform import test


class NoiseLayersTest(test.TestCase):

  def test_GaussianNoise(self):
    with self.test_session():
      testing_utils.layer_test(
          keras.layers.GaussianNoise,
          kwargs={'stddev': 1.},
          input_shape=(3, 2, 3))

  def test_GaussianDropout(self):
    with self.test_session():
      testing_utils.layer_test(
          keras.layers.GaussianDropout,
          kwargs={'rate': 0.5},
          input_shape=(3, 2, 3))

  def test_AlphaDropout(self):
    with self.test_session():
      testing_utils.layer_test(
          keras.layers.AlphaDropout,
          kwargs={'rate': 0.2},
          input_shape=(3, 2, 3))


if __name__ == '__main__':
  test.main()
