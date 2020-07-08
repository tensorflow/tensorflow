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
"""Tests the get_layer_policy function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.mixed_precision.experimental import get_layer_policy
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.platform import test


class GetLayerPolicyTest(test.TestCase):

  def test_get_layer_policy(self):
    layer = core.Dense(4)
    self.assertEqual(get_layer_policy.get_layer_policy(layer).name, 'float32')

    p = policy.Policy('mixed_float16')
    layer = core.Dense(4, dtype=p)
    self.assertIs(get_layer_policy.get_layer_policy(layer), p)

    layer = core.Dense(4, dtype='float64')
    self.assertEqual(get_layer_policy.get_layer_policy(layer).name, 'float64')

  def test_error(self):
    with self.assertRaisesRegex(
        ValueError, 'get_policy can only be called on a layer, but got: 1'):
      get_layer_policy.get_layer_policy(1)


if __name__ == '__main__':
  base_layer_utils.enable_v2_dtype_behavior()
  test.main()
