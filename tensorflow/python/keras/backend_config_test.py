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
"""Tests for backend_config."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend
from tensorflow.python.keras import backend_config
from tensorflow.python.keras import combinations
from tensorflow.python.platform import test


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class BackendConfigTest(test.TestCase):

  def test_backend(self):
    self.assertEqual(backend.backend(), 'tensorflow')

  def test_epsilon(self):
    epsilon = 1e-2
    backend_config.set_epsilon(epsilon)
    self.assertEqual(backend_config.epsilon(), epsilon)
    backend_config.set_epsilon(1e-7)
    self.assertEqual(backend_config.epsilon(), 1e-7)

  def test_floatx(self):
    floatx = 'float64'
    backend_config.set_floatx(floatx)
    self.assertEqual(backend_config.floatx(), floatx)
    backend_config.set_floatx('float32')
    self.assertEqual(backend_config.floatx(), 'float32')

  def test_image_data_format(self):
    image_data_format = 'channels_first'
    backend_config.set_image_data_format(image_data_format)
    self.assertEqual(backend_config.image_data_format(), image_data_format)
    backend_config.set_image_data_format('channels_last')
    self.assertEqual(backend_config.image_data_format(), 'channels_last')


if __name__ == '__main__':
  test.main()
