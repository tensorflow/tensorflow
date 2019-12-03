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
# pylint: disable=protected-access
"""Unit tests for serializing Keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class CustomLayer(keras.layers.Layer):

  def __init__(self, unused_a):
    super(CustomLayer, self).__init__()


class ModelSerializationTest(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types(exclude_models=['subclass'])
  def test_model_config_always_saved(self):
    layer = CustomLayer(None)
    with self.assertRaisesRegexp(NotImplementedError,
                                 'must override `get_config`.'):
      layer.get_config()
    model = testing_utils.get_model_from_layers([layer], input_shape=(3,))
    properties = model._trackable_saved_model_saver.python_properties
    self.assertIsNotNone(properties['config'])


if __name__ == '__main__':
  test.main()
