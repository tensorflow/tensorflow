# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for layer_parser module."""

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.platform import test

from tensorflow.tools.model_summary_cli import layer_parser


class GetLayerInfoTest(test.TestCase):

  def test_dense_layer_info(self):
    layer = layers.Dense(10, input_shape=(5,), name='test_dense')
    layer.build((None, 5))
    info = layer_parser.get_layer_info(layer)

    self.assertEqual(info['name'], 'test_dense')
    self.assertEqual(info['class_name'], 'Dense')
    self.assertEqual(info['trainable'], True)
    self.assertEqual(info['params'], 60)  # 5*10 weights + 10 biases
    self.assertIsNone(info['params_note'])

  def test_unbuilt_layer_info(self):
    layer = layers.Dense(10, name='unbuilt')
    info = layer_parser.get_layer_info(layer)

    self.assertEqual(info['params'], 0)
    self.assertEqual(info['params_note'], 'unused')


class GetModelInfoTest(test.TestCase):

  def test_sequential_model_info(self):
    model = models.Sequential([
        layers.Dense(10, input_shape=(5,), name='dense_1'),
        layers.Dense(1, name='dense_2')
    ], name='test_model')

    info = layer_parser.get_model_info(model)

    self.assertEqual(info['name'], 'test_model')
    self.assertEqual(info['class_name'], 'Sequential')
    self.assertTrue(info['is_sequential'])
    self.assertEqual(len(info['layers']), 2)
    self.assertEqual(info['total_params'], 71)  # (5*10+10) + (10*1+1)

  def test_functional_model_info(self):
    inputs = layers.Input(shape=(5,), name='input')
    x = layers.Dense(10, name='dense')(inputs)
    outputs = layers.Dense(1, name='output')(x)
    model = models.Model(inputs, outputs, name='functional_model')

    info = layer_parser.get_model_info(model)

    self.assertEqual(info['name'], 'functional_model')
    self.assertEqual(info['class_name'], 'Functional')
    self.assertFalse(info['is_sequential'])
    self.assertTrue(info['is_graph_network'])


if __name__ == '__main__':
  test.main()
