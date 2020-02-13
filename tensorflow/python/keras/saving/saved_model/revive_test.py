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
"""Tests reviving models from config and SavedModel.

These tests ensure that a model revived from a combination of config and
SavedModel have the expected structure.
"""
# TODO(kathywu): Move relevant tests from saved_model_test to

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class SubclassedModelNoConfig(keras.Model):

  def __init__(self, a, b):
    super(SubclassedModelNoConfig, self).__init__()

    self.a = a
    self.b = b
    self.shared = CustomLayerNoConfig(a, b)
    self.all_layers = [
        self.shared,
        CustomLayerWithConfig(a + 1, b + 2),
        CustomLayerNoConfig(a + 3, b + 4),
        keras.Sequential([
            # TODO(b/145029112): Bug with losses when there are shared layers.
            # self.shared,  <-- Enable when bug is fixed.
            CustomLayerNoConfig(a + 5, b + 6)
        ])]

  def call(self, inputs):
    x = inputs
    for layer in self.all_layers:
      x = layer(x)
    return x


class SubclassedModelWithConfig(SubclassedModelNoConfig):

  def get_config(self):
    return {'a': self.a,
            'b': self.b}

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class CustomLayerNoConfig(keras.layers.Layer):

  def __init__(self, a, b, name=None):
    super(CustomLayerNoConfig, self).__init__(name=name)
    self.a = variables.Variable(a, name='a')
    self.b = b
    def a_regularizer():
      return self.a * 2
    self.add_loss(a_regularizer)

  def build(self, input_shape):
    self.c = variables.Variable(
        constant_op.constant(1.0, shape=input_shape[1:]), name=self.name+'_c')

  def call(self, inputs):
    self.add_loss(math_ops.reduce_sum(inputs), inputs)
    return inputs + self.c


class CustomLayerWithConfig(CustomLayerNoConfig):

  def get_config(self):
    return {'a': backend.get_value(self.a),
            'b': self.b,
            'name': self.name}


class TestModelRevive(keras_parameterized.TestCase):

  def setUp(self):
    super(TestModelRevive, self).setUp()
    self.path = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, self.path, ignore_errors=True)

  def _save_model_dir(self, dirname='saved_model'):
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    return os.path.join(temp_dir, dirname)

  def _assert_revived_correctness(self, model, revived):
    self.assertAllEqual(model.input_names, revived.input_names)
    self.assertAllEqual(model.output_names, revived.output_names)
    self.assertTrue(all([
        i.shape.as_list() == r.shape.as_list() and i.dtype == r.dtype
        for (i, r) in zip(model.inputs, revived.inputs)]))
    self.assertTrue(all([
        i.shape.as_list() == r.shape.as_list() and i.dtype == r.dtype
        for (i, r) in zip(model.outputs, revived.outputs)]))

    self.assertAllClose(self.evaluate(model.weights),
                        self.evaluate(revived.weights))
    input_arr = constant_op.constant(
        np.random.random((2, 2, 3)).astype(np.float32))

    self.assertAllClose(model(input_arr), revived(input_arr))
    self.assertAllClose(sum(model.losses), sum(revived.losses))
    self.assertAllClose(len(model.losses), len(revived.losses))
    model_layers = {layer.name: layer for layer in model.layers}
    revived_layers = {layer.name: layer for layer in revived.layers}
    self.assertAllEqual(model_layers.keys(), revived_layers.keys())

    for name in model_layers:
      model_layer = model_layers[name]
      revived_layer = revived_layers[name]
      self.assertEqual(model_layer.name, revived_layer.name)
      self.assertEqual(model_layer.dtype, revived_layer.dtype)
      self.assertEqual(model_layer.trainable, revived_layer.trainable)
      if 'WithConfig' in type(model_layer).__name__:
        self.assertEqual(type(model_layer), type(revived_layer))
      else:
        # When loading layers from SavedModel, a new class is dynamically
        # created with the same name.
        self.assertEqual(type(model_layer).__name__,
                         type(revived_layer).__name__)

  @keras_parameterized.run_with_all_model_types
  def test_revive(self):
    input_shape = None
    if testing_utils.get_model_type() == 'functional':
      input_shape = (2, 3)

    layer_with_config = CustomLayerWithConfig(1., 2)
    layer_without_config = CustomLayerNoConfig(3., 4)
    subclassed_with_config = SubclassedModelWithConfig(4., 6.)
    subclassed_without_config = SubclassedModelNoConfig(7., 8.)

    inputs = keras.Input((2, 3))
    x = CustomLayerWithConfig(1., 2)(inputs)
    x = CustomLayerNoConfig(3., 4)(x)
    x = SubclassedModelWithConfig(4., 6.)(x)
    x = SubclassedModelNoConfig(7., 8.)(x)
    inner_model_functional = keras.Model(inputs, x)

    inner_model_sequential = keras.Sequential(
        [CustomLayerWithConfig(1., 2),
         CustomLayerNoConfig(3., 4),
         SubclassedModelWithConfig(4., 6.),
         SubclassedModelNoConfig(7., 8.)])

    class SubclassedModel(keras.Model):

      def __init__(self):
        super(SubclassedModel, self).__init__()
        self.all_layers = [CustomLayerWithConfig(1., 2),
                           CustomLayerNoConfig(3., 4),
                           SubclassedModelWithConfig(4., 6.),
                           SubclassedModelNoConfig(7., 8.)]

      def call(self, inputs):
        x = inputs
        for layer in self.all_layers:
          x = layer(x)
        return x

    inner_model_subclassed = SubclassedModel()

    layers = [layer_with_config,
              layer_without_config,
              subclassed_with_config,
              subclassed_without_config,
              inner_model_functional,
              inner_model_sequential,
              inner_model_subclassed]
    model = testing_utils.get_model_from_layers(
        layers, input_shape=input_shape)

    # The inputs attribute must be defined in order to save the model.
    if not model.inputs:
      model._set_inputs(tensor_spec.TensorSpec((None, 2, 3)))

    # Test that the correct checkpointed values are loaded, whether the layer is
    # created from the config or SavedModel.
    layer_with_config.c.assign(2 * layer_with_config.c)
    layer_without_config.c.assign(3 * layer_without_config.c)

    model.save(self.path, save_format='tf')
    revived = keras_load.load(self.path)
    self._assert_revived_correctness(model, revived)

  def test_revive_subclassed_with_nested_model(self):
    model = SubclassedModelNoConfig(1., 2.)
    model._set_inputs(tensor_spec.TensorSpec((None, 2, 3)))
    model.save(self.path, save_format='tf')
    revived = keras_load.load(self.path)
    self._assert_revived_correctness(model, revived)


if __name__ == '__main__':
  ops.enable_eager_execution()
  with generic_utils.CustomObjectScope({
      'CustomLayerWithConfig': CustomLayerWithConfig,
      'SubclassedModelWithConfig': SubclassedModelWithConfig}):
    test.main()
