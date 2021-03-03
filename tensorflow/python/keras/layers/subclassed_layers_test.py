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
"""Tests for Keras subclassed layers utilizing desired user syntax."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
@keras_parameterized.run_with_all_model_types
class SubclassedLayersTest(keras_parameterized.TestCase):

  def test_simple_build_with_constant(self):

    class BuildConstantLayer(keras.layers.Layer):

      def build(self, input_shape):
        self.b = ops.convert_to_tensor_v2_with_dispatch(2.0)

      def call(self, inputs):
        return self.b * inputs

    layer = BuildConstantLayer()
    model = testing_utils.get_model_from_layers(
        [layer, keras.layers.Dense(1)], input_shape=(1,))

    x = ops.convert_to_tensor_v2_with_dispatch([[3.0]])
    self.assertEqual(
        tf_utils.is_symbolic_tensor(model(x)), not context.executing_eagerly())
    self.assertEqual(
        tf_utils.is_symbolic_tensor(layer(x)), not context.executing_eagerly())
    self.assertAllClose(keras.backend.get_value(layer(x)), [[6.0]])

  def test_build_with_derived_constant(self):

    class BuildDerivedConstantLayer(keras.layers.Layer):

      def build(self, input_shape):
        a = ops.convert_to_tensor_v2_with_dispatch(1.0)
        b = 2.0 * a
        self.variable = variables.Variable(b)
        self.constant = ops.convert_to_tensor_v2_with_dispatch(self.variable)

      def call(self, inputs):
        return self.variable * self.constant * inputs

    layer = BuildDerivedConstantLayer()
    model = testing_utils.get_model_from_layers(
        [layer, keras.layers.Dense(1)], input_shape=(1,))

    x = ops.convert_to_tensor_v2_with_dispatch([[3.0]])
    self.assertEqual(
        tf_utils.is_symbolic_tensor(model(x)), not context.executing_eagerly())
    self.assertEqual(
        tf_utils.is_symbolic_tensor(layer(x)), not context.executing_eagerly())
    self.assertAllClose(keras.backend.get_value(layer(x)), [[12.0]])


if __name__ == '__main__':
  test.main()
