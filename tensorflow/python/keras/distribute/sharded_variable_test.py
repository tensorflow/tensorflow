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
"""Tests for ShardedVariable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


class ShardedVariableTest(test.TestCase):

  def test_keras_layer_setattr(self):

    class Layer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        variables1 = [
            variables_lib.Variable([0]),
            variables_lib.Variable([1]),
        ]
        variables2 = [
            variables_lib.Variable([2], trainable=False),
            variables_lib.Variable([3], trainable=False),
        ]
        self.w = sharded_variable.ShardedVariable(variables1)
        self.b = sharded_variable.ShardedVariable(variables2)

    layer = Layer()

    self.assertLen(layer.trainable_weights, 2)
    self.assertEqual(layer.trainable_weights[0], [0])
    self.assertEqual(layer.trainable_weights[1], [1])
    self.assertLen(layer.non_trainable_weights, 2)
    self.assertEqual(layer.non_trainable_weights[0], [2])
    self.assertEqual(layer.non_trainable_weights[1], [3])
    self.assertAllEqual(layer.weights,
                        layer.trainable_weights + layer.non_trainable_weights)
    self.assertAllEqual(layer.trainable_weights, layer.trainable_variables)
    self.assertAllEqual(layer.weights, layer.variables)

    checkpoint_deps = set(dep.ref for dep in layer._checkpoint_dependencies)
    self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))

  def test_keras_layer_add_weight(self):

    class Layer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        self.w = self.add_weight(
            shape=(2,), initializer=lambda shape, dtype: [0, 1], trainable=True)
        self.b = self.add_weight(
            shape=(2,),
            initializer=lambda shape, dtype: [2, 3],
            trainable=False)

    def sharded_variable_creator(next_creator, **kwargs):
      v1_value = kwargs['initial_value']()[0:1]
      v2_value = kwargs['initial_value']()[1:]

      kwargs['initial_value'] = v1_value
      kwargs['shape'] = (1,)
      v1 = next_creator(**kwargs)

      kwargs['initial_value'] = v2_value
      kwargs['shape'] = (1,)
      v2 = next_creator(**kwargs)

      return sharded_variable.ShardedVariable([v1, v2])

    with variable_scope.variable_creator_scope(sharded_variable_creator):
      layer = Layer()

    self.assertLen(layer.trainable_weights, 2)
    self.assertEqual(layer.trainable_weights[0], [0])
    self.assertEqual(layer.trainable_weights[1], [1])
    self.assertLen(layer.non_trainable_weights, 2)
    self.assertEqual(layer.non_trainable_weights[0], [2])
    self.assertEqual(layer.non_trainable_weights[1], [3])
    self.assertAllEqual(layer.weights,
                        layer.trainable_weights + layer.non_trainable_weights)
    self.assertAllEqual(layer.trainable_weights, layer.trainable_variables)
    self.assertAllEqual(layer.weights, layer.variables)

    checkpoint_deps = set(dep.ref for dep in layer._checkpoint_dependencies)
    self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
