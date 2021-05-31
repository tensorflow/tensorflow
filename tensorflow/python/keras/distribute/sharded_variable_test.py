# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for ClusterCoordinator and Keras models."""
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.framework import constant_op
from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


class ShardedVariableTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        multi_worker_testing_utils.make_parameter_server_cluster(3, 2),
        variable_partitioner=sharded_variable.FixedShardsPartitioner(2))

  def assert_list_all_equal(self, list1, list2):
    """Used in lieu of `assertAllEqual`.

    This is used to replace standard `assertAllEqual` for the cases where
    `list1` and `list2` contain `AggregatingVariable`. Lists with
    `AggregatingVariable` are not convertible to numpy array via `np.array`
    calls as numpy would raise `ValueError: setting an array element with a
    sequence.`

    Args:
      list1: The first list to compare equality.
      list2: The second list to compare equality.
    """
    for lhs, rhs in zip(list1, list2):
      self.assertEqual(lhs, rhs)

  def test_keras_layer_setattr(self):

    class Layer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        self.w = variables_lib.Variable([0, 1])
        self.b = variables_lib.Variable([2, 3], trainable=False)

    with self.strategy.scope():
      layer = Layer()

    self.assertLen(layer.trainable_weights, 2)
    self.assertEqual(layer.trainable_weights[0], [0])
    self.assertEqual(layer.trainable_weights[1], [1])
    self.assertLen(layer.non_trainable_weights, 2)
    self.assertEqual(layer.non_trainable_weights[0], [2])
    self.assertEqual(layer.non_trainable_weights[1], [3])
    self.assert_list_all_equal(
        layer.weights, layer.trainable_weights + layer.non_trainable_weights)
    self.assert_list_all_equal(layer.trainable_weights,
                               layer.trainable_variables)
    self.assert_list_all_equal(layer.weights, layer.variables)

    checkpoint_deps = set(dep.ref for dep in layer._checkpoint_dependencies)
    self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))

  def test_keras_layer_add_weight(self):

    class Layer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        self.w = self.add_weight(
            shape=(2,),
            initializer=lambda shape, dtype: constant_op.constant([0., 1.],),
            trainable=True)
        self.b = self.add_weight(
            shape=(2,),
            initializer=lambda shape, dtype: constant_op.constant([2., 3.]),
            trainable=False)

    with self.strategy.scope():
      layer = Layer()

    self.assertLen(layer.trainable_weights, 2)
    self.assertEqual(layer.trainable_weights[0], [0.])
    self.assertEqual(layer.trainable_weights[1], [1.])
    self.assertLen(layer.non_trainable_weights, 2)
    self.assertEqual(layer.non_trainable_weights[0], [2.])
    self.assertEqual(layer.non_trainable_weights[1], [3.])
    self.assert_list_all_equal(
        layer.weights, layer.trainable_weights + layer.non_trainable_weights)
    self.assert_list_all_equal(layer.trainable_weights,
                               layer.trainable_variables)
    self.assert_list_all_equal(layer.weights, layer.variables)

    checkpoint_deps = set(dep.ref for dep in layer._checkpoint_dependencies)
    self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
