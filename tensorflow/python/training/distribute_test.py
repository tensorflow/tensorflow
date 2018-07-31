# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test DistributionStrategy, TowerContext, and supporting APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.training import distribute


class _TestTowerContext(distribute.TowerContext):

  def merge_call(self, fn, *args, **kwargs):
    return kwargs["test_arg"]


def _get_test_variable(name, synchronization, aggregation):
  return {
      "name": name,
      "synchronization": synchronization,
      "aggregation": aggregation
  }


class _TestStrategy(distribute.DistributionStrategy):

  def _call_for_each_tower(self, fn, *args, **kwargs):
    with _TestTowerContext(self, tower_id=0):
      return fn(*args, **kwargs)

  def _create_variable(self, next_creator, *args, **kwargs):
    return _get_test_variable(kwargs["name"], kwargs["synchronization"],
                              kwargs["aggregation"])


def _assert_in_default_state(t):
  t.assertIs(distribute._default_tower_context,
             distribute.get_tower_context())
  t.assertIs(None, distribute.get_cross_tower_context())
  t.assertIs(distribute._default_distribution_strategy,
             distribute.get_distribution_strategy())
  t.assertFalse(distribute.has_distribution_strategy())


class TestStrategyTest(test.TestCase):

  def testCallForEachTower(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()

    def run_fn():
      tower_context = distribute.get_tower_context()
      self.assertTrue(tower_context is not None)
      self.assertIs(None, distribute.get_cross_tower_context())
      self.assertTrue(distribute.has_distribution_strategy())
      self.assertIs(dist, distribute.get_distribution_strategy())
      self.assertEqual("foo", tower_context.merge_call(None, test_arg="foo"))
      expected_value = _get_test_variable(
          "bar", variable_scope.VariableSynchronization.AUTO,
          variable_scope.VariableAggregation.NONE)
      self.assertDictEqual(expected_value,
                           variable_scope.variable(1.0, name="bar"))

    with self.assertRaises(RuntimeError):
      dist.call_for_each_tower(run_fn)
    with dist.scope():
      dist.call_for_each_tower(run_fn)
    _assert_in_default_state(self)

  def testScope(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()
    with dist.scope():
      self.assertIs(None, distribute.get_tower_context())
      self.assertIs(dist, distribute.get_cross_tower_context())
      self.assertTrue(distribute.has_distribution_strategy())
      self.assertIs(dist, distribute.get_distribution_strategy())
      expected_value = _get_test_variable(
          "baz", variable_scope.VariableSynchronization.AUTO,
          variable_scope.VariableAggregation.NONE)
      self.assertDictEqual(expected_value,
                           variable_scope.variable(1.0, name="baz"))
    _assert_in_default_state(self)

  def testSettingSynchronizationAndAggregation(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()
    with dist.scope():
      expected_value = _get_test_variable(
          "baz", variable_scope.VariableSynchronization.ON_WRITE,
          variable_scope.VariableAggregation.MEAN)
      self.assertDictEqual(
          expected_value,
          variable_scope.variable(
              1.0,
              name="baz",
              synchronization=variable_scope.VariableSynchronization.ON_WRITE,
              aggregation=variable_scope.VariableAggregation.MEAN))
    _assert_in_default_state(self)


class DefaultDistributionStrategyTest(test.TestCase):

  def testMergeCall(self):
    _assert_in_default_state(self)

    def merge_fn(dist, s):
      self.assertIs(distribute._default_distribution_strategy, dist)
      self.assertIs(None, distribute.get_tower_context())
      self.assertIs(dist, distribute.get_cross_tower_context())
      self.assertIs(dist, distribute.get_distribution_strategy())
      self.assertFalse(distribute.has_distribution_strategy())
      return "foo_" + s

    tower_ctx = distribute.get_tower_context()
    self.assertIs(distribute._default_tower_context, tower_ctx)
    self.assertEqual("foo_bar", tower_ctx.merge_call(merge_fn, "bar"))
    _assert_in_default_state(self)


if __name__ == "__main__":
  test.main()
