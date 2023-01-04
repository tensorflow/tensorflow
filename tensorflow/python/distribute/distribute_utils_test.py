# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utility functions in distribute_utils."""

import collections
import collections.abc

from absl.testing import parameterized
import wrapt

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model.model_utils import mode_keys


def _nested_value(d):
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


class RegroupAndSelectDeviceTest(test.TestCase, parameterized.TestCase):

  def _is_per_replica(self, result, expected, klass=values.PerReplica):
    self.assertIsInstance(result, klass)
    for i, exp in enumerate(expected):
      self.assertEqual(exp, result.values[i])

  def testNested(self):
    result = distribute_utils.regroup((_nested_value("1"), _nested_value("2")))
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 3)
    self._is_per_replica(result[0], ["a1", "a2"])
    self._is_per_replica(result[2], ["h1", "h2"])

    self.assertIsInstance(result[1], list)
    self.assertLen(result[1], 3)
    self._is_per_replica(result[1][0], ["b1", "b2"])
    self._is_per_replica(result[1][2], ["g1", "g2"])

    self.assertIsInstance(result[1][1], dict)
    self.assertEqual(set(["c", "e"]), set(result[1][1].keys()))
    self._is_per_replica(result[1][1]["c"], ["d1", "d2"])
    self._is_per_replica(result[1][1]["e"], ["f1", "f2"])

    # Also test that we can undo the merge using select_replica()
    self.assertEqual(_nested_value("1"),
                     distribute_utils.select_replica(0, result))
    self.assertEqual(_nested_value("2"),
                     distribute_utils.select_replica(1, result))
    # select_device_mirrored() should fail due to non-mirrored values
    with self.assertRaises(TypeError):
      distribute_utils.select_replica_mirrored(0, result)
    with self.assertRaises(TypeError):
      distribute_utils.select_replica_mirrored(1, result)

  def testRegroupKeepsDictBasedClass(self):
    class DictBasedClass(dict):
      """Dummy class inherited from a dict."""

    result = distribute_utils.regroup(
        (DictBasedClass(a="a1", b="b1"), DictBasedClass(a="a2", b="b2")))
    self.assertIsInstance(result, DictBasedClass)
    self._is_per_replica(result["a"], ["a1", "a2"])
    self._is_per_replica(result["b"], ["b1", "b2"])

  def testRegroupCollectionsMapping(self):

    class CollectionsMappingBasedClass(collections.abc.Mapping):
      """Class inherited from collections.abc.Mapping."""

      def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

      def __getitem__(self, key):
        return self._d.__getitem__(key)

      def __iter__(self):
        return iter(self._d)

      def __len__(self):
        return len(self._d)

    result = distribute_utils.regroup(
        (CollectionsMappingBasedClass(a="a1", b="b1"),
         CollectionsMappingBasedClass(a="a2", b="b2")))
    self.assertIsInstance(result, CollectionsMappingBasedClass)
    self._is_per_replica(result["a"], ["a1", "a2"])
    self._is_per_replica(result["b"], ["b1", "b2"])

  def testWrapClass(self):
    # Normally a mirrored value would be the same across devices, but
    # for a test it is convenient to be able to tell the values apart.
    result = distribute_utils.regroup((_nested_value("1"), _nested_value("2")),
                                      values.Mirrored)
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 3)
    self._is_per_replica(result[0], ["a1", "a2"], values.Mirrored)
    self._is_per_replica(result[2], ["h1", "h2"], values.Mirrored)

    self.assertIsInstance(result[1], list)
    self.assertLen(result[1], 3)
    self._is_per_replica(result[1][0], ["b1", "b2"], values.Mirrored)
    self._is_per_replica(result[1][2], ["g1", "g2"], values.Mirrored)

    self.assertIsInstance(result[1][1], dict)
    self.assertEqual(set(["c", "e"]), set(result[1][1].keys()))
    self._is_per_replica(result[1][1]["c"], ["d1", "d2"], values.Mirrored)
    self._is_per_replica(result[1][1]["e"], ["f1", "f2"], values.Mirrored)

    # Also test that we can undo the merge using select_replica()
    self.assertEqual(_nested_value("1"),
                     distribute_utils.select_replica(0, result))
    self.assertEqual(_nested_value("2"),
                     distribute_utils.select_replica(1, result))
    # Values are marked as mirrored, so select_device_mirrored() is allowed.
    self.assertEqual(_nested_value("1"),
                     distribute_utils.select_replica_mirrored(0, result))
    self.assertEqual(_nested_value("2"),
                     distribute_utils.select_replica_mirrored(1, result))

  def testWrapAListOfTwoTuples(self):
    result = distribute_utils.regroup([("1", "2"), ("3", "4")])
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 2)
    self._is_per_replica(result[0], ("1", "3"), values.PerReplica)
    self._is_per_replica(result[1], ("2", "4"), values.PerReplica)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_one_cpu,
          ],
          mode=["graph", "eager"],
      ))
  def testMirroredContainer(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          1., aggregation=variable_scope.VariableAggregation.SUM)
    self.assertTrue(distribute_utils.is_distributed_variable(v))
    self.assertTrue(distribute_utils.is_distributed_variable(
        distribute_utils.regroup(v.values)))

  def testSameId(self):
    foo = object()
    result = distribute_utils.regroup((("a", foo), ("b", foo)))
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 2)
    self._is_per_replica(result[0], ["a", "b"])
    self.assertIs(foo, result[1])

    # Test select_replica(), should undo the merge done by regroup().
    result_0 = distribute_utils.select_replica(0, result)
    self.assertIsInstance(result_0, tuple)
    self.assertLen(result_0, 2)
    self.assertEqual("a", result_0[0])
    self.assertIs(foo, result_0[1])
    result_1 = distribute_utils.select_replica(1, result)
    self.assertIsInstance(result_1, tuple)
    self.assertLen(result_1, 2)
    self.assertEqual("b", result_1[0])
    self.assertIs(foo, result_1[1])

  def testOneDevice(self):
    result = distribute_utils.regroup((_nested_value("1"),))
    # On one device regroup() and select_replica() are basically identity.
    self.assertEqual(_nested_value("1"), result)
    self.assertEqual(_nested_value("1"),
                     distribute_utils.select_replica(0, result))

  def testNamedTuple(self):

    # We include toy implementations of Scaffold and EstimatorSpec to
    # avoid a dependency on Estimator here.

    class Scaffold(object):
      pass

    class EstimatorSpec(collections.namedtuple(
        "EstimatorSpec", ["mode", "loss", "train_op", "scaffold"])):

      def __new__(cls, mode, loss, train_op, scaffold=None):
        return super(EstimatorSpec, cls).__new__(
            cls, mode=mode, loss=loss, train_op=train_op,
            scaffold=scaffold or Scaffold())

    with context.graph_mode(), ops.Graph().as_default():
      created_estimator_specs = []

      for device_id in range(3):
        spec = EstimatorSpec(
            mode=mode_keys.EstimatorModeKeys.TRAIN,
            loss=constant_op.constant(device_id / 2),
            train_op=array_ops.identity(constant_op.constant(device_id)))
        created_estimator_specs.append(spec)

      merged_estimator_spec = distribute_utils.regroup(created_estimator_specs)

      self.assertIsInstance(merged_estimator_spec, EstimatorSpec)
      self.assertEqual(mode_keys.EstimatorModeKeys.TRAIN,
                       merged_estimator_spec.mode)
      for device_id in range(3):
        self.assertEqual(created_estimator_specs[device_id].loss,
                         merged_estimator_spec.loss.values[device_id])
        self.assertEqual(created_estimator_specs[device_id].train_op,
                         merged_estimator_spec.train_op.values[device_id])
        # Scaffold is populated by `EstimatorSpec.__new__`.
        self.assertEqual(created_estimator_specs[device_id].scaffold,
                         merged_estimator_spec.scaffold.values[device_id])
        self.assertIsInstance(created_estimator_specs[device_id].scaffold,
                              Scaffold)
        # Also test that we can undo the merge using select_replica()
        self.assertEqual(created_estimator_specs[device_id],
                         distribute_utils.select_replica(
                             device_id, merged_estimator_spec))

  def testWrappedNamedTuple(self):
    Point = collections.namedtuple("Point", ["x", "y"])
    point1 = Point(x=0, y=2)
    point2 = Point(x=1, y=3)
    wrapped1 = wrapt.ObjectProxy(point1)
    wrapped2 = wrapt.ObjectProxy(point2)
    result = distribute_utils.regroup([wrapped1, wrapped2])
    self.assertEqual(result.x.values, (0, 1))
    self.assertEqual(result.y.values, (2, 3))

if __name__ == "__main__":
  test.main()
