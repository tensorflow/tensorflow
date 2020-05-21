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
"""Tests for the distributed values library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model.model_utils import mode_keys
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.tracking import util as trackable_utils
from tensorflow.python.types import core
from tensorflow.python.util import nest


class DistributedValuesTest(test.TestCase, parameterized.TestCase):

  def testGetEager(self):
    one = constant_op.constant(1)
    two = constant_op.constant(2)
    v = values.DistributedValues((one, two))
    self.assertEqual(one, v._get())
    with distribute_lib.ReplicaContext(None, 1):
      self.assertEqual(two, v._get())

  def testGetGraph(self):
    with context.graph_mode(), ops.Graph().as_default():
      one = constant_op.constant(1)
      two = constant_op.constant(2)
      v = values.DistributedValues((one, two))
      self.assertEqual(one, v._get())
      with distribute_lib.ReplicaContext(None, 1):
        self.assertEqual(two, v._get())

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueFromTensor(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    single_value = constant_op.constant(1)
    def value_fn(ctx):
      del ctx
      return single_value

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    self.assertAllEqual(
        distribution.experimental_local_results(distributed_values),
        constant_op.constant(1., shape=(distribution.num_replicas_in_sync)))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueSingleNumpyArrayConstant(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    array_value = np.array([1., 2., 3.])
    def value_fn(ctx):
      del ctx
      return array_value

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    local_results = distribution.experimental_local_results(distributed_values)
    self.assertLen(local_results, distribution.num_replicas_in_sync)
    for result in local_results:
      self.assertAllEqual(result, [1., 2., 3.])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueTupleConstant(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    tuple_value = (1., 2., 3.)
    def value_fn(ctx):
      del ctx
      return tuple_value
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    local_results = distribution.experimental_local_results(distributed_values)
    for result in local_results:
      self.assertAllEqual(result, (1., 2., 3.))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueNestedStructurePerReplica(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    tuple_value = (1., 2., 3.)
    def value_fn(ctx):
      per_replica = []
      for val in tuple_value:
        per_replica.append(val * ctx.replica_id_in_sync_group)
      return per_replica
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    for i in range(distribution.num_replicas_in_sync):
      self.assertAllEqual(
          values.select_replica(i, distributed_values),
          (1. * i, 2. * i, 3. * i))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueSpareTensor(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    def value_fn(ctx):
      del ctx
      return sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    local_results = distribution.experimental_local_results(distributed_values)
    for i in range(distribution.num_replicas_in_sync):
      self.assertAllEqual(
          sparse_ops.sparse_tensor_to_dense(local_results[i]),
          [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueExtractFromArray(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    multiple_values = range(distribution.num_replicas_in_sync)
    def value_fn(ctx):
      return multiple_values[ctx.replica_id_in_sync_group]
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    local_results = distribution.experimental_local_results(distributed_values)
    for i in range(distribution.num_replicas_in_sync):
      self.assertAllEqual(local_results[i], i)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueAndRun(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    @def_function.function
    def run():
      multiple_values = range(distribution.num_replicas_in_sync)
      def value_fn(ctx):
        return multiple_values[ctx.replica_id_in_sync_group]
      distributed_values = (
          distribution.experimental_distribute_values_from_function(value_fn))

      def computation(x):
        return math_ops.square(x)

      outputs = distribution.experimental_local_results(
          distribution.run(computation,
                           args=(distributed_values,)))
      return outputs

    local_results = run()

    for i in range(distribution.num_replicas_in_sync):
      self.assertAllEqual(local_results[i], i**2)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              # TODO(b/137795644): support CentralStroageStrategy
              # strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["eager"]
      ))
  def testMakeDistributedValueDefaultDevicePlacement(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    multiple_values = []
    for i in range(distribution.num_replicas_in_sync):
      multiple_values.append(constant_op.constant(1.0))

    def value_fn(ctx):
      return multiple_values[ctx.replica_id_in_sync_group]
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    for i in range(distribution.num_replicas_in_sync):
      self.assertAllEqual(distributed_values._values[i].device,
                          "/job:localhost/replica:0/task:0/device:CPU:0")

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              # TODO(b/137795644): support CentralStroageStrategy
              # strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["eager"]
      ))
  def testMakeDistributedValueExplicitDevicePlacement(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    worker_devices = distribution.extended.worker_devices
    multiple_values = []
    for i in range(distribution.num_replicas_in_sync):
      with ops.device(worker_devices[i]):
        multiple_values.append(array_ops.identity(1.0))

    def value_fn(ctx):
      return multiple_values[ctx.replica_id_in_sync_group]
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    for i in range(distribution.num_replicas_in_sync):
      self.assertAllEqual(distributed_values._values[i].device,
                          worker_devices[i])


class DistributedDelegateTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testGetAttr(self):
    class Foo(object):

      def __init__(self, x):
        self.x = x

    v = values.DistributedDelegate((Foo(7), Foo(8)))
    self.assertEqual(7, v.x)
    with self.assertRaises(AttributeError):
      _ = v.y

  @test_util.run_in_graph_and_eager_modes
  def testOperatorOverride(self):
    v = values.DistributedDelegate((7, 8))
    # v should act like int(7).
    self.assertEqual(8, v + 1)
    self.assertEqual(10, 3 + v)
    self.assertEqual(14, v + v)
    self.assertEqual(5, v - 2)
    self.assertEqual(6, 13 - v)
    self.assertEqual(0, v - v)
    self.assertEqual(14, v * 2)
    self.assertEqual(21, 3 * v)
    self.assertEqual(49, v * v)
    self.assertEqual(3.5, v / 2)
    self.assertEqual(1.5, 10.5 / v)
    self.assertEqual(3, v // 2)
    self.assertEqual(2, 15 // v)
    self.assertEqual(1, v % 2)
    self.assertEqual(2, 16 % v)
    # pylint: disable=g-generic-assert
    self.assertTrue(v < 12)
    self.assertTrue(v <= 12)
    self.assertFalse(v > 12)
    self.assertFalse(v >= 12)
    self.assertFalse(12 < v)
    self.assertFalse(12 <= v)
    self.assertTrue(12 > v)
    self.assertTrue(12 >= v)
    # pylint: enable=g-generic-assert
    self.assertEqual(3, v & 3)
    self.assertEqual(3, 11 & v)
    self.assertEqual(15, v | 8)
    self.assertEqual(23, 16 | v)
    self.assertEqual(4, v ^ 3)
    self.assertEqual(12, 11 ^ v)
    self.assertEqual(343, pow(v, 3))
    self.assertEqual(3, pow(v, 3, 10))
    self.assertEqual(128, pow(2, v))
    self.assertEqual(-7, -v)
    self.assertEqual(~7, ~v)
    self.assertEqual(7, abs(v))
    with self.assertRaises(TypeError):
      _ = v[2]


def _device_str(d):
  return "/device:GPU:" + str(d)


def _nested_value(d):
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


def _make_mirrored_val(init_val=5.0):
  v = []
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, _ in zip(devices, ["v", "v/replica"]):
    with ops.device(d):
      v.append(constant_op.constant(init_val))
  return values.Mirrored(v)


def _make_mirrored():
  v = []
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))
  mirrored = values.MirroredVariable(
      None, v, variable_scope.VariableAggregation.SUM)
  return mirrored


class RegroupAndSelectDeviceTest(test.TestCase, parameterized.TestCase):

  def _is_per_replica(self, result, expected, klass=values.PerReplica):
    self.assertIsInstance(result, klass)
    for i, exp in enumerate(expected):
      self.assertEqual(exp, result.values[i])

  def testNested(self):
    result = values.regroup((_nested_value("1"), _nested_value("2")))
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
                     values.select_replica(0, result))
    self.assertEqual(_nested_value("2"),
                     values.select_replica(1, result))
    # select_device_mirrored() should fail due to non-mirrored values
    with self.assertRaises(TypeError):
      values.select_replica_mirrored(0, result)
    with self.assertRaises(TypeError):
      values.select_replica_mirrored(1, result)

  def testRegroupKeepsDictBasedClass(self):
    class DictBasedClass(dict):
      """Dummy class inherited from a dict."""

    result = values.regroup(
        (DictBasedClass(a="a1", b="b1"), DictBasedClass(a="a2", b="b2")))
    self.assertIsInstance(result, DictBasedClass)
    self._is_per_replica(result["a"], ["a1", "a2"])
    self._is_per_replica(result["b"], ["b1", "b2"])

  def testWrapClass(self):
    # Normally a mirrored value would be the same across devices, but
    # for a test it is convenient to be able to tell the values apart.
    result = values.regroup((_nested_value("1"), _nested_value("2")),
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
                     values.select_replica(0, result))
    self.assertEqual(_nested_value("2"),
                     values.select_replica(1, result))
    # Values are marked as mirrored, so select_device_mirrored() is allowed.
    self.assertEqual(_nested_value("1"),
                     values.select_replica_mirrored(0, result))
    self.assertEqual(_nested_value("2"),
                     values.select_replica_mirrored(1, result))

  def testWrapAListOfTwoTuples(self):
    result = values.regroup([("1", "2"), ("3", "4")])
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
    self.assertTrue(values.is_distributed_variable(v))
    self.assertTrue(values.is_distributed_variable(values.regroup(v.values)))

  def testSameId(self):
    foo = object()
    result = values.regroup((("a", foo), ("b", foo)))
    self.assertIsInstance(result, tuple)
    self.assertLen(result, 2)
    self._is_per_replica(result[0], ["a", "b"])
    self.assertIs(foo, result[1])

    # Test select_replica(), should undo the merge done by regroup().
    result_0 = values.select_replica(0, result)
    self.assertIsInstance(result_0, tuple)
    self.assertLen(result_0, 2)
    self.assertEqual("a", result_0[0])
    self.assertIs(foo, result_0[1])
    result_1 = values.select_replica(1, result)
    self.assertIsInstance(result_1, tuple)
    self.assertLen(result_1, 2)
    self.assertEqual("b", result_1[0])
    self.assertIs(foo, result_1[1])

  def testOneDevice(self):
    result = values.regroup((_nested_value("1"),))
    # On one device regroup() and select_replica() are basically identity.
    self.assertEqual(_nested_value("1"), result)
    self.assertEqual(_nested_value("1"), values.select_replica(0, result))

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

      merged_estimator_spec = values.regroup(created_estimator_specs)

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
                         values.select_replica(device_id,
                                               merged_estimator_spec))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_one_cpu,
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.tpu_strategy,
            strategy_combinations.central_storage_strategy_with_two_gpus,
        ],
        synchronization=[
            variables_lib.VariableSynchronization.ON_READ,
            variables_lib.VariableSynchronization.ON_WRITE,
        ],
        aggregation=[
            variables_lib.VariableAggregation.MEAN,
            variables_lib.VariableAggregation.SUM,
            variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
        ],
        mode=["graph", "eager"]))
class DistributedVariableTest(test.TestCase, parameterized.TestCase):

  def testExtendsVariable(self, distribution, synchronization, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          1., synchronization=synchronization, aggregation=aggregation)
    self.assertIsInstance(v, variables_lib.Variable)

  def testCheckpointing(self, distribution, synchronization, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          constant_op.constant([1., 2., 3., 4]),
          synchronization=synchronization,
          aggregation=aggregation)

    self.evaluate(v.initializer)
    before_save = self.evaluate(v.read_value())

    # Save random weights into checkpoint.
    checkpoint = trackable_utils.Checkpoint(v=v)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    with self.test_session():
      save_path = checkpoint.save(prefix)

    # Assign inverted value.
    self.evaluate(v.assign(constant_op.constant([4., 3., 2., 1.])))
    after_assign = self.evaluate(v.read_value())
    self.assertNotAllClose(before_save, after_assign)

    # Restore from the checkpoint.
    with self.test_session():
      checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    after_restore = self.evaluate(v)
    self.assertAllClose(before_save, after_restore)

  def testTraceback(self, distribution, synchronization, aggregation):
    if context.executing_eagerly():
      self.skipTest("does not apply to eager")
    with distribution.scope():
      variable_scope.get_variable(
          name="testVar",
          initializer=1.,
          use_resource=True,
          synchronization=synchronization,
          aggregation=aggregation)
      with self.assertRaisesRegex(ValueError,
                                  "Variable testVar already exists"):
        variable_scope.get_variable(
            name="testVar",
            initializer=1.,
            use_resource=True,
            synchronization=synchronization,
            aggregation=aggregation)

  def testSelectReplica(self, distribution, synchronization, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          1., synchronization=synchronization, aggregation=aggregation)
    self.assertIs(v, values.select_replica(0, v))

  def testIsTensorLike(self, distribution, synchronization, aggregation):
    if isinstance(distribution.extended,
                  tpu_strategy.TPUExtended) and context.executing_eagerly():
      self.skipTest("TPU doesn't support pure eager")

    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)
    # In cross replica context.
    self.assertIsInstance(v, core.Tensor)
    # In replica context.
    distribution.run(
        lambda v: self.assertIsInstance(v, core.Tensor), args=(v,))

  def testAssignReturnValueIsTensorLike(self, distribution, synchronization,
                                        aggregation):
    if isinstance(distribution.extended, tpu_strategy.TPUExtended):
      if context.executing_eagerly():
        self.skipTest("TPU doesn't support pure eager")
      else:
        self.skipTest("b/152076846")

    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)

    def assert_is_tensor_like(v):
      # We can't use Python literals because they are treated as non-distributed
      # values is not allowed when aggregation is SUM. See
      # `cross_device_ops.reduce_non_distributed_value`.
      delta = array_ops.identity(1.)
      self.assertIsInstance(v.assign(delta), core.Tensor)
      self.assertIsInstance(v.assign_sub(delta), core.Tensor)
      self.assertIsInstance(v.assign_add(delta), core.Tensor)

    # In cross replica context we return a PerReplica which is not Tensor like
    # all the time yet.
    if (synchronization == variables_lib.VariableSynchronization.ON_READ and
        aggregation != variables_lib.VariableAggregation.SUM):
      assert_is_tensor_like(v)

    # In replica context.
    distribution.run(assert_is_tensor_like, args=(v,))

  def testAssignSignature(self, distribution, synchronization, aggregation):
    # This test verifies assign*() can be called in the same way as normal
    # variables.
    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)

      def assign():
        one = constant_op.constant(1.)
        v.assign(one, True, "assign", False)
        # TODO(b/154017756): SyncOnReadVariable.assign() doesn't support passing
        # value as a keyword argument.
        v.assign(one, use_locking=True, name="assign", read_value=False)
        v.assign_add(one, True, "assign", False)
        v.assign_add(one, use_locking=True, name="assign", read_value=False)
        v.assign_sub(one, True, "assign", False)
        v.assign_sub(one, use_locking=True, name="assign", read_value=False)
        # Return something for graph mode to fetch.
        return constant_op.constant(1)

      self.evaluate(variables_lib.global_variables_initializer())
      if not (synchronization == variables_lib.VariableSynchronization.ON_READ
              and aggregation == variables_lib.VariableAggregation.SUM):
        self.evaluate(distribution.experimental_local_results(assign()))
      if not (isinstance(distribution.extended, tpu_strategy.TPUExtended) and
              context.executing_eagerly()):
        self.evaluate(
            distribution.experimental_local_results(distribution.run(assign)))


class MirroredVariableTest(test.TestCase, parameterized.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    mirrored = _make_mirrored()
    v = mirrored.values[0]
    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableOnAnotherDevice(self):
    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    mirrored = values.MirroredVariable(
        None, (v,), variable_scope.VariableAggregation.MEAN)

    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)

  def _assign_mirrored(self, v, new):
    for var, n in zip(v.values, new):
      self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveAndRestoreMirroredOneGraph(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.cached_session(config=self.config) as sess:
      mirrored = _make_mirrored()
      v = mirrored.values

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path, saver = self._save_return_saver(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(mirrored, [5., 6.])

      # Restores the saved value of 3. to both variables.
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  def _save_mirrored(self):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      mirrored = _make_mirrored()

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path = self._save(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(mirrored, [5., 6.])
    return save_path

  def _save_normal(self):
    """Save variables without mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(3.))

      # Saves the current value of var, 3.
      save_path = self._save(sess, var)

      # Change the values between save and restore.
      self.evaluate(var.assign(5.))
    return save_path

  def _restore_normal(self, save_path):
    """Restore to variables without mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3. to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3., self.evaluate(var))

  def _restore_mirrored(self, save_path):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      mirrored = _make_mirrored()
      v = mirrored.values

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [7., 8.])

      # Restores the saved value of 3. to both variables.
      saver = saver_lib.Saver(var_list=[mirrored])
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveMirroredRestoreMirrored(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored()
    self._restore_mirrored(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveMirroredRestoreNormal(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored()
    self._restore_normal(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveNormalRestoreMirrored(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_normal()
    self._restore_mirrored(save_path)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_gpu,
          ],
          mode=["graph"]))
  def testFetchAMirroredVariable(self, distribution):
    with self.session(graph=ops.Graph()) as sess, distribution.scope():
      with ops.device("/device:GPU:0"):
        v = variable_scope.get_variable(
            name="v", initializer=1., use_resource=True)
      mirrored = values.MirroredVariable(
          distribution, (v,), variable_scope.VariableAggregation.MEAN)
      sess.run(variables_lib.global_variables_initializer())
      sess.run({"complicated": mirrored})

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          mode=["graph", "eager"]))
  def testValueInReplicaContext(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          1., aggregation=variables_lib.VariableAggregation.MEAN)
      self.evaluate(variables_lib.global_variables_initializer())

      @def_function.function
      def f():
        with ops.control_dependencies([v.assign_add(1.)]):
          return v.value()

      results = self.evaluate(
          distribution.experimental_local_results(
              distribution.run(f)))
      for value in results:
        self.assertEqual(2., value)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          mode=["graph", "eager"]))
  def testAssignOutOfScope(self, distribution):
    with distribution.scope():
      mirrored = variables_lib.Variable(1.)
    self.evaluate(mirrored.assign(3.))
    self.assertEqual(self.evaluate(mirrored.read_value()), 3.)
    for component in mirrored.values:
      self.assertEqual(self.evaluate(component.read_value()), 3.)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testAssignAggregationMeanDTypeNonFloat(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          1,
          aggregation=variable_scope.VariableAggregation.MEAN,
          dtype=dtypes.int32)
    self.evaluate(v.initializer)

    @def_function.function
    def assign():
      ctx = distribution_strategy_context.get_replica_context()
      return v.assign(ctx.replica_id_in_sync_group)

    # disallow assign() with distributed value in replica context.
    with self.assertRaisesRegexp(ValueError,
                                 "Cannot update non-float variables"):
      self.evaluate(
          distribution.experimental_local_results(
              distribution.run(assign)))

    # allow assign() with same value in replica context.
    @def_function.function
    def assign_same():
      return v.assign(2)

    self.evaluate(
        distribution.experimental_local_results(
            distribution.run(assign_same)))
    self.assertEqual(self.evaluate(v.read_value()), 2)

    # allow assign() with mirrored variable in replica context.
    with distribution.scope():
      v2 = variables_lib.Variable(
          3,
          aggregation=variable_scope.VariableAggregation.SUM,
          dtype=dtypes.int32)
    self.evaluate(v2.initializer)

    @def_function.function
    def assign_mirrored():
      return v.assign(v2)

    self.evaluate(
        distribution.experimental_local_results(
            distribution.run(assign_mirrored)))
    self.assertEqual(self.evaluate(v.read_value()), 3)

    # allow assign() in cross replica context.
    with distribution.scope():
      self.evaluate(v.assign(4))
      self.assertEqual(self.evaluate(v.read_value()), 4)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          mode=["eager"]))
  def testInitializedToSameValueInsideEagerRun(self, distribution):
    v = [None]

    @def_function.function
    def step():

      def f():
        if v[0] is None:
          v[0] = variables_lib.Variable(random_ops.random_normal([]))

      distribution.run(f)

    context.set_global_seed(None)
    step()
    vals = self.evaluate(v[0].values)
    self.assertAllEqual(vals[0], vals[1])

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          mode=["graph", "eager"]))
  def testAggregationOnlyFirstReplica(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          15.,
          synchronization=variables_lib.VariableSynchronization.ON_WRITE,
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def assign():
      ctx = distribution_strategy_context.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      return v.assign(math_ops.cast(replica_id, dtypes.float32))
    per_replica_results = self.evaluate(distribution.experimental_local_results(
        distribution.run(assign)))
    # The per-replica values should always match the first replicas value.
    self.assertAllEqual(
        array_ops.zeros(distribution.num_replicas_in_sync, dtypes.float32),
        per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          mode=["eager"]))
  def testInitScope(self, distribution):

    class C(object):
      pass

    obj = C()
    obj.w = None
    obj.v = None

    @def_function.function
    def assign():
      with ops.init_scope():
        if obj.w is None:
          obj.w = variables_lib.Variable(
              0, aggregation=variables_lib.VariableAggregation.MEAN)
          obj.v = variables_lib.Variable(
              obj.w.read_value(),
              aggregation=variables_lib.VariableAggregation.MEAN)

      return obj.v.assign_add(2)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(distribution.run(assign)))
    self.assertAllEqual([2, 2], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          mode=["graph", "eager"]))
  def testAssignAdd(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          1, aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def assign():
      return v.assign_add(2)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(distribution.run(assign)))
    # The per-replica values should always match the first replicas value.
    self.assertAllEqual([3, 3], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testScatterSub(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          [0., 0., 0.], aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_sub():
      ctx = distribution_strategy_context.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      value = indexed_slices.IndexedSlices(
          values=array_ops.stack([
              math_ops.cast(replica_id, dtypes.float32),
              math_ops.cast(replica_id + 1, dtypes.float32)
          ]),
          indices=array_ops.stack([replica_id, replica_id + 1]),
          dense_shape=(3,))
      return v.scatter_sub(value)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(scatter_sub)))
    self.assertAllEqual([[0., -1., -1.], [0., -1., -1.]], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testScatterAdd(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          [0, 0, 0], aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_add():
      ctx = distribution_strategy_context.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      value = indexed_slices.IndexedSlices(
          values=array_ops.stack([replica_id, replica_id + 1]),
          indices=array_ops.stack([replica_id, replica_id + 1]),
          dense_shape=(3,))
      return v.scatter_add(value)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(scatter_add)))
    self.assertAllEqual([[0, 2, 2], [0, 2, 2]], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testScatterDiv(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          [1, 6, 1], aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_div():
      ctx = distribution_strategy_context.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      value = indexed_slices.IndexedSlices(
          values=array_ops.reshape(replica_id + 2, [1]),
          indices=array_ops.reshape(replica_id, [1]),
          dense_shape=(3,))
      return v.scatter_div(value)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(scatter_div)))
    self.assertAllEqual([[0, 2, 1], [0, 2, 1]], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testScatterMul(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          [2., 1., 1.], aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_mul():
      ctx = distribution_strategy_context.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      value = indexed_slices.IndexedSlices(
          values=array_ops.reshape(
              math_ops.cast(replica_id + 2, dtypes.float32), [1]),
          indices=array_ops.reshape(replica_id, [1]),
          dense_shape=(3,))
      return v.scatter_mul(value)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(scatter_mul)))
    self.assertAllClose([[2., 1.5, 1.], [2., 1.5, 1.]], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testScatterMin(self, distribution):
    with distribution.scope():
      v1 = variables_lib.Variable(
          [0, 2, 0], aggregation=variables_lib.VariableAggregation.SUM)
      v2 = variables_lib.Variable(
          [0, 2, 0],
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def scatter_min(v):
      value = indexed_slices.IndexedSlices(
          values=array_ops.identity([1]),
          indices=array_ops.identity([1]),
          dense_shape=(3,))
      return v.scatter_min(value)

    with self.assertRaisesRegex(NotImplementedError, "scatter_min.*"):
      self.evaluate(
          distribution.experimental_local_results(
              distribution.run(scatter_min, args=(v1,))))

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(scatter_min, args=(v2,))))
    self.assertAllClose([[0, 1, 0], [0, 1, 0]], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testScatterMax(self, distribution):
    with distribution.scope():
      v1 = variables_lib.Variable(
          [0, 0, 0], aggregation=variables_lib.VariableAggregation.SUM)
      v2 = variables_lib.Variable(
          [0, 0, 0],
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def scatter_max(v):
      value = indexed_slices.IndexedSlices(
          values=array_ops.identity([1]),
          indices=array_ops.identity([0]),
          dense_shape=(3,))
      return v.scatter_max(value)

    with self.assertRaisesRegex(NotImplementedError, "scatter_max.*"):
      self.evaluate(
          distribution.experimental_local_results(
              distribution.run(scatter_max, args=(v1,))))

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(scatter_max, args=(v2,))))
    self.assertAllClose([[1, 0, 0], [1, 0, 0]], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testScatterUpdate(self, distribution):
    with distribution.scope():
      v1 = variables_lib.Variable(
          [0, 0, 0], aggregation=variables_lib.VariableAggregation.SUM)
      v2 = variables_lib.Variable(
          [0, 0, 0],
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def scatter_update(v):
      value = indexed_slices.IndexedSlices(
          values=array_ops.identity([3]),
          indices=array_ops.identity([1]),
          dense_shape=(3,))
      return v.scatter_update(value)

    with self.assertRaisesRegex(NotImplementedError, "scatter_update.*"):
      self.evaluate(
          distribution.experimental_local_results(
              distribution.run(scatter_update, args=(v1,))))

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(scatter_update, args=(v2,))))
    self.assertAllClose([[0, 3, 0], [0, 3, 0]], per_replica_results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph", "eager"]))
  def testScatterOpsInCrossReplicaContext(self, distribution):
    with distribution.scope():
      v1 = variables_lib.Variable(
          [1, 1, 1], aggregation=variables_lib.VariableAggregation.SUM)
      v2 = variables_lib.Variable([1, 1, 1])
    self.evaluate(variables_lib.global_variables_initializer())

    value = indexed_slices.IndexedSlices(
        values=array_ops.identity([2]),
        indices=array_ops.identity([0]),
        dense_shape=(3,))
    with distribution.scope():
      self.evaluate(v1.scatter_add(value))
      self.assertAllEqual([3, 1, 1], self.evaluate(v1.read_value()))

      self.evaluate(v2.scatter_min(value))
      self.assertAllEqual([1, 1, 1], self.evaluate(v2.read_value()))


_TPU_STRATEGIES = (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)


def _make_replica_local(method, strategy=None):
  if strategy is None:
    devices = ("/device:GPU:0", "/device:CPU:0")
  else:
    devices = strategy.extended.worker_devices

  v = []
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))

  if (strategy is not None) and isinstance(strategy, _TPU_STRATEGIES):
    var_cls = tpu_values.TPUSyncOnReadVariable
  else:
    var_cls = values.SyncOnReadVariable
  replica_local = var_cls(strategy, v, method)
  return v, replica_local


class SyncOnReadVariablePropertiesTest(test.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")
    v, replica_local = _make_replica_local(
        variable_scope.VariableAggregation.SUM)

    self.assertEqual(v[0].constraint, replica_local.constraint)
    self.assertEqual(v[0].name, replica_local.name)
    self.assertEqual(v[0].dtype, replica_local.dtype)
    self.assertEqual(v[0].shape, replica_local.shape)
    self.assertEqual(variable_scope.VariableAggregation.SUM,
                     replica_local.aggregation)

  @test_util.run_v2_only
  def testCanPassToDefFun(self):
    @def_function.function
    def add1(x):
      return x + 1

    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    replica_local = values.SyncOnReadVariable(
        None, (v,), variable_scope.VariableAggregation.MEAN)
    self.assertEqual(2., self.evaluate(add1(replica_local)))


def mirrored_and_tpu_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.tpu_strategy,
      ],
      mode=["graph", "eager"])


# TODO(b/144432582): Add variable aggregation type to combinations to simplify
# tests.
def strategy_and_run_tf_function_combinations():
  # Test the combination of different strategies and whether a tf.function
  # is passed into strategy.run."""
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
      ],
      mode=["graph", "eager"],
      experimental_run_tf_function=[True, False]) + combinations.combine(
          distribution=[
              strategy_combinations.tpu_strategy,
          ],
          mode=["graph", "eager"],
          experimental_run_tf_function=[True])


class SyncOnReadVariableTest(test.TestCase, parameterized.TestCase):

  def _assign_replica_local(self, v, new):
    for var, n in zip(v, new):
      with ops.device(var.device):
        self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testTensorConversion(self, distribution):
    with context.graph_mode():
      _, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)
      converted = ops.convert_to_tensor(replica_local, as_ref=False)
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

      converted = ops.convert_to_tensor(replica_local, as_ref=True)
      # Resources variable are converted to tensors as well when as_ref is True.
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreReplicaLocalSumOneGraph(self, distribution):
    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 7.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])

        # Restores the saved value of 7. which gets divided equally
        # between the variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreReplicaLocalMeanOneGraph(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])

        # Restores the saved value of 3.5 to both variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _save_replica_local_mean(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])
    return save_path

  def _save_replica_local_sum(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [1.5, 2.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])
    return save_path

  def _save_normal(self):
    """Save variables without mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(3.5))

      # Saves the current value of var, 3.5.
      save_path = self._save(sess, var)

      # Change the values between save and restore.
      self.evaluate(var.assign(5.))
    return save_path

  def _restore_normal(self, save_path):
    """Restore to variables without mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3.5 to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3.5, self.evaluate(var))

  def _restore_replica_local_mean(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _restore_replica_local_sum(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([1.75, 1.75], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalRestoreReplicaLocalMean(self, distribution):
    save_path = self._save_replica_local_mean(distribution)
    self._restore_replica_local_mean(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalRestoreReplicaLocalSum(self, distribution):
    save_path = self._save_replica_local_sum(distribution)
    self._restore_replica_local_sum(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalMeanRestoreNormal(self, distribution):
    save_path = self._save_replica_local_mean(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalSumRestoreNormal(self, distribution):
    save_path = self._save_replica_local_sum(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreReplicaLocalMean(self, distribution):
    save_path = self._save_normal()
    self._restore_replica_local_mean(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreReplicaLocalSum(self, distribution):
    save_path = self._save_normal()
    self._restore_replica_local_sum(save_path, distribution)

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssign(self, distribution, experimental_run_tf_function):

    def assign(fn, v, update_value, cross_replica):
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return distribution.experimental_local_results(
            distribution.run(update_fn))

    updates = [("assign", 1.), ("assign_add", 1.), ("assign_sub", -1.)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # VariableAggregation.SUM in cross-replica mode is tested below,
      # VariableAggregation.NONE in cross-replica mode is not supported.
      if cross_replica and aggregation in [
          variables_lib.VariableAggregation.SUM,
          variables_lib.VariableAggregation.NONE,
      ]:
        continue
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(self.evaluate(component.read_value()),
                            self.evaluate(array_ops.ones_like(component)))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignDtypeConversion(self, distribution,
                                experimental_run_tf_function):

    def assign(fn, v, update_value, cross_replica):
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return distribution.experimental_local_results(
            distribution.run(update_fn))

    updates = [("assign", 1), ("assign_add", 1), ("assign_sub", -1)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # VariableAggregation.SUM in cross-replica mode is tested below,
      # VariableAggregation.NONE in cross-replica mode is not supported.
      if cross_replica and aggregation in [
          variables_lib.VariableAggregation.SUM,
          variables_lib.VariableAggregation.NONE,
      ]:
        continue
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(self.evaluate(component.read_value()),
                            self.evaluate(array_ops.ones_like(component)))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testAssignWithAggregationSum(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(v.assign(1. * distribution.num_replicas_in_sync))
    for component in v._values:
      self.assertAllEqual(self.evaluate(component.read_value()),
                          self.evaluate(array_ops.ones_like(component)))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testAssignAddSubWithAggregationSum(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(variables_lib.global_variables_initializer())
    with self.assertRaisesRegex(
        ValueError, "SyncOnReadVariable does not support "):
      self.evaluate(v.assign_add(1.))
    with self.assertRaisesRegex(
        ValueError, "SyncOnReadVariable does not support "):
      self.evaluate(v.assign_sub(1.))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testReadValueInReplicaContext(self, distribution,
                                    experimental_run_tf_function):
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      if experimental_run_tf_function:
        read_var_fn = def_function.function(v.read_value)
      else:
        read_var_fn = v.read_value
      results = self.evaluate(
          distribution.experimental_local_results(
              distribution.run(read_var_fn)))
      for component, value in zip(v._values, results):
        self.assertAllEqual(self.evaluate(component.read_value()), value)

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testReadValueInCrossReplicaContext(self, distribution,
                                         experimental_run_tf_function):
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        # variables_lib.VariableAggregation.MEAN,
        # variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      if isinstance(distribution, _TPU_STRATEGIES):
        resolver = tpu_cluster_resolver.TPUClusterResolver("")
        tpu_strategy_util.initialize_tpu_system(resolver)
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())

      def assign(v=v):
        ctx = distribution_strategy_context.get_replica_context()
        replica_id = ctx.replica_id_in_sync_group
        return v.assign(math_ops.cast(replica_id, dtypes.float32))

      if experimental_run_tf_function:
        assign = def_function.function(assign)

      self.evaluate(
          distribution.experimental_local_results(distribution.run(assign)))
      num_replicas = distribution.num_replicas_in_sync
      sum_of_replica_values = num_replicas * (num_replicas - 1) / 2.
      if aggregation == variables_lib.VariableAggregation.SUM:
        expected = sum_of_replica_values
      elif aggregation == variables_lib.VariableAggregation.MEAN:
        expected = sum_of_replica_values / num_replicas
      else:
        expected = 0
      self.assertEqual(expected, self.evaluate(v.read_value()), aggregation)
      self.assertEqual(expected, self.evaluate(v.value()), aggregation)
      self.assertEqual(expected, self.evaluate(v), aggregation)
      self.assertEqual(expected, self.evaluate(array_ops.identity(v)),
                       aggregation)

  # TODO(b/145574622): Re-enable this test once ReduceOp argument is
  # respected on GPUs.
  @combinations.generate(strategy_and_run_tf_function_combinations())
  def disable_testAllReduce(self, distribution,
                            experimental_run_tf_function):
    with distribution.scope():
      v = variable_scope.variable(
          2.,
          synchronization=variables_lib.VariableSynchronization.ON_WRITE,
          aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(variables_lib.global_variables_initializer())

    def all_reduce():
      ctx = distribution_strategy_context.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      return ctx.all_reduce("SUM", v) + math_ops.cast(replica_id,
                                                      dtypes.float32)

    if experimental_run_tf_function:
      all_reduce = def_function.function(all_reduce)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(distribution.run(all_reduce)))
    expected_result = []
    for i in range(distribution.num_replicas_in_sync):
      expected_result.append(2.0 * distribution.num_replicas_in_sync +
                             1.0 * i)
    self.assertEqual(per_replica_results, tuple(expected_result))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignPerReplicaBeforeRead(self, distribution,
                                     experimental_run_tf_function):
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())

      def assign(var=v):
        ctx = distribution_strategy_context.get_replica_context()
        replica_id = ctx.replica_id_in_sync_group
        return var.assign(math_ops.cast(replica_id, dtypes.float32))

      if experimental_run_tf_function:
        assign = def_function.function(assign)

      per_replica_results = self.evaluate(
          distribution.experimental_local_results(distribution.run(assign)))
      expected_result = []
      for i in range(distribution.num_replicas_in_sync):
        expected_result.append(1.0 * i)
      self.assertEqual(per_replica_results, tuple(expected_result))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testReadValueWithAggregationNoneInCrossReplicaContext(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.NONE)
    self.evaluate(variables_lib.global_variables_initializer())
    with self.assertRaisesRegex(
        ValueError, "Could not convert from .* VariableAggregation\\.NONE"):
      self.evaluate(v.read_value())

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testInitializedToSameValueInsideEagerRun(self, distribution):
    if not context.executing_eagerly(): self.skipTest("eager only")

    v = [None]
    @def_function.function
    def step():
      def f():
        if v[0] is None:
          v[0] = variables_lib.Variable(
              random_ops.random_normal([]),
              synchronization=variables_lib.VariableSynchronization.ON_READ)

      distribution.run(f)

    context.set_global_seed(None)
    step()
    vals = self.evaluate(v[0].values)
    self.assertAllEqual(vals[0], vals[1])


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        aggregation=[
            variables_lib.VariableAggregation.MEAN,
            variables_lib.VariableAggregation.SUM,
            variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
        ],
        mode=["graph", "eager"]))
class SyncOnReadScatterReplicaTest(test.TestCase, parameterized.TestCase):

  def testScatterSub(self, distribution, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          [1., 1., 1.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[0.], [1.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[1.], [2.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_sub, args=(delta,)))

  def testScatterAdd(self, distribution, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          [1., 1., 1.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[0.], [1.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[1.], [2.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_add, args=(delta,)))

  def testScatterDiv(self, distribution, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          [2., 6., 1.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[2.], [2.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[3.], [3.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_div, args=(delta,)))

  def testScatterMul(self, distribution, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          [2., 1., 1.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[2.], [3.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[4.], [5.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_mul, args=(delta,)))

  def testScatterMin(self, distribution, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          [3., 4., 5.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[1.], [8.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[9.], [2.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_min, args=(delta,)))

  def testScatterMax(self, distribution, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          [3., 4., 5.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[1.], [8.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[9.], [2.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_max, args=(delta,)))

  def testScatterUpdate(self, distribution, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          [0., 0., 0.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[1.], [2.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[3.], [4.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_min, args=(delta,)))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.central_storage_strategy_with_two_gpus
        ],
        mode=["graph", "eager"]))
class AggregatingVariableTest(test.TestCase, parameterized.TestCase):

  def testAssignOutOfScope(self, distribution):
    with distribution.scope():
      aggregating = variables_lib.Variable(1.)
    self.assertIsInstance(aggregating, values.AggregatingVariable)
    self.evaluate(aggregating.assign(3.))
    self.assertEqual(self.evaluate(aggregating.read_value()), 3.)
    self.assertEqual(self.evaluate(aggregating._v.read_value()), 3.)

  def testAssignAdd(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          1, aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def assign():
      return v.assign_add(2)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.experimental_run_v2(assign)))
    self.assertAllEqual([3], per_replica_results)


class MirroredTest(test.TestCase):

  def testAddOp(self):
    if context.num_gpus() < 1:
      self.skipTest("A GPU is not available for this test.")
    mirrored_val = _make_mirrored_val(init_val=3.)

    self.assertEqual(self.evaluate(constant_op.constant(6.)),
                     self.evaluate(mirrored_val + mirrored_val))
    self.assertEqual(self.evaluate(constant_op.constant(4.)),
                     self.evaluate(mirrored_val + 1))
    self.assertEqual(self.evaluate(mirrored_val + 1),
                     self.evaluate(math_ops.add(mirrored_val, 1)))
    self.assertEqual(type(mirrored_val + 1),
                     type(math_ops.add(mirrored_val, 1)))


class PerReplicaTest(test.TestCase, parameterized.TestCase):

  def testTypeSpec(self):
    vals = (constant_op.constant(1.),)
    per_replica = values.PerReplica(vals)

    spec = per_replica._type_spec
    self.assertEqual(spec._value_specs,
                     (tensor_spec.TensorSpec([], dtypes.float32),))

  def testTypeSpecRoundTrip(self):
    vals = (constant_op.constant(1.),)
    per_replica = values.PerReplica(vals)

    spec = per_replica._type_spec
    tensor_list = spec._to_components(per_replica)
    reconstructed = spec._from_components(tensor_list)

    self.assertAllEqual(per_replica.values, reconstructed.values)

  def testTypeSpecNest(self):
    vals = (constant_op.constant(1.), constant_op.constant([5., 6.0]),)
    per_replica = values.PerReplica(vals)

    # Note: nest.map_structure exercises nest.flatten and
    # nest.pack_sequence_as.
    result = nest.map_structure(
        lambda t: t + 10, per_replica, expand_composites=True)

    self.assertLen(result.values, 2)
    self.assertAllEqual(result.values[0], 11.)
    self.assertAllEqual(result.values[1], [15., 16.0])

  @test_util.run_in_graph_and_eager_modes
  def testIsGraphTensor(self):
    per_replica = values.PerReplica((constant_op.constant(1.),))
    for t in nest.flatten(per_replica, expand_composites=True):
      self.assertEqual(hasattr(t, "graph"), not context.executing_eagerly())

  def testDoesNotTriggerFunctionTracing(self):
    traces = []

    @def_function.function
    def f(x):
      traces.append(None)  # Only happens on trace.
      return x

    per_replica = values.PerReplica((constant_op.constant(1.),))

    # Trace once.
    f(per_replica)
    self.assertNotEmpty(traces)
    del traces[:]

    per_replica_spec = per_replica._type_spec
    for _ in range(5):
      vals = per_replica_spec._to_components(per_replica)
      vals = [v * 2 for v in vals]
      per_replica = per_replica_spec._from_components(vals)

      output = f(per_replica)
      self.assertIsInstance(output, values.PerReplica)
      self.assertAllEqual(output._values, per_replica._values)
      self.assertEmpty(traces)  # Make sure we're not re-tracing `f`.

  def testFunctionCanReturnPerReplica(self):
    f = def_function.function(lambda x: x)
    x = values.PerReplica((constant_op.constant(1.),))
    y = f(x)
    self.assertIsNot(x, y)
    nest.map_structure(self.assertAllEqual, x, y, expand_composites=True)
    self.assertEqual(x._type_spec, y._type_spec)

  @test_util.run_in_graph_and_eager_modes
  def testCondWithTensorValues(self):
    per_replica_1 = values.PerReplica((constant_op.constant("a"),))
    per_replica_2 = values.PerReplica((constant_op.constant(["b", "c"]),))
    condition = array_ops.placeholder_with_default(True, [])

    result = control_flow_ops.cond(
        condition, lambda: per_replica_1, lambda: per_replica_2)

    self.assertLen(result.values, 1)
    self.assertAllEqual(result.values[0], "a")

  @test_util.run_in_graph_and_eager_modes
  def testCondWithValuesConvertibleToTensor(self):
    per_replica_1 = values.PerReplica(("a",))
    per_replica_2 = values.PerReplica(("b",))
    condition = array_ops.placeholder_with_default(True, [])

    result = control_flow_ops.cond(
        condition, lambda: per_replica_1, lambda: per_replica_2)

    self.assertLen(result.values, 1)
    self.assertAllEqual(result.values[0], "a")

  @test_util.build_as_function_and_v1_graph
  def testCondWithValuesNotConvertibleToTensor(self):
    per_replica_1 = values.PerReplica(({"a"},))
    per_replica_2 = values.PerReplica(({"b", "c"},))
    condition = array_ops.placeholder(dtypes.bool, [])

    with self.assertRaisesRegex(TypeError, "Could not build a TypeSpec for"):
      control_flow_ops.cond(
          condition, lambda: per_replica_1, lambda: per_replica_2)


def _make_index_slices(values, indices, dense_shape=None):
  if dense_shape:
    dense_shape = array_ops.identity(dense_shape)
  return indexed_slices.IndexedSlices(
      array_ops.identity(values), array_ops.identity(indices), dense_shape)


if __name__ == "__main__":
  test.main()
