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

import copy
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util as ds_test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import saver as saver_lib


def _device_str(d):
  return "/device:GPU:" + str(d)


def _nested_value(d):
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


def mirrored_and_tpu_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
          strategy_combinations.tpu_strategy_spmd,
      ],
      mode=["graph", "eager"])


class DistributedValuesTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
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
        ds_test_util.gather(distribution, distributed_values),
        constant_op.constant(1., shape=(distribution.num_replicas_in_sync)))

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
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
    self.assertAllEqual(
        ds_test_util.gather(distribution, distributed_values).numpy(),
        [[1., 2., 3.]] * distribution.num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
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
    distributed_values = ds_test_util.gather(distribution, distributed_values)

    # Expected output for 2 replicas:
    # ([1.0, 1.0], [2.0, 2.0], [3.0, 3.0])
    expected = tuple([v for i in range(distribution.num_replicas_in_sync)]
                     for v in tuple_value)
    self.assertAllEqual(distributed_values, expected)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
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
      return tuple(per_replica)
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    distributed_values = ds_test_util.gather(distribution, distributed_values)

    # Expected output for 2 replicas:
    # ([0.0, 1.0], [0.0, 2.0], [0.0, 3.0])
    expected = tuple([v * i for i in range(distribution.num_replicas_in_sync)]
                     for v in tuple_value)
    self.assertAllEqual(distributed_values, expected)

  # NOTE(priyag): Cannot test this with MultiWorkerMirroredStrategy because
  # collective ops do not support SparseTensors.
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
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
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
    distributed_values = ds_test_util.gather(distribution, distributed_values)
    expected = range(distribution.num_replicas_in_sync)
    self.assertAllEqual(distributed_values, expected)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
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

      outputs = ds_test_util.gather(
          distribution,
          distribution.run(computation, args=(distributed_values,)))
      return outputs

    results = run()

    expected = [i**2 for i in range(distribution.num_replicas_in_sync)]
    self.assertAllEqual(results, expected)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ] + strategy_combinations.multiworker_strategies,
          mode=["eager"]))
  def testMakeDistributedValueDefaultDevicePlacement(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    def value_fn(ctx):
      del ctx
      return constant_op.constant(1.0)
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    default_device = array_ops.identity(constant_op.constant(1.0)).device
    for i in range(len(distribution.extended.worker_devices)):
      self.assertAllEqual(distributed_values._values[i].device, default_device)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ] + strategy_combinations.multiworker_strategies,
          mode=["eager"],
          op_type=[constant_op.constant, array_ops.identity]))
  def testMakeDistributedValueExplicitDevicePlacement(self, distribution,
                                                      op_type):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    worker_devices = distribution.extended.worker_devices
    def value_fn(ctx):
      # In multi client setup, worker_devices is just the devices on that
      # worker.
      worker_device_id = ctx.replica_id_in_sync_group % len(worker_devices)
      with ops.device(worker_devices[worker_device_id]):
        return op_type(1.0)

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    for i in range(len(distribution.extended.worker_devices)):
      self.assertAllEqual(distributed_values._values[i].device,
                          worker_devices[i])


class PerReplicaTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ] + strategy_combinations.multiworker_strategies,
          mode=["eager"]))
  def testUsePerReplicaInvalidContextGivesError(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    multiple_values = range(distribution.num_replicas_in_sync)
    def value_fn(ctx):
      return multiple_values[ctx.replica_id_in_sync_group]
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    with self.assertRaisesRegex(ValueError, "not inside a replica context"):
      math_ops.cast(distributed_values, dtypes.float32)


class PerWorkerResourceTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(dataset_fn_as_tf_function=[True, False]))
  def testMapFnTracing(self, dataset_fn_as_tf_function):
    # For a PerWorkerResource to correctly behave when used in dataset.map,
    # it has to be that the map_fn is not traced only once such that
    # PerWorkerResource.local_table can return the correct resource. This test
    # can detect the potential breakage of this behavior on TAP.
    self._traced_once = 0

    def map_fn(x):
      self._traced_once += 1
      return x

    def dataset_fn():
      dataset = dataset_ops.DatasetV2.from_tensors([0, 1, 2]).repeat().batch(
          2, drop_remainder=True)
      dataset = dataset.map(map_fn)
      return dataset

    datasets = []
    number_of_input_pipelines = 5

    if dataset_fn_as_tf_function:
      dataset_fn = def_function.function(dataset_fn)
      expected_tracing_times = 1
    else:
      expected_tracing_times = number_of_input_pipelines

    for _ in range(number_of_input_pipelines):
      datasets.append(dataset_fn())

    self.assertEqual(self._traced_once, expected_tracing_times)


class DistributedDelegateTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testGetAttr(self):
    class Foo(object):

      def __init__(self, x):
        self.x = x

    v = values_lib.DistributedDelegate((Foo(7), Foo(8)))
    self.assertEqual(7, v.x)
    with self.assertRaises(AttributeError):
      _ = v.y

  @test_util.run_in_graph_and_eager_modes
  def testOperatorOverride(self):
    v = values_lib.DistributedDelegate((7, 8))
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

  @test_util.run_in_graph_and_eager_modes
  def testCopy(self):

    class Foo(object):

      def __init__(self, x):
        self.x = x

    v = values_lib.DistributedDelegate((Foo(7), Foo(8)))
    v_shallow_copy = copy.copy(v)
    self.assertEqual(v.x, v_shallow_copy.x)
    v_deep_copy = copy.deepcopy(v)
    self.assertEqual(v.x, v_deep_copy.x)


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
    var_cls = values_lib.SyncOnReadVariable
  replica_local = var_cls(strategy, v, method)
  return v, replica_local


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

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu
          ],
          mode=["eager"]))
  def testCanPassToDefFun(self, distribution):

    @def_function.function
    def add1(x):
      return x + 1.

    with distribution.scope():
      v = variables_lib.Variable(
          1.,
          aggregation=variables_lib.VariableAggregation.MEAN,
          synchronization=variables_lib.VariableSynchronization.ON_READ)

    self.assertEqual(2., self.evaluate(add1(v)))

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

  @combinations.generate(combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ], mode=["eager"]))
  def testValueInCrossReplicaContext(self, distribution):
    value_list, replica_local = _make_replica_local(
        variable_scope.VariableAggregation.ONLY_FIRST_REPLICA, distribution)

    self.assertIsInstance(replica_local.value(), ops.Tensor)
    self.assertEqual(self.evaluate(replica_local.value()),
                     self.evaluate(value_list[0].value()))

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy_packed_var,
          ],
          mode=["eager"]))
  def testValueInDefaultReplicaContext(self, distribution):
    with distribution.scope():
      v1 = variables_lib.Variable(
          0.0,
          aggregation=variables_lib.VariableAggregation.SUM,
          synchronization=variables_lib.VariableSynchronization.ON_READ)
      v2 = variables_lib.Variable(
          0.0,
          aggregation=variables_lib.VariableAggregation.SUM,
          synchronization=variables_lib.VariableSynchronization.ON_READ)

    @def_function.function
    def replica_fn():
      v1.assign_add(1.0)
      v2.assign_add(2.0)

    distribution.run(replica_fn)
    sum_v = v1 + v2
    self.assertEqual(sum_v, 6.0)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.tpu_strategy_packed_var,
          ],
          mode=["eager"]))
  def testReplicatedValueNameDeterministic(self, distribution):
    with distribution.scope():
      v1 = variables_lib.Variable(0.0, name="test_var_1")
      v2 = variables_lib.Variable(0.0, name="test_var_2")

    def fn():
      v1.assign_add(1.0)
      v2.assign_add(2.0)
      return v1 + v2

    @def_function.function
    def dist_run_fn():
      a = distribution.run(fn)
      return a

    concrete_fn = dist_run_fn.get_concrete_function()
    inputs = concrete_fn.graph.inputs
    self.assertLen(inputs, 2)
    # Before cl/433948982, input name will include a non-deterministic uid,
    # e.g. "test_var_1_139726389910864/handle/inputs_0:0"
    self.assertEqual(inputs[0].name, "test_var_1/handle/inputs_0:0")
    self.assertEqual(inputs[1].name, "test_var_2/handle/inputs_0:0")

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


if __name__ == "__main__":
  ds_test_util.main()
