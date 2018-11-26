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
"""Multi-GPU tests for MirroredStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.contrib.distribute.python import strategy_test_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import training as keras_training
from tensorflow.python.keras.layers import core as keras_core
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import server_lib


GPU_TEST = "test_gpu" in sys.argv[0]


@combinations.generate(combinations.combine(
    distribution=[
        combinations.mirrored_strategy_with_gpu_and_cpu,
        combinations.mirrored_strategy_with_two_gpus,
        combinations.core_mirrored_strategy_with_gpu_and_cpu,
        combinations.core_mirrored_strategy_with_two_gpus],
    mode=["graph", "eager"]))
class MirroredTwoDeviceDistributionTest(strategy_test_lib.DistributionTestBase,
                                        parameterized.TestCase):

  def testMinimizeLoss(self, distribution):
    if context.executing_eagerly():
      self._test_minimize_loss_eager(distribution)
    else:
      self._test_minimize_loss_graph(distribution)

  def testReplicaId(self, distribution):
    self._test_replica_id(distribution)

  def testNumReplicasInSync(self, distribution):
    self.assertEqual(2, distribution.num_replicas_in_sync)

  def testCallAndMergeExceptions(self, distribution):
    self._test_call_and_merge_exceptions(distribution)

  def testRunRegroupError(self, distribution):
    def run_fn():
      replica_id = int(self.evaluate(_replica_id()))
      # Generates a list with different lengths on different devices.
      # Will fail in _regroup() (if more than one device).
      return list(range(replica_id))

    with distribution.scope(), self.assertRaises(AssertionError):
      distribution.extended.call_for_each_replica(run_fn)

  def testReduceToCpu(self, distribution):
    with distribution.scope():
      result = distribution.extended.call_for_each_replica(_replica_id)
      reduced = distribution.reduce(
          reduce_util.ReduceOp.SUM,
          result,
          destinations="/device:CPU:0")
      unwrapped = distribution.unwrap(reduced)
      self.assertEqual(1, len(unwrapped))
      expected = sum(range(distribution.num_replicas_in_sync))
      self.assertEqual(expected, self.evaluate(unwrapped[0]))

  def testMakeInputFnIterator(self, distribution):
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=2,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    iterator = distribution.make_input_fn_iterator(input_fn)
    self._test_input_fn_iterator(iterator, distribution.extended.worker_devices,
                                 expected_values)

  def testGlobalStepUpdate(self, distribution):
    self._test_global_step_update(distribution)


def one_device_combinations():
  return combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_one_cpu,
          combinations.mirrored_strategy_with_one_gpu,
          combinations.core_mirrored_strategy_with_one_cpu,
          combinations.core_mirrored_strategy_with_one_gpu],
      mode=["graph", "eager"])


class MirroredOneDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.NamedDistribution(
              "Mirrored1CPU",
              lambda: mirrored_strategy.MirroredStrategy(["/device:CPU:0"]),
              required_gpus=1),
          combinations.mirrored_strategy_with_one_gpu,
          combinations.NamedDistribution(
              "CoreMirrored1CPU",
              lambda: mirrored_strategy.CoreMirroredStrategy(["/device:CPU:0"]),
              required_gpus=1),
          combinations.core_mirrored_strategy_with_one_gpu],
      mode=["graph", "eager"]))
  def testReduceToMultipleDestinations(self, distribution):
    with distribution.scope():
      reduced = distribution.extended.reduce_to(
          reduce_util.ReduceOp.SUM,
          1.0,
          destinations=["/device:CPU:0", "/device:GPU:0"])
      unwrapped = distribution.unwrap(reduced)
      self.assertLen(unwrapped, 2)
      self.assertEqual(1.0, self.evaluate(unwrapped[0]))

  @combinations.generate(one_device_combinations())
  def testMinimizeLoss(self, distribution):
    if context.executing_eagerly():
      self._test_minimize_loss_eager(distribution)
    else:
      self._test_minimize_loss_graph(distribution)

  @combinations.generate(one_device_combinations())
  def testReplicaId(self, distribution):
    self._test_replica_id(distribution)

  @combinations.generate(one_device_combinations())
  def testCallAndMergeExceptions(self, distribution):
    self._test_call_and_merge_exceptions(distribution)


class MirroredStrategyVariableCreatorStackTest(
    test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      distribution=[combinations.mirrored_strategy_with_gpu_and_cpu,
                    combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=["graph"]))
  def testCreatorStacksAreThreadLocal(self, distribution):
    def model_fn():
      replica_id_str = str(self.evaluate(_replica_id()))

      def thread_creator_fn(next_creator, *args, **kwargs):
        return next_creator(*args, **kwargs) + ":thread_" + replica_id_str

      with variable_scope.variable_creator_scope(thread_creator_fn):
        # Create a variable in this scope.
        v = variable_scope.variable(1.0)

        # This will pause the current thread, and execute the other thread.
        ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    def main_thread_creator(next_creator, *args, **kwargs):
      # We are not using the underlying next_creator for test purposes.
      del next_creator, args, kwargs
      return "main_thread"

    with context.graph_mode(), \
        distribution.scope(), \
        variable_scope.variable_creator_scope(main_thread_creator):
      result = distribution.extended.call_for_each_replica(model_fn)
      result = distribution.unwrap(result)
      expected = ["main_thread:thread_0", "main_thread:thread_1"]
      self.assertEqual(expected, result)


@combinations.generate(combinations.combine(
    distribution=[
        combinations.mirrored_strategy_with_gpu_and_cpu,
        combinations.core_mirrored_strategy_with_gpu_and_cpu],
    mode=["graph", "eager"]))
class MirroredStrategyVariableCreationTest(test.TestCase):

  def testSingleVariable(self, distribution):
    def model_fn():
      # This variable should be created only once across the threads because of
      # special variable_creator functions used by
      # `distribution.extended.call_for_each_replica`.
      v = variable_scope.variable(1.0, name="foo")
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertIsInstance(result, values.MirroredVariable)
      self.assertEqual("foo:0", result.name)

  def testUnnamedVariable(self, distribution):
    def model_fn():
      v = variable_scope.variable(1.0)
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertIsInstance(result, values.MirroredVariable)
      # Default name of "Variable" will be used.
      self.assertEqual("Variable:0", result.name)

  def testMultipleVariables(self, distribution):
    def model_fn():
      vs = []
      for i in range(5):
        vs.append(variable_scope.variable(1.0, name="foo" + str(i)))
      ds_context.get_replica_context().merge_call(lambda _: _)
      return vs

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      for i, v in enumerate(result):
        self.assertIsInstance(v, values.MirroredVariable)
        self.assertEqual("foo" + str(i) + ":0", v.name)

  def testMultipleVariablesWithSameCanonicalName(self, distribution):
    def model_fn():
      vs = []
      vs.append(variable_scope.variable(1.0, name="foo/bar"))
      vs.append(variable_scope.variable(1.0, name="foo_1/bar"))
      vs.append(variable_scope.variable(1.0, name="foo_1/bar_1"))
      vs.append(variable_scope.variable(1.0, name="foo/bar_1"))
      ds_context.get_replica_context().merge_call(lambda _: _)
      return vs

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      for v in result:
        self.assertIsInstance(v, values.MirroredVariable)
      self.assertEqual(4, len(result))
      self.assertEqual("foo/bar:0", result[0].name)
      self.assertEqual("foo_1/bar:0", result[1].name)
      self.assertEqual("foo_1/bar_1:0", result[2].name)
      self.assertEqual("foo/bar_1:0", result[3].name)

  def testVariableWithSameCanonicalNameAcrossThreads(self, distribution):
    def model_fn():
      replica_id = self.evaluate(_replica_id())
      v = variable_scope.variable(1.0, name="foo_" + str(replica_id))
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertIsInstance(result, values.MirroredVariable)
      # The resulting mirrored variable will use the name from the first device.
      self.assertEqual("foo_0:0", result.name)

  def testWithLayers(self, distribution):
    def model_fn(features):
      with variable_scope.variable_scope("common"):
        layer1 = core.Dense(1)
        layer1(features)
        layer2 = core.Dense(1)
        layer2(features)
        # This will pause the current thread, and execute the other thread.
        ds_context.get_replica_context().merge_call(lambda _: _)
        layer3 = core.Dense(1)
        layer3(features)
        return [(layer1.kernel, layer1.bias),
                (layer2.kernel, layer2.bias),
                (layer3.kernel, layer3.bias)]

    ds = distribution.distribute_dataset(
        lambda: dataset_ops.Dataset.from_tensors([[1.]]).repeat(10))
    if context.executing_eagerly():
      iterator = ds.make_one_shot_iterator()
    else:
      iterator = ds.make_initializable_iterator()
      self.evaluate([iterator.initializer])

    features = iterator.get_next()

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(
          model_fn, args=(features,))
      suffixes = ["", "_1", "_2"]
      for (kernel, bias), suffix in zip(result, suffixes):
        self.assertIsInstance(kernel, values.MirroredVariable)
        self.assertEqual("common/dense" + suffix + "/kernel:0", kernel.name)
        self.assertIsInstance(bias, values.MirroredVariable)
        self.assertEqual("common/dense" + suffix + "/bias:0", bias.name)

  def testWithVariableAndVariableScope(self, distribution):
    def model_fn():
      v0 = variable_scope.variable(1.0, name="var0", aggregation=None)
      with variable_scope.variable_scope("common"):
        v1 = variable_scope.variable(1.0, name="var1")
        # This will pause the current thread, and execute the other thread.
        ds_context.get_replica_context().merge_call(lambda _: _)
        v2 = variable_scope.variable(
            1.0,
            name="var2",
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        v3 = variable_scope.variable(
            1.0,
            name="var3",
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation=variable_scope.VariableAggregation.MEAN)

      return v0, v1, v2, v3

    with distribution.scope():
      v = variable_scope.variable(1.0, name="var-main0")
      self.assertEqual("var-main0:0", v.name)

      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertEqual(4, len(result))
      v0, v1, v2, v3 = result
      self.assertIsInstance(v0, values.MirroredVariable)
      self.assertEqual("var0:0", v0.name)
      self.assertIsInstance(v1, values.MirroredVariable)
      self.assertEqual("common/var1:0", v1.name)
      self.assertIsInstance(v2, values.ReplicaLocalVariable)
      self.assertEqual("common/var2:0", v2.name)
      self.assertEqual(variable_scope.VariableAggregation.SUM, v2.aggregation)
      self.assertIsInstance(v3, values.MirroredVariable)
      self.assertEqual("common/var3:0", v3.name)
      self.assertEqual(variable_scope.VariableAggregation.MEAN, v3.aggregation)

  def testWithGetVariableAndVariableScope(self, distribution):
    def model_fn():
      v0 = variable_scope.get_variable("var0", [1])
      with variable_scope.variable_scope("common"):
        v1 = variable_scope.get_variable("var1", [1])
        # This will pause the current thread, and execute the other thread.
        ds_context.get_replica_context().merge_call(lambda _: _)
        v2 = variable_scope.get_variable(
            "var2", [1],
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        v3 = variable_scope.get_variable(
            "var3", [1],
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation=variable_scope.VariableAggregation.MEAN)

      return v0, v1, v2, v3

    with distribution.scope():
      with variable_scope.variable_scope("main"):
        v = variable_scope.get_variable("var-main0", [1])
        self.assertEqual("main/var-main0:0", v.name)

        result = distribution.extended.call_for_each_replica(model_fn)
        self.assertEqual(4, len(result))
        v0, v1, v2, v3 = result
        self.assertIsInstance(v0, values.MirroredVariable)
        self.assertEqual("main/var0:0", v0.name)
        self.assertIsInstance(v1, values.MirroredVariable)
        self.assertEqual("main/common/var1:0", v1.name)
        self.assertIsInstance(v2, values.ReplicaLocalVariable)
        self.assertEqual("main/common/var2:0", v2.name)
        self.assertEqual(variable_scope.VariableAggregation.SUM,
                         v2.aggregation)
        self.assertIsInstance(v3, values.MirroredVariable)
        self.assertEqual("main/common/var3:0", v3.name)
        self.assertEqual(variable_scope.VariableAggregation.MEAN,
                         v3.aggregation)

  def testOnlyFirstReplicaUpdatesVariables(self, distribution):
    def create_fn():
      aggregation = variable_scope.VariableAggregation.ONLY_FIRST_REPLICA
      v0 = variable_scope.variable(
          2.0,
          name="on_read",
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=aggregation)
      v1 = variable_scope.variable(
          3.0,
          name="on_write",
          synchronization=variable_scope.VariableSynchronization.ON_WRITE,
          aggregation=aggregation)
      return v0, v1

    devices = ["/device:GPU:0", "/device:CPU:0"]
    with distribution.scope():
      v0, v1 = distribution.extended.call_for_each_replica(create_fn)
      self.evaluate(v0.initializer)
      self.assertEqual(2.0, self.evaluate(v0.get(devices[0])))
      self.assertEqual(2.0, self.evaluate(v0.get(devices[1])))
      self.assertEqual(2.0, self.evaluate(distribution.extended.read_var(v0)))
      self.evaluate(v1.initializer)
      self.assertEqual(3.0, self.evaluate(v1.get(devices[0])))
      self.assertEqual(3.0, self.evaluate(v1.get(devices[1])))
      self.assertEqual(3.0, self.evaluate(distribution.extended.read_var(v1)))

      def replica_id_plus_one():
        return math_ops.cast(_replica_id() + 1, dtype=dtypes.float32)

      # Update using the assign_add member function.
      def update_member_fn():
        update0 = v0.assign_add(5.0 * replica_id_plus_one())
        update1 = v1.assign_add(7.0 * replica_id_plus_one())
        return update0, update1

      update0a, update1a = distribution.extended.call_for_each_replica(
          update_member_fn)

      # Update "sync on read" variable.
      self.evaluate(distribution.group(update0a))
      self.assertEqual(2.0 + 5.0, self.evaluate(v0.get(devices[0])))
      # Writes are not synchronized for "sync on read" variables,
      # so device[1] can end up with a different value.
      self.assertEqual(2.0 + 2*5.0, self.evaluate(v0.get(devices[1])))
      # Always reads from device 0.
      self.assertEqual(2.0 + 5.0, self.evaluate(
          distribution.extended.read_var(v0)))

      # Update "sync on write" variable.
      self.evaluate(distribution.group(update1a))
      self.assertEqual(3.0 + 7.0, self.evaluate(v1.get(devices[0])))
      # Writes are synchronized for v1, only the argument to assign_add on
      # device[0] is used.
      self.assertEqual(3.0 + 7.0, self.evaluate(v1.get(devices[1])))
      self.assertEqual(3.0 + 7.0, self.evaluate(
          distribution.extended.read_var(v1)))

      # Update using state_ops.assign_add global function.
      def update_state_ops_fn():
        update0 = state_ops.assign_add(v0, 11.0 * replica_id_plus_one())
        update1 = state_ops.assign_add(v1, 13.0 * replica_id_plus_one())
        return update0, update1

      update0b, update1b = distribution.extended.call_for_each_replica(
          update_state_ops_fn)
      self.evaluate(distribution.group(update0b))

      # Update "sync on read" variable.
      self.assertEqual(2.0 + 5.0 + 11.0, self.evaluate(v0.get(devices[0])))
      self.assertEqual(2.0 + 2*5.0 + 2*11.0, self.evaluate(v0.get(devices[1])))
      self.assertEqual(2.0 + 5.0 + 11.0, self.evaluate(
          distribution.extended.read_var(v0)))

      # Update "sync on write" variable.
      self.evaluate(distribution.group(update1b))
      self.assertEqual(3.0 + 7.0 + 13.0, self.evaluate(v1.get(devices[0])))
      self.assertEqual(3.0 + 7.0 + 13.0, self.evaluate(v1.get(devices[1])))
      self.assertEqual(3.0 + 7.0 + 13.0, self.evaluate(
          distribution.extended.read_var(v1)))

  def testNoneSynchronizationWithGetVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "`NONE` variable synchronization mode is not "
          "supported with `Mirrored` distribution strategy. Please change "
          "the `synchronization` for variable: v"):
        variable_scope.get_variable(
            "v", [1],
            synchronization=variable_scope.VariableSynchronization.NONE)

  def testNoneSynchronizationWithVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "`NONE` variable synchronization mode is not "
          "supported with `Mirrored` distribution strategy. Please change "
          "the `synchronization` for variable: v"):
        variable_scope.variable(
            1.0,
            name="v",
            synchronization=variable_scope.VariableSynchronization.NONE)

  def testInvalidSynchronizationWithVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable synchronization mode: Invalid for "
          "variable: v"):
        variable_scope.variable(1.0, name="v", synchronization="Invalid")

  def testInvalidAggregationWithGetVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable aggregation mode: invalid for "
          "variable: v"):
        variable_scope.get_variable(
            "v", [1],
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation="invalid")

  def testInvalidAggregationWithVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable aggregation mode: invalid for "
          "variable: v"):
        variable_scope.variable(
            1.0,
            name="v",
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation="invalid")

  def testNonMatchingVariableCreation(self, distribution):
    def model_fn(name):
      v = variable_scope.variable(1.0, name=name)
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      names = values.DistributedValues({
          "/device:CPU:0": "foo",
          "/device:GPU:0": "bar"
      })
      with self.assertRaises(RuntimeError):
        _ = distribution.extended.call_for_each_replica(model_fn, args=(names,))

  def testReplicaLocalVariable(self, distribution):
    all_v_sum = {}
    all_v_mean = {}
    components_sum = {}
    components_mean = {}

    def model_fn():
      replica_id = self.evaluate(_replica_id())
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM)
      v_mean = variable_scope.variable(
          4.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.MEAN)
      self.assertTrue(isinstance(v_sum, values.ReplicaLocalVariable))
      self.assertTrue(isinstance(v_mean, values.ReplicaLocalVariable))
      updates = [v_sum.assign_add(2.0 + replica_id),
                 v_mean.assign(6.0 * replica_id)]
      all_v_sum[replica_id] = v_sum
      all_v_mean[replica_id] = v_mean
      c_sum = v_sum.get()
      c_mean = v_mean.get()
      components_sum[replica_id] = c_sum
      components_mean[replica_id] = c_mean
      self.assertIsNot(v_sum, c_sum)
      self.assertIsNot(v_mean, c_mean)
      return updates, v_sum, v_mean, c_sum, c_mean

    with distribution.scope():
      # Create "sum" and "mean" versions of ReplicaLocalVariables.
      ret_ops, ret_v_sum, ret_v_mean, regrouped_sum, regrouped_mean = (
          distribution.extended.call_for_each_replica(model_fn))
      # Should see the same wrapping instance in all replicas.
      self.assertIs(all_v_sum[0], ret_v_sum)
      self.assertIs(all_v_mean[0], ret_v_mean)
      self.assertIs(all_v_sum[0], all_v_sum[1])
      self.assertIs(all_v_mean[0], all_v_mean[1])

      # Regroup should recover the same wrapper.
      self.assertIs(ret_v_sum, regrouped_sum)
      self.assertIs(ret_v_mean, regrouped_mean)
      self.assertIsNot(components_sum[0], components_sum[1])
      self.assertIsNot(components_mean[0], components_mean[1])

      # Apply updates
      self.evaluate(variables.global_variables_initializer())
      self.evaluate([y for x in ret_ops for y in distribution.unwrap(x)])
      expected_sum = 0.0
      expected_mean = 0.0
      for i, d in enumerate(distribution.extended.worker_devices):
        # Should see different values on different devices.
        v_sum_value = self.evaluate(ret_v_sum.get(d).read_value())
        v_mean_value = self.evaluate(ret_v_mean.get(d).read_value())
        expected = i + 3.0
        self.assertEqual(expected, v_sum_value)
        expected_sum += expected
        expected = i * 6.0
        self.assertEqual(expected, v_mean_value)
        expected_mean += expected
      expected_mean /= len(distribution.extended.worker_devices)

      # Without get(device), should return the value you get by
      # applying the reduction across all replicas (whether you use
      # read_var(), get(), or nothing).
      self.assertEqual(expected_sum, self.evaluate(
          distribution.extended.read_var(ret_v_sum)))
      self.assertEqual(expected_mean, self.evaluate(
          distribution.extended.read_var(ret_v_mean)))
      self.assertEqual(expected_sum, self.evaluate(ret_v_sum.get()))
      self.assertEqual(expected_mean, self.evaluate(ret_v_mean.get()))
      self.assertEqual(expected_sum, self.evaluate(ret_v_sum))
      self.assertEqual(expected_mean, self.evaluate(ret_v_mean))

  # TODO(priyag): Update this test to work in eager mode as well.
  def testDynamicRnnVariables(self, distribution):
    def model_fn():
      inputs = constant_op.constant(2 * [2 * [[0.0, 1.0, 2.0, 3.0, 4.0]]])
      cell_fw = rnn_cell_impl.LSTMCell(300)
      cell_bw = rnn_cell_impl.LSTMCell(300)
      (outputs, _) = rnn.bidirectional_dynamic_rnn(
          cell_fw,
          cell_bw,
          inputs,
          dtype=dtypes.float32)
      return outputs

    with context.graph_mode(), distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      # Two variables are created by the RNN layer.
      self.assertEqual(2, len(result))
      for v in result:
        self.assertIsInstance(v, values.DistributedValues)
        _, v1 = distribution.unwrap(v)
        self.assertStartsWith(v1._op.name, "replica_1/")

  def testReplicaLocalVariableUpdate(self, distribution):
    def model_fn():
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM)
      self.assertTrue(isinstance(v_sum, values.ReplicaLocalVariable))
      return v_sum

    def update(var, value):
      return var.assign(value)

    with distribution.scope():
      ret_v_sum = distribution.extended.call_for_each_replica(model_fn)

      # Initialize variables.
      self.evaluate(variables.global_variables_initializer())
      # Assert that the aggregated value of the replica local vars is the sum
      # of the individual values before running the update ops.
      self.assertEqual(1.0, self.evaluate(ret_v_sum.get(
          distribution.extended.worker_devices[0]).read_value()))
      self.assertEqual(2.0, self.evaluate(ret_v_sum))

      # Apply updates.
      update_ops = distribution.extended.update(
          ret_v_sum, update, args=(5.0,), group=False)
      self.evaluate(update_ops)
      # Assert that the aggregated value of the replica local vars is the sum
      # of the individual values after running the update ops.
      self.assertEqual(5.0, self.evaluate(ret_v_sum.get(
          distribution.extended.worker_devices[0]).read_value()))
      self.assertEqual(10.0, self.evaluate(ret_v_sum))


@combinations.generate(combinations.combine(
    distribution=[
        combinations.mirrored_strategy_with_gpu_and_cpu,
        combinations.core_mirrored_strategy_with_gpu_and_cpu],
    mode=["graph"]))
class MirroredStrategyNameScopeTest(test.TestCase):
  # NOTE(priyag): Names and name scopes are ignored in eager, hence we are not
  # testing this in eager mode.

  def testNameScope(self, distribution):
    def model_fn():
      with ops.name_scope("foo"):
        a = constant_op.constant(1.0, name="a")
        ds_context.get_replica_context().merge_call(lambda _: _)
        b = constant_op.constant(1.0, name="b")
      return a, b

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        result = distribution.extended.call_for_each_replica(model_fn)
        self.assertEqual(2, len(result))
        for v, name in zip(result, ["a", "b"]):
          self.assertIsInstance(v, values.DistributedValues)
          v0, v1 = distribution.unwrap(v)
          self.assertEqual("main/foo/" + name + ":0", v0.name)
          self.assertEqual("main/replica_1/foo/" + name + ":0", v1.name)

  def testWithDefaultName(self, distribution):
    def model_fn():
      with ops.name_scope(None, "foo"):
        a = constant_op.constant(1.0, name="a")
        ds_context.get_replica_context().merge_call(lambda _: _)
        b = constant_op.constant(2.0, name="b")
      return a, b

    with context.graph_mode(), distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertEqual(2, len(result))
      for v, name in zip(result, ["a", "b"]):
        self.assertIsInstance(v, values.DistributedValues)
        v0, v1 = distribution.unwrap(v)
        self.assertEqual("foo/" + name + ":0", v0.name)
        self.assertEqual("replica_1/foo/" + name + ":0", v1.name)

  # variable_scope.variable() respects name scopes when creating
  # variables. On the other hand variable_scope.get_variable() ignores name
  # scopes when creating variables. We test both methods of creating variables
  # to make sure that we have the same variable names in both cases.
  def testNameScopeWithVariable(self, distribution):
    def in_cross_replica(_):
      c = variable_scope.variable(1.0, name="c")
      return c

    def model_fn():
      b = variable_scope.variable(1.0, name="b")
      with ops.name_scope("foo"):
        c = ds_context.get_replica_context().merge_call(in_cross_replica)
      return b, c

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        a = variable_scope.variable(1.0, name="a")
        result = distribution.extended.call_for_each_replica(model_fn)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = distribution.unwrap(a)
      b0, b1 = distribution.unwrap(result_b)
      c0, c1 = distribution.unwrap(result_c)
      self.assertEqual("main/a:0", a0.name)
      self.assertEqual("main/a/replica_1:0", a1.name)
      self.assertEqual("main/b:0", b0.name)
      self.assertEqual("main/b/replica_1:0", b1.name)
      self.assertEqual("main/foo/c:0", c0.name)
      self.assertEqual("main/foo/c/replica_1:0", c1.name)

  def testNameScopeWithGetVariable(self, distribution):
    def in_cross_replica(_):
      c = variable_scope.get_variable("c", [1])
      return c

    def model_fn():
      b = variable_scope.get_variable("b", [1])
      with ops.name_scope("foo"):
        c = ds_context.get_replica_context().merge_call(in_cross_replica)
      return b, c

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        a = variable_scope.get_variable("a", [1])
        result = distribution.extended.call_for_each_replica(model_fn)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = distribution.unwrap(a)
      b0, b1 = distribution.unwrap(result_b)
      c0, c1 = distribution.unwrap(result_c)
      self.assertEqual("a:0", a0.name)
      self.assertEqual("a/replica_1:0", a1.name)
      self.assertEqual("b:0", b0.name)
      self.assertEqual("b/replica_1:0", b1.name)
      self.assertEqual("c:0", c0.name)
      self.assertEqual("c/replica_1:0", c1.name)


@combinations.generate(combinations.combine(
    distribution=[
        combinations.NamedDistribution(
            "Mirrored3Devices",
            # pylint: disable=g-long-lambda
            lambda: mirrored_strategy.MirroredStrategy(
                ["/device:GPU:0", "/device:GPU:1", "/device:CPU:0"]),
            required_gpus=2),
        combinations.NamedDistribution(
            "CoreMirrored3Devices",
            # pylint: disable=g-long-lambda
            lambda: mirrored_strategy.CoreMirroredStrategy(
                ["/device:GPU:0", "/device:GPU:1", "/device:CPU:0"]),
            required_gpus=2)],
    mode=["graph", "eager"]))
class MirroredThreeDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):

  def testThreeDevices(self, distribution):
    def model_fn():
      v = variable_scope.variable(1.0, name="foo")
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertIsInstance(result, values.MirroredVariable)
      self.assertEqual("foo:0", result.name)


@combinations.generate(combinations.combine(
    distribution=[
        combinations.mirrored_strategy_with_gpu_and_cpu,
        combinations.core_mirrored_strategy_with_gpu_and_cpu],
    mode=["graph", "eager"]))
class MirroredVariableUpdateTest(test.TestCase):
  # The following tests check assign, assign_add and assign_sub on Mirrored
  # variables in replica and cross replica context.

  def testAssignMirroredVarReplicaContextWithoutAggregationType(self,
                                                                distribution):
    # Test that we always have an aggregation type set on the mirrored variable
    # if we assign to it in replica mode.
    def var_fn():
      v = variable_scope.variable(1.0, name="foo")
      return v

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        return mirrored_var.assign(5.0)

      with self.assertRaisesRegexp(
          ValueError, "You must specify an aggregation method to update a "
                      "MirroredVariable in Replica Context."):
        self.evaluate(distribution.unwrap(
            distribution.extended.call_for_each_replica(model_fn)))

  def testAssignMirroredVarReplicaContextWithSum(self, distribution):
    # Test that we don't reduce a non-per-replica value with the "sum"
    # aggregation type.
    def var_fn():
      v = variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.SUM)
      return v

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        return mirrored_var.assign(5.0)

      with self.assertRaisesRegexp(
          ValueError, "A non-DistributedValues value 5.0 cannot be reduced "
          "with the given reduce op ReduceOp.SUM."):
        self.evaluate(distribution.unwrap(
            distribution.extended.call_for_each_replica(model_fn)))

  def testAssignMirroredVarCrossDeviceContext(self, distribution):
    def var_fn():
      return variable_scope.variable(1.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign(6.0))
      self.assertEqual(6.0, mirrored_var_result)

  def testAssignMirroredVarReplicaContext(self, distribution):
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(
            ds_context.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign(value)

      self.evaluate(distribution.unwrap(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(0.5, self.evaluate(mirrored_var))

  def testAssignMirroredVarReplicaContextWithSingleValue(self, distribution):
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        return mirrored_var.assign(5.0)

      self.evaluate(distribution.unwrap(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(5.0, self.evaluate(mirrored_var))

  def testAssignAddMirroredVarCrossDeviceContext(self, distribution):
    def var_fn():
      return variable_scope.variable(1.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      # read_value == True
      mirrored_var_result = self.evaluate(
          mirrored_var.assign_add(6.0, read_value=True))
      self.assertEqual(7.0, mirrored_var_result)
      self.assertEqual(7.0, self.evaluate(mirrored_var.get("/device:CPU:0")))
      self.assertEqual(7.0, self.evaluate(mirrored_var.get("/device:GPU:0")))

      # read_value == False
      self.evaluate(mirrored_var.assign_add(2.0, read_value=False))
      self.assertEqual(9.0, self.evaluate(mirrored_var.get("/device:CPU:0")))
      self.assertEqual(9.0, self.evaluate(mirrored_var.get("/device:GPU:0")))

  def testAssignAddMirroredVarReplicaContext(self, distribution):
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(
            ds_context.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign_add(value)

      self.evaluate(distribution.unwrap(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(1.5, self.evaluate(mirrored_var))

  def testAssignAddMirroredVarReplicaContextWithSingleValue(self, distribution):
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        return mirrored_var.assign_add(5.0)

      self.evaluate(distribution.unwrap(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(6.0, self.evaluate(mirrored_var))

  def testAssignSubMirroredVarCrossDeviceContext(self, distribution):
    def var_fn():
      return variable_scope.variable(5.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign_sub(2.0))
      self.assertEqual(3.0, mirrored_var_result)
      self.assertEqual(3.0, self.evaluate(mirrored_var.get("/device:GPU:0")))
      self.assertEqual(3.0, self.evaluate(mirrored_var.get("/device:CPU:0")))

  def testAssignSubMirroredVarReplicaContext(self, distribution):
    def var_fn():
      return variable_scope.variable(
          5.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(
            ds_context.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign_sub(value)

      self.evaluate(distribution.unwrap(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(4.5, self.evaluate(mirrored_var))

  def testAssignSubMirroredVarReplicaContextWithSingleValue(self, distribution):
    def var_fn():
      return variable_scope.variable(
          5.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))

      def model_fn():
        return mirrored_var.assign_sub(1.0)

      self.evaluate(distribution.unwrap(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(4.0, self.evaluate(mirrored_var))


@combinations.generate(combinations.combine(
    distribution=[
        combinations.mirrored_strategy_with_gpu_and_cpu,
        combinations.core_mirrored_strategy_with_gpu_and_cpu],
    mode=["graph", "eager"]))
class MirroredAndReplicaLocalVariableInitializerTest(test.TestCase):

  def testAssignMirroredVarInitializer(self, distribution):
    # This test is not eager compatible since in eager variables are initialized
    # upon construction instead of once the initialization op is run.
    with context.graph_mode():
      def var_fn():
        v = variable_scope.variable(1.0, name="foo")
        return v

      with distribution.scope():
        mirrored_var = distribution.extended.call_for_each_replica(var_fn)
        self.assertIsInstance(mirrored_var, values.MirroredVariable)
        self.assertFalse(self.evaluate(mirrored_var.is_initialized()))
        self.evaluate(mirrored_var.initializer)
        self.assertTrue(self.evaluate(mirrored_var.is_initialized()))

  def testAssignReplicaLocalVarInitializer(self, distribution):
    # This test is not eager compatible since in eager variables are initialized
    # upon construction instead of once the initialization op is run.
    with context.graph_mode():
      def model_fn():
        v_sum = variable_scope.variable(
            1.0,
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        self.assertTrue(isinstance(v_sum, values.ReplicaLocalVariable))
        return v_sum

      with distribution.scope():
        replica_local_var = distribution.extended.call_for_each_replica(
            model_fn)
        self.assertTrue(isinstance(replica_local_var,
                                   values.ReplicaLocalVariable))
        self.assertFalse(self.evaluate(replica_local_var.is_initialized()))
        self.evaluate(replica_local_var.initializer)
        self.assertTrue(self.evaluate(replica_local_var.is_initialized()))


@combinations.generate(combinations.combine(
    distribution=[
        combinations.mirrored_strategy_with_gpu_and_cpu,
        combinations.core_mirrored_strategy_with_gpu_and_cpu],
    mode=["graph", "eager"]))
class ReplicaLocalVariableAssignTest(test.TestCase):

  def testAssignReplicaLocalVarSumAggregation(self, distribution):
    def model_fn():
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM)
      return v_sum

    with distribution.scope():
      replica_local_var = distribution.extended.call_for_each_replica(model_fn)
      self.assertTrue(isinstance(replica_local_var,
                                 values.ReplicaLocalVariable))
      self.evaluate(variables.global_variables_initializer())
      # Each replica has a value of 1.0 assigned to it in replica context.
      # When we read the value using `read_var` we should see the SUM of each of
      # values on each of the replicas.
      self.assertEqual(2.0, self.evaluate(
          distribution.read_var(replica_local_var)))
      # Assigning 6.0 in cross replica context will assign a value of
      # 6.0/num_replicas to each replica.
      tlv_ops = replica_local_var.assign(6.0)
      self.evaluate(tlv_ops)
      # On reading the replica local var we should get the assigned value back.
      # The value on all the replicas are added before being returned by
      # `read_var`.
      self.assertEqual(6.0, self.evaluate(
          distribution.read_var(replica_local_var)))

  def testAssignReplicaLocalVarMeanAggregation(self, distribution):
    def model_fn():
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.MEAN)
      return v_sum

    with distribution.scope():
      replica_local_var = distribution.extended.call_for_each_replica(model_fn)
      self.assertTrue(isinstance(replica_local_var,
                                 values.ReplicaLocalVariable))
      self.evaluate(variables.global_variables_initializer())
      # Each replica has a value of 1.0 assigned to it in replica context.
      # When we read the value using `read_var` we should see the MEAN of values
      # on all replicas which is the value assigned in replica context.
      self.assertEqual(1.0, self.evaluate(
          distribution.read_var(replica_local_var)))
      tlv_ops = replica_local_var.assign(6.0)
      self.evaluate(tlv_ops)
      # On reading the replica local var we should get the MEAN of all values
      # which is equal to the value assigned.
      self.assertEqual(6.0, self.evaluate(
          distribution.read_var(replica_local_var)))


class MockModel(object):

  def __init__(self, two_variables=False):
    self.variables = []
    self.variables.append(variable_scope.variable(1.25, name="dummy_var1"))
    if two_variables:
      self.variables.append(variable_scope.variable(2.0, name="dummy_var2"))

  def __call__(self, factor=2):
    x = factor * self.variables[0]
    if len(self.variables) > 1:
      x += self.variables[1]
    return x


class MiniModel(keras_training.Model):
  """Minimal model for mnist.

  Useful for testing and debugging on slow TPU simulators.
  """

  def __init__(self):
    super(MiniModel, self).__init__(name="")
    self.fc = keras_core.Dense(1, name="fc", kernel_initializer="ones",
                               bias_initializer="ones")

  def call(self, inputs, training=True):
    inputs = array_ops.ones([1, 10])
    return self.fc(inputs)


@combinations.generate(combinations.combine(
    distribution=[
        combinations.mirrored_strategy_with_gpu_and_cpu,
        combinations.core_mirrored_strategy_with_gpu_and_cpu],
    mode=["graph", "eager"]))
class MirroredStrategyDefunTest(test.TestCase):

  def _call_and_check(self, distribution, model_fn, inputs, expected_result,
                      defuns, two_variables=False):
    cpu_dev = device_util.canonicalize("CPU:0")
    gpu_dev = device_util.canonicalize("GPU:0")
    devices = [cpu_dev, gpu_dev]

    with distribution.scope():
      mock_model = MockModel(two_variables)
      self.evaluate(variables.global_variables_initializer())

      result = distribution.extended.call_for_each_replica(
          model_fn, args=[mock_model] + inputs)
      for device in devices:
        device_result = values.select_device(device, result)
        device_expected_result = values.select_device(device, expected_result)
        self.assertAllClose(device_expected_result,
                            self.evaluate(device_result))

      for defun in defuns:
        # PolymorphicFunctions are specialized to the current device stack, so
        # call_for_each has one trace per device. To check that the expected set
        # of variables was accessed on each trace, we first retrieve each
        # device-specific graph function.
        per_replica_graph_functions = (
            distribution.extended.call_for_each_replica(
                defun.get_concrete_function, args=[mock_model] + inputs))
        for device in devices:
          graph_function = per_replica_graph_functions.get(device=device)
          self.assertEqual(set(mock_model.variables),
                           set(graph_function.graph.variables))

  def testVariableInDefun(self, distribution):
    @function.defun
    def times_two(mock_model):
      return mock_model()

    def model_fn(mock_model):
      return times_two(mock_model)

    self._call_and_check(distribution, model_fn, [], 2.5, [times_two])

  def testVariableInNestedDefun(self, distribution):
    @function.defun
    def times_two(mock_model):
      return mock_model()

    @function.defun
    def two_x_plus_one(mock_model):
      return times_two(mock_model) + 1

    def model_fn(mock_model):
      return two_x_plus_one(mock_model)

    self._call_and_check(distribution, model_fn, [], 3.5,
                         [times_two, two_x_plus_one])

  def testTwoVariablesInNestedDefun(self, distribution):
    @function.defun
    def fn1(mock_model):
      return mock_model()

    @function.defun
    def fn2(mock_model):
      return fn1(mock_model) + 1

    def model_fn(mock_model):
      return fn2(mock_model)

    self._call_and_check(distribution, model_fn, [], 5.5, [fn1, fn2],
                         two_variables=True)

  def testGradientTapeOverNestedDefuns(self, distribution):
    @function.defun
    def fn1(mock_model):
      return mock_model()

    @function.defun
    def fn2(mock_model):
      return fn1(mock_model) + 1

    def model_fn(mock_model):
      with backprop.GradientTape(persistent=True) as gtape:
        result = fn2(mock_model)
      grads = gtape.gradient(result,
                             [v.get() for v in mock_model.variables])
      return grads

    self._call_and_check(distribution, model_fn, [], [2.0, 1.0], [fn1, fn2],
                         two_variables=True)

  def testPassPerReplica(self, distribution):
    @function.defun
    def fn1(mock_model, factor):
      return mock_model(factor)

    factors = values.PerReplica({"CPU:0": 5.0, "GPU:0": 3.0})
    expected_result = values.PerReplica({"CPU:0": 5.0 * 1.25,
                                         "GPU:0": 3.0 * 1.25})
    self._call_and_check(distribution, fn1, [factors], expected_result, [fn1])

  def testTrain(self, distribution):
    with distribution.scope():
      mock_model = MiniModel()
      mock_model.call = function.defun(mock_model.call)

      def loss_fn(ctx):
        del ctx
        return mock_model(array_ops.ones([1, 10]))

      gradients_fn = backprop.implicit_grad(loss_fn)
      gradients_fn = optimizer_lib.get_filtered_grad_fn(gradients_fn)
      grads_and_vars = distribution.extended.call_for_each_replica(
          gradients_fn, args=(None,))

      optimizer = gradient_descent.GradientDescentOptimizer(0.25)
      update_ops = optimizer._distributed_apply(distribution, grads_and_vars)  # pylint: disable=protected-access

      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(update_ops)

      updated_var_values = self.evaluate(mock_model.variables)
      # All variables start at 1.0 and get two updates of 0.25.
      self.assertAllEqual(0.5 * np.ones([10, 1]), updated_var_values[0])
      self.assertAllEqual([0.5], updated_var_values[1])


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.CoreMirroredStrategy(
                    num_gpus_per_worker=context.num_gpus()),
                required_gpus=1),
            combinations.NamedDistribution(
                "CoreMirrored",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.CoreMirroredStrategy(
                    num_gpus_per_worker=context.num_gpus()),
                required_gpus=1)
        ],
        mode=["graph"]))
class MultiWorkerMirroredStrategyTest(
    multi_worker_test_base.MultiWorkerTestBase,
    strategy_test_lib.DistributionTestBase):

  def _configure_distribution_strategy(self, distribution):
    cluster_spec = server_lib.ClusterSpec({
        "worker": ["/job:worker/task:0", "/job:worker/task:1"]
    })
    distribution.configure(cluster_spec=cluster_spec)

  def test_num_replicas_in_sync(self, distribution):
    self._configure_distribution_strategy(distribution)
    # We calculate the total number of gpus across the workers(2) specified in
    # the cluster spec.
    self.assertEqual(context.num_gpus() * 2, distribution.num_replicas_in_sync)

  def testMinimizeLossGraph(self, distribution):
    self._configure_distribution_strategy(distribution)
    self._test_minimize_loss_graph(distribution, learning_rate=0.05)

  def testDeviceScope(self, distribution):
    """Test the device scope of multi-worker MirroredStrategy."""
    self._configure_distribution_strategy(distribution)
    with distribution.scope():
      a = constant_op.constant(1.)
      with ops.device("/cpu:0"):
        b = constant_op.constant(1.)
      self.assertEqual(a.device, "/job:worker/task:0")
      self.assertEqual(b.device, "/job:worker/task:0/device:CPU:0")

  def testMakeInputFnIterator(self, distribution):
    self._configure_distribution_strategy(distribution)
    dataset_fn = lambda: dataset_ops.Dataset.range(100)
    num_gpus = context.num_gpus()
    num_workers = 2

    expected_values = [[i+j for j in range(num_gpus)] * num_workers
                       for i in range(0, 100, num_gpus)]

    with context.graph_mode(), self.cached_session() as sess:
      # `expected_input_pipeline_id` is None because the input_fn will be called
      # multiple times, each with a different input_pipeline_id.
      input_fn = self._input_fn_to_test_input_context(
          dataset_fn,
          expected_num_replicas_in_sync=num_workers*num_gpus,
          expected_num_input_pipelines=num_workers,
          expected_input_pipeline_id=None)
      iterator = distribution.make_input_fn_iterator(input_fn)
      self._test_input_fn_iterator(
          iterator, distribution.extended.worker_devices, expected_values, sess)

  def testUpdateConfigProto(self, distribution):
    distribution.configure(cluster_spec={"worker": ["fake1", "fake2"]})

    config_proto = config_pb2.ConfigProto()
    new_config = distribution.update_config_proto(config_proto)

    # Verify isolate_session_state
    self.assertTrue(new_config.isolate_session_state)


class MultiWorkerMirroredStrategyTestWithChief(
    multi_worker_test_base.MultiWorkerTestBase,
    strategy_test_lib.DistributionTestBase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers and 1 chief."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=2, num_ps=0, has_chief=True)
    cls._default_target = "grpc://" + cls._cluster_spec["chief"][0]

  def testMinimizeLossGraph(self):
    strategy = mirrored_strategy.MirroredStrategy(
        num_gpus_per_worker=context.num_gpus())
    strategy.configure(cluster_spec=self._cluster_spec)
    self._test_minimize_loss_graph(strategy, learning_rate=0.05)

  def testMinimizeLossGraphCoreMirroredStrategy(self):
    strategy = mirrored_strategy.CoreMirroredStrategy(
        num_gpus_per_worker=context.num_gpus())
    strategy.configure(cluster_spec=self._cluster_spec)
    self._test_minimize_loss_graph(strategy, learning_rate=0.05)


def _replica_id():
  replica_id = ds_context.get_replica_context().replica_id_in_sync_group
  if not isinstance(replica_id, ops.Tensor):
    replica_id = constant_op.constant(replica_id)
  return replica_id


if __name__ == "__main__":
  test.main()
