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

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import strategy_test_lib
from tensorflow.contrib.distribute.python import values
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import core
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import distribute as distribute_lib


GPU_TEST = "test_gpu" in sys.argv[0]


class MirroredTwoDeviceDistributionTest(strategy_test_lib.DistributionTestBase):

  def _get_distribution_strategy(self):
    devices = ["/device:CPU:0", "/device:GPU:0"]
    if GPU_TEST:
      self.assertGreater(context.num_gpus(), 0)
      if context.num_gpus() > 1:
        devices = ["/device:GPU:0", "/device:GPU:1"]
    print(self.id().split(".")[-1], "devices:", ", ".join(devices))
    return mirrored_strategy.MirroredStrategy(devices)

  def testMinimizeLossEager(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")
    self._test_minimize_loss_eager(self._get_distribution_strategy())

  def testMinimizeLossGraph(self):
    soft_placement = not GPU_TEST
    print("testMinimizeLossGraph soft_placement:", soft_placement)
    self._test_minimize_loss_graph(
        self._get_distribution_strategy(), soft_placement=soft_placement)

  def testMapReduce(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")
    self._test_map_reduce(self._get_distribution_strategy())

  def testDeviceIndex(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")
    self._test_device_index(self._get_distribution_strategy())

  def testTowerId(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")
    self._test_tower_id(self._get_distribution_strategy())

  def testNumTowers(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")
    self.assertEqual(2, self._get_distribution_strategy().num_towers)

  @test_util.run_in_graph_and_eager_modes
  def testCallAndMergeExceptions(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")
    self._test_call_and_merge_exceptions(self._get_distribution_strategy())

  @test_util.run_in_graph_and_eager_modes
  def testRunRegroupError(self):

    def run_fn(device_id):
      # Generates a list with different lengths on different devices.
      # Will fail in _regroup() (if more than one device).
      return list(range(device_id))

    dist = self._get_distribution_strategy()
    with dist.scope(), self.assertRaises(AssertionError):
      dist.call_for_each_tower(run_fn, dist.worker_device_index)

  @test_util.run_in_graph_and_eager_modes
  def testReduceToCpu(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")

    def run_fn(device_id):
      return device_id

    dist = self._get_distribution_strategy()
    with dist.scope():
      result = dist.call_for_each_tower(run_fn, dist.worker_device_index)
      reduced = dist.reduce(
          variable_scope.VariableAggregation.SUM,
          result,
          destinations="/device:CPU:0")
      unwrapped = dist.unwrap(reduced)
      self.assertEqual(1, len(unwrapped))
      expected = sum(range(len(dist.worker_devices)))
      self.assertEqual(expected, self.evaluate(unwrapped[0]))

  @test_util.run_in_graph_and_eager_modes()
  def testReduceToMultipleDestinations(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")

    devices = ["/device:GPU:0"]
    if GPU_TEST:
      self.assertGreater(context.num_gpus(), 0)
    print(self.id().split(".")[-1], "devices:", ", ".join(devices))

    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      reduced = dist.reduce(
          variable_scope.VariableAggregation.SUM,
          1.0,
          destinations=["/device:CPU:0", "/device:GPU:0"])
      unwrapped = dist.unwrap(reduced)
      self.assertEqual(2, len(unwrapped))
      self.assertEqual(1.0, self.evaluate(unwrapped[0]))


class MirroredStrategyVariableCreationTest(test.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  def _skip_eager_if_gpus_less_than(self, num_gpus):
    if context.num_gpus() < num_gpus and context.executing_eagerly():
      self.skipTest("Enough GPUs not available for this test in eager mode.")

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSingleVariable(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn():
      # This variable should be created only once across the threads because of
      # special variable_creator functions used by `dist.call_for_each_tower`.
      v = variable_scope.variable(1.0, name="foo")
      distribute_lib.get_tower_context().merge_call(lambda _: _)
      return v

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      self.assertIsInstance(result, values.MirroredVariable)
      self.assertEquals("foo:0", result.name)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testUnnamedVariable(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn():
      v = variable_scope.variable(1.0)
      distribute_lib.get_tower_context().merge_call(lambda _: _)
      return v

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      self.assertIsInstance(result, values.MirroredVariable)
      # Default name of "Variable" will be used.
      self.assertEquals("Variable:0", result.name)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testMultipleVariables(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn():
      vs = []
      for i in range(5):
        vs.append(variable_scope.variable(1.0, name="foo" + str(i)))
      distribute_lib.get_tower_context().merge_call(lambda _: _)
      return vs

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      for i, v in enumerate(result):
        self.assertIsInstance(v, values.MirroredVariable)
        self.assertEquals("foo" + str(i) + ":0", v.name)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testMultipleVariablesWithSameCanonicalName(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn():
      vs = []
      vs.append(variable_scope.variable(1.0, name="foo/bar"))
      vs.append(variable_scope.variable(1.0, name="foo_1/bar"))
      vs.append(variable_scope.variable(1.0, name="foo_1/bar_1"))
      vs.append(variable_scope.variable(1.0, name="foo/bar_1"))
      distribute_lib.get_tower_context().merge_call(lambda _: _)
      return vs

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      for v in result:
        self.assertIsInstance(v, values.MirroredVariable)
      self.assertEquals(4, len(result))
      self.assertEquals("foo/bar:0", result[0].name)
      self.assertEquals("foo_1/bar:0", result[1].name)
      self.assertEquals("foo_1/bar_1:0", result[2].name)
      self.assertEquals("foo/bar_1:0", result[3].name)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableWithSameCanonicalNameAcrossThreads(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn(device_id):
      v = variable_scope.variable(1.0, name="foo_" + str(device_id))
      distribute_lib.get_tower_context().merge_call(lambda _: _)
      return v

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      result = dist.call_for_each_tower(
          model_fn, dist.worker_device_index, run_concurrently=False)
      self.assertIsInstance(result, values.MirroredVariable)
      # The resulting mirrored variable will use the name from the first device.
      self.assertEquals("foo_0:0", result.name)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testWithLayers(self):
    self._skip_eager_if_gpus_less_than(1)
    def model_fn(features):
      with variable_scope.variable_scope("common"):
        layer1 = core.Dense(1)
        layer1(features)
        layer2 = core.Dense(1)
        layer2(features)
        # This will pause the current thread, and execute the other thread.
        distribute_lib.get_tower_context().merge_call(lambda _: _)
        layer3 = core.Dense(1)
        layer3(features)
        return [(layer1.kernel, layer1.bias),
                (layer2.kernel, layer2.bias),
                (layer3.kernel, layer3.bias)]

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])
    features = dist.distribute_dataset(
        lambda: dataset_ops.Dataset.from_tensors([[1.]]).repeat(10)
    ).make_one_shot_iterator().get_next()

    with dist.scope():
      result = dist.call_for_each_tower(
          model_fn, features, run_concurrently=False)
      suffixes = ["", "_1", "_2"]
      for (kernel, bias), suffix in zip(result, suffixes):
        self.assertIsInstance(kernel, values.MirroredVariable)
        self.assertEquals("common/dense" + suffix + "/kernel:0", kernel.name)
        self.assertIsInstance(bias, values.MirroredVariable)
        self.assertEquals("common/dense" + suffix + "/bias:0", bias.name)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testWithVariableAndVariableScope(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn():
      v0 = variable_scope.variable(1.0, name="var0", aggregation=None)
      with variable_scope.variable_scope("common"):
        v1 = variable_scope.variable(1.0, name="var1")
        # This will pause the current thread, and execute the other thread.
        distribute_lib.get_tower_context().merge_call(lambda _: _)
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

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      v = variable_scope.variable(1.0, name="var-main0")
      self.assertEquals("var-main0:0", v.name)

      result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      self.assertEquals(4, len(result))
      v0, v1, v2, v3 = result
      self.assertIsInstance(v0, values.MirroredVariable)
      self.assertEquals("var0:0", v0.name)
      self.assertIsInstance(v1, values.MirroredVariable)
      self.assertEquals("common/var1:0", v1.name)
      self.assertIsInstance(v2, values.TowerLocalVariable)
      self.assertEquals("common/var2:0", v2.name)
      self.assertEquals(variable_scope.VariableAggregation.SUM, v2.aggregation)
      self.assertIsInstance(v3, values.MirroredVariable)
      self.assertEquals("common/var3:0", v3.name)
      self.assertEquals(variable_scope.VariableAggregation.MEAN, v3.aggregation)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testWithGetVariableAndVariableScope(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn():
      v0 = variable_scope.get_variable("var0", [1])
      with variable_scope.variable_scope("common"):
        v1 = variable_scope.get_variable("var1", [1])
        # This will pause the current thread, and execute the other thread.
        distribute_lib.get_tower_context().merge_call(lambda _: _)
        v2 = variable_scope.get_variable(
            "var2", [1],
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        v3 = variable_scope.get_variable(
            "var3", [1],
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation=variable_scope.VariableAggregation.MEAN)

      return v0, v1, v2, v3

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      with variable_scope.variable_scope("main"):
        v = variable_scope.get_variable("var-main0", [1])
        self.assertEquals("main/var-main0:0", v.name)

        result = dist.call_for_each_tower(model_fn, run_concurrently=False)
        self.assertEquals(4, len(result))
        v0, v1, v2, v3 = result
        self.assertIsInstance(v0, values.MirroredVariable)
        self.assertEquals("main/var0:0", v0.name)
        self.assertIsInstance(v1, values.MirroredVariable)
        self.assertEquals("main/common/var1:0", v1.name)
        self.assertIsInstance(v2, values.TowerLocalVariable)
        self.assertEquals("main/common/var2:0", v2.name)
        self.assertEquals(variable_scope.VariableAggregation.SUM,
                          v2.aggregation)
        self.assertIsInstance(v3, values.MirroredVariable)
        self.assertEquals("main/common/var3:0", v3.name)
        self.assertEquals(variable_scope.VariableAggregation.MEAN,
                          v3.aggregation)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testNoneSynchronizationWithGetVariable(self):
    self._skip_eager_if_gpus_less_than(1)
    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      with self.assertRaisesRegexp(
          ValueError, "`NONE` variable synchronization mode is not "
          "supported with `Mirrored` distribution strategy. Please change "
          "the `synchronization` for variable: v"):
        variable_scope.get_variable(
            "v", [1],
            synchronization=variable_scope.VariableSynchronization.NONE)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testNoneSynchronizationWithVariable(self):
    self._skip_eager_if_gpus_less_than(1)
    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      with self.assertRaisesRegexp(
          ValueError, "`NONE` variable synchronization mode is not "
          "supported with `Mirrored` distribution strategy. Please change "
          "the `synchronization` for variable: v"):
        variable_scope.variable(
            1.0,
            name="v",
            synchronization=variable_scope.VariableSynchronization.NONE)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testInvalidSynchronizationWithVariable(self):
    self._skip_eager_if_gpus_less_than(1)
    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable synchronization mode: Invalid for "
          "variable: v"):
        variable_scope.variable(1.0, name="v", synchronization="Invalid")

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testInvalidAggregationWithGetVariable(self):
    self._skip_eager_if_gpus_less_than(1)
    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable aggregation mode: invalid for "
          "variable: v"):
        variable_scope.get_variable(
            "v", [1],
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation="invalid")

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testInvalidAggregationWithVariable(self):
    self._skip_eager_if_gpus_less_than(1)
    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable aggregation mode: invalid for "
          "variable: v"):
        variable_scope.variable(
            1.0,
            name="v",
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation="invalid")

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testThreeDevices(self):
    self._skip_eager_if_gpus_less_than(2)

    def model_fn():
      v = variable_scope.variable(1.0, name="foo")
      distribute_lib.get_tower_context().merge_call(lambda _: _)
      return v

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:GPU:1", "/device:CPU:0"])

    with dist.scope():
      result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      self.assertIsInstance(result, values.MirroredVariable)
      self.assertEquals("foo:0", result.name)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testNonMatchingVariableCreation(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn(name):
      v = variable_scope.variable(1.0, name=name)
      distribute_lib.get_tower_context().merge_call(lambda _: _)
      return v

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      names = values.DistributedValues({
          "/device:CPU:0": "foo",
          "/device:GPU:0": "bar"
      })
      with self.assertRaises(RuntimeError):
        _ = dist.call_for_each_tower(model_fn, names, run_concurrently=False)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testTowerLocalVariable(self):
    self._skip_eager_if_gpus_less_than(1)

    all_v_sum = {}
    all_v_mean = {}
    components_sum = {}
    components_mean = {}

    def model_fn(device_id):
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM)
      v_mean = variable_scope.variable(
          4.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.MEAN)
      self.assertTrue(isinstance(v_sum, values.TowerLocalVariable))
      self.assertTrue(isinstance(v_mean, values.TowerLocalVariable))
      updates = [v_sum.assign_add(2.0 + device_id),
                 v_mean.assign(6.0 * device_id)]
      all_v_sum[device_id] = v_sum
      all_v_mean[device_id] = v_mean
      c_sum = v_sum.get()
      c_mean = v_mean.get()
      components_sum[device_id] = c_sum
      components_mean[device_id] = c_mean
      self.assertIsNot(v_sum, c_sum)
      self.assertIsNot(v_mean, c_mean)
      return updates, v_sum, v_mean, c_sum, c_mean

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      # Create "sum" and "mean" versions of TowerLocalVariables.
      ret_ops, ret_v_sum, ret_v_mean, regrouped_sum, regrouped_mean = (
          dist.call_for_each_tower(
              model_fn, dist.worker_device_index, run_concurrently=False))
      # Should see the same wrapping instance in all towers.
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
      self.evaluate([y for x in ret_ops for y in dist.unwrap(x)])
      expected_sum = 0.0
      expected_mean = 0.0
      for i, d in enumerate(dist.worker_devices):
        # Should see different values on different devices.
        v_sum_value = self.evaluate(ret_v_sum.get(d).read_value())
        v_mean_value = self.evaluate(ret_v_mean.get(d).read_value())
        expected = i + 3.0
        self.assertEqual(expected, v_sum_value)
        expected_sum += expected
        expected = i * 6.0
        self.assertEqual(expected, v_mean_value)
        expected_mean += expected
      expected_mean /= len(dist.worker_devices)

      # Without get(device), should return the value you get by
      # applying the reduction across all towers (whether you use
      # read_var(), get(), or nothing).
      self.assertEqual(expected_sum, self.evaluate(dist.read_var(ret_v_sum)))
      self.assertEqual(expected_mean, self.evaluate(dist.read_var(ret_v_mean)))
      self.assertEqual(expected_sum, self.evaluate(ret_v_sum.get()))
      self.assertEqual(expected_mean, self.evaluate(ret_v_mean.get()))
      self.assertEqual(expected_sum, self.evaluate(ret_v_sum))
      self.assertEqual(expected_mean, self.evaluate(ret_v_mean))

  # NOTE(priyag): Names and name scopes are ignored in eager, hence we are not
  # testing this in eager mode.

  def testNameScope(self):
    def model_fn():
      with ops.name_scope("foo"):
        a = constant_op.constant(1.0, name="a")
        distribute_lib.get_tower_context().merge_call(lambda _: _)
        b = constant_op.constant(1.0, name="b")
      return a, b

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with context.graph_mode(), dist.scope():
      with ops.name_scope("main"):
        result = dist.call_for_each_tower(model_fn, run_concurrently=False)
        self.assertEquals(2, len(result))
        for v, name in zip(result, ["a", "b"]):
          self.assertIsInstance(v, values.DistributedValues)
          v0, v1 = dist.unwrap(v)
          self.assertEquals("main/foo/" + name + ":0", v0.name)
          self.assertEquals("main/tower_1/foo/" + name + ":0", v1.name)

  def testWithDefaultName(self):
    def model_fn():
      with ops.name_scope(None, "foo"):
        a = constant_op.constant(1.0, name="a")
        distribute_lib.get_tower_context().merge_call(lambda _: _)
        b = constant_op.constant(2.0, name="b")
      return a, b

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with context.graph_mode(), dist.scope():
      result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      self.assertEquals(2, len(result))
      for v, name in zip(result, ["a", "b"]):
        self.assertIsInstance(v, values.DistributedValues)
        v0, v1 = dist.unwrap(v)
        self.assertEquals("foo/" + name + ":0", v0.name)
        self.assertEquals("tower_1/foo/" + name + ":0", v1.name)

  # variable_scope.variable() respects name scopes when creating
  # variables. On the other hand variable_scope.get_variable() ignores name
  # scopes when creating variables. We test both methods of creating variables
  # to make sure that we have the same variable names in both cases.
  def testNameScopeWithVariable(self):
    def in_cross_tower(_):
      c = variable_scope.variable(1.0, name="c")
      return c

    def model_fn():
      b = variable_scope.variable(1.0, name="b")
      with ops.name_scope("foo"):
        c = distribute_lib.get_tower_context().merge_call(in_cross_tower)
      return b, c

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with context.graph_mode(), dist.scope():
      with ops.name_scope("main"):
        a = variable_scope.variable(1.0, name="a")
        result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = dist.unwrap(a)
      b0, b1 = dist.unwrap(result_b)
      c0, c1 = dist.unwrap(result_c)
      self.assertEquals("main/a:0", a0.name)
      self.assertEquals("main/a/replica_1:0", a1.name)
      self.assertEquals("main/b:0", b0.name)
      self.assertEquals("main/b/replica_1:0", b1.name)
      self.assertEquals("main/foo/c:0", c0.name)
      self.assertEquals("main/foo/c/replica_1:0", c1.name)

  def testNameScopeWithGetVariable(self):
    def in_cross_tower(_):
      c = variable_scope.get_variable("c", [1])
      return c

    def model_fn():
      b = variable_scope.get_variable("b", [1])
      with ops.name_scope("foo"):
        c = distribute_lib.get_tower_context().merge_call(in_cross_tower)
      return b, c

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with context.graph_mode(), dist.scope():
      with ops.name_scope("main"):
        a = variable_scope.get_variable("a", [1])
        result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = dist.unwrap(a)
      b0, b1 = dist.unwrap(result_b)
      c0, c1 = dist.unwrap(result_c)
      self.assertEquals("a:0", a0.name)
      self.assertEquals("a/replica_1:0", a1.name)
      self.assertEquals("b:0", b0.name)
      self.assertEquals("b/replica_1:0", b1.name)
      self.assertEquals("c:0", c0.name)
      self.assertEquals("c/replica_1:0", c1.name)

  def testDynamicRnnVariables(self):
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

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with context.graph_mode(), dist.scope():
      result = dist.call_for_each_tower(model_fn, run_concurrently=False)
      # Two variables are created by the RNN layer.
      self.assertEquals(2, len(result))
      for v in result:
        self.assertIsInstance(v, values.DistributedValues)
        _, v1 = dist.unwrap(v)
        self.assertStartsWith(v1.name, "tower_1/")

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testTowerLocalVariableUpdate(self):
    with context.graph_mode():

      def model_fn():
        v_sum = variable_scope.variable(
            1.0,
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        self.assertTrue(isinstance(v_sum, values.TowerLocalVariable))
        return v_sum

      dist = mirrored_strategy.MirroredStrategy(
          ["/device:GPU:0", "/device:GPU:1"])

      def update(var, value):
        return var.assign(value)

      with dist.scope():
        ret_v_sum = dist.call_for_each_tower(model_fn, run_concurrently=False)
        update_ops = dist.unwrap(dist.update(ret_v_sum, update, 5.0))

        # Initialize variables.
        self.evaluate(variables.global_variables_initializer())
        # Assert that the aggregated value of the tower local vars is the sum of
        # the individual values before running the update ops.
        self.assertEquals(1.0, self.evaluate(
            ret_v_sum.get(dist._devices[0]).read_value()))
        self.assertEquals(2.0, self.evaluate(ret_v_sum))

        # Apply updates.
        self.evaluate(update_ops)
        # Assert that the aggregated value of the tower local vars is the sum of
        # the individual values after running the update ops.
        self.assertEquals(5.0, self.evaluate(
            ret_v_sum.get(dist._devices[0]).read_value()))
        self.assertEquals(10.0, self.evaluate(ret_v_sum))


class MirroredVariableUpdateTest(test.TestCase):
  # The following tests check assign, assign_add and assign_sub on Mirrored
  # variables in tower and cross tower context.
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  def _skip_eager_if_gpus_less_than(self, num_gpus):
    if context.num_gpus() < num_gpus and context.executing_eagerly():
      self.skipTest("Enough GPUs not available for this test in eager mode.")

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testAssignMirroredVarTowerContextWithoutAggregationType(self):
    # Test that we always have an aggregation type set on the mirrored variable
    # if we assign to it in tower mode.
    self._skip_eager_if_gpus_less_than(1)
    def var_fn():
      v = variable_scope.variable(1.0, name="foo")
      return v

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      mirrored_var = dist.call_for_each_tower(var_fn, run_concurrently=False)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        return mirrored_var.assign(5.0)

      with self.assertRaisesRegexp(
          ValueError, "You must specify an aggregation method to update a "
                      "MirroredVariable in Tower Context."):
        self.evaluate(dist.unwrap(dist.call_for_each_tower(model_fn)))

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testAssignMirroredVarTowerContextWithSum(self):
    # Test that we don't reduce a non-per-device value with the "sum"
    # aggregation type.
    self._skip_eager_if_gpus_less_than(1)
    def var_fn():
      v = variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.SUM)
      return v

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      mirrored_var = dist.call_for_each_tower(var_fn, run_concurrently=False)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        return mirrored_var.assign(5.0)

      with self.assertRaisesRegexp(
          ValueError, "A non PerDevice value cannot be reduced with the given "
          "aggregation."):
        self.evaluate(dist.unwrap(dist.call_for_each_tower(model_fn)))

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testAssignMirroredVarCrossTowerContext(self):
    self._skip_eager_if_gpus_less_than(1)
    def var_fn():
      return variable_scope.variable(1.0, name="foo")

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      mirrored_var = dist.call_for_each_tower(var_fn, run_concurrently=False)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEquals(1.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign(6.0))
      self.assertEquals(6.0, mirrored_var_result)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testAssignMirroredVarTowerContext(self):
    self._skip_eager_if_gpus_less_than(1)
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      mirrored_var = dist.call_for_each_tower(var_fn, run_concurrently=False)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEquals(1.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(distribute_lib.get_tower_context().tower_id,
                              mirrored_var.dtype)
        return mirrored_var.assign(value)

      self.evaluate(dist.unwrap(dist.call_for_each_tower(
          model_fn, run_concurrently=False)))
      self.assertEquals(0.5, self.evaluate(mirrored_var))

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testAssignAddMirroredVarCrossTowerContext(self):
    self._skip_eager_if_gpus_less_than(1)
    def var_fn():
      return variable_scope.variable(1.0, name="foo")

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      mirrored_var = dist.call_for_each_tower(var_fn, run_concurrently=False)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEquals(1.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign_add(6.0))
      self.assertEquals(7.0, mirrored_var_result)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testAssignAddMirroredVarTowerContext(self):
    self._skip_eager_if_gpus_less_than(1)
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      mirrored_var = dist.call_for_each_tower(var_fn, run_concurrently=False)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEquals(1.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(distribute_lib.get_tower_context().tower_id,
                              mirrored_var.dtype)
        return mirrored_var.assign_add(value)

      self.evaluate(dist.unwrap(dist.call_for_each_tower(
          model_fn, run_concurrently=False)))
      self.assertEquals(1.5, self.evaluate(mirrored_var))

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testAssignSubMirroredVarCrossTowerContext(self):
    self._skip_eager_if_gpus_less_than(1)
    def var_fn():
      return variable_scope.variable(5.0, name="foo")

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      mirrored_var = dist.call_for_each_tower(var_fn, run_concurrently=False)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEquals(5.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign_sub(2.0))
      self.assertEquals(3.0, mirrored_var_result)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testAssignSubMirroredVarTowerContext(self):
    self._skip_eager_if_gpus_less_than(1)
    def var_fn():
      return variable_scope.variable(
          5.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      mirrored_var = dist.call_for_each_tower(var_fn, run_concurrently=False)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEquals(5.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(distribute_lib.get_tower_context().tower_id,
                              mirrored_var.dtype)
        return mirrored_var.assign_sub(value)

      self.evaluate(dist.unwrap(dist.call_for_each_tower(
          model_fn, run_concurrently=False)))
      self.assertEquals(4.5, self.evaluate(mirrored_var))


class MirroredAndTowerLocalVariableInitializerTest(test.TestCase):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  def testAssignMirroredVarInitializer(self):
    # This test is not eager compatible since in eager variables are initialized
    # upon construction instead of once the initialization op is run.
    with context.graph_mode():
      def var_fn():
        v = variable_scope.variable(1.0, name="foo")
        return v

      dist = mirrored_strategy.MirroredStrategy(
          ["/device:GPU:0", "/device:CPU:0"])

      with dist.scope():
        mirrored_var = dist.call_for_each_tower(var_fn)
        self.assertIsInstance(mirrored_var, values.MirroredVariable)
        self.assertFalse(self.evaluate(mirrored_var.is_initialized()))
        self.evaluate(mirrored_var.initializer)
        self.assertTrue(self.evaluate(mirrored_var.is_initialized()))

  def testAssignTowerLocalVarInitializer(self):
    # This test is not eager compatible since in eager variables are initialized
    # upon construction instead of once the initialization op is run.
    with context.graph_mode():
      def model_fn():
        v_sum = variable_scope.variable(
            1.0,
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        self.assertTrue(isinstance(v_sum, values.TowerLocalVariable))
        return v_sum

      dist = mirrored_strategy.MirroredStrategy(
          ["/device:GPU:0", "/device:CPU:0"])

      with dist.scope():
        tower_local_var = dist.call_for_each_tower(model_fn)
        self.assertTrue(isinstance(tower_local_var, values.TowerLocalVariable))
        self.assertFalse(self.evaluate(tower_local_var.is_initialized()))
        self.evaluate(tower_local_var.initializer)
        self.assertTrue(self.evaluate(tower_local_var.is_initialized()))

if __name__ == "__main__":
  test.main()
