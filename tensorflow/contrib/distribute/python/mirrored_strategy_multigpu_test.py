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
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import core
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

  @test_util.run_in_graph_and_eager_modes()
  def testCallAndMergeExceptions(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")
    self._test_call_and_merge_exceptions(self._get_distribution_strategy())

  @test_util.run_in_graph_and_eager_modes()
  def testRunRegroupError(self):

    def run_fn(device_id):
      # Generates a list with different lengths on different devices.
      # Will fail in _regroup() (if more than one device).
      return list(range(device_id))

    dist = self._get_distribution_strategy()
    with dist.scope(), self.assertRaises(AssertionError):
      dist.call_for_each_tower(run_fn, dist.worker_device_index)

  @test_util.run_in_graph_and_eager_modes()
  def testReduceToCpu(self):
    if not GPU_TEST:
      self.skipTest("Not GPU test")

    def run_fn(device_id):
      return device_id

    dist = self._get_distribution_strategy()
    with dist.scope():
      result = dist.call_for_each_tower(run_fn, dist.worker_device_index)
      reduced = dist.reduce("sum", result, destinations="/device:CPU:0")
      unwrapped = dist.unwrap(reduced)
      self.assertEqual(1, len(unwrapped))
      expected = sum(range(len(dist.worker_devices)))
      self.assertEqual(expected, self.evaluate(unwrapped[0]))


@test_util.with_c_api
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
  def testWithGetVariableAndVariableScope(self):
    self._skip_eager_if_gpus_less_than(1)

    def model_fn():
      v0 = variable_scope.get_variable("var-thread0", [1])
      with variable_scope.variable_scope("common"):
        v1 = variable_scope.get_variable("var-thread1", [1])
        # This will pause the current thread, and execute the other thread.
        distribute_lib.get_tower_context().merge_call(lambda _: _)
        v2 = variable_scope.get_variable("var-thread2", [1])

      return v0, v1, v2

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)
    with dist.scope():
      with variable_scope.variable_scope("main"):
        v = variable_scope.get_variable("var-main0", [1])
        self.assertEquals("main/var-main0:0", v.name)

        result = dist.call_for_each_tower(model_fn, run_concurrently=False)
        self.assertEquals(3, len(result))
        v0, v1, v2 = result
        self.assertIsInstance(v0, values.MirroredVariable)
        self.assertEquals("main/var-thread0:0", v0.name)
        self.assertIsInstance(v1, values.MirroredVariable)
        self.assertEquals("main/common/var-thread1:0", v1.name)
        self.assertIsInstance(v2, values.MirroredVariable)
        self.assertEquals("main/common/var-thread2:0", v2.name)

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

    def model_fn(device_id):
      tower_context = distribute_lib.get_tower_context()
      with tower_context.tower_local_var_scope("sum"):
        v_sum = variable_scope.variable(1.0)
      with tower_context.tower_local_var_scope("mean"):
        v_mean = variable_scope.variable(4.0)
      self.assertTrue(isinstance(v_sum, values.TowerLocalVariable))
      self.assertTrue(isinstance(v_mean, values.TowerLocalVariable))
      updates = [v_sum.assign_add(2.0 + device_id),
                 v_mean.assign(6.0 * device_id)]
      all_v_sum[device_id] = v_sum
      all_v_mean[device_id] = v_mean
      return updates, v_sum, v_mean

    dist = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:CPU:0"])

    with dist.scope():
      # Create "sum" and "mean" versions of TowerLocalVariables.
      ret_ops, ret_v_sum, ret_v_mean = dist.call_for_each_tower(
          model_fn, dist.worker_device_index, run_concurrently=False)
      # Should see the same wrapping instance in all towers.
      self.assertIs(all_v_sum[0], ret_v_sum)
      self.assertIs(all_v_mean[0], ret_v_mean)
      for i in range(1, dist.num_towers):
        self.assertIs(all_v_sum[0], all_v_sum[1])
        self.assertIs(all_v_mean[0], all_v_mean[1])

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
      # fetch(), get(), or nothing).
      self.assertEqual(expected_sum, self.evaluate(dist.fetch(ret_v_sum)))
      self.assertEqual(expected_mean, self.evaluate(dist.fetch(ret_v_mean)))
      self.assertEqual(expected_sum, self.evaluate(ret_v_sum.get()))
      self.assertEqual(expected_mean, self.evaluate(ret_v_mean.get()))
      if not context.executing_eagerly():
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


if __name__ == "__main__":
  test.main()
