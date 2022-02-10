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
"""Tests for training.moving_averages when using a DistributionStrategy."""

from absl.testing import parameterized

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.training import moving_averages


all_distributions = [
    strategy_combinations.default_strategy,
    strategy_combinations.one_device_strategy,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
    strategy_combinations.tpu_strategy,
    strategy_combinations.multi_worker_mirrored_2x1_cpu,
    strategy_combinations.multi_worker_mirrored_2x1_gpu,
    strategy_combinations.multi_worker_mirrored_2x2_gpu,
    strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
    strategy_combinations.multi_worker_mirrored_4x1_cpu,
]

all_combinations = combinations.combine(
    distribution=all_distributions, mode=["graph"])

all_combinations_eager = combinations.combine(
    distribution=all_distributions, mode=["eager"], use_function=[True, False])


class AssignMovingAveragesTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(all_combinations)
  def testReplicaModeWithoutZeroDebias(self, distribution):
    replica_id = [0]

    def replica_fn():
      var = variables.Variable([10.0, 11.0])
      val = constant_op.constant([1.0 + replica_id[0], 2.0 - replica_id[0]])
      replica_id[0] += 1
      decay = 0.25
      assign = moving_averages.assign_moving_average(
          var, val, decay, zero_debias=False)
      return var, assign

    with distribution.scope():
      var, assign = distribution.extended.call_for_each_replica(replica_fn)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([10.0, 11.0], self.evaluate(var))
      self.evaluate(distribution.experimental_local_results(assign))
      # Mean of val across calls to replica_fn().
      average_val = [1.0 + 0.5 * (replica_id[0] - 1),
                     2.0 - 0.5 * (replica_id[0] - 1)]
      val_weight = 1.0 - 0.25
      self.assertAllClose(
          [10.0 * 0.25 + average_val[0] * val_weight,
           11.0 * 0.25 + average_val[1] * val_weight],
          self.evaluate(var))

  @combinations.generate(all_combinations)
  def testReplicaMode(self, distribution):
    replica_id = [0]

    def replica_fn():
      var = variables.Variable([0.0, 0.0])
      val = constant_op.constant([1.0 + replica_id[0], 2.0 - replica_id[0]])
      replica_id[0] += 1
      decay = 0.25
      assign = moving_averages.assign_moving_average(var, val, decay)
      return var, assign.op

    with distribution.scope():
      var, assign_op = distribution.extended.call_for_each_replica(replica_fn)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([0.0, 0.0], self.evaluate(var))
      self.evaluate(distribution.experimental_local_results(assign_op))
      # Mean of val across calls to replica_fn().
      average_val = [1.0 + 0.5 * (replica_id[0] - 1),
                     2.0 - 0.5 * (replica_id[0] - 1)]
      self.assertAllClose(average_val, self.evaluate(var))

  @combinations.generate(all_combinations)
  def testCrossDeviceWithoutZeroDebias(self, distribution):
    with distribution.scope():
      var = variables.Variable([10.0, 11.0])
      val = constant_op.constant([1.0, 2.0])
      decay = 0.25
      # NOTE(josh11b): We currently generate an error if val is a PerReplica
      # value.
      assign = moving_averages.assign_moving_average(
          var, val, decay, zero_debias=False)

      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([10.0, 11.0], self.evaluate(var))
      self.evaluate(assign)
      average_val = [1.0, 2.0]
      val_weight = 1.0 - 0.25
      self.assertAllClose(
          [10.0 * 0.25 + average_val[0] * val_weight,
           11.0 * 0.25 + average_val[1] * val_weight],
          self.evaluate(var))
      # Also try assign.op.
      self.evaluate(assign.op)
      orig_weight = 0.25 * 0.25
      val_weight = 1.0 - orig_weight
      self.assertAllClose(
          [10.0 * orig_weight + average_val[0] * val_weight,
           11.0 * orig_weight + average_val[1] * val_weight],
          self.evaluate(var))

  @combinations.generate(all_combinations)
  def testCrossDevice(self, distribution):
    with distribution.scope():
      var = variables.Variable([0.0, 0.0])
      val = variables.Variable([1.0, 2.0])
      decay = 0.25
      # NOTE(josh11b): We currently generate an error if val is a PerReplica
      # value.
      assign = moving_averages.assign_moving_average(var, val, decay)

      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([0.0, 0.0], self.evaluate(var))
      self.evaluate(assign)
      self.assertAllClose([1.0, 2.0], self.evaluate(var))

  @combinations.generate(all_combinations_eager)
  def testUpdateContext(self, distribution, use_function):
    with distribution.scope():
      var1 = variables.Variable([0.0, 0.0])
      var2 = variables.Variable([0.0, 0.0])
      var3 = variables.Variable([0.0, 0.0])

      def update_fn(v, value):
        v.assign_add(value)
        moving_averages.assign_moving_average(var2, [2.0, 4.0], decay=0.25)
        moving_averages.assign_moving_average(
            var3, [2.0, 4.0], decay=0.25, zero_debias=False)

      distribution.extended.update(var1, update_fn, ([1.0, 1.0],))

      self.assertAllClose([2.0, 4.0], var2.read_value())
      self.assertAllClose([1.5, 3.0], var3.read_value())

  @combinations.generate(all_combinations)
  def testAssignVariable(self, distribution):

    def replica_fn():
      var = variables.Variable([10.0, 11.0])
      # Here we expect to check the case when input value are variable.
      val = variables.Variable([1., 2.])
      decay = 0.25
      assign = moving_averages.assign_moving_average(
          var, val, decay, zero_debias=False)
      return var, assign

    with distribution.scope():
      var, assign = distribution.extended.call_for_each_replica(replica_fn)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([10.0, 11.0], self.evaluate(var))
      self.evaluate(distribution.experimental_local_results(assign))
      self.assertAllClose(
          [10 * 0.25 + 1. * (1 - 0.25), 11 * 0.25 + 2. * (1 - 0.25)],
          self.evaluate(var))


class ExponentialMovingAverageTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(all_combinations_eager)
  def testReplicaContextEager(self, distribution, use_function):
    if not use_function and strategy_test_lib.is_tpu_strategy(distribution):
      self.skipTest("TPUStrategy doesn't support pure eager execution.")
    if isinstance(distribution,
                  collective_all_reduce_strategy.CollectiveAllReduceStrategy):
      self.skipTest("b/160194267: Cannot do variable.assign([0.5]) in replica "
                    "context with MultiWorkerMirroredStrategy.")
    with distribution.scope():
      w = variables.Variable([1.0],
                             name="w",
                             aggregation=variables.VariableAggregation.MEAN)
      ema = moving_averages.ExponentialMovingAverage(0.8)

      def fn():

        def _ema_replica_fn_eager():
          ema.apply([w])
          w.assign_sub([0.5])
          ema.apply([w])
          return ema.average(w)

        return distribution.run(_ema_replica_fn_eager)

      if use_function:
        fn = def_function.function(fn)
      ema_w = fn()
    self.assertAllClose(
        self.evaluate(distribution.experimental_local_results(ema_w))[0],
        [0.89999998])

  @combinations.generate(all_combinations_eager)
  def testCrossReplicaContextEager(self, distribution, use_function):
    with distribution.scope():
      w = variables.Variable([1.0],
                             name="w",
                             aggregation=variables.VariableAggregation.MEAN)
      ema = moving_averages.ExponentialMovingAverage(0.8)

      def fn():
        ema.apply([w])
        w.assign_sub([0.5])
        ema.apply([w])
        return ema.average(w)

      if use_function:
        fn = def_function.function(fn)
      avg = fn()
    self.assertAllClose(
        self.evaluate(distribution.experimental_local_results(avg))[0],
        [0.89999998])

  def _ema_replica_fn_graph(self):
    w = variables.Variable([1.0],
                           name="w",
                           aggregation=variables.VariableAggregation.MEAN)
    ema = moving_averages.ExponentialMovingAverage(0.8)
    w_apply = ema.apply([w])
    w_assign = w.assign_sub([0.5])
    return w_assign, w_apply, ema.average(w)

  @combinations.generate(all_combinations)
  def testReplicaContextGraph(self, distribution):
    if strategy_test_lib.is_tpu_strategy:
      self.skipTest("b/139550827: Cannot do variable.assign in replica context "
                    "of TPUStrategy")
    if isinstance(distribution,
                  collective_all_reduce_strategy.CollectiveAllReduceStrategy):
      self.skipTest("b/160194267: Cannot do variable.assign([0.5]) in replica "
                    "context with MultiWorkerMirroredStrategy.")
    with distribution.scope():
      w_assign, w_apply, ema_w = distribution.run(
          self._ema_replica_fn_graph)
    self.assertEqual(ema_w.name, "w/ExponentialMovingAverage:0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(distribution.experimental_local_results(w_apply))
    self.evaluate(distribution.experimental_local_results(w_assign))
    self.evaluate(distribution.experimental_local_results(w_apply))
    self.assertAllClose(
        self.evaluate(distribution.experimental_local_results(ema_w))[0],
        [0.89999998])

  @combinations.generate(all_combinations)
  def testCrossReplicaContextGraph(self, distribution):
    with distribution.scope():
      w_assign, w_apply, ema_w = self._ema_replica_fn_graph()
    self.assertEqual(ema_w.name, "w/ExponentialMovingAverage:0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(distribution.experimental_local_results(w_apply))
    self.evaluate(distribution.experimental_local_results(w_assign))
    self.evaluate(distribution.experimental_local_results(w_apply))
    self.assertAllClose(
        self.evaluate(distribution.experimental_local_results(ema_w))[0],
        [0.89999998])


if __name__ == "__main__":
  # TODO(b/172304955): enable logical devices.
  test_util.main(config_logical_devices=False)
