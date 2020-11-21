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

import itertools

import uuid
from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.training import checkpoint_management as ckpt_manager
from tensorflow.python.training.tracking import util as trackable_utils


_TPU_STRATEGIES = (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)


def strategy_and_run_tf_function_combinations():
  # Test the combination of different strategies and whether a tf.function
  # is passed into strategy.run."""
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
      ],
      mode=["graph", "eager"],
      experimental_run_tf_function=[True, False],
      use_var_policy=[True, False]) + combinations.combine(
          distribution=[
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
          ],
          mode=["graph", "eager"],
          experimental_run_tf_function=[True],
          use_var_policy=[True, False])


def strategy_with_var_policy():
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ],
      mode=["graph", "eager"],
      use_var_policy=[True, False])


class OnWriteVariableSync(test.TestCase, parameterized.TestCase):

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
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      if (not cross_replica and aggregation ==
          variables_lib.VariableAggregation.SUM):
        continue
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(self.evaluate(component.read_value()),
                            self.evaluate(array_ops.ones_like(component)))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignOnWriteVar(self, distribution, experimental_run_tf_function):

    with distribution.scope():
      v_to_assign = variable_scope.variable(
          2., aggregation=variables_lib.VariableAggregation.MEAN)
      v_to_assign_sub = variable_scope.variable(
          -2., aggregation=variables_lib.VariableAggregation.MEAN)

    def assign(fn, v, update_value, cross_replica):
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return distribution.experimental_local_results(
            distribution.run(update_fn))

    updates = [("assign", v_to_assign), ("assign_add", v_to_assign),
               ("assign_sub", v_to_assign_sub)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      if aggregation == variables_lib.VariableAggregation.SUM:
        continue
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(2.0, self.evaluate(component.read_value()))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignPerReplicaVal(self, distribution, experimental_run_tf_function):

    if isinstance(distribution, _TPU_STRATEGIES):
      self.skipTest("Assigning PerReplica values is not supported. See"
                    " sponge/80ba41f8-4220-4516-98ce-bbad48f9f11a.")

    with distribution.scope():
      per_replica_value = values.PerReplica(
          [constant_op.constant(2.0),
           constant_op.constant(2.0)])
      per_replica_sub_value = values.PerReplica(
          [constant_op.constant(-2.0),
           constant_op.constant(-2.0)])

    def assign(fn, v, update_value, cross_replica):
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return distribution.experimental_local_results(
            distribution.run(update_fn))

    updates = [("assign", per_replica_value), ("assign_add", per_replica_value),
               ("assign_sub", per_replica_sub_value)]
    # We don't support assigning PerReplica valus to vars in replica context
    # with aggregation=NONE.
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      if cross_replica:
        # We don't support assigning PerReplica values to MirroredVariables in
        # cross replica context
        continue
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      if aggregation == variables_lib.VariableAggregation.SUM:
        expected = 4.0
      else:
        expected = 2.0
      for component in v._values:
        self.assertAllEqual(expected, self.evaluate(component.read_value()))

  @combinations.generate(strategy_with_var_policy())
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
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      with distribution.scope():
        v = variable_scope.variable(
            2.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())

      if experimental_run_tf_function:
        read_var_fn = def_function.function(v.read_value)
      else:
        read_var_fn = v.read_value

      results = read_var_fn()
      for component in v._values:
        self.assertEqual(self.evaluate(component.read_value()),
                         self.evaluate(results))

  @combinations.generate(strategy_with_var_policy())
  def testAssignOutOfScope(self, distribution):
    with distribution.scope():
      mirrored = variables_lib.Variable(1.)
    self.evaluate(mirrored.assign(3.))
    self.assertEqual(self.evaluate(mirrored.read_value()), 3.)
    for component in mirrored.values:
      self.assertEqual(self.evaluate(component.read_value()), 3.)

  @combinations.generate(strategy_with_var_policy())
  def testInitializedToSameValueInsideEagerRun(self, distribution):
    if not context.executing_eagerly(): self.skipTest("eager only test")
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

  @combinations.generate(strategy_with_var_policy())
  def testAggregationOnlyFirstReplica(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          15.,
          synchronization=variables_lib.VariableSynchronization.ON_WRITE,
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def assign():
      ctx = ds_context.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      return v.assign(math_ops.cast(replica_id, dtypes.float32))
    per_replica_results = self.evaluate(distribution.experimental_local_results(
        distribution.run(assign)))
    # The per-replica values should always match the first replicas value.
    self.assertAllEqual(
        array_ops.zeros(distribution.num_replicas_in_sync, dtypes.float32),
        per_replica_results)

  @combinations.generate(strategy_with_var_policy())
  def testInitScope(self, distribution):
    if not context.executing_eagerly(): self.skipTest("eager only")

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
              0., aggregation=variables_lib.VariableAggregation.MEAN)
          obj.v = variables_lib.Variable(
              obj.w.read_value(),
              aggregation=variables_lib.VariableAggregation.MEAN)
          self.evaluate(variables_lib.global_variables_initializer())

      return obj.v.assign_add(2.)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(distribution.run(assign)))
    self.assertAllEqual([2., 2.], per_replica_results)

  @combinations.generate(strategy_with_var_policy())
  def testOperatorOverride(self, distribution):

    with distribution.scope():
      v = variable_scope.variable(
          1, aggregation=variables_lib.VariableAggregation.SUM)
      self.evaluate(variables_lib.global_variables_initializer())

    self.assertEqual(2, self.evaluate(v + 1))

    @def_function.function
    def add():
      return v + 1

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(distribution.run(add)))
    self.assertAllEqual([2, 2], per_replica_results)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ],
          mode=["eager"],
          use_var_policy=[True, False]))
  def testSaveAndRestoreOnWrite(self, strategy):
    aggregation = [
        variable_scope.VariableAggregation.NONE,
        variable_scope.VariableAggregation.ONLY_FIRST_REPLICA,
        variable_scope.VariableAggregation.SUM,
        variable_scope.VariableAggregation.MEAN
    ]
    for agg in aggregation:
      v_normal_restore = variables_lib.Variable(1.0)
      v_normal_save = variables_lib.Variable(3.0)
      with strategy.scope():
        v_on_write = variables_lib.Variable(2.0, aggregation=agg)

        # Save ONWRITE Restore ONWRITE
        # Save
        ckpt = trackable_utils.Checkpoint(var=v_on_write)
        manager = ckpt_manager.CheckpointManager(
            ckpt, "/tmp/ckpt_" + str(uuid.uuid4()), max_to_keep=None)
        manager.save()
        # Restore
        ckpt.restore(manager.latest_checkpoint)
        self.assertEqual(2.0, self.evaluate(v_on_write._values[0]))
        self.assertEqual(2.0, self.evaluate(v_on_write.read_value()))

        # Save Mirrored Restore Normal
        # We've already saved Mirrored, so we only need to restore normal
        ckpt_normal = trackable_utils.Checkpoint(var=v_normal_restore)
        ckpt_normal.restore(manager.latest_checkpoint)
        self.assertEqual(2.0, self.evaluate(v_on_write._values[0]))
        self.assertEqual(2.0, self.evaluate(v_normal_restore.read_value()))

        # Save Normal Restore Mirrored
        # Save
        ckpt = trackable_utils.Checkpoint(var=v_normal_save)
        manager_2 = ckpt_manager.CheckpointManager(
            ckpt, "/tmp/ckptckpt_" + str(uuid.uuid4()), max_to_keep=None)
        manager_2.save()
        # Restore
        ckpt_on_write = trackable_utils.Checkpoint(var=v_on_write)
        ckpt_on_write.restore(manager_2.latest_checkpoint)
        self.assertEqual(3.0, self.evaluate(v_on_write._values[0]))
        self.assertEqual(3.0, self.evaluate(v_on_write.read_value()))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"],
        use_var_policy=[True, False]))
class OnWriteVariableSyncScatterTests(test.TestCase, parameterized.TestCase):

  def testScatterSub(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          [0., 0., 0.], aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_sub():
      ctx = ds_context.get_replica_context()
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

  def testScatterAdd(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          [0, 0, 0], aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_add():
      ctx = ds_context.get_replica_context()
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

  def testScatterDiv(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          [1, 6, 1], aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_div():
      ctx = ds_context.get_replica_context()
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

  def testScatterMul(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(
          [2., 1., 1.], aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_mul():
      ctx = ds_context.get_replica_context()
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


class OnReadVariableSyncTest(test.TestCase, parameterized.TestCase):

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
  def testAssignOnReadVar(self, distribution, experimental_run_tf_function):

    with distribution.scope():
      v_to_assign = variable_scope.variable(
          2., aggregation=variables_lib.VariableAggregation.MEAN)
      v_to_assign_sub = variable_scope.variable(
          -2., aggregation=variables_lib.VariableAggregation.MEAN)

    def assign(fn, v, update_value, cross_replica):
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return distribution.experimental_local_results(
            distribution.run(update_fn))

    updates = [("assign", v_to_assign), ("assign_add", v_to_assign),
               ("assign_sub", v_to_assign_sub)]
    expected_cross_replica = {
        variables_lib.VariableAggregation.SUM: 1.0,
        variables_lib.VariableAggregation.MEAN: 2.0,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA: 2.0
    }
    expected_replica = {
        variables_lib.VariableAggregation.SUM: 2.0,
        variables_lib.VariableAggregation.MEAN: 2.0,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA: 2.0
    }
    # aggregation=NONE is not supported for OnReadVariables.
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      if aggregation == variables_lib.VariableAggregation.SUM:
        continue
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      if cross_replica:
        for component in v._values:
          self.assertAllEqual(expected_cross_replica.get(aggregation),
                              self.evaluate(component.read_value()))
      else:
        for component in v._values:
          self.assertAllEqual(expected_replica.get(aggregation),
                              self.evaluate(component.read_value()))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignPerReplicaVal(self, distribution, experimental_run_tf_function):

    if isinstance(distribution, _TPU_STRATEGIES):
      self.skipTest("Assigning PerReplica values is not supported. See"
                    " sponge/80ba41f8-4220-4516-98ce-bbad48f9f11a.")

    self.skipTest("We don't support assiging PerReplica values in cross "
                  "replica context or replica context. see error in "
                  "sponge/2b2e54c1-eda6-4534-82e1-c73b1dcd517f.")

    with distribution.scope():
      per_replica_value = values.PerReplica(
          [constant_op.constant(2.0),
           constant_op.constant(2.0)])

    def assign(fn, v, update_value, cross_replica):
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return distribution.experimental_local_results(
            distribution.run(update_fn))

    updates = [("assign", per_replica_value)]
    # We don't support assigning PerReplica valus to vars in replica context
    # with aggregation=NONE.
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      # with self.assertRaisesRegex(ValueError, "Attempt to convert a value "):
      self.evaluate(assign(fn, v, update_value, cross_replica))
      if aggregation == variables_lib.VariableAggregation.SUM:
        expected = 4.0
      else:
        expected = 2.0
      for component in v._values:
        self.assertAllEqual(expected, self.evaluate(component.read_value()))

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

  @combinations.generate(strategy_with_var_policy())
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

  @combinations.generate(strategy_with_var_policy())
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
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
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
        ctx = ds_context.get_replica_context()
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
      ctx = ds_context.get_replica_context()
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
        ctx = ds_context.get_replica_context()
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

  @combinations.generate(strategy_with_var_policy())
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

  @combinations.generate(strategy_with_var_policy())
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

  @combinations.generate(strategy_with_var_policy())
  def testOperatorOverride(self, distribution):

    with distribution.scope():
      v = variable_scope.variable(
          0.0,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.MEAN)
      self.evaluate(variables_lib.global_variables_initializer())

      @def_function.function
      def assign():
        ctx = ds_context.get_replica_context()
        replica_id = ctx.replica_id_in_sync_group
        return v.assign(math_ops.cast(replica_id, dtypes.float32))

      # Assign different replicas with different values.
      self.evaluate(distribution.experimental_local_results(
          distribution.run(assign)))
      self.assertEqual(1.5, self.evaluate(v + 1))

      @def_function.function
      def add():
        return v + 1

      per_replica_results = self.evaluate(
          distribution.experimental_local_results(distribution.run(add)))
      self.assertAllEqual([1, 2], per_replica_results)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ],
          mode=["eager"],
          use_var_policy=[True, False]))
  def testSaveAndRestoreOnRead(self, strategy):
    aggregation = [variable_scope.VariableAggregation.SUM,
                   variable_scope.VariableAggregation.MEAN]
    for agg in aggregation:
      v_normal_restore = variables_lib.Variable(1.0)
      v_normal_save = variables_lib.Variable(2.0)

      with strategy.scope():
        v_on_read = variables_lib.Variable(
            1.0, synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=agg)

        @def_function.function
        def assign_fn():
          cluster_resolver = strategy.cluster_resolver
          replica_ctx = ds_context.get_replica_context()
          if ((cluster_resolver and cluster_resolver.task_type == "worker") or
              math_ops.equal(replica_ctx.replica_id_in_sync_group,
                             constant_op.constant(1))):
            v_on_read.assign(3.)  # pylint:disable=cell-var-from-loop
          else:
            v_on_read.assign(4.)  # pylint:disable=cell-var-from-loop

        strategy.run(assign_fn)

        # Save ONREAD, restore ONREAD
        # Saves v[0] + v[1] = 7 for SUM and 3.5 for MEAN.
        ckpt = trackable_utils.Checkpoint(var=v_on_read)
        manager = ckpt_manager.CheckpointManager(
            ckpt, "/tmp/ckpt_" + str(uuid.uuid4()), max_to_keep=None)
        manager.save()
        # Restores a value of 7/2 = 3.5 for SUM and 3.5 for MEAN.
        ckpt.restore(manager.latest_checkpoint)
        self.assertEqual(3.5, self.evaluate(v_on_read._values[0]))

        # Save ONREAD, restore normal
        ckpt_normal = trackable_utils.Checkpoint(var=v_normal_restore)
        ckpt_normal.restore(manager.latest_checkpoint)
        if agg == variable_scope.VariableAggregation.SUM:
          self.assertEqual(7.0, self.evaluate(v_normal_restore.read_value()))
        else:
          self.assertEqual(3.5, self.evaluate(v_normal_restore.read_value()))

        # Save normal, restore ONREAD
        ckpt = trackable_utils.Checkpoint(var=v_normal_save)
        manager = ckpt_manager.CheckpointManager(
            ckpt, "/tmp/ckpt_" + str(uuid.uuid4()), max_to_keep=None)
        manager.save()
        # Restores a value of 2/2 = 1.0 for SUM and 2.0 for MEAN.
        ckpt_on_read = trackable_utils.Checkpoint(var=v_on_read)
        ckpt_on_read.restore(manager.latest_checkpoint)
        if agg == variable_scope.VariableAggregation.SUM:
          self.assertEqual(1.0, self.evaluate(v_on_read._values[0]))
        else:
          self.assertEqual(2.0, self.evaluate(v_on_read._values[0]))


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
        mode=["graph", "eager"],
        use_var_policy=[True, False]))
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


if __name__ == "__main__":
  test_util.main()
