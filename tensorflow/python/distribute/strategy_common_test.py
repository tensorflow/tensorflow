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
"""Tests for common methods in strategy classes."""

from absl.testing import parameterized

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ] + strategy_combinations.all_strategies,
        mode=['eager']))
class StrategyTest(test.TestCase, parameterized.TestCase):

  def testCaptureReplicaId(self, strategy):
    m = {}

    @def_function.function
    def f():
      return distribute_lib.get_replica_context().replica_id_in_sync_group

    @def_function.function
    def g():
      # Make g() a stateful function so it's traced twice.
      if m.get('v', None) is None:
        m['v'] = variables.Variable(0.)
      return strategy.run(f)

    g()

  def testMergeCallInitScope(self, strategy):
    with strategy.scope():

      @def_function.function
      def fn():

        def merge_fn(unused_strat):

          y = constant_op.constant(11)
          return y

        def replica_fn():

          with ops.init_scope():
            y = distribute_lib.get_replica_context().merge_call(merge_fn)
            z = y + 1
            return z

        return strategy.run(replica_fn)

      result = strategy.experimental_local_results(fn())
      self.assertAllClose(result, [12] * _get_num_replicas_per_client(strategy))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.tpu_strategy
        ],
        mode=['graph', 'eager']))
class StrategyLocalResultTest(test.TestCase):

  def testLocalResultForDictionary(self, distribution):

    @def_function.function
    def model_fn():
      return {'a': constant_op.constant(1.), 'b': constant_op.constant(2.)}

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ({'a': 1., 'b': 2.}, {'a': 1., 'b': 2.}))

  def testLocalResultForList(self, distribution):

    @def_function.function
    def model_fn():
      return [constant_op.constant(1.), constant_op.constant(2.)]

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ([1., 2.], [1., 2.]))

  def testLocalResultForTuple(self, distribution):

    @def_function.function
    def model_fn():
      return (constant_op.constant(1.), constant_op.constant(2.),
              constant_op.constant(3.))

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ((1., 2., 3.), (1., 2., 3.)))

  def testLocalResultForNestedStruct(self, distribution):

    @def_function.function
    def model_fn():
      return ({
          'a': constant_op.constant(1.),
          'b': constant_op.constant(2.)
      }, {
          'a': constant_op.constant(4.),
          'b': constant_op.constant(6.)
      })

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, (({
          'a': 1.,
          'b': 2.
      }, {
          'a': 4.,
          'b': 6.
      }), ({
          'a': 1.,
          'b': 2.
      }, {
          'a': 4.,
          'b': 6.
      })))

  def testLocalResultForNestedStructWithoutTensor(self, distribution):

    @def_function.function
    def model_fn():
      return {'a': 1., 'b': 2.}

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertIsInstance(v, tuple)
      self.assertAllEqual(v, ({'a': 1., 'b': 2.}, {'a': 1., 'b': 2.}))

  def testLocalResultForScalarValue(self, distribution):

    @def_function.function
    def model_fn():
      return distribution.extended._get_local_replica_id(
          distribute_lib.get_replica_context().replica_id_in_sync_group)

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertIsInstance(v, tuple)
      self.assertEqual(v, (0, 1))

  def testLocalResultForDictionaryDifferentReplicas(self, distribution):

    @def_function.function
    def model_fn():
      replica_id = distribution.extended._get_local_replica_id(
          distribute_lib.get_replica_context().replica_id_in_sync_group)
      return {
          'a': math_ops.cast(replica_id + 1, dtype=float),
          'b': math_ops.cast(replica_id + 2, dtype=float)
      }

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertAllEqual(got, ({'a': 1., 'b': 2.}, {'a': 2., 'b': 3.}))

  def testLocalResultForTensor(self, distribution):

    @def_function.function
    def model_fn():
      return constant_op.constant([2., 3.])

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertAllEqual(v, ([2., 3.], [2., 3.]))


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ] + strategy_combinations.all_strategies,
        mode=['eager']))
class ReduceTest(test.TestCase, parameterized.TestCase):

  def testBasic(self, strategy):
    per_replica_value = strategy.experimental_distribute_values_from_function(
        lambda _: array_ops.ones((), dtypes.float32))

    def fn_eager():

      return strategy.reduce(
          reduce_util.ReduceOp.SUM, value=per_replica_value, axis=None)

    fn_graph = def_function.function(fn_eager)
    # Run reduce under the strategy scope to explicitly enter
    # strategy default_device scope.
    with strategy.scope():
      self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
      self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)

    # Run reduce without a strategy scope to implicitly enter
    # strategy default_device scope.
    self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
    self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)

  def testAxis(self, strategy):

    @def_function.function
    def fn():
      return constant_op.constant([1., 2.])

    x = strategy.run(fn)

    x_m = strategy.reduce(reduce_util.ReduceOp.MEAN, x, axis=0)
    self.assertEqual(1.5, x_m)
    x_s = strategy.reduce(reduce_util.ReduceOp.SUM, x, axis=0)
    self.assertEqual(3 * strategy.num_replicas_in_sync, x_s)


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.default_strategy,
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
            strategy_combinations.tpu_strategy,
            strategy_combinations.tpu_strategy_packed_var,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
        ],
        update_fn=['assign', 'assign_add', 'assign_sub'],
        tf_function=[True, False],
        mode=['eager']))
class ReplicaCtxUpdateTest(test.TestCase, parameterized.TestCase):

  def testDenseUpdate(self, strategy, tf_function, update_fn):
    if strategy_test_lib.is_tpu_strategy(strategy) and (not tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')
    with strategy.scope():
      distributed_variable1 = variables.Variable(5.0)

    def replica_fn():
      value = array_ops.constant(2.)
      python_literal = 1.
      replica_context = distribute_lib.get_replica_context()
      fn_sets = {
          'assign': lambda var, value: var.assign(value),
          'assign_add': lambda var, value: var.assign_add(value),
          'assign_sub': lambda var, value: var.assign_sub(value),
      }
      replica_context._update(
          distributed_variable1, fn_sets[update_fn], args=(value,))
      replica_context._update(
          distributed_variable1, fn_sets[update_fn], args=(python_literal,))

    if tf_function:
      replica_fn = def_function.function(replica_fn)
    strategy.run(replica_fn)

    expected_result = {'assign': 1., 'assign_add': 8., 'assign_sub': 2.}
    self.assertAllEqual(
        strategy.experimental_local_results(distributed_variable1),
        [expected_result[update_fn]] * _get_num_replicas_per_client(strategy))


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
            strategy_combinations.tpu_strategy,
        ] + strategy_combinations.strategies_minus_tpu,
        tf_function=[combinations.tf_function, combinations.no_tf_function],
        mode=['eager']))
class ReplicaCtxAllReduceTest(test.TestCase, parameterized.TestCase):

  def testDense(self, strategy, tf_function):
    if (strategy_test_lib.is_tpu_strategy(strategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value = array_ops.identity(1.0)
        reduced = strategy.extended._replica_ctx_all_reduce(
            reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    self.assertEqual(got, 1.0 * strategy.num_replicas_in_sync)

  def testSparse(self, strategy, tf_function):
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value = indexed_slices.IndexedSlices(
            values=array_ops.identity([[1.0]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
        reduced = strategy.extended._replica_ctx_all_reduce(
            reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    expect = indexed_slices.IndexedSlices(
        values=array_ops.identity([[1.0 * strategy.num_replicas_in_sync]]),
        indices=array_ops.identity([0]),
        dense_shape=array_ops.identity([5, 1]))
    self.assertAllEqual(
        ops.convert_to_tensor(got), ops.convert_to_tensor(expect))

  def testNestedInput(self, strategy, tf_function):
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value = (array_ops.identity(1.0),
                 indexed_slices.IndexedSlices(
                     values=array_ops.identity([[1.0]]),
                     indices=array_ops.identity([0]),
                     dense_shape=array_ops.identity([5, 1])),
                 array_ops.identity(2.0),
                 indexed_slices.IndexedSlices(
                     values=array_ops.identity([[2.0]]),
                     indices=array_ops.identity([1]),
                     dense_shape=array_ops.identity([5, 1])))
        reduced = strategy.extended._replica_ctx_all_reduce(
            reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    expect = (1.0 * strategy.num_replicas_in_sync,
              indexed_slices.IndexedSlices(
                  values=array_ops.identity(
                      [[1.0 * strategy.num_replicas_in_sync]]),
                  indices=array_ops.identity([0]),
                  dense_shape=array_ops.identity([5, 1])),
              2.0 * strategy.num_replicas_in_sync,
              indexed_slices.IndexedSlices(
                  values=array_ops.identity(
                      [[2.0 * strategy.num_replicas_in_sync]]),
                  indices=array_ops.identity([1]),
                  dense_shape=array_ops.identity([5, 1])))

    self.assertAllClose(
        nest.map_structure(ops.convert_to_tensor, got),
        nest.map_structure(ops.convert_to_tensor, expect))

  def testSyncOnReadVariableInput(self, strategy, tf_function):
    if (not strategy_test_lib.is_mirrored_strategy(strategy) and
        not strategy_test_lib.is_multi_worker_mirrored_strategy(strategy) and
        not strategy_test_lib.is_tpu_strategy(strategy)):
      self.skipTest('Skip strategies not using SyncOnReadVariables.')
    if (strategy_test_lib.is_tpu_strategy(strategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')
    if (strategy_test_lib.is_multi_worker_mirrored_strategy(strategy) and
        tf_function is combinations.tf_function):
      self.skipTest('Skip MWMS + graph combination until b/228512201 is fixed.')

    with strategy.scope():
      var = variables.Variable(
          0.0,
          synchronization=variables.VariableSynchronization.ON_READ,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)

    @tf_function
    def replica_fn():
      replica_context = distribute_lib.get_replica_context()
      replica_id = replica_context.replica_id_in_sync_group
      var.assign(math_ops.cast(replica_id, dtype=float) * 3.0)

      return replica_context.all_reduce(reduce_util.ReduceOp.SUM, var)

    if strategy_test_lib.is_multi_worker_mirrored_strategy(strategy):
      client_local_replica_num = strategy.extended._num_devices_per_worker
    else:
      client_local_replica_num = strategy.num_replicas_in_sync

    workers_num = strategy.num_replicas_in_sync
    expected_sum = sum(range(workers_num)) * 3.0

    # Expand the values on each replica if multiple devices are used; otherwise
    # simple read the value of the Tensor.
    result = strategy.run(replica_fn)
    if hasattr(result, 'values'):
      result = result.values
    result = nest.flatten(result)

    # Iterate through all replicas and verify the reduce sum result.
    for i in range(client_local_replica_num):
      self.assertEqual(result[i].numpy(), expected_sum)


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
            strategy_combinations.tpu_strategy,
        ] + strategy_combinations.strategies_minus_tpu,
        tf_function=[combinations.tf_function, combinations.no_tf_function],
        mode=['eager']))
class AllReduceTest(test.TestCase, parameterized.TestCase):

  def testDense(self, strategy, tf_function):
    if (strategy_test_lib.is_tpu_strategy(strategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value = array_ops.identity(1.0)
        rep_ctx = distribute_lib.get_replica_context()
        reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    self.assertEqual(got, 1.0 * strategy.num_replicas_in_sync)

  def testSparse(self, strategy, tf_function):
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value = indexed_slices.IndexedSlices(
            values=array_ops.identity([[1.0]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
        rep_ctx = distribute_lib.get_replica_context()
        reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.MEAN, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]

    if not strategy_test_lib.is_tpu_strategy(strategy):
      self.assertIsInstance(got, indexed_slices.IndexedSlices)
    expect = indexed_slices.IndexedSlices(
        values=array_ops.identity([[1.0]]),
        indices=array_ops.identity([0]),
        dense_shape=array_ops.identity([5, 1]))
    self.assertAllEqual(
        ops.convert_to_tensor(got), ops.convert_to_tensor(expect))

  def testSparseTuple(self, strategy, tf_function):
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value1 = indexed_slices.IndexedSlices(
            values=array_ops.identity([[1.0]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
        value2 = indexed_slices.IndexedSlices(
            values=array_ops.identity([[2.0]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
        rep_ctx = distribute_lib.get_replica_context()
        reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, [value1, value2])
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]

    if not strategy_test_lib.is_tpu_strategy(strategy):
      for g in got:
        self.assertIsInstance(g, indexed_slices.IndexedSlices)
    expect = [
        indexed_slices.IndexedSlices(
            values=array_ops.identity([[1.0 * strategy.num_replicas_in_sync]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1])),
        indexed_slices.IndexedSlices(
            values=array_ops.identity([[2.0 * strategy.num_replicas_in_sync]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
    ]
    self.assertAllEqual(
        nest.map_structure(ops.convert_to_tensor, got),
        nest.map_structure(ops.convert_to_tensor, expect))

  def testNestedInput(self, strategy, tf_function):
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value = (array_ops.identity(1.0),
                 indexed_slices.IndexedSlices(
                     values=array_ops.identity([[1.0]]),
                     indices=array_ops.identity([0]),
                     dense_shape=array_ops.identity([5, 1])),
                 array_ops.identity(2.0),
                 indexed_slices.IndexedSlices(
                     values=array_ops.identity([[2.0]]),
                     indices=array_ops.identity([1]),
                     dense_shape=array_ops.identity([5, 1])))
        rep_ctx = distribute_lib.get_replica_context()
        reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    expect = (1.0 * strategy.num_replicas_in_sync,
              indexed_slices.IndexedSlices(
                  values=array_ops.identity(
                      [[1.0 * strategy.num_replicas_in_sync]]),
                  indices=array_ops.identity([0]),
                  dense_shape=array_ops.identity([5, 1])),
              2.0 * strategy.num_replicas_in_sync,
              indexed_slices.IndexedSlices(
                  values=array_ops.identity(
                      [[2.0 * strategy.num_replicas_in_sync]]),
                  indices=array_ops.identity([1]),
                  dense_shape=array_ops.identity([5, 1])))

    self.assertAllClose(
        nest.map_structure(ops.convert_to_tensor, got),
        nest.map_structure(ops.convert_to_tensor, expect))


def _make_indexed_slices(values, indices, dense_shape):
  tensor = indexed_slices.IndexedSlices(
      values=constant_op.constant(values),
      indices=constant_op.constant(indices),
      dense_shape=constant_op.constant(dense_shape))
  return tensor


def _get_num_replicas_per_client(strategy):
  if isinstance(strategy, CollectiveAllReduceStrategy):
    resolver = strategy.cluster_resolver
    return max(nest.flatten(resolver.num_accelerators())[0], 1)
  else:
    return strategy.num_replicas_in_sync


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ],
        mode=['eager']))
class DistributedCollectiveAllReduceStrategyTest(
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):

  def testDatasetFromFunction(self, strategy):
    def dataset_fn(input_context):
      global_batch_size = 10
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = dataset_ops.DatasetV2.range(100).repeat().batch(batch_size)
      return d.shard(input_context.num_input_pipelines,
                     input_context.input_pipeline_id)

    expected_sum_on_workers = {'chief': 10, 'worker': 35}
    input_iterator = iter(
        strategy.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(iterator):
      return strategy.experimental_local_results(iterator.get_next())

    result = run(input_iterator)
    sum_value = math_ops.reduce_sum(result)
    self.assertEqual(
        sum_value.numpy(),
        expected_sum_on_workers[multi_worker_test_base.get_task_type()])

  def testSimpleInputFromDatasetLastPartialBatch(self, strategy):
    global_batch_size = 8
    dataset = dataset_ops.DatasetV2.range(14).batch(
        global_batch_size, drop_remainder=False)
    input_iterator = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(input_iterator):
      return strategy.run(lambda x: x, args=(next(input_iterator),))

    # Let the complete batch go.
    run(input_iterator)

    # `result` is an incomplete batch
    result = run(input_iterator)
    expected_data_on_workers = {'chief': [8, 9, 10], 'worker': [11, 12, 13]}
    self.assertAllEqual(
        expected_data_on_workers[multi_worker_test_base.get_task_type()],
        result.numpy(),
    )

  def testSimpleInputFromFnLastPartialBatch(self, strategy):

    def dataset_fn(input_context):
      global_batch_size = 8
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      dataset = dataset_ops.DatasetV2.range(14).batch(
          batch_size, drop_remainder=False)
      return dataset.shard(input_context.num_input_pipelines,
                           input_context.input_pipeline_id)

    input_iterator = iter(
        strategy.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(input_iterator):
      return strategy.run(lambda x: x, args=(next(input_iterator),))

    # Let the complete batch go.
    run(input_iterator)
    # `result` is an incomplete batch
    result = run(input_iterator)

    expected_data_on_worker = {'chief': [8, 9, 10, 11], 'worker': [12, 13]}
    self.assertAllEqual(
        expected_data_on_worker[multi_worker_test_base.get_task_type()],
        result.numpy())

  def testReduceHostTensor(self, strategy):
    reduced = strategy.reduce(
        reduce_util.ReduceOp.SUM, array_ops.identity(1.), axis=None)
    self.assertEqual(reduced.numpy(), 2.)

  def testReduceToHostTensor(self, strategy):
    value = array_ops.identity(1.)
    reduced = strategy.extended.reduce_to(reduce_util.ReduceOp.SUM, value,
                                          value)
    self.assertEqual(reduced.numpy(), 2.)

  def testBatchReduceToHostTensor(self, strategy):
    value = array_ops.identity(1.)
    reduced = strategy.extended.batch_reduce_to(reduce_util.ReduceOp.SUM,
                                                [(value, value),
                                                 (value, value)])
    self.assertAllEqual([2., 2.], reduced)

  def testReduceDeviceTensors(self, strategy):
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.reduce(reduce_util.ReduceOp.SUM, value, axis=None)
    self.assertEqual(reduced.numpy(), 2.)

  def testReduceToDeviceTensors(self, strategy):
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.extended.reduce_to(reduce_util.ReduceOp.SUM, value,
                                          value)
    self.assertEqual(reduced.numpy(), 2.)

  def testBatchReduceToDeviceTensors(self, strategy):
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.extended.batch_reduce_to(reduce_util.ReduceOp.SUM,
                                                [(value, value),
                                                 (value, value)])
    self.assertAllEqual([2., 2.], reduced)

  # TODO(crccw): add a test that mixes device and host tensors after multi
  # worker strategy combinations can run on a fixed number of GPUs.


class StrategyClusterResolverTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu] +
          strategy_combinations.all_strategies,
          mode=['eager']))
  def testClusterResolverProperty(self, strategy):
    # CollectiveAllReduceStrategy and TPUStrategy must have a cluster resolver.
    # `None` otherwise.
    resolver = strategy.cluster_resolver
    if (not isinstance(strategy, CollectiveAllReduceStrategy) and
        not strategy_test_lib.is_tpu_strategy(strategy)):
      self.assertIsNone(resolver)
      return

    with strategy.scope():
      self.assertIs(strategy.cluster_resolver, resolver)

    self.assertTrue(hasattr(resolver, 'cluster_spec'))
    self.assertTrue(hasattr(resolver, 'master'))
    self.assertTrue(hasattr(resolver, 'num_accelerators'))
    self.assertTrue(hasattr(resolver, 'task_id'))
    self.assertTrue(hasattr(resolver, 'task_type'))
    if isinstance(strategy, CollectiveAllReduceStrategy):
      self.assertEqual(resolver.task_id, 0)
      self.assertAllInSet(resolver.task_type, ['chief', 'worker'])


if __name__ == '__main__':
  test_util.main()
