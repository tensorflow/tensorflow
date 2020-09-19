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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy
from tensorflow.python.distribute.tpu_strategy import TPUStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
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

  def testSimpleReduce(self, strategy):
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

  def testCaptureReplicaId(self, strategy):
    m = {}

    @def_function.function
    def f():
      return ds_context.get_replica_context().replica_id_in_sync_group

    @def_function.function
    def g():
      # Make g() a stateful function so it's traced twice.
      if m.get('v', None) is None:
        m['v'] = variables.Variable(0.)
      return strategy.run(f)

    g()


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ],
        mode=['eager'],
        pure_eager=[True, False]))
class GatherTest(test.TestCase, parameterized.TestCase):

  def _gather_same_shape_and_verify(self, value_on_replica, axis, pure_eager,
                                    strategy):
    distributed_values = strategy.experimental_distribute_values_from_function(
        lambda _: array_ops.identity(value_on_replica))

    def run():
      return strategy._gather(distributed_values, axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    all_results = [
        value_on_replica for _ in range(strategy.num_replicas_in_sync)
    ]
    expected_result = array_ops.concat(all_results, axis=axis)
    self.assertAllEqual(run().numpy(), expected_result)

  def testGatherPerReplicaDense1D0Axis(self, strategy, pure_eager):
    """A DistributedValues object with two tensors of shape [3] on each replica gathers to a tensor of [6]."""
    single_value = constant_op.constant([1, 2, 3])
    axis = 0
    self._gather_same_shape_and_verify(single_value, axis, pure_eager, strategy)

  def testGatherPerReplicaDense2D0Axis(self, strategy, pure_eager):
    """A DistributedValues object with two tensors of [1, 3] on each replica gathers along 0th dim to a tensor of [2, 3]."""
    single_value = constant_op.constant([[1, 2, 3]])
    axis = 0
    self._gather_same_shape_and_verify(single_value, axis, pure_eager, strategy)

  def testGatherPerReplicaDense2D1Axis(self, strategy, pure_eager):
    """A DistributedValues object with two tensors of [1, 3] on each replica gathers along 1st dim to a tensor of [1, 6]."""
    single_value = constant_op.constant([[1, 2, 3]])
    axis = 1
    self._gather_same_shape_and_verify(single_value, axis, pure_eager, strategy)

  def testGatherPerReplicaDense3D0Axis(self, strategy, pure_eager):
    """A DistributedValues object with two tensors of [1, 2, 2] on each replica gathers along 0th dim to a tensor of [2, 2, 2]."""
    single_value = constant_op.constant([[[1, 2], [1, 2]]])
    axis = 0
    self._gather_same_shape_and_verify(single_value, axis, pure_eager, strategy)

  def testGatherPerReplicaDense3D1Axis(self, strategy, pure_eager):
    """A DistributedValues object with two tensors of [1, 2, 2] on each replica gathers along 1nd dimension to a tensor of [1, 4, 2]."""
    single_value = constant_op.constant([[[1, 2], [1, 2]]])
    axis = 1
    self._gather_same_shape_and_verify(single_value, axis, pure_eager, strategy)

  def testGatherPerReplicaDense3D2Axis(self, strategy, pure_eager):
    """A DistributedValues object with two tensors of [1, 2, 2] on each replica gathers along 2nd dimension to a tensor of [1, 2, 4]."""
    single_value = constant_op.constant([[[1, 2], [1, 2]]])
    axis = 2
    self._gather_same_shape_and_verify(single_value, axis, pure_eager, strategy)

  def testGatherDiffShapeAtAxis0(self, strategy, pure_eager):
    """Different `Axis`-th (0) dimension: shape [1, 1], [2, 1] -> [3, 1]."""

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(ctx.replica_id_in_sync_group + 1, 1))

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn)
    axis = 0

    def run():
      return strategy._gather(distributed_values, axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    if strategy.num_replicas_in_sync == 1:
      expected_result = constant_op.constant(1, shape=(1, 1))
    elif strategy.num_replicas_in_sync == 2:
      expected_result = constant_op.constant(1, shape=(3, 1))
    elif strategy.num_replicas_in_sync == 4:
      expected_result = constant_op.constant(1, shape=(10, 1))
    else:
      # should follow expected_result = constant_op.constant(
      #    1, shape=(sum(range(strategy.num_replicas_in_sync + 1)), 1))
      raise ValueError('Add your own expect according to num_replicas_in sync')

    self.assertAllEqual(run().numpy(), expected_result)

  def testGatherDiffShapeAtAxis1(self, strategy, pure_eager):
    """Different `Axis`-th (non-0) dimension: shape [1, 1], [1, 2] -> [1, 3]."""

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn)
    axis = 1

    def run():
      return strategy._gather(distributed_values, axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    if strategy.num_replicas_in_sync == 1:
      expected_result = constant_op.constant(1, shape=(1, 1))
    elif strategy.num_replicas_in_sync == 2:
      expected_result = constant_op.constant(1, shape=(1, 3))
    elif strategy.num_replicas_in_sync == 4:
      expected_result = constant_op.constant(1, shape=(1, 10))
    else:
      # should follow expected_result = constant_op.constant(
      #   1, shape=(1, sum(range(strategy.num_replicas_in_sync + 1))))
      raise ValueError('Add your own expect according to num_replicas_in sync')

    self.assertAllEqual(run().numpy(), expected_result)

  def testGatherRaiseDiffShapeAtNonAxis(self, strategy, pure_eager):
    """Different at non-`axis`-th dimension : [1, 1], [1, 2], 0th -> raise error."""
    if _get_num_devices_per_worker(strategy) > 1:
      self.skipTest('b/167331966')
    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn)
    axis = 0

    def run():
      return strategy._gather(distributed_values, axis=axis)

    error_message = 'Shape mismatch'
    if not pure_eager:
      run = def_function.function(run)

    with self.assertRaisesRegex(errors.InvalidArgumentError, error_message):
      run()

  def testGatherRaiseSparsePerReplicaMultiWorker(self, strategy, pure_eager):
    if strategy.num_replicas_in_sync != 2:
      self.skipTest('Test for two replicas.')
    dense_shape = [5, 2]
    if multi_worker_test_base.get_task_type() == 'chief':
      t0 = _make_indexed_slices(
          values=[[1., 2.]], indices=[2], dense_shape=dense_shape)
    if multi_worker_test_base.get_task_type() == 'worker':
      t0 = _make_indexed_slices(
          values=[[3., 4.], [5., 6.]], indices=[1, 3], dense_shape=dense_shape)

    def run(value):
      return strategy._gather(value, axis=0)

    with self.assertRaisesRegex(
        NotImplementedError,
        r'gather/all_gather does not support IndexedSlices'):
      if pure_eager:
        run(t0)
      else:
        def_function.function(run)(t0)

  def testGatherRaiseDifferentRank(self, strategy, pure_eager):
    """Different rank: [1,], [1, 2] -> raise error."""
    if strategy.num_replicas_in_sync <= 1:
      self.skipTest('Test for more than 1 replicas.')
    if _get_num_devices_per_worker(strategy) > 1:
      self.skipTest('b/167331966')
    def value_fn(ctx):
      return array_ops.ones(shape=(range(1, ctx.replica_id_in_sync_group + 2)))

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn)
    axis = 0

    def run():
      return strategy._gather(distributed_values, axis=axis)

    error_message = 'Shape mismatch'

    if not pure_eager:
      run = def_function.function(run)

    with self.assertRaisesRegex(errors.InvalidArgumentError, error_message):
      run()


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ],
        mode=['eager'],
        pure_eager=[True, False]))
class AllGatherTest(test.TestCase, parameterized.TestCase):

  def _all_gather_same_shape_and_verify(self, value_on_replica, axis,
                                        pure_eager, strategy):
    per_replica_value = strategy.experimental_distribute_values_from_function(
        lambda _: array_ops.identity(value_on_replica))

    def replica_fn(per_replica_value):
      ctx = ds_context.get_replica_context()
      local_value = array_ops.identity(per_replica_value)
      return ctx._all_gather(local_value, axis=axis)

    if not pure_eager:
      replica_fn = def_function.function(replica_fn)

    result = strategy.experimental_local_results(
        strategy.run(replica_fn, args=(per_replica_value,)))

    all_value = [value_on_replica for _ in range(strategy.num_replicas_in_sync)]
    expect = array_ops.concat(all_value, axis=axis)
    expected_result = [expect] * _get_num_devices_per_worker(strategy)

    self.assertAllClose(result, expected_result)

  def testAllGatherPerReplicaDense1D0Axis(self, strategy, pure_eager):
    """all_gather(..., axis=0,...) a DistributedValues with a Tensor of shape (3,) on two replica returns a PerReplica of tensor(s) with shape (6,)."""
    single_value = constant_op.constant([1, 2, 3], dtype=dtypes.float32)
    axis = 0
    self._all_gather_same_shape_and_verify(single_value, axis, pure_eager,
                                           strategy)

  def testAllGatherPerReplicaDense2D0Axis(self, strategy, pure_eager):
    """all_gather(..., axis=0,...) a DistributedValues with a Tensor of shape (1,3) on two replica returns PerReplica of tensor(s) with shape (2,3)."""
    single_value = constant_op.constant([[1, 2, 3]])
    axis = 0
    self._all_gather_same_shape_and_verify(single_value, axis, pure_eager,
                                           strategy)

  def testAllGatherPerReplicaDense2D1Axis(self, strategy, pure_eager):
    """all_gather(..., axis=1,...) a DistributedValues with a Tensor of shape (1,3) on two replica returns PerReplica of tensor(s) with shape (1,6)."""
    single_value = constant_op.constant([[1, 2, 3]])
    axis = 1
    self._all_gather_same_shape_and_verify(single_value, axis, pure_eager,
                                           strategy)

  def testAllGatherPerReplicaDense3D0Axis(self, strategy, pure_eager):
    """all_gather(..., axis=0,...) a DistributedValues with a Tensor of shape (1,2,2) on two replica returns PerReplica of tensor(s) with shape (2,2,2)."""
    single_value = constant_op.constant([[[1, 2], [1, 2]]])
    axis = 0
    self._all_gather_same_shape_and_verify(single_value, axis, pure_eager,
                                           strategy)

  def testAllGatherPerReplicaDense3D1Axis(self, strategy, pure_eager):
    """all_gather(..., axis=1,...) a DistributedValues with a Tensor of shape (1,2,2) on two replica returns PerReplica of tensor(s) with shape (1,4,2)."""
    single_value = constant_op.constant([[[1, 2], [1, 2]]])
    axis = 1
    self._all_gather_same_shape_and_verify(single_value, axis, pure_eager,
                                           strategy)

  def testAllGatherPerReplicaDense3D2Axis(self, strategy, pure_eager):
    """all_gather(..., axis=2,...) a DistributedValues with a Tensor of shape (1,2,2) on two replica returns PerReplica of tensor(s) with shape (1,2,4)."""
    single_value = constant_op.constant([[[1, 2], [1, 2]]])
    axis = 2
    self._all_gather_same_shape_and_verify(single_value, axis, pure_eager,
                                           strategy)

  def testAllGatherDiffShapeAtAxis0(self, strategy, pure_eager):
    """Different `Axis==0`-th dimension: shape [1, 1], [2, 1] -> [3, 1]."""

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(ctx.replica_id_in_sync_group + 1, 1))

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    if strategy.num_replicas_in_sync == 1:
      expect = constant_op.constant(1, shape=(1, 1))
    elif strategy.num_replicas_in_sync == 2:
      expect = constant_op.constant(1, shape=(3, 1))
    elif strategy.num_replicas_in_sync == 4:
      expect = constant_op.constant(1, shape=(10, 1))
    else:
      # should follow expect = constant_op.constant(
      #     1, shape=(sum(range(strategy.num_replicas_in_sync + 1)), 1))
      raise ValueError('Add your own expect according to num_replicas_in sync')

    def run(value):
      value_identity = array_ops.identity(value)
      ctx = ds_context.get_replica_context()
      return ctx._all_gather(value_identity, axis=0)

    if not pure_eager:
      run = def_function.function(run)

    expected_result = [expect] * _get_num_devices_per_worker(strategy)
    result = strategy.experimental_local_results(
        strategy.run(run, args=(per_replica_value,)))
    self.assertAllEqual(result, expected_result)

  def testAllGatherDiffShapeAtAxis1(self, strategy, pure_eager):
    """Different `Axis`-th (not 0th) dimension: shape [1, 1], [1, 2] -> [1, 3]."""

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    if strategy.num_replicas_in_sync == 1:
      expect = constant_op.constant(1, shape=(1, 1))
    elif strategy.num_replicas_in_sync == 2:
      expect = constant_op.constant(1, shape=(1, 3))
    elif strategy.num_replicas_in_sync == 4:
      expect = constant_op.constant(1, shape=(1, 10))
    else:
      # should follow expect = constant_op.constant(
      #    1, shape=(1, sum(range(strategy.num_replicas_in_sync + 1))))
      raise ValueError('Add your own expect according to num_replicas_in sync')

    def run(value):
      value_identity = array_ops.identity(value)
      ctx = ds_context.get_replica_context()
      return ctx._all_gather(value_identity, axis=1)

    if not pure_eager:
      run = def_function.function(run)

    expected_result = [expect] * _get_num_devices_per_worker(strategy)
    result = strategy.experimental_local_results(
        strategy.run(run, args=(per_replica_value,)))
    self.assertAllEqual(result, expected_result)

  def testAllGatherNest(self, strategy, pure_eager):
    axis = 1

    def value_fn(ctx):
      value = constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))
      return value
    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    if strategy.num_replicas_in_sync == 1:
      expect_1 = constant_op.constant(1, shape=(1, 1))
    elif strategy.num_replicas_in_sync == 2:
      expect_1 = constant_op.constant(1, shape=(1, 3))
    elif strategy.num_replicas_in_sync == 4:
      expect_1 = constant_op.constant(1, shape=(1, 10))
    else:
      # should follow expect_1 = constant_op.constant(
      #    1, shape=(1, sum(range(strategy.num_replicas_in_sync + 1))))
      raise ValueError('Add your own expect according to num_replicas_in sync')

    expected_per_replica_1 = [expect_1] * _get_num_devices_per_worker(strategy)

    value_2 = constant_op.constant([[[1, 2], [1, 2]]])

    if strategy.num_replicas_in_sync == 1:
      expect_2 = constant_op.constant([[[1, 2], [1, 2]]])
    elif strategy.num_replicas_in_sync == 2:
      expect_2 = constant_op.constant([[[1, 2], [1, 2], [1, 2], [1, 2]]])
    elif strategy.num_replicas_in_sync == 4:
      expect_2 = constant_op.constant([[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
                                        [1, 2], [1, 2], [1, 2]]])
    else:
      # should follow expect_2 = array_ops.concat(
      #    [value_2 for _ in range(strategy.num_replicas_in_sync)], axis=axis)
      raise ValueError('Add your own expect according to num_replicas_in sync')

    expected_per_replica_2 = [expect_2] * _get_num_devices_per_worker(strategy)

    def run(value):
      value_1 = array_ops.identity(value)
      value_3 = array_ops.identity(value_2)
      ctx = ds_context.get_replica_context()
      return ctx._all_gather([value_1, value_3], axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    result = strategy.run(run, args=(per_replica_value,))
    self.assertAllEqual(
        strategy.experimental_local_results(result[0]), expected_per_replica_1)
    self.assertAllEqual(
        strategy.experimental_local_results(result[1]), expected_per_replica_2)

  def testAllGatherNest1D0Axis(self, strategy, pure_eager):
    """all_gather(..., axis=0,...) a nest of DistributedValues."""
    single_value = constant_op.constant([1, 2, 3])
    axis = 0

    def run():
      value_identity = array_ops.identity(single_value)
      ctx = ds_context.get_replica_context()
      return ctx._all_gather([value_identity, value_identity], axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    all_value = [single_value for _ in range(strategy.num_replicas_in_sync)]
    expect = array_ops.concat(all_value, axis=axis)
    expected_per_replica = [expect] * _get_num_devices_per_worker(strategy)

    result = strategy.run(run)
    for gathered_result in result:
      self.assertAllEqual(
          strategy.experimental_local_results(gathered_result),
          expected_per_replica)

  def testAllGatherRaiseDiffShapeAtNonAxis(self, strategy, pure_eager):
    """Different at non-`axis`-th dimension : [2, 1], [1, 1], all_gather(...axis=1...) -> raise error."""
    if _get_num_devices_per_worker(strategy) > 1:
      self.skipTest('b/167331966')

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    def run(value):
      value_identity = array_ops.identity(value)
      ctx = ds_context.get_replica_context()
      return ctx._all_gather(value_identity, axis=0)

    if not pure_eager:
      run = def_function.function(run)

    with self.assertRaisesRegex(errors.InvalidArgumentError, r'Shape mismatch'):
      strategy.run(run, args=(per_replica_value,))

  def testAllGatherRaiseSparsePerReplica(self, strategy, pure_eager):
    # all_gather supports sparse when using tf.function, because sparse tensors
    # are converted to dense in
    # third_party/tensorflow/python/ops/custom_gradient.py _graph_mode_decorator
    if strategy.num_replicas_in_sync != 2:
      self.skipTest('Test for two replicas.')
    dense_shape = [5, 2]
    t0 = _make_indexed_slices(
        values=[[1., 2.]], indices=[2], dense_shape=dense_shape)

    def replica_fn(value):
      ctx = ds_context.get_replica_context()
      return ctx._all_gather(value, axis=0)

    with self.assertRaisesRegex(
        NotImplementedError,
        r'gather/all_gather does not support IndexedSlices'):
      strategy.run(replica_fn, args=(t0,))

  def testAllGatherRaiseDifferentRank(self, strategy, pure_eager):
    """Different rank: [1,], [1, 2] -> raise error."""
    if strategy.num_replicas_in_sync <= 1:
      self.skipTest('Test for more than 1 replicas.')
    if _get_num_devices_per_worker(strategy) > 1:
      self.skipTest('b/167331966')
    def value_fn(ctx):
      return array_ops.ones(shape=(range(1, ctx.replica_id_in_sync_group + 2)))

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    def run(value):
      value_identity = array_ops.identity(value)
      ctx = ds_context.get_replica_context()
      return ctx._all_gather(value_identity, axis=0)

    error_message = 'Shape mismatch'

    if not pure_eager:
      run = def_function.function(run)

    with self.assertRaisesRegex(errors.InvalidArgumentError, error_message):
      strategy.run(run, args=(per_replica_value,))


def _make_indexed_slices(values, indices, dense_shape):
  tensor = ops.IndexedSlices(
      values=constant_op.constant(values),
      indices=constant_op.constant(indices),
      dense_shape=constant_op.constant(dense_shape))
  return tensor


def _get_num_devices_per_worker(strategy):
  """Returns the number of workers in the current cluster for multi-worker."""
  resolver = strategy.cluster_resolver
  return max(nest.flatten(resolver.num_accelerators())[0], 1)


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
        strategy.experimental_distribute_datasets_from_function(dataset_fn))

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
        result.numpy(),
        expected_data_on_workers[multi_worker_test_base.get_task_type()])

  def testSimpleInputFromFnLastPartialBatch(self, strategy):

    def dataset_fn(input_context):
      global_batch_size = 8
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      dataset = dataset_ops.DatasetV2.range(14).batch(
          batch_size, drop_remainder=False)
      return dataset.shard(input_context.num_input_pipelines,
                           input_context.input_pipeline_id)

    input_iterator = iter(
        strategy.experimental_distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(input_iterator):
      return strategy.run(lambda x: x, args=(next(input_iterator),))

    # Let the complete batch go.
    run(input_iterator)
    # `result` is an incomplete batch
    result = run(input_iterator)

    expected_data_on_worker = {'chief': [8, 9, 10, 11], 'worker': [12, 13]}
    self.assertAllEqual(
        result.numpy(),
        expected_data_on_worker[multi_worker_test_base.get_task_type()])

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
    self.assertAllEqual(reduced, [2., 2.])

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
    self.assertAllEqual(reduced, [2., 2.])

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
    if not isinstance(strategy, CollectiveAllReduceStrategy) and not isinstance(
        strategy, TPUStrategy):
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
  v2_compat.enable_v2_behavior()
  combinations.main()
