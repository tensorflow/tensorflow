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
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@tf_test_util.with_eager_op_as_function
@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.default_strategy,
            strategy_combinations.one_device_strategy,
            strategy_combinations.one_device_strategy_gpu,
            strategy_combinations.central_storage_strategy_with_two_gpus,
            strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
            strategy_combinations.mirrored_strategy_with_one_cpu,
            strategy_combinations.mirrored_strategy_with_one_gpu,
            strategy_combinations.mirrored_strategy_with_two_gpus,
            strategy_combinations.mirrored_strategy_with_two_cpus,
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ],
        mode=['eager'],
        pure_eager=[True, False]) + combinations.combine(
            strategy=[
                strategy_combinations.tpu_strategy,
                strategy_combinations.tpu_strategy_packed_var,
                strategy_combinations.tpu_strategy_one_step,
                strategy_combinations.cloud_tpu_strategy,
            ],
            mode=['eager'],
            pure_eager=[False]))
class GatherTest(test.TestCase, parameterized.TestCase):

  def _gather_same_shape_and_verify(self, value_on_replica, axis, pure_eager,
                                    strategy):
    distributed_values = strategy.experimental_distribute_values_from_function(
        lambda _: array_ops.identity(value_on_replica))

    def run():
      return strategy.gather(distributed_values, axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    all_results = [
        value_on_replica for _ in range(strategy.num_replicas_in_sync)
    ]
    expected_result = array_ops.concat(all_results, axis=axis)
    self.assertAllEqual(expected_result, run().numpy())

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
      return strategy.gather(distributed_values, axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    expected_result = constant_op.constant(
        1, shape=(sum(range(strategy.num_replicas_in_sync + 1)), 1))

    self.assertAllEqual(expected_result, run().numpy())

  def testGatherDiffShapeAtAxis1(self, strategy, pure_eager):
    """Different `Axis`-th (non-0) dimension: shape [1, 1], [1, 2] -> [1, 3]."""

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn)
    axis = 1

    def run():
      return strategy.gather(distributed_values, axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    expected_result = constant_op.constant(
        1, shape=(1, sum(range(strategy.num_replicas_in_sync + 1))))

    self.assertAllEqual(expected_result, run().numpy())

  def testGatherRaiseDiffShapeAtNonAxis(self, strategy, pure_eager):
    """Different at non-`axis`-th dimension : [1, 1], [1, 2], 0th -> raise error."""
    if isinstance(strategy, CollectiveAllReduceStrategy
                 ) and _get_num_replicas_per_client(strategy) > 1:
      self.skipTest('b/167331966')

    if strategy.num_replicas_in_sync <= 1:
      self.skipTest('Test for more than 1 replica only.')

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn)
    axis = 0

    def run():
      return strategy.gather(distributed_values, axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    if isinstance(strategy, CollectiveAllReduceStrategy):
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  r'Shape mismatch'):
        run()
    elif isinstance(strategy,
                    (mirrored_strategy.MirroredStrategy,
                     central_storage_strategy.CentralStorageStrategy)):
      with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                  r'Dimension \d in both shapes must be equal'):
        run()

  def testGatherRaiseSparse(self, strategy, pure_eager):
    dense_shape = [5, 2]
    t0 = _make_indexed_slices(
        values=[[1., 2.]], indices=[2], dense_shape=dense_shape)

    def run(value):
      return strategy.gather(value, axis=0)

    with self.assertRaisesRegex(
        NotImplementedError,
        r'gather does not support IndexedSlices'):
      if pure_eager:
        run(t0)
      else:
        def_function.function(run)(t0)

  def testGatherRaiseDifferentRank(self, strategy, pure_eager):
    """Different rank: [1,], [1, 2] -> raise error."""
    if strategy.num_replicas_in_sync <= 1:
      self.skipTest('Test for more than 1 replicas.')
    if isinstance(strategy, CollectiveAllReduceStrategy
                 ) and _get_num_replicas_per_client(strategy) > 1:
      self.skipTest('b/167331966')
    def value_fn(ctx):
      return array_ops.ones(shape=(range(1, ctx.replica_id_in_sync_group + 2)))

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn)
    axis = 0

    def run():
      return strategy.gather(distributed_values, axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    if isinstance(strategy, CollectiveAllReduceStrategy):
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  r'Shape mismatch'):
        run()
    elif isinstance(
        strategy,
        (mirrored_strategy.MirroredStrategy,
         central_storage_strategy.CentralStorageStrategy)):
      if pure_eager:
        with self.assertRaises(errors.InvalidArgumentError) as e:
          run()
        # Different error message depending on whether collective ops is used.
        self.assertRegexMatch(
            str(e.exception),
            ['Ranks of all input tensors should match', 'Shape mismatch'])
      else:
        with self.assertRaises((errors.InvalidArgumentError, ValueError)) as e:
          run()
        self.assertRegexMatch(
            str(e.exception),
            [r'Shape must be rank \d but is rank \d', 'Shape mismatch'])
    elif _is_tpu_strategy(strategy) and pure_eager:
      with self.assertRaisesRegex(ValueError,
                                  r'Dimension \d in both shapes must be equal'):
        run()
    else:
      with self.assertRaisesRegex(ValueError,
                                  r'Shape must be rank \d but is rank \d'):
        run()

  # Ideally, here we should split them into another test class, AllGatherTest.
  # But doing that makes two initialize_tpu_system() calls and one of them times
  # out, on Kokoro. Integrating two into one avoids it.
  def _all_gather_same_shape_and_verify(self, value_on_replica, axis,
                                        pure_eager, strategy):
    per_replica_value = strategy.experimental_distribute_values_from_function(
        lambda _: array_ops.identity(value_on_replica))

    def replica_fn(per_replica_value):
      ctx = distribute_lib.get_replica_context()
      local_value = array_ops.identity(per_replica_value)
      return ctx.all_gather(local_value, axis=axis)

    if not pure_eager:
      replica_fn = def_function.function(replica_fn)

    result = strategy.experimental_local_results(
        strategy.run(replica_fn, args=(per_replica_value,)))

    all_value = [value_on_replica for _ in range(strategy.num_replicas_in_sync)]
    expect = array_ops.concat(all_value, axis=axis)
    expected_result = [expect] * _get_num_replicas_per_client(strategy)

    self.assertAllClose(expected_result, result)

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

  def testAllGatherDiffValueTPU(self, strategy, pure_eager):
    # Test for TPU only since it can't be tested via testAllGatherDiffShape*
    if not _is_tpu_strategy(strategy):
      self.skipTest('Test for TPU only. For other strategies case already'
                    ' covered in other tests')

    data = [[1], [2], [3], [4], [5], [6], [7], [8]]

    axis = 0
    dataset = dataset_ops.DatasetV2.from_tensor_slices(data).batch(8)
    input_iterator = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def replica_fn(per_replica_value):
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather(array_ops.identity(per_replica_value), axis=axis)

    result = strategy.experimental_local_results(
        strategy.run(replica_fn, args=(next(input_iterator),)))

    expected_result = [data] * _get_num_replicas_per_client(strategy)
    self.assertAllClose(expected_result, result)

  def testAllGatherDiffShapeAtAxis0(self, strategy, pure_eager):
    """Different `Axis==0`-th dimension: shape [1, 1], [2, 1] -> [3, 1]."""

    if _is_tpu_strategy(strategy):
      self.skipTest('TPU does not support all_gather different shapes')

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(ctx.replica_id_in_sync_group + 1, 1))

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    expect = constant_op.constant(
        1, shape=(sum(range(strategy.num_replicas_in_sync + 1)), 1))

    def run(value):
      value_identity = array_ops.identity(value)
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather(value_identity, axis=0)

    if not pure_eager:
      run = def_function.function(run)

    expected_result = [expect] * _get_num_replicas_per_client(strategy)
    result = strategy.experimental_local_results(
        strategy.run(run, args=(per_replica_value,)))
    self.assertAllEqual(expected_result, result)

  def testAllGatherDiffShapeAtAxis1(self, strategy, pure_eager):
    """Different `Axis`-th (not 0th) dimension: shape [1, 1], [1, 2] -> [1, 3]."""
    if _is_tpu_strategy(strategy):
      self.skipTest('TPU does not support all_gather different shapes')

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    expect = constant_op.constant(
        1, shape=(1, sum(range(strategy.num_replicas_in_sync + 1))))

    def run(value):
      value_identity = array_ops.identity(value)
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather(value_identity, axis=1)

    if not pure_eager:
      run = def_function.function(run)

    expected_result = [expect] * _get_num_replicas_per_client(strategy)
    result = strategy.experimental_local_results(
        strategy.run(run, args=(per_replica_value,)))
    self.assertAllEqual(expected_result, result)

  def testAllGatherNest(self, strategy, pure_eager):
    if _is_tpu_strategy(strategy):
      self.skipTest('TPU does not support all_gather different shapes')

    axis = 1

    def value_fn(ctx):
      value = constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))
      return value
    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    expect_1 = constant_op.constant(
        1, shape=(1, sum(range(strategy.num_replicas_in_sync + 1))))

    expected_per_replica_1 = [expect_1] * _get_num_replicas_per_client(strategy)

    value_2 = constant_op.constant([[[1, 2], [1, 2]]])

    expect_2 = array_ops.concat(
        [value_2 for _ in range(strategy.num_replicas_in_sync)], axis=axis)

    expected_per_replica_2 = [expect_2] * _get_num_replicas_per_client(strategy)

    def run(value):
      value_1 = array_ops.identity(value)
      value_3 = array_ops.identity(value_2)
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather([value_1, value_3], axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    result = strategy.run(run, args=(per_replica_value,))
    self.assertAllEqual(expected_per_replica_1,
                        strategy.experimental_local_results(result[0]))
    self.assertAllEqual(expected_per_replica_2,
                        strategy.experimental_local_results(result[1]))

  def testAllGatherNest1D0Axis(self, strategy, pure_eager):
    """all_gather(..., axis=0,...) a nest of DistributedValues."""
    single_value = constant_op.constant([1, 2, 3])
    axis = 0

    def run():
      value_identity = array_ops.identity(single_value)
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather([value_identity, value_identity], axis=axis)

    if not pure_eager:
      run = def_function.function(run)

    all_value = [single_value for _ in range(strategy.num_replicas_in_sync)]
    expect = array_ops.concat(all_value, axis=axis)
    expected_per_replica = [expect] * _get_num_replicas_per_client(strategy)

    result = strategy.run(run)
    for gathered_result in result:
      self.assertAllEqual(expected_per_replica,
                          strategy.experimental_local_results(gathered_result))

  def testAllGatherRaiseDiffShapeAtNonAxis(self, strategy, pure_eager):
    """Different at non-`axis`-th dimension : [2, 1], [1, 1], all_gather(...axis=1...) -> raise error."""
    if _is_tpu_strategy(strategy):
      self.skipTest('TODO(b/169108777): raise a clear error message in xla.')

    if isinstance(strategy, CollectiveAllReduceStrategy
                 ) and _get_num_replicas_per_client(strategy) > 1:
      self.skipTest('b/167331966')

    if strategy.num_replicas_in_sync <= 1:
      self.skipTest('Test for more than 1 replica only.')

    def value_fn(ctx):
      return constant_op.constant(
          1, shape=(1, ctx.replica_id_in_sync_group + 1))

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    def run(value):
      value_identity = array_ops.identity(value)
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather(value_identity, axis=0)

    if not pure_eager:
      run = def_function.function(run)

    if isinstance(strategy, CollectiveAllReduceStrategy):
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  r'Shape mismatch'):
        strategy.run(run, args=(per_replica_value,))
    elif isinstance(strategy,
                    (mirrored_strategy.MirroredStrategy,
                     central_storage_strategy.CentralStorageStrategy)):
      with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                  r'Dimension \d in both shapes must be equal'):
        strategy.run(run, args=(per_replica_value,))

  def testAllGatherRaiseSparse(self, strategy, pure_eager):
    dense_shape = [5, 2]
    t0 = _make_indexed_slices(
        values=[[1., 2.]], indices=[2], dense_shape=dense_shape)

    def replica_fn(value):
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather(value, axis=0)

    with self.assertRaisesRegex(
        NotImplementedError,
        r'all_gather does not support IndexedSlices'):
      if not pure_eager:
        strategy.run(def_function.function(replica_fn), args=(t0,))
      else:
        strategy.run(replica_fn, args=(t0,))

  def testAllGatherRaiseDifferentRank(self, strategy, pure_eager):
    """Different rank: [1,], [1, 2] -> raise error."""
    if _is_tpu_strategy(strategy):
      self.skipTest('TODO(b/169108777): raise a clear error message in xla.')

    if strategy.num_replicas_in_sync <= 1:
      self.skipTest('Test for more than 1 replicas.')
    if isinstance(strategy, CollectiveAllReduceStrategy
                 ) and _get_num_replicas_per_client(strategy) > 1:
      self.skipTest('b/167331966')
    def value_fn(ctx):
      return array_ops.ones(shape=(range(1, ctx.replica_id_in_sync_group + 2)))

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)

    def run(value):
      value_identity = array_ops.identity(value)
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather(value_identity, axis=0)

    if not pure_eager:
      run = def_function.function(run)

    if isinstance(strategy, CollectiveAllReduceStrategy):
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  r'Shape mismatch'):
        strategy.run(run, args=(per_replica_value,))
    elif isinstance(strategy,
                    (mirrored_strategy.MirroredStrategy,
                     central_storage_strategy.CentralStorageStrategy)):
      if pure_eager:
        with self.assertRaises(errors.InvalidArgumentError) as e:
          strategy.run(run, args=(per_replica_value,))
        # Different error message depending on whether collective ops is used.
        self.assertRegexMatch(
            str(e.exception),
            ['Ranks of all input tensors should match', 'Shape mismatch'])
      else:
        with self.assertRaises((errors.InvalidArgumentError, ValueError)) as e:
          strategy.run(run, args=(per_replica_value,))
        self.assertRegexMatch(
            str(e.exception),
            [r'Shape must be rank \d but is rank \d', 'Shape mismatch'])
    else:
      with self.assertRaisesRegex(ValueError,
                                  r'Dimension \d in both shapes must be equal'):
        strategy.run(run, args=(per_replica_value,))

  def testAllGatherGradient(self, strategy, pure_eager):
    if pure_eager:
      self.skipTest('`tf.gradients` is not supported with eager execution '
                    'without using tf.functions.')

    def all_gather_fn(value):
      axis = 1
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather(array_ops.identity(value), axis)

    gradient_comp = sum(range(1, strategy.num_replicas_in_sync + 1))
    gradient = [[gradient_comp], [gradient_comp]]
    grads_for_all_replicas = [gradient] * _get_num_replicas_per_client(strategy)

    @def_function.function
    def step(c):
      x = constant_op.constant([[3.], [5.]])
      mid = all_gather_fn(x)
      y = mid * c
      return gradients_impl.gradients_v2(y, [x])[0]

    def value_fn(ctx):
      x = [1., 2., 3., 4., 5., 6., 7., 8.]
      return array_ops.constant([x[ctx.replica_id_in_sync_group]])

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)
    result = strategy.experimental_local_results(
        strategy.run(step, args=(per_replica_value,)))

    self.assertAllEqual(grads_for_all_replicas, result)

  def testAllGatherGradientNest(self, strategy, pure_eager):
    if pure_eager:
      self.skipTest('`tf.gradients` is not supported with eager execution '
                    'without using tf.functions.')

    def all_gather_fn(value):
      axis = 1
      ctx = distribute_lib.get_replica_context()
      return ctx.all_gather(array_ops.identity(value), axis)

    gradient_comp = sum(range(1, strategy.num_replicas_in_sync + 1))
    gradient = [[gradient_comp], [gradient_comp]]
    grads_for_all_replicas = [gradient] * _get_num_replicas_per_client(strategy)

    @def_function.function
    def step(c):
      x = constant_op.constant([[3.], [5.]])
      y = constant_op.constant([[2.], [4.]])
      mid = all_gather_fn([x, y])
      y = mid * c
      return gradients_impl.gradients_v2(y, [x])[0]

    def value_fn(ctx):
      x = [1., 2., 3., 4., 5., 6., 7., 8.]
      return array_ops.constant([x[ctx.replica_id_in_sync_group]])

    per_replica_value = strategy.experimental_distribute_values_from_function(
        value_fn)
    result = strategy.experimental_local_results(
        strategy.run(step, args=(per_replica_value,)))

    self.assertAllEqual(grads_for_all_replicas, result)


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


def _is_tpu_strategy(strategy):
  return isinstance(strategy,
                    (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1,
                     tpu_strategy.TPUStrategyV2))


if __name__ == '__main__':
  test_util.main()
