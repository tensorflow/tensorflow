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
"""Types specific to tf.distribute."""

from tensorflow.python.util.tf_export import tf_export

# TODO(mdan, anjalisridhar): Decide the location of this file.


class Iterable(object):
  """Interface for distributed objects that admit iteration/reduction."""

  def __iter__(self):
    pass

  # TODO(mdan): Describe this contract.
  def reduce(self, initial_state, reduce_func):
    """Reduces this iterable object to a single element.

    The transformation calls `reduce_func` successively on each element.
    The `initial_state` argument is used for the initial state and the final
    state is returned as the result.

    Args:
      initial_state: An element representing the initial state of the
        reduction.
      reduce_func: A function that maps `(old_state, input_element)` to
        `new_state`. The structure of `new_state` must match the structure of
        `old_state`. For the first element, `old_state` is `initial_state`.

    Returns:
      The final state of the transformation.
    """


class Iterator(object):
  """Interface for distributed iterators."""

  def get_next(self):
    """Unlike __next__, this may use a non-raising mechanism."""

  def __next__(self):
    pass

  def __iter__(self):
    pass


@tf_export("distribute.DistributedValues", v1=[])
class DistributedValues(object):
  """Base class for representing distributed values.

  A subclass instance of `tf.distribute.DistributedValues` is created when
  creating variables within a distribution strategy, iterating a
  `tf.distribute.DistributedDataset` or through `tf.distribute.Strategy.run`.
  This base class should never be instantiated directly.
  `tf.distribute.DistributedValues` contains a value per replica. Depending on
  the subclass, the values could either be synced on update, synced on demand,
  or never synced.

  Two representative types of `tf.distribute.DistributedValues` are
  `tf.types.experimental.PerReplica` and `tf.types.experimental.Mirrored`
  values.

  `PerReplica` values exist on the worker devices, with a different value for
  each replica. They are produced by iterating through a distributed dataset
  returned by `tf.distribute.Strategy.experimental_distribute_dataset` (Example
  1, below) and `tf.distribute.Strategy.distribute_datasets_from_function`. They
  are also the typical result returned by `tf.distribute.Strategy.run` (Example
  2).

  `Mirrored` values are like `PerReplica` values, except we know that the value
  on all replicas are the same. `Mirrored` values are kept synchronized by the
  distribution strategy in use, while `PerReplica` values are left
  unsynchronized. `Mirrored` values typically represent model weights. We can
  safely read a `Mirrored` value in a cross-replica context by using the value
  on any replica, while PerReplica values should not be read or manipulated in
  a cross-replica context."

  `tf.distribute.DistributedValues` can be reduced via `strategy.reduce` to
  obtain a single value across replicas (Example 4), used as input into
  `tf.distribute.Strategy.run` (Example 3), or collected to inspect the
  per-replica values using `tf.distribute.Strategy.experimental_local_results`
  (Example 5).

  Example usages:

  1. Created from a `tf.distribute.DistributedDataset`:

  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> distributed_values
  PerReplica:{
    0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.], dtype=float32)>,
    1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>
  }

  2. Returned by `run`:

  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> @tf.function
  ... def run():
  ...   ctx = tf.distribute.get_replica_context()
  ...   return ctx.replica_id_in_sync_group
  >>> distributed_values = strategy.run(run)
  >>> distributed_values
  PerReplica:{
    0: <tf.Tensor: shape=(), dtype=int32, numpy=0>,
    1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
  }

  3. As input into `run`:

  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> @tf.function
  ... def run(input):
  ...   return input + 1.0
  >>> updated_value = strategy.run(run, args=(distributed_values,))
  >>> updated_value
  PerReplica:{
    0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>,
    1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([7.], dtype=float32)>
  }

  4. As input into `reduce`:

  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> reduced_value = strategy.reduce(tf.distribute.ReduceOp.SUM,
  ...                                 distributed_values,
  ...                                 axis = 0)
  >>> reduced_value
  <tf.Tensor: shape=(), dtype=float32, numpy=11.0>

  5. How to inspect per-replica values locally:

  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> per_replica_values = strategy.experimental_local_results(
  ...    distributed_values)
  >>> per_replica_values
  (<tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.], dtype=float32)>,
   <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>)

  """


@tf_export("types.experimental.distributed.PerReplica", v1=[])
class PerReplica(DistributedValues):
  """Holds a distributed value: a map from replica id to unsynchronized values.

  `PerReplica` values exist on the worker devices, with a different value for
  each replica. They can be produced many ways, often by iterating through a
  distributed dataset returned by
  `tf.distribute.Strategy.experimental_distribute_dataset` and
  `tf.distribute.Strategy.distribute_datasets_from_function`. They are also the
  typical result returned by `tf.distribute.Strategy.run`.
  """


@tf_export("types.experimental.distributed.Mirrored", v1=[])
class Mirrored(DistributedValues):
  """Holds a distributed value: a map from replica id to synchronized values.

  `Mirrored` values are `tf.distribute.DistributedValues` for which we know that
  the value on all replicas is the same. `Mirrored` values are kept synchronized
  by the distribution strategy in use, while `tf.types.experimental.PerReplica`
  values are left unsynchronized. `Mirrored` values typically represent model
  weights. We can safely read a `Mirrored` value in a cross-replica context by
  using the value on any replica, while `PerReplica` values should not be read
  or manipulated directly by the user in a cross-replica context.
  """
