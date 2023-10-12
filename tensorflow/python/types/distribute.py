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
from tensorflow.tools.docs import doc_controls

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


@tf_export("distribute.DistributedIterator", v1=[])
class DistributedIteratorInterface(Iterator):
  """An iterator over `tf.distribute.DistributedDataset`.

  `tf.distribute.DistributedIterator` is the primary mechanism for enumerating
  elements of a `tf.distribute.DistributedDataset`. It supports the Python
  Iterator protocol, which means it can be iterated over using a for-loop or by
  fetching individual elements explicitly via `get_next()`.

  You can create a `tf.distribute.DistributedIterator` by calling `iter` on
  a `tf.distribute.DistributedDataset` or creating a python loop over a
  `tf.distribute.DistributedDataset`.

  Visit the [tutorial](https://www.tensorflow.org/tutorials/distribute/input)
  on distributed input for more examples and caveats.
  """

  def get_next(self):
    """Returns the next input from the iterator for all replicas.

    Example use:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> dataset = tf.data.Dataset.range(100).batch(2)
    >>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
    >>> dist_dataset_iterator = iter(dist_dataset)
    >>> @tf.function
    ... def one_step(input):
    ...   return input
    >>> step_num = 5
    >>> for _ in range(step_num):
    ...   strategy.run(one_step, args=(dist_dataset_iterator.get_next(),))
    >>> strategy.experimental_local_results(dist_dataset_iterator.get_next())
    (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([10])>,
     <tf.Tensor: shape=(1,), dtype=int64, numpy=array([11])>)

    Returns:
      A single `tf.Tensor` or a `tf.distribute.DistributedValues` which contains
      the next input for all replicas.

    Raises:
      `tf.errors.OutOfRangeError`: If the end of the iterator has been reached.
    """
    raise NotImplementedError(
        "DistributedIterator.get_next() must be implemented in descendants.")

  @property
  def element_spec(self):
    # pylint: disable=line-too-long
    """The type specification of an element of `tf.distribute.DistributedIterator`.

    Example usage:

    >>> global_batch_size = 16
    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> dataset = tf.data.Dataset.from_tensors(([1.],[2])).repeat(100).batch(global_batch_size)
    >>> distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    >>> distributed_iterator.element_spec
    (PerReplicaSpec(TensorSpec(shape=(None, 1), dtype=tf.float32, name=None),
                    TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)),
     PerReplicaSpec(TensorSpec(shape=(None, 1), dtype=tf.int32, name=None),
                    TensorSpec(shape=(None, 1), dtype=tf.int32, name=None)))

    Returns:
      A nested structure of `tf.TypeSpec` objects matching the structure of an
      element of this `tf.distribute.DistributedIterator`. This returned value
      is typically a `tf.distribute.DistributedValues` object and specifies the
      `tf.TensorSpec` of individual components.
    """
    raise NotImplementedError(
        "DistributedIterator.element_spec() must be implemented in descendants")

  def get_next_as_optional(self):
    # pylint: disable=line-too-long
    """Returns a `tf.experimental.Optional` that contains the next value for all replicas.

    If the `tf.distribute.DistributedIterator` has reached the end of the
    sequence, the returned `tf.experimental.Optional` will have no value.

    Example usage:

    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> global_batch_size = 2
    >>> steps_per_loop = 2
    >>> dataset = tf.data.Dataset.range(10).batch(global_batch_size)
    >>> distributed_iterator = iter(
    ...     strategy.experimental_distribute_dataset(dataset))
    >>> def step_fn(x):
    ...   # train the model with inputs
    ...   return x
    >>> @tf.function
    ... def train_fn(distributed_iterator):
    ...   for _ in tf.range(steps_per_loop):
    ...     optional_data = distributed_iterator.get_next_as_optional()
    ...     if not optional_data.has_value():
    ...       break
    ...     per_replica_results = strategy.run(step_fn, args=(optional_data.get_value(),))
    ...     tf.print(strategy.experimental_local_results(per_replica_results))
    >>> train_fn(distributed_iterator)
    ... # ([0 1], [2 3])
    ... # ([4], [])

    Returns:
      An `tf.experimental.Optional` object representing the next value from the
      `tf.distribute.DistributedIterator` (if it has one) or no value.
    """
    # pylint: enable=line-too-long
    raise NotImplementedError(
        "get_next_as_optional() not implemented in descendants")


@tf_export("distribute.DistributedDataset", v1=[])
class DistributedDatasetInterface(Iterable):
  # pylint: disable=line-too-long
  """Represents a dataset distributed among devices and machines.

  A `tf.distribute.DistributedDataset` could be thought of as a "distributed"
  dataset. When you use `tf.distribute` API to scale training to multiple
  devices or machines, you also need to distribute the input data, which leads
  to a `tf.distribute.DistributedDataset` instance, instead of a
  `tf.data.Dataset` instance in the non-distributed case. In TF 2.x,
  `tf.distribute.DistributedDataset` objects are Python iterables.

  Note: `tf.distribute.DistributedDataset` instances are *not* of type
  `tf.data.Dataset`. It only supports two usages we will mention below:
  iteration and `element_spec`. We don't support any other APIs to transform or
  inspect the dataset.

  There are two APIs to create a `tf.distribute.DistributedDataset` object:
  `tf.distribute.Strategy.experimental_distribute_dataset(dataset)`and
  `tf.distribute.Strategy.distribute_datasets_from_function(dataset_fn)`.
  *When to use which?* When you have a `tf.data.Dataset` instance, and the
  regular batch splitting (i.e. re-batch the input `tf.data.Dataset` instance
  with a new batch size that is equal to the global batch size divided by the
  number of replicas in sync) and autosharding (i.e. the
  `tf.data.experimental.AutoShardPolicy` options) work for you, use the former
  API. Otherwise, if you are *not* using a canonical `tf.data.Dataset` instance,
  or you would like to customize the batch splitting or sharding, you can wrap
  these logic in a `dataset_fn` and use the latter API. Both API handles
  prefetch to device for the user. For more details and examples, follow the
  links to the APIs.


  There are two main usages of a `DistributedDataset` object:

  1. Iterate over it to generate the input for a single device or multiple
  devices, which is a `tf.distribute.DistributedValues` instance. To do this,
  you can:

    * use a pythonic for-loop construct:

      >>> global_batch_size = 4
      >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
      >>> dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(4).batch(global_batch_size)
      >>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
      >>> @tf.function
      ... def train_step(input):
      ...   features, labels = input
      ...   return labels - 0.3 * features
      >>> for x in dist_dataset:
      ...   # train_step trains the model using the dataset elements
      ...   loss = strategy.run(train_step, args=(x,))
      ...   print("Loss is", loss)
      Loss is PerReplica:{
        0: tf.Tensor(
      [[0.7]
       [0.7]], shape=(2, 1), dtype=float32),
        1: tf.Tensor(
      [[0.7]
       [0.7]], shape=(2, 1), dtype=float32)
      }

      Placing the loop inside a `tf.function` will give a performance boost.
      However `break` and `return` are currently not supported if the loop is
      placed inside a `tf.function`. We also don't support placing the loop
      inside a `tf.function` when using
      `tf.distribute.experimental.MultiWorkerMirroredStrategy` or
      `tf.distribute.experimental.TPUStrategy` with multiple workers.

    * use `__iter__` to create an explicit iterator, which is of type
      `tf.distribute.DistributedIterator`

      >>> global_batch_size = 4
      >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
      >>> train_dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(50).batch(global_batch_size)
      >>> train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
      >>> @tf.function
      ... def distributed_train_step(dataset_inputs):
      ...   def train_step(input):
      ...     loss = tf.constant(0.1)
      ...     return loss
      ...   per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
      ...   return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
      >>> EPOCHS = 2
      >>> STEPS = 3
      >>> for epoch in range(EPOCHS):
      ...   total_loss = 0.0
      ...   num_batches = 0
      ...   dist_dataset_iterator = iter(train_dist_dataset)
      ...   for _ in range(STEPS):
      ...     total_loss += distributed_train_step(next(dist_dataset_iterator))
      ...     num_batches += 1
      ...   average_train_loss = total_loss / num_batches
      ...   template = ("Epoch {}, Loss: {:.4f}")
      ...   print (template.format(epoch+1, average_train_loss))
      Epoch 1, Loss: 0.2000
      Epoch 2, Loss: 0.2000


    To achieve a performance improvement, you can also wrap the `strategy.run`
    call with a `tf.range` inside a `tf.function`. This runs multiple steps in a
    `tf.function`. Autograph will convert it to a `tf.while_loop` on the worker.
    However, it is less flexible comparing with running a single step inside
    `tf.function`. For example, you cannot run things eagerly or arbitrary
    python code within the steps.


  2. Inspect the `tf.TypeSpec` of the data generated by `DistributedDataset`.

    `tf.distribute.DistributedDataset` generates
    `tf.distribute.DistributedValues` as input to the devices. If you pass the
    input to a `tf.function` and would like to specify the shape and type of
    each Tensor argument to the function, you can pass a `tf.TypeSpec` object to
    the `input_signature` argument of the `tf.function`. To get the
    `tf.TypeSpec` of the input, you can use the `element_spec` property of the
    `tf.distribute.DistributedDataset` or `tf.distribute.DistributedIterator`
    object.

    For example:

    >>> global_batch_size = 4
    >>> epochs = 1
    >>> steps_per_epoch = 1
    >>> mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> dataset = tf.data.Dataset.from_tensors(([2.])).repeat(100).batch(global_batch_size)
    >>> dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    >>> @tf.function(input_signature=[dist_dataset.element_spec])
    ... def train_step(per_replica_inputs):
    ...   def step_fn(inputs):
    ...     return tf.square(inputs)
    ...   return mirrored_strategy.run(step_fn, args=(per_replica_inputs,))
    >>> for _ in range(epochs):
    ...   iterator = iter(dist_dataset)
    ...   for _ in range(steps_per_epoch):
    ...     output = train_step(next(iterator))
    ...     print(output)
    PerReplica:{
      0: tf.Tensor(
    [[4.]
     [4.]], shape=(2, 1), dtype=float32),
      1: tf.Tensor(
    [[4.]
     [4.]], shape=(2, 1), dtype=float32)
    }


  Visit the [tutorial](https://www.tensorflow.org/tutorials/distribute/input)
  on distributed input for more examples and caveats.
  """

  def __iter__(self):
    """Creates an iterator for the `tf.distribute.DistributedDataset`.

    The returned iterator implements the Python Iterator protocol.

    Example usage:

    >>> global_batch_size = 4
    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4]).repeat().batch(global_batch_size)
    >>> distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    >>> print(next(distributed_iterator))
    PerReplica:{
      0: tf.Tensor([1 2], shape=(2,), dtype=int32),
      1: tf.Tensor([3 4], shape=(2,), dtype=int32)
    }

    Returns:
      An `tf.distribute.DistributedIterator` instance for the given
      `tf.distribute.DistributedDataset` object to enumerate over the
      distributed data.
    """
    raise NotImplementedError("Must be implemented in descendants")

  @property
  def element_spec(self):
    """The type specification of an element of this `tf.distribute.DistributedDataset`.

    Example usage:

    >>> global_batch_size = 16
    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> dataset = tf.data.Dataset.from_tensors(([1.],[2])).repeat(100).batch(global_batch_size)
    >>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
    >>> dist_dataset.element_spec
    (PerReplicaSpec(TensorSpec(shape=(None, 1), dtype=tf.float32, name=None),
                    TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)),
     PerReplicaSpec(TensorSpec(shape=(None, 1), dtype=tf.int32, name=None),
                    TensorSpec(shape=(None, 1), dtype=tf.int32, name=None)))

    Returns:
      A nested structure of `tf.TypeSpec` objects matching the structure of an
      element of this `tf.distribute.DistributedDataset`. This returned value is
      typically a `tf.distribute.DistributedValues` object and specifies the
      `tf.TensorSpec` of individual components.
    """
    raise NotImplementedError(
        "DistributedDataset.element_spec must be implemented in descendants.")

  @doc_controls.do_not_generate_docs
  def reduce(self, initial_state, reduce_func):
    raise NotImplementedError(
        "DistributedDataset.reduce must be implemented in descendants.")
