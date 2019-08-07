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
"""Class implementing a single machine parameter server strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.util.tf_export import tf_export


@tf_export("distribute.experimental.CentralStorageStrategy", v1=[])
class CentralStorageStrategy(distribute_lib.Strategy):
  """A one-machine strategy that puts all variables on a single device.

  Variables are assigned to local CPU or the only GPU. If there is more
  than one GPU, compute operations (other than variable update operations)
  will be replicated across all GPUs.

  For Example:
  ```
  strategy = tf.distribute.experimental.CentralStorageStrategy()
  # Create a dataset
  ds = tf.data.Dataset.range(5).batch(2)
  # Distribute that dataset
  dist_dataset = strategy.experimental_distribute_dataset(ds)

  with strategy.scope():
    @tf.function
    def train_step(val):
      return val + 1

    # Iterate over the distributed dataset
    for x in dist_dataset:
      # process dataset elements
      strategy.experimental_run_v2(train_step, args=(x,))
  ```
  """

  def __init__(self, compute_devices=None, parameter_device=None):
    extended = parameter_server_strategy.ParameterServerStrategyExtended(
        self,
        compute_devices=compute_devices,
        parameter_device=parameter_device)
    """Initializes the strategy with optional device strings.

    Args:
    compute_devices: an optional list of strings for device to replicate models
      on. If this is not provided, all local GPUs will be used; if there is no
      GPU, local CPU will be used.
    parameter_device: an optional device string for which device to put
      variables on. The default one is CPU or GPU if there is only one.
    """
    super(CentralStorageStrategy, self).__init__(extended)

  @classmethod
  def _from_num_gpus(cls, num_gpus):
    return cls(device_util.local_devices_from_num_gpus(num_gpus))

  def experimental_distribute_dataset(self, dataset):  # pylint: disable=useless-super-delegation
    """Distributes a tf.data.Dataset instance provided via dataset.

    The returned dataset is a wrapped strategy dataset which creates a
    multidevice iterator under the hood. It prefetches the input data to the
    specified devices on the worker. The returned distributed dataset can be
    iterated over similar to how regular datasets can.

    NOTE: Currently, the user cannot add any more transformations to a
    distributed dataset.

    For Example:
    ```
    strategy = tf.distribute.CentralStorageStrategy()  # with 1 CPU and 1 GPU
    dataset = tf.data.Dataset.range(10).batch(2)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    for x in dist_dataset:
      print(x)  # Prints PerReplica values [0, 1], [2, 3],...

    ```
    Args:
      dataset: `tf.data.Dataset` to be prefetched to device.

    Returns:
      A "distributed `Dataset`" that the caller can iterate over.
    """
    return super(CentralStorageStrategy, self).experimental_distribute_dataset(
        dataset)

  def experimental_distribute_datasets_from_function(self, dataset_fn):  # pylint: disable=useless-super-delegation
    """Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

    `dataset_fn` will be called once for each worker in the strategy. In this
    case, we only have one worker so `dataset_fn` is called once. Each replica
    on this worker will then dequeue a batch of elements from this local
    dataset.

    The `dataset_fn` should take an `tf.distribute.InputContext` instance where
    information about batching and input replication can be accessed.

    For Example:
    ```
    def dataset_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)
      return d.shard(
          input_context.num_input_pipelines, input_context.input_pipeline_id)

    inputs = strategy.experimental_distribute_datasets_from_function(dataset_fn)

    for batch in inputs:
      replica_results = strategy.experimental_run_v2(replica_fn, args=(batch,))
    ```

    IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a
    per-replica batch size, unlike `experimental_distribute_dataset`, which uses
    the global batch size.  This may be computed using
    `input_context.get_per_replica_batch_size`.

    Args:
      dataset_fn: A function taking a `tf.distribute.InputContext` instance and
        returning a `tf.data.Dataset`.

    Returns:
      A "distributed `Dataset`", which the caller can iterate over like regular
      datasets.
    """
    return super(
        CentralStorageStrategy,
        self).experimental_distribute_datasets_from_function(dataset_fn)

  def experimental_local_results(self, value):  # pylint: disable=useless-super-delegation
    """Returns the list of all local per-replica values contained in `value`.

    In `CentralStorageStrategy` there is a single worker so the value returned
    will be all the values on that worker.

    Args:
      value: A value returned by `experimental_run()`, `experimental_run_v2()`,
        `extended.call_for_each_replica()`, or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
    """
    return super(CentralStorageStrategy, self).experimental_local_results(value)

  def experimental_run_v2(self, fn, args=(), kwargs=None):  # pylint: disable=useless-super-delegation
    """Run `fn` on each replica, with the given arguments.

    In `CentralStorageStrategy`, `fn` is  called on each of the compute
    replicas, with the provided "per replica" arguments specific to that device.

    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.

    Returns:
      Return value from running `fn`.
    """
    return super(CentralStorageStrategy, self).experimental_run_v2(fn, args,
                                                                   kwargs)

  def reduce(self, reduce_op, value, axis):  # pylint: disable=useless-super-delegation
    """Reduce `value` across replicas.

    Given a per-replica value returned by `experimental_run_v2`, say a
    per-example loss, the batch will be divided across all the replicas. This
    function allows you to aggregate across replicas and optionally also across
    batch elements.  For example, if you have a global batch size of 8 and 2
    replicas, values for examples `[0, 1, 2, 3]` will be on replica 0 and
    `[4, 5, 6, 7]` will be on replica 1. By default, `reduce` will just
    aggregate across replicas, returning `[0+4, 1+5, 2+6, 3+7]`. This is useful
    when each replica is computing a scalar or some other value that doesn't
    have a "batch" dimension (like a gradient). More often you will want to
    aggregate across the global batch, which you can get by specifying the batch
    dimension as the `axis`, typically `axis=0`. In this case it would return a
    scalar `0+1+2+3+4+5+6+7`.

    If there is a last partial batch, you will need to specify an axis so
    that the resulting shape is consistent across replicas. So if the last
    batch has size 6 and it is divided into [0, 1, 2, 3] and [4, 5], you
    would get a shape mismatch unless you specify `axis=0`. If you specify
    `tf.distribute.ReduceOp.MEAN`, using `axis=0` will use the correct
    denominator of 6. Contrast this with computing `reduce_mean` to get a
    scalar value on each replica and this function to average those means,
    which will weigh some values `1/8` and others `1/4`.

    For Example:
    ```
    strategy = tf.distribute.experimental.CentralStorageStrategy(
        compute_devices=['CPU:0', 'GPU:0'], parameter_device='CPU:0')
    ds = tf.data.Dataset.range(10)
    # Distribute that dataset
    dist_dataset = strategy.experimental_distribute_dataset(ds)

    with strategy.scope():
      @tf.function
      def train_step(val):
        # pass through
        return val

      # Iterate over the distributed dataset
      for x in dist_dataset:
        result = strategy.experimental_run_v2(train_step, args=(x,))

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, result,
                             axis=None).numpy()
    # result: array([ 4,  6,  8, 10])

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, result, axis=0).numpy()
    # result: 28
    ```

    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: A "per replica" value, e.g. returned by `experimental_run_v2` to
        be combined into a single tensor.
      axis: Specifies the dimension to reduce along within each
        replica's tensor. Should typically be set to the batch dimension, or
        `None` to only reduce across replicas (e.g. if the tensor has no batch
        dimension).

    Returns:
      A `Tensor`.
    """
    return super(CentralStorageStrategy, self).reduce(reduce_op, value, axis)


@tf_export(v1=["distribute.experimental.CentralStorageStrategy"])  # pylint: disable=missing-docstring
class CentralStorageStrategyV1(distribute_lib.StrategyV1):

  __doc__ = CentralStorageStrategy.__doc__

  def __init__(self, compute_devices=None, parameter_device=None):
    super(CentralStorageStrategyV1, self).__init__(
        parameter_server_strategy.ParameterServerStrategyExtended(
            self,
            compute_devices=compute_devices,
            parameter_device=parameter_device))
  __init__.__doc__ = CentralStorageStrategy.__init__.__doc__
