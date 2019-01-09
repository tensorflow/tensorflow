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
"""Contrib version of MirroredStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import mirrored_strategy


# pylint: disable=protected-access,invalid-name
_call_for_each_replica = mirrored_strategy._call_for_each_replica
_create_mirrored_variable = mirrored_strategy._create_mirrored_variable
all_local_devices = mirrored_strategy.all_local_devices
CoreMirroredStrategy = mirrored_strategy.MirroredStrategy
CoreMirroredExtended = mirrored_strategy.MirroredExtended
# pylint: enable=protected-access,invalid-name


class MirroredStrategy(distribute_lib.DistributionStrategy):
  """Mirrors vars to distribute across multiple devices and machines.

  *** contrib version ***

  This strategy uses one replica per device and sync replication for its
  multi-GPU version.

  When `cluster_spec` is given by the `configure` method., it turns into the
  mulit-worker version that works on multiple workers with in-graph replication.
  Note: `configure` will be called by higher-level APIs if running in
  distributed environment.

  There are several important concepts for distributed TensorFlow, e.g.
  `client`, `job`, `task`, `cluster`, `in-graph replication` and
  `synchronous training` and they have already been defined in the
  [TensorFlow's documentation](https://www.tensorflow.org/deploy/distributed).
  The distribution strategy inherits these concepts as well and in addition to
  that we also clarify several more concepts:

  * **In-graph replication**: the `client` creates a single `tf.Graph` that
    specifies tasks for devices on all workers. The `client` then creates a
    client session which will talk to the `master` service of a `worker`. Then
    the `master` will partition the graph and distribute the work to all
    participating workers.
  * **Worker**: A `worker` is a TensorFlow `task` that usually maps to one
    physical machine. We will have multiple `worker`s with different `task`
    index. They all do similar things except for one worker checkpointing model
    variables, writing summaries, etc. in addition to its ordinary work.

  The multi-worker version of this class maps one replica to one device on a
  worker. It mirrors all model variables on all replicas. For example, if you
  have two `worker`s and each `worker` has 4 GPUs, it will create 8 copies of
  the model variables on these 8 GPUs. Then like in MirroredStrategy, each
  replica performs their computation with their own copy of variables unless in
  cross-replica model where variable or tensor reduction happens.

  Args:
    devices: a list of device strings.
    num_gpus: number of GPUs. For local training, either specify `devices` or
      `num_gpus`. In distributed training, this must be specified as number of
      GPUs on each worker.
    num_gpus_per_worker: number of GPUs per worker. This is the same as
      `num_gpus` and only one of `num_gpus` and `num_gpus_per_worker` can be
      specified.
    cross_device_ops: optional, a descedant of `CrossDeviceOps`. If this is not
      set, the `configure` method will try to find the best one.
    auto_shard_dataset: whether to auto-shard the dataset when there are
      multiple workers.
    cross_tower_ops: Deprecated alias for `cross_device_ops`.
  """

  def __init__(self,
               devices=None,
               num_gpus=None,
               num_gpus_per_worker=None,
               cross_device_ops=None,
               auto_shard_dataset=False,
               cross_tower_ops=None):
    assert not (cross_device_ops and cross_tower_ops)
    if num_gpus is not None and num_gpus_per_worker is not None:
      raise ValueError(
          "You cannot specify both `num_gpus` and `num_gpus_per_worker`.")
    if num_gpus is None:
      num_gpus = num_gpus_per_worker
    extended = MirroredExtended(self, devices, num_gpus,
                                cross_device_ops or cross_tower_ops,
                                auto_shard_dataset)
    super(MirroredStrategy, self).__init__(extended)

  # Override to change the documentation to reflect the different handling of
  # global vs. local batch size between core and contrib.
  def experimental_make_numpy_iterator(  # pylint: disable=useless-super-delegation
      self, numpy_input, batch_size, num_epochs=1, shuffle=1024, session=None):
    """Makes an iterator for input provided via a nest of numpy arrays.

    NOTE: The `batch_size` argument here has different behavior for this
    contrib version of `MirroredStrategy`.

    Args:
      numpy_input: A nest of NumPy input arrays that will be distributed evenly
        across all replicas.
      batch_size: The number of entries from the array we should consume in one
        step of the computation, across all replicas. This is the per-replica
        batch size. The global batch size will be this times
        `num_replicas_in_sync`.
      num_epochs: The number of times to iterate through the examples. A value
        of `None` means repeat forever.
      shuffle: Size of buffer to use for shuffling the input examples.
        Use `None` to disable shuffling.
      session: (TensorFlow v1.x graph execution only) A session used for
        initialization.

    Returns:
      An `tf.distribute.InputIterator` which returns inputs for each step of the
      computation.  User should call `initialize` on the returned iterator.
    """
    return super(MirroredStrategy, self).experimental_make_numpy_iterator(
        numpy_input, batch_size, num_epochs, shuffle, session)


class MirroredExtended(CoreMirroredExtended):
  """Implementation of (contrib) MirroredStrategy."""

  def __init__(self,
               container_strategy,
               devices=None,
               num_gpus_per_worker=None,
               cross_device_ops=None,
               auto_shard_dataset=False):
    if devices is None:
      devices = mirrored_strategy.all_local_devices(num_gpus_per_worker)
    elif num_gpus_per_worker is not None:
      raise ValueError(
          "Must only specify one of `devices` and `num_gpus_per_worker`.")
    super(MirroredExtended, self).__init__(container_strategy, devices,
                                           cross_device_ops)
    self._auto_shard_dataset = auto_shard_dataset

  def _make_dataset_iterator(self, dataset):
    """Make iterator from dataset without splitting the batch.

    This implementation is different than the one in
    `tf.distribute.MirroredStrategy` for purposes of backward compatibility.
    We treat the incoming dataset's batch size as per replica batch size.

    Args:
      dataset: `tf.data.Dataset` for input.
    Returns:
      An `InputIterator` which returns inputs for each step of the computation.
    """
    return input_lib.DatasetIterator(dataset, self._input_workers)

  def _distribute_dataset(self, dataset_fn):
    if self._local_mode:
      return input_lib.PerReplicaDataset(
          self._call_dataset_fn(dataset_fn), self._input_workers, 0)
    else:
      return input_lib.MultiWorkerDataset(
          functools.partial(self._call_dataset_fn, dataset_fn),
          self._input_workers,
          auto_shard=self._auto_shard_dataset)

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    return False
