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
"""Classes implementing a mirrored DistributionStrategy for multiple workers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from tensorflow.contrib.distribute.python import values
from tensorflow.contrib.distribute.python.mirrored_strategy import MirroredStrategy
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.training import device_util
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest


# TODO(yuefengz): support between-graph replication.
# TODO(yuefengz): merge this class into its base class.
# TODO(yuefengz): in some cases, we probably want to use configure method to
# configure this class.
# TODO(yuefengz): MirroredStrategy.worker_devices may be confusing after the
# class is introduced.
class MultiWorkerMirroredStrategy(MirroredStrategy):
  """Mirrored strategy that works on multiple workers with in-graph replication.

  There are several important concepts for distributed TensorFlow, e.g.
  `client`, `job`, 'task', `cluster`, `in-graph replication` and
  'synchronous training' and they have already been defined in the
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

  This class maps one tower to one device on a worker. It mirrors all model
  variables on all towers. For example, if you have two `worker`s and each
  `worker` has 4 GPUs, it will create 8 copies of the model variables on these 8
  GPUs. Then like in MirroredStrategy, each tower performs their computation
  with their own copy of variables unless in cross-tower model where variable or
  tensor reduction happens.
  """

  def __init__(self,
               num_gpus_per_worker=1,
               worker_job_name=None,
               num_workers=None,
               cluster=None,
               cross_tower_ops=None,
               prefetch_on_device=None):
    """Initialize the strategy object.

    Args:
      num_gpus_per_worker: number of GPUs per work. If it is zero, the local
        CPU will be used.
      worker_job_name: the job name for `worker`, typically just 'worker'.
      num_workers: the number of workers. If it is 0, it regenerates to
        single-worker MirroredStrategy.
      cluster: a `tf.train.ClusterSpec` object or a dict that can be used to
        construct a `tf.train.ClusterSpec` object or a `tf.train.ClusterDef`
        proto buffer. It is an alternative way to initialize this object.
      cross_tower_ops: the cross tower ops to use. If None, a default one will
        be used. If configure method is called, a best one for the configuration
        will be chosen.
      prefetch_on_device: a boolean to specify whether to prefetech input to
        each worker's devices.

    Raises:
      ValueError: if got an unexpected `cluster`.
    """
    if cluster is None:
      self._workers = [
          '/job:%s/task:%d' % (worker_job_name, task_index)
          for task_index in range(num_workers)
      ]
    else:
      if isinstance(cluster, (dict, cluster_pb2.ClusterDef)):
        cluster_spec = server_lib.ClusterSpec(cluster)
      elif isinstance(cluster, server_lib.ClusterSpec):
        cluster_spec = cluster
      else:
        raise ValueError(
            "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
            '`tf.train.ClusterDef` object')

      self._workers = []
      for job in sorted(cluster_spec.jobs):
        for task in range(cluster_spec.num_tasks(job)):
          self._workers.append('/job:%s/task:%d' % (job, task))

    self._num_gpus_per_worker = num_gpus_per_worker
    if num_gpus_per_worker > 0:
      self._worker_device_map = {
          worker: [
              device_util.canonicalize(worker + '/device:GPU:%d' % gpu)
              for gpu in range(num_gpus_per_worker)
          ] for worker in self._workers
      }
    else:
      self._worker_device_map = {
          worker: [device_util.canonicalize(worker, '/device:CPU:0')]
          for worker in self._workers
      }
    self._devices = nest.flatten(self._worker_device_map)

    super(MultiWorkerMirroredStrategy, self).__init__(
        devices=self._devices, prefetch_on_device=prefetch_on_device)

    # Setting `_default_device` will add a device scope in the
    # distribution.scope. We set the default device to the first worker. When
    # users specify device under distribution.scope by
    #   with tf.device("/cpu:0"):
    #     ...
    # their ops will end up on the cpu device of its first worker, e.g.
    # "/job:worker/task:0/device:CPU:0". Note this is not used in tower mode.
    self._default_device = self._workers[0]

  def distribute_dataset(self, dataset_fn):
    return values.MultiWorkerDataset(
        partial(self._call_dataset_fn, dataset_fn), self._worker_device_map,
        self._prefetch_on_device)
