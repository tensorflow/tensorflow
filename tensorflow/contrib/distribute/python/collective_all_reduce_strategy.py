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
"""Class CollectiveAllReduceStrategy implementing DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
from tensorflow.contrib.distribute.python import cross_tower_utils
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import values
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import tf_logging as logging


# TODO(yuefengz): support in-graph replication.
class CollectiveAllReduceStrategy(mirrored_strategy.MirroredStrategy):
  """Distribution strategy that uses collective ops for all-reduce.

  It is similar to the MirroredStrategy but it uses collective ops for
  reduction.

  When `cluster_spec` is given by the `configure` method, it turns into the
  mulit-worker version that works on multiple workers with between-graph
  replication.

  Note: `configure` will be called by higher-level APIs if running in
  distributed environment.
  """

  def __init__(self, num_gpus_per_worker=0):
    """Initializes the object.

    Args:
      num_gpus_per_worker: number of local GPUs or GPUs per worker, the default
        is 0 meaning CPU only.
    """
    self._num_gpus_per_worker = num_gpus_per_worker
    self._initialize_local_worker(num_gpus_per_worker)

  def _initialize_local_worker(self, num_gpus_per_worker):
    """Initializes the object for local training."""
    self._is_chief = True
    self._num_workers = 1

    if num_gpus_per_worker:
      local_devices = [
          "/device:GPU:%d" % i for i in range(num_gpus_per_worker)
      ]
    else:
      local_devices = ["/device:CPU:0"]

    self._collective_keys = cross_tower_utils.CollectiveKeys()
    super(CollectiveAllReduceStrategy, self).__init__(
        devices=local_devices,
        cross_tower_ops=cross_tower_ops_lib.CollectiveAllReduce(
            num_workers=1,
            num_gpus_per_worker=num_gpus_per_worker,
            collective_keys=self._collective_keys))

    self._cluster_spec = None
    self._task_type = None
    self._task_id = None

    logging.info("CollectiveAllReduceStrategy with local_devices = %r",
                 local_devices)

  def _initialize_multi_worker(self, num_gpus_per_worker, cluster_spec,
                               task_type, task_id):
    """Initializes the object for multi-worker training."""
    if task_type is None or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`")
    if task_type not in ["chief", "worker"]:
      raise ValueError(
          "Unrecognized task_type: %r, valid task types are: \"chief\", "
          "\"worker\"." % task_type)
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
    self._num_workers = len(cluster_spec.as_dict().get("worker", [])) + len(
        cluster_spec.as_dict().get("chief", []))
    if not self._num_workers:
      raise ValueError("No `worker` or `chief` tasks can be found in "
                       "`cluster_spec`.")

    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)

    worker_device = "/job:%s/task:%d" % (task_type, task_id)
    if num_gpus_per_worker:
      local_devices = [
          "%s/device:GPU:%d" % (worker_device, i)
          for i in range(num_gpus_per_worker)
      ]
    else:
      local_devices = [worker_device]

    self._collective_keys = cross_tower_utils.CollectiveKeys()
    super(CollectiveAllReduceStrategy, self).__init__(
        devices=local_devices,
        cross_tower_ops=cross_tower_ops_lib.CollectiveAllReduce(
            num_workers=self._num_workers,
            num_gpus_per_worker=num_gpus_per_worker,
            collective_keys=self._collective_keys))

    # Add a default device so that ops without specified devices will not end up
    # on other workers.
    self._default_device = "/job:%s/task:%d" % (task_type, task_id)

    self._cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
    self._task_type = task_type
    self._task_id = task_id

    logging.info(
        "Multi-worker CollectiveAllReduceStrategy with "
        "cluster_spec = %r, task_type = %r, task_id = %r, "
        "num_workers = %r, local_devices = %r", cluster_spec.as_dict(),
        task_type, task_id, self._num_workers, local_devices)

  def _create_variable(self, next_creator, *args, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    devices = self._get_devices_from(colocate_with)
    group_size = len(devices) * self._num_workers
    group_key = self._collective_keys.get_group_key(self._devices)

    def _real_mirrored_creator(devices, *args, **kwargs):
      """Creates one MirroredVariable on the current worker."""
      index = {}
      collective_instance_key = self._collective_keys.get_instance_key(
          key_id=kwargs["name"])
      if "initial_value" not in kwargs:
        raise ValueError("Initial value must be specified.")
      initial_value = kwargs["initial_value"]
      if callable(initial_value):
        initial_value_fn = initial_value
      else:
        initial_value_fn = lambda: initial_value

      for i, d in enumerate(devices):
        with ops.device(d):
          if i > 0:
            # Give replicas meaningful distinct names:
            var0name = index[devices[0]].name.split(":")[0]
            # We append a / to variable names created on towers with id > 0 to
            # ensure that we ignore the name scope and instead use the given
            # name as the absolute name of the variable.
            kwargs["name"] = "%s/replica_%d/" % (var0name, i)

          # The initial value fn makes sure variables all initialized to
          # same values. The first device of the chief worker will send their
          # variable values to other devices and other workers.
          def _overridden_initial_value_fn(device=d, index=i):  # pylint: disable=g-missing-docstring
            with ops.device(device):
              initial_value = initial_value_fn()
              assert not callable(initial_value)
              initial_value = ops.convert_to_tensor(initial_value)

              if self._is_chief and index == 0:
                bcast_send = collective_ops.broadcast_send(
                    initial_value, initial_value.shape, initial_value.dtype,
                    group_size, group_key, collective_instance_key)
                with ops.control_dependencies([bcast_send]):
                  return array_ops.identity(initial_value)
              else:
                return collective_ops.broadcast_recv(
                    initial_value.shape, initial_value.dtype, group_size,
                    group_key, collective_instance_key)

          kwargs["initial_value"] = _overridden_initial_value_fn

          with context.context().device_policy(context.DEVICE_PLACEMENT_SILENT):
            v = next_creator(*args, **kwargs)

          assert not isinstance(v, values.DistributedVariable)
          index[d] = v
      return index

    # pylint: disable=protected-access
    return mirrored_strategy._create_mirrored_variable(
        devices, _real_mirrored_creator, *args, **kwargs)

  def distribute_dataset(self, dataset_fn):
    """Distributes the dataset to each local GPU."""
    # TODO(yuefengz): shard the dataset.
    return values.PerDeviceDataset(
        self._call_dataset_fn(dataset_fn), self._devices, True)

  def configure(self,
                session_config=None,
                cluster_spec=None,
                task_type=None,
                task_id=None):
    """Configures the object.

    Args:
      session_config: a @{tf.ConfigProto}
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type, such as "worker".
      task_id: the current task id.

    Raises:
      ValueError: if `task_type` is not in the `cluster_spec`.
    """
    if not self._cluster_spec and cluster_spec:
      # If a `cluster_spec` is already passed in, do nothing here.
      # TODO(yuefengz): check `cluster_spec` is the same if this object has
      # already been initialized with a `cluster_spec`.
      self._initialize_multi_worker(self._num_gpus_per_worker, cluster_spec,
                                    task_type, task_id)

    if not session_config or not self._cluster_spec:
      return

    assert self._task_type
    assert self._task_id is not None

    # Collective group leader is needed for collective ops to coordinate
    # workers.
    if "chief" in self._cluster_spec.jobs:
      session_config.experimental.collective_group_leader = (
          "/job:chief/replica:0/task:0")
    else:
      if "worker" not in self._cluster_spec.jobs:
        raise ValueError(
            "You must have `chief` or `worker` jobs in the `cluster_spec`.")
      session_config.experimental.collective_group_leader = (
          "/job:worker/replica:0/task:0")

    # The device filters prevent communication between workers.
    del session_config.device_filters[:]
    session_config.device_filters.append(
        "/job:%s/task:%d" % (self._task_type, self._task_id))

    # The scoped_allocator_optimization is to optimize graphs for collective
    # ops.
    rewrite_options = session_config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append("CollectiveReduce")

  @property
  def between_graph(self):
    return True

  @property
  def should_init(self):
    return True

  @property
  def should_checkpoint(self):
    return self._is_chief

  @property
  def should_save_summary(self):
    return self._is_chief
