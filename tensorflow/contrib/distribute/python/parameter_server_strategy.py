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
"""Classes implementing a multi-worker ps DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import values
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import device_setter
from tensorflow.python.training import device_util
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest

_LOCAL_CPU = "/device:CPU:0"
_LOCAL_GPU_0 = "/device:GPU:0"


def _normalize_cluster_spec(cluster_spec):
  """Makes `cluster_spec` into a `ClusterSpec` object."""
  if isinstance(cluster_spec, (dict, cluster_pb2.ClusterDef)):
    return server_lib.ClusterSpec(cluster_spec)
  elif not isinstance(cluster_spec, server_lib.ClusterSpec):
    raise ValueError(
        "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
        "`tf.train.ClusterDef` object")
  return cluster_spec


# TODO(yuefengz): maybe cache variables on local CPU.
# TODO(yuefengz): we may want to set session options to disallow communication
# between workers.
class ParameterServerStrategy(distribute_lib.DistributionStrategy):
  """A parameter server DistributionStrategy.

  This strategy class works for both local training and between-graph replicated
  training for multiple workers. If `cluster_spec` is specified, either passed
  in to __init__() method or parsed from the
  ["TF_CONFIG" environment
  variable](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig),
  variables and updates to those variables are assigned to parameter servers and
  other operations are assigned to workers. If `cluster_spec` is not set, it
  becomes local training where variables are assigned to local CPU or the only
  GPU. When each worker has more than one GPU, operations will be replicated on
  these GPUs. In both cases, operations are replicated but variables are not and
  these workers share a common view for which paramater server a variable is
  assigned to.

  This class assumes between-graph replication will be used and works on a graph
  for a particular worker.

  It is expected to call `call_for_each_tower(fn, *args, **kwargs)` for any
  operations which potentially can be replicated across towers (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) Always use `tf.get_variable` instead of `tf.Variable` which is not able
  to refer to the same variable on different towers.

  2) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  3) It is also not recommended to open a colocation scope (i.e. calling
  `tf.colocate_with`) under the strategy's scope. For colocating variables,
  use `distribution.colocate_vars_with` instead. Colocation of ops will possibly
  create conflicts of device assignement.
  """

  def __init__(self,
               num_gpus_per_worker=0,
               cluster_spec=None,
               task_type=None,
               task_id=None):
    """Initiailizes this strategy.

    Args:
      num_gpus_per_worker: number of local GPUs or GPUs per worker.
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type.
      task_id: the current task id.
    """
    super(ParameterServerStrategy, self).__init__()
    self._num_gpus_per_worker = num_gpus_per_worker
    if cluster_spec:
      cluster_spec = _normalize_cluster_spec(cluster_spec)
    self._cluster_spec = cluster_spec

    # We typically don't need to do all-reduce in this strategy.
    self._cross_tower_ops = (
        cross_tower_ops_lib.ReductionToOneDeviceCrossTowerOps(
            reduce_to_device=_LOCAL_CPU))

    self._initialize_devices(num_gpus_per_worker, cluster_spec, task_type,
                             task_id)

  def _initialize_devices(self, num_gpus_per_worker, cluster_spec, task_type,
                          task_id):
    """Initialize internal devices.

    It creates variable devices and compute devices. Variables and operations
    will be assigned to them respectively. We have one compute device per tower.
    The variable device is a device function or device string. The default
    variable device assigns variables to parameter servers in a round-robin
    fashion.

    Args:
      num_gpus_per_worker: number of local GPUs or GPUs per worker.
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type.
      task_id: the current task id.

    Raises:
      ValueError: if the cluster_spec doesn't have ps jobs.
    """
    self._task_type = task_type or "worker"
    self._task_id = task_id or 0
    self._worker_device = "/job:%s/task:%d" % (self._task_type, self._task_id)

    # TODO(yuefengz): maybe clearer to split it into two classes, one for
    # the distribuetd case and one for the local case, once we have the factory
    # class/method.

    # Define compute devices which is a list of device strings and one for each
    # tower. When there are GPUs, replicate operations on these GPUs. Otherwise,
    # place operations on CPU.
    if cluster_spec is None:
      # Local mode.
      if num_gpus_per_worker > 0:
        self._compute_devices = list(
            map("/device:GPU:{}".format, range(num_gpus_per_worker)))
      else:
        self._compute_devices = [_LOCAL_CPU]
    else:
      # Distributed mode.
      if num_gpus_per_worker > 0:
        self._compute_devices = [
            "%s/device:GPU:%d" % (self._worker_device, i)
            for i in range(num_gpus_per_worker)
        ]
      else:
        self._compute_devices = [self._worker_device]

    self._compute_devices = list(
        map(device_util.resolve, self._compute_devices))
    self._canonical_compute_device_set = set(self._compute_devices)

    # Define variable device which is a device string in the local case and a
    # device function in the distributed case. It is used to open a device scope
    # where varibles are defined.
    # The `_parameter_devices` is needed for the `parameter_devices` property
    # and is a list of all variable devices.
    if cluster_spec is None:
      # Local mode. If there is only one GPU, put everything on that GPU.
      # Otherwise, place variables on CPU.
      if num_gpus_per_worker == 1:
        assert len(list(self._compute_devices)) == 1
        self._variable_device = _LOCAL_GPU_0
        self._parameter_devices = [_LOCAL_GPU_0]
      else:
        self._variable_device = _LOCAL_CPU
        self._parameter_devices = [_LOCAL_CPU]
    else:
      # Distributed mode. Place variables on ps jobs in a round-robin fashion.
      # Note that devices returned from `replica_device_setter` are not
      # canonical and therefore we don't canonicalize all variable devices to
      # make them consistent.
      # TODO(yuefengz): support passing a strategy object to control variable
      # assignment.
      # TODO(yuefengz): merge the logic of replica_device_setter into this
      # class.
      num_ps_replicas = len(cluster_spec.as_dict().get("ps", []))
      if num_ps_replicas == 0:
        raise ValueError("The cluster spec needs to have `ps` jobs.")
      self._variable_device = device_setter.replica_device_setter(
          ps_tasks=num_ps_replicas,
          worker_device=self._worker_device,
          merge_devices=True,
          cluster=cluster_spec)

      # Parameter devices are all tasks of the "ps" job.
      self._parameter_devices = map("/job:ps/task:{}".format,
                                    range(num_ps_replicas))

    # Define the default device in cross-tower mode. In the distributed case, we
    # set the default device to the corresponding worker to prevent these ops
    # from being placed on other workers.
    if cluster_spec is None:
      self._default_device = None
    else:
      self._default_device = self._worker_device

  def distribute_dataset(self, dataset_fn):
    """Distributes the dataset to each local GPU."""
    return values.PerDeviceDataset(
        self._call_dataset_fn(dataset_fn), self._compute_devices, True)

  def _broadcast(self, tensor, destinations):
    if not cross_tower_ops_lib.check_destinations(destinations):
      destinations = self._compute_devices
    return self._cross_tower_ops.broadcast(tensor, destinations)

  # TODO(yuefengz): not all ops in device_setter.STANDARD_PS_OPS will go through
  # this creator, such as "MutableHashTable".
  def _create_variable(self, next_creator, *args, **kwargs):
    if "colocate_with" in kwargs:
      with ops.device(None):
        with ops.colocate_with(kwargs["colocate_with"]):
          return next_creator(*args, **kwargs)

    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(self._variable_device):
        return next_creator(*args, **kwargs)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    # pylint: disable=protected-access
    return mirrored_strategy._call_for_each_tower(self, fn, *args, **kwargs)

  def _verify_destinations_not_different_worker(self, destinations):
    if destinations is None:
      return
    for d in cross_tower_ops_lib.get_devices_from(destinations):
      d_spec = tf_device.DeviceSpec.from_string(d)
      if d_spec.job == self._task_type and d_spec.task != self._task_id:
        raise ValueError(
            "Cannot reduce to another worker: %r, current worker is %r" %
            (d, self._worker_device))

  def _reduce(self, aggregation, value, destinations):
    self._verify_destinations_not_different_worker(destinations)
    if not isinstance(value, values.DistributedValues):
      # pylint: disable=protected-access
      return mirrored_strategy._reduce_non_distributed_value(
          self, aggregation, value, destinations)

    return self._cross_tower_ops.reduce(
        aggregation, value, destinations=destinations)

  def _batch_reduce(self, aggregation, value_destination_pairs):
    for _, destinations in value_destination_pairs:
      self._verify_destinations_not_different_worker(destinations)
    return self._cross_tower_ops.batch_reduce(aggregation,
                                              value_destination_pairs)

  def _select_single_value(self, structured):
    """Select any single values in `structured`."""

    def _select_fn(x):  # pylint: disable=g-missing-docstring
      if isinstance(x, values.Mirrored):
        if len(x.devices) == 1:
          return list(x._index.values())[0]  # pylint: disable=protected-access
        else:
          raise ValueError(
              "You cannot update variable with a Mirrored object with multiple "
              "components %r when using ParameterServerStrategy. You must "
              "specify a single value or a Mirrored with a single value." % x)
      elif isinstance(x, values.PerDevice):
        raise ValueError(
            "You cannot update variable with a PerDevice object %r when using "
            "ParameterServerStrategy. You must specify a single value or a "
            "Mirrored with a single value" % x)
      else:
        return x

    return nest.map_structure(_select_fn, structured)

  def _update(self, var, fn, *args, **kwargs):
    if not isinstance(var, resource_variable_ops.ResourceVariable):
      raise ValueError(
          "You can not update `var` %r. It must be a Variable." % var)
    with ops.colocate_with(var), distribute_lib.UpdateContext(var.device):
      return fn(var, *self._select_single_value(args),
                **self._select_single_value(kwargs))

  # TODO(yuefengz): does it need to call _select_single_value?
  def _update_non_slot(self, colocate_with, fn, *args, **kwargs):
    with ops.device(
        colocate_with.device), distribute_lib.UpdateContext(colocate_with):
      return fn(*args, **kwargs)

  def _unwrap(self, val):
    if isinstance(val, values.DistributedValues):
      # Return in a deterministic order.
      if set(val.devices) == self._canonical_compute_device_set:
        return [val.get(device=d) for d in self._compute_devices]
      return [val.get(device=d) for d in sorted(val.devices)]
    return [val]

  def value_container(self, val):
    return values.value_container(val)

  def read_var(self, var):
    # No need to distinguish between normal variables and tower-local variables.
    return array_ops.identity(var)

  def configure(self, session_config=None):
    del session_config

    # Use TF_CONFIG to get the cluster spec and the current job.
    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    cluster_spec = _normalize_cluster_spec(tf_config.get("cluster", {}))

    task_env = tf_config.get("task", {})
    if task_env:
      task_type = task_env.get("type", "worker")
      task_id = int(task_env.get("index", "0"))
    else:
      task_type = "worker"
      task_id = None

    # Set the devices if cluster_spec is defined in TF_CONFIG but not passed in
    # the constructor.
    if not self._cluster_spec and cluster_spec:
      self._cluster_spec = cluster_spec
      self._initialize_devices(self._num_gpus_per_worker, cluster_spec,
                               task_type, task_id)

  @property
  def num_towers(self):
    return len(self._compute_devices)

  @property
  def worker_devices(self):
    # Make a copy to prevent users from accidentally mutating our copy.
    return list(self._compute_devices)

  @property
  def parameter_devices(self):
    return list(self._parameter_devices)

  def non_slot_devices(self, var_list):
    return min(var_list, key=lambda x: x.name)
