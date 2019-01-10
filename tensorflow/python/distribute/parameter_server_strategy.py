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

import copy

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest

_LOCAL_CPU = "/device:CPU:0"
_LOCAL_GPU_0 = "/device:GPU:0"


# TODO(yuefengz): maybe cache variables on local CPU.
class ParameterServerStrategy(distribute_lib.DistributionStrategy):
  """A parameter server DistributionStrategy.

  This strategy class works for both local training and between-graph replicated
  training for multiple workers. It uses `TFConfigClusterResolver` to detect
  configurations for multi-worker training. In multi-worker training mode, i.e.
  `TFConfigClusterResolver` has detected 'TF_CONFIG' environment variable and
  'TF_CONFIG' has a cluster spec, variables and updates to those variables are
  assigned to parameter servers and other operations are assigned to workers.
  In local training mode, variables are assigned to local CPU or the only GPU.
  When each worker has more than one GPU, operations will be replicated on these
  GPUs. In both cases, operations are replicated but variables are not and these
  workers share a common view for which paramater server a variable is assigned
  to.

  This class assumes between-graph replication will be used and works on a graph
  for a particular worker. Note that each graph and worker is independent.
  This means that while each worker will synchronously compute a single gradient
  update across all GPUs, updates between workers proceed asynchronously.
  Operations that occur only on the first replica (such as incrementing the
  global step), will occur on the first replica *of every worker*.

  It is expected to call `call_for_each_replica(fn, ...)` for any
  operations which potentially can be replicated across replicas (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  2) It is also not recommended to open a colocation scope (i.e. calling
  `tf.colocate_with`) under the strategy's scope. For colocating variables,
  use `distribution.colocate_vars_with` instead. Colocation of ops will possibly
  create conflicts of device assignment.
  """

  def __init__(self):
    """Initializes this strategy with default TFConfigClusterResolver."""
    super(ParameterServerStrategy, self).__init__(
        ParameterServerStrategyExtended(self))


class ParameterServerStrategyExtended(
    distribute_lib.DistributionStrategyExtended):
  """Implementation of ParameterServerStrategy."""

  def __init__(self,
               container_strategy,
               cluster_resolver=TFConfigClusterResolver()):
    super(ParameterServerStrategyExtended, self).__init__(container_strategy)
    self._initialize_strategy(cluster_resolver)

    # We typically don't need to do all-reduce in this strategy.
    self._cross_device_ops = (
        cross_device_ops_lib.ReductionToOneDeviceCrossDeviceOps(
            reduce_to_device=_LOCAL_CPU))

  def _initialize_strategy(self, cluster_resolver):
    if cluster_resolver.cluster_spec().as_dict():
      self._initialize_multi_worker(cluster_resolver)
    else:
      self._initialize_local(cluster_resolver)
    # Save the num_gpus_per_worker for configure method.
    self._num_gpus_per_worker = cluster_resolver.num_accelerators()

  def _initialize_multi_worker(self, cluster_resolver):
    """Initialize devices for multiple workers.

    It creates variable devices and compute devices. Variables and operations
    will be assigned to them respectively. We have one compute device per
    replica. The variable device is a device function or device string. The
    default variable device assigns variables to parameter servers in a
    round-robin fashion.

    Args:
      cluster_resolver: a descendant of `ClusterResolver` object.

    Raises:
      ValueError: if the cluster doesn't have ps jobs.
    """
    num_gpus = cluster_resolver.num_accelerators()
    cluster_spec = cluster_resolver.cluster_spec()
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_index
    if not task_type or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`")
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
    assert cluster_spec.as_dict()

    worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._input_host_device = numpy_dataset.SingleDevice(worker_device)

    # Define compute devices which is a list of device strings and one for each
    # replica. When there are GPUs, replicate operations on these GPUs.
    # Otherwise, place operations on CPU.
    if num_gpus > 0:
      compute_devices = tuple(
          "%s/device:GPU:%d" % (worker_device, i) for i in range(num_gpus))
    else:
      compute_devices = (worker_device,)

    self._device_map = values.ReplicaDeviceMap(compute_devices)
    self._input_workers = input_lib.InputWorkers(
        self._device_map, [(worker_device, compute_devices)])

    # In distributed mode, place variables on ps jobs in a round-robin fashion.
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
        worker_device=worker_device,
        merge_devices=True,
        cluster=cluster_spec)

    # The `_parameter_devices` is needed for the `parameter_devices` property
    # and is a list of all variable devices. Here parameter devices are all
    # tasks of the "ps" job.
    self._parameter_devices = tuple(map("/job:ps/task:{}".format,
                                        range(num_ps_replicas)))

    # Add a default device so that ops without specified devices will not end up
    # on other workers.
    self._default_device = worker_device

    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id

    logging.info(
        "Multi-worker ParameterServerStrategy with "
        "cluster_spec = %r, task_type = %r, task_id = %r, "
        "num_ps_replicas = %r, is_chief = %r, device_map = %r, "
        "variable_device = %r", cluster_spec.as_dict(), task_type, task_id,
        num_ps_replicas, self._is_chief, self._device_map,
        self._variable_device)

  def _initialize_local(self, cluster_resolver):
    """Initialize internal devices for local training."""
    worker_device = device_util.canonicalize("/device:CPU:0")
    self._input_host_device = numpy_dataset.SingleDevice(worker_device)
    num_gpus = cluster_resolver.num_accelerators()
    # Define compute devices which is a list of device strings and one for each
    # replica. When there are GPUs, replicate operations on these GPUs.
    # Otherwise, place operations on CPU.
    if num_gpus > 0:
      compute_devices = tuple(map("/device:GPU:{}".format, range(num_gpus)))
    else:
      compute_devices = (_LOCAL_CPU,)

    self._device_map = values.ReplicaDeviceMap(compute_devices)
    self._input_workers = input_lib.InputWorkers(
        self._device_map, [(worker_device, compute_devices)])

    # If there is only one GPU, put everything on that GPU. Otherwise, place
    # variables on CPU.
    if num_gpus == 1:
      assert len(compute_devices) == 1
      self._variable_device = _LOCAL_GPU_0
      self._parameter_devices = (_LOCAL_GPU_0,)
    else:
      self._variable_device = _LOCAL_CPU
      self._parameter_devices = (_LOCAL_CPU,)

    self._is_chief = True
    self._cluster_spec = None
    self._task_type = None
    self._task_id = None

    logging.info(
        "ParameterServerStrategy with compute_devices = %r, "
        "variable_device = %r", compute_devices, self._variable_device)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    values.validate_colocate(colocate_with_variable, self)

  def _distribute_dataset(self, dataset_fn):
    """Distributes the dataset to each local GPU."""
    return input_lib.PerReplicaDataset(
        self._call_dataset_fn(dataset_fn),
        self._input_workers,
        0,
        prefetch_on_device=True)

  def _make_dataset_iterator(self, dataset):
    return input_lib.DatasetIterator(dataset, self._input_workers,
                                     self._num_replicas_in_sync)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    """Distributes the dataset to each local GPU."""
    if self._cluster_spec:
      input_pipeline_id = multi_worker_util.id_in_cluster(
          self._cluster_spec, self._task_type, self._task_id)
      num_input_pipelines = multi_worker_util.worker_count(
          self._cluster_spec, self._task_type)
    else:
      input_pipeline_id = 0
      num_input_pipelines = 1
    input_context = distribute_lib.InputContext(
        num_input_pipelines=num_input_pipelines,
        input_pipeline_id=input_pipeline_id,
        num_replicas_in_sync=self._num_replicas_in_sync)
    return input_lib.InputFunctionIterator(input_fn, self._input_workers,
                                           [input_context])

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, self._input_host_device, session)

  def _broadcast_to(self, tensor, destinations):
    # This is both a fast path for Python constants, and a way to delay
    # converting Python values to a tensor until we know what type it
    # should be converted to. Otherwise we have trouble with:
    #   global_step.assign_add(1)
    # since the `1` gets broadcast as an int32 but global_step is int64.
    if isinstance(tensor, (float, int)):
      return tensor
    if not cross_device_ops_lib.check_destinations(destinations):
      # TODO(josh11b): Use current logical device instead of 0 here.
      destinations = values.LogicalDeviceSpec(
          device_map=self._device_map, logical_device=0)
    return self._cross_device_ops.broadcast(tensor, destinations)

  def _allow_variable_partition(self):
    return not context.executing_eagerly()

  # TODO(yuefengz): not all ops in device_setter.STANDARD_PS_OPS will go through
  # this creator, such as "MutableHashTable".
  def _create_variable(self, next_creator, *args, **kwargs):
    if self._num_replicas_in_sync > 1:
      aggregation = kwargs.pop("aggregation", vs.VariableAggregation.NONE)
      if aggregation not in (
          vs.VariableAggregation.NONE,
          vs.VariableAggregation.SUM,
          vs.VariableAggregation.MEAN,
          vs.VariableAggregation.ONLY_FIRST_REPLICA
      ):
        raise ValueError("Invalid variable aggregation mode: " + aggregation +
                         " for variable: " + kwargs["name"])

      def var_creator(*args, **kwargs):
        """Create an AggregatingVariable and fix up collections."""
        # Record what collections this variable should be added to.
        collections = kwargs.pop("collections", None)
        if collections is None:
          collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        kwargs["collections"] = []

        # Create and wrap the variable.
        v = next_creator(*args, **kwargs)
        wrapped = values.AggregatingVariable(
            self._container_strategy(), v, aggregation)

        # Add the wrapped variable to the requested collections.
        # The handling of eager mode and the global step matches
        # ResourceVariable._init_from_args().
        if not context.executing_eagerly():
          g = ops.get_default_graph()
          # If "trainable" is True, next_creator() will add the contained
          # variable to the TRAINABLE_VARIABLES collection, so we manually
          # remove it and replace with the wrapper. We can't set "trainable"
          # to False for next_creator() since that causes functions like
          # implicit_gradients to skip those variables.
          if kwargs.get("trainable", True):
            collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
            l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
            l.remove(v)
          g.add_to_collections(collections, wrapped)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, wrapped)

        return wrapped
    else:
      var_creator = next_creator

    if "colocate_with" in kwargs:
      colocate_with = kwargs["colocate_with"]
      if isinstance(colocate_with, numpy_dataset.SingleDevice):
        with ops.device(colocate_with.device):
          return var_creator(*args, **kwargs)
      with ops.device(None):
        with ops.colocate_with(colocate_with):
          return var_creator(*args, **kwargs)

    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(self._variable_device):
        return var_creator(*args, **kwargs)

  def _call_for_each_replica(self, fn, args, kwargs):
    # pylint: disable=protected-access
    return mirrored_strategy._call_for_each_replica(
        self._container_strategy(), self._device_map, fn, args, kwargs)

  def _verify_destinations_not_different_worker(self, destinations):
    if not self._cluster_spec:
      return
    if destinations is None:
      return
    for d in cross_device_ops_lib.get_devices_from(destinations):
      d_spec = tf_device.DeviceSpec.from_string(d)
      if d_spec.job == self._task_type and d_spec.task != self._task_id:
        raise ValueError(
            "Cannot reduce to another worker: %r, current worker is %r" %
            (d, self._input_workers.worker_devices[0]))

  def _reduce_to(self, reduce_op, value, destinations):
    self._verify_destinations_not_different_worker(destinations)
    if not isinstance(value, values.DistributedValues):
      # pylint: disable=protected-access
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, self._device_map, value, destinations)
    return self._cross_device_ops.reduce(
        reduce_op, value, destinations=destinations)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs):
    for _, destinations in value_destination_pairs:
      self._verify_destinations_not_different_worker(destinations)
    return self._cross_device_ops.batch_reduce(reduce_op,
                                               value_destination_pairs)

  def _select_single_value(self, structured):
    """Select any single values in `structured`."""

    def _select_fn(x):  # pylint: disable=g-missing-docstring
      if isinstance(x, values.Mirrored):
        if len(x.devices) == 1:
          return x.primary
        else:
          raise ValueError(
              "You cannot update variable with a Mirrored object with multiple "
              "components %r when using ParameterServerStrategy. You must "
              "specify a single value or a Mirrored with a single value." % x)
      elif isinstance(x, values.PerReplica):
        raise ValueError(
            "You cannot update variable with a PerReplica object %r when using "
            "ParameterServerStrategy. You must specify a single value or a "
            "Mirrored with a single value" % x)
      else:
        return x

    return nest.map_structure(_select_fn, structured)

  def _update(self, var, fn, args, kwargs, group):
    if isinstance(var, values.AggregatingVariable):
      var = var.get()
    if not isinstance(var, resource_variable_ops.ResourceVariable):
      raise ValueError(
          "You can not update `var` %r. It must be a Variable." % var)
    with ops.colocate_with(var), distribute_lib.UpdateContext(var.device):
      result = fn(var, *self._select_single_value(args),
                  **self._select_single_value(kwargs))
      if group:
        return result
      else:
        return nest.map_structure(self._unwrap, result)

  # TODO(yuefengz): does it need to call _select_single_value?
  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    with ops.device(
        colocate_with.device), distribute_lib.UpdateContext(colocate_with):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._unwrap, result)

  def _unwrap(self, val):
    if isinstance(val, values.DistributedValues):
      return val.values
    return (val,)

  def value_container(self, val):
    if (hasattr(val, "_aggregating_container") and
        not isinstance(val, values.AggregatingVariable)):
      wrapper = val._aggregating_container()  # pylint: disable=protected-access
      if wrapper is not None:
        return wrapper
    return val

  def read_var(self, var):
    # No need to distinguish between normal variables and replica-local
    # variables.
    return array_ops.identity(var)

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    """Configures the strategy class.

    The strategy object will be re-initialized if `cluster_spec` is given but
    was not passed in the constructor.

    Args:
      session_config: not used currently.
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type.
      task_id: the current task id.

    Raises:
      ValueError: if `cluster_spec` is given but `task_type` or `task_id` is
        not.
    """
    if cluster_spec:
      # Use the num_gpus_per_worker recorded in constructor since _configure
      # doesn't take num_gpus.
      cluster_resolver = SimpleClusterResolver(
          cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
          task_type=task_type,
          task_index=task_id,
          num_accelerators=self._num_gpus_per_worker)
      self._initialize_multi_worker(cluster_resolver)

    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

  def _update_config_proto(self, config_proto):
    updated_config = copy.deepcopy(config_proto)
    if not self._cluster_spec:
      updated_config.isolate_session_state = True
      return updated_config

    updated_config.isolate_session_state = False

    assert self._task_type
    assert self._task_id is not None

    # The device filters prevent communication between workers.
    if self._task_type not in ["chief", "worker"]:
      return updated_config
    del updated_config.device_filters[:]
    updated_config.device_filters.extend(
        ["/job:%s/task:%d" % (self._task_type, self._task_id), "/job:ps"])
    return updated_config

  @property
  def _num_replicas_in_sync(self):
    return self._device_map.num_replicas_in_graph

  @property
  def worker_devices(self):
    return self._device_map.all_devices

  @property
  def worker_devices_by_replica(self):
    return self._device_map.devices_by_replica

  @property
  def parameter_devices(self):
    return self._parameter_devices

  def non_slot_devices(self, var_list):
    return min(var_list, key=lambda x: x.name)

  @property
  def experimental_between_graph(self):
    # TODO(yuefengz): Should this return False in the local case?
    return True

  @property
  def experimental_should_init(self):
    return self._is_chief

  @property
  def should_checkpoint(self):
    return self._is_chief

  @property
  def should_save_summary(self):
    return self._is_chief

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `distribute_dataset` and `make_input_fn_iterator` assume per-replica
    batching.

    Returns:
      Boolean.
    """
    return True
