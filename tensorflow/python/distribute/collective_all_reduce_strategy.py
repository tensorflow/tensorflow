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

import copy

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


# TODO(yuefengz): support in-graph replication.
@tf_export("distribute.experimental.MultiWorkerMirroredStrategy", v1=[])
class CollectiveAllReduceStrategy(distribute_lib.Strategy):
  """Distribution strategy that uses collective ops for all-reduce.

  It is similar to MirroredStrategy but it uses collective ops for reduction.

  By default it uses all local GPUs or CPU for single-worker training.

  When 'TF_CONFIG' environment variable is given, it parses cluster_spec,
  task_type and task_id from 'TF_CONFIG' and turns into a multi-worker strategy
  which mirrores models on GPUs of all machines in a cluster. In the current
  implementation, it uses all GPUs in a cluster and it assumes all workers have
  the same number of GPUs.

  It supports both eager mode and graph mode. However, for eager mode, it has to
  set up the eager context in its constructor and therefore all ops in eager
  mode have to run after the strategy object is created.

  Args:
    communication: optional Enum of type
      `distribute.experimental.CollectiveCommunication`.  This provides a way
      for the user to override the choice of collective op communication.
      Possible values include `AUTO`, `RING`, and `NCCL`.
  """

  def __init__(
      self,
      communication=cross_device_ops_lib.CollectiveCommunication.AUTO):
    """Initializes the object."""
    super(CollectiveAllReduceStrategy, self).__init__(
        CollectiveAllReduceExtended(
            self,
            communication=communication))


@tf_export(v1=["distribute.experimental.MultiWorkerMirroredStrategy"])
class CollectiveAllReduceStrategyV1(distribute_lib.StrategyV1):

  __doc__ = CollectiveAllReduceStrategy.__doc__

  def __init__(
      self,
      communication=cross_device_ops_lib.CollectiveCommunication.AUTO):
    """Initializes the object."""
    super(CollectiveAllReduceStrategyV1, self).__init__(
        CollectiveAllReduceExtended(
            self,
            communication=communication))


class CollectiveAllReduceExtended(mirrored_strategy.MirroredExtended):
  """Implementation of CollectiveAllReduceStrategy."""

  def __init__(self,
               container_strategy,
               communication,
               cluster_resolver=TFConfigClusterResolver()):
    distribute_lib.StrategyExtendedV1.__init__(self, container_strategy)
    assert isinstance(
        communication,
        cross_device_ops_lib.CollectiveCommunication)
    self._communication = communication
    self._initialize_strategy(cluster_resolver)
    assert isinstance(self._get_cross_device_ops(),
                      cross_device_ops_lib.CollectiveAllReduce)

  def _initialize_strategy(self, cluster_resolver):
    if cluster_resolver.cluster_spec().as_dict():
      self._initialize_multi_worker(cluster_resolver)
    else:
      self._initialize_local(cluster_resolver)

  def _initialize_local(self, cluster_resolver):
    """Initializes the object for local training."""
    self._is_chief = True
    self._num_workers = 1

    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    if isinstance(cluster_resolver, TFConfigClusterResolver):
      num_gpus = context.num_gpus()
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)

    if num_gpus:
      local_devices = tuple("/device:GPU:%d" % i for i in range(num_gpus))
    else:
      local_devices = ("/device:CPU:0",)
    self._worker_device = device_util.canonicalize("/device:CPU:0")
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)

    self._collective_keys = cross_device_utils.CollectiveKeys()
    super(CollectiveAllReduceExtended, self)._initialize_local(local_devices)
    # TODO(yuefengz): remove num_gpus_per_worker from CollectiveAllReduce.
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        num_workers=self._num_workers,
        num_gpus_per_worker=num_gpus,
        collective_keys=self._collective_keys)

    self._cluster_spec = None
    self._task_type = None
    self._task_id = None

    # This is a mark to tell whether we are running with standalone client or
    # independent worker. Right now with standalone client, strategy object is
    # created as local strategy and then turn into multi-worker strategy via
    # configure call.
    self._local_or_standalone_client_mode = True

    # Save the num_gpus_per_worker and rpc_layer for configure method.
    self._num_gpus_per_worker = num_gpus
    self._rpc_layer = cluster_resolver.rpc_layer

    logging.info("CollectiveAllReduceStrategy with local_devices = %r",
                 local_devices)

  def _initialize_multi_worker(self, cluster_resolver):
    """Initializes the object for multi-worker training."""
    # TODO(yuefengz): The `num_gpus` is only for this particular task. It
    # assumes all workers have the same number of GPUs. We should remove this
    # assumption by querying all tasks for their numbers of GPUs.
    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    if isinstance(cluster_resolver, TFConfigClusterResolver):
      num_gpus = context.num_gpus()
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)

    cluster_spec = multi_worker_util.normalize_cluster_spec(
        cluster_resolver.cluster_spec())
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    if task_type is None or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`.")

    self._num_workers = multi_worker_util.worker_count(cluster_spec, task_type)
    if not self._num_workers:
      raise ValueError("No `worker`, `chief` or `evaluator` tasks can be found "
                       "in `cluster_spec`.")

    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)

    self._worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)
    if num_gpus:
      local_devices = tuple("%s/device:GPU:%d" % (self._worker_device, i)
                            for i in range(num_gpus))
    else:
      local_devices = (self._worker_device,)

    self._collective_keys = cross_device_utils.CollectiveKeys()
    super(CollectiveAllReduceExtended, self)._initialize_local(local_devices)
    self._input_workers = input_lib.InputWorkers(
        self._device_map, [(self._worker_device, self.worker_devices)])
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        num_workers=self._num_workers,
        num_gpus_per_worker=num_gpus,
        collective_keys=self._collective_keys)

    # Add a default device so that ops without specified devices will not end up
    # on other workers.
    self._default_device = "/job:%s/task:%d" % (task_type, task_id)

    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id

    # Save the num_gpus_per_worker and rpc_layer for configure method.
    self._num_gpus_per_worker = num_gpus
    self._rpc_layer = cluster_resolver.rpc_layer

    logging.info(
        "Multi-worker CollectiveAllReduceStrategy with cluster_spec = %r, "
        "task_type = %r, task_id = %r, num_workers = %r, local_devices = %r, "
        "communication = %s", cluster_spec.as_dict(), task_type,
        task_id, self._num_workers, local_devices,
        self._communication)

    if (context.executing_eagerly() and
        not getattr(self, "_std_server_started", False) and
        not getattr(self, "_local_or_standalone_client_mode", False)):
      # Checking _local_or_standalone_client_mode as well because we should not
      # create the std server in standalone client mode.
      config_proto = config_pb2.ConfigProto()
      config_proto = self._update_config_proto(config_proto)
      server_def = tensorflow_server_pb2.ServerDef(
          cluster=cluster_spec.as_cluster_def(),
          default_session_config=config_proto,
          job_name=task_type,
          task_index=task_id,
          protocol=cluster_resolver.rpc_layer or "grpc")
      context.context().enable_collective_ops(server_def)
      self._std_server_started = True
      logging.info(
          "Enabled multi-worker collective ops with available devices: %r",
          context.context().devices())

  def _create_variable(self, next_creator, *args, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      device_map = self._device_map
      logical_device = 0  # TODO(josh11b): Get logical device from scope here.
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(*args, **kwargs)
    else:
      device_map = colocate_with.device_map
      logical_device = colocate_with.logical_device

    def _real_mirrored_creator(devices, *args, **kwargs):
      """Creates one MirroredVariable on the current worker."""
      unique_var_name = ops.get_default_graph().unique_name(
          kwargs["name"], mark_as_used=False).rstrip("/")
      # pylint: disable=protected-access
      collective_instance_key = self._collective_keys.get_instance_key(
          key_id=unique_var_name)
      # Only the first device participles in the broadcast of initial values.
      group_key = self._collective_keys.get_group_key([devices[0]])
      group_size = self._num_workers
      if "initial_value" not in kwargs:
        raise ValueError("Initial value must be specified.")
      initial_value = kwargs["initial_value"]
      if callable(initial_value):
        initial_value_fn = initial_value
      else:
        initial_value_fn = lambda: initial_value

      value_list = []
      for i, d in enumerate(devices):
        with ops.init_scope(), ops.device(d):
          if i == 0:
            # The initial value fn makes sure variables all initialized to
            # same values. The first device of the chief worker will send their
            # variable values to other workers.
            def _overridden_initial_value_fn(device=d, index=i):  # pylint: disable=g-missing-docstring
              with ops.device(device):
                initial_value = initial_value_fn()
                assert not callable(initial_value)
                initial_value = ops.convert_to_tensor(initial_value)

                assert index == 0, index
                if self._num_workers > 1:
                  if self._is_chief:
                    bcast_send = collective_ops.broadcast_send(
                        initial_value, initial_value.shape, initial_value.dtype,
                        group_size, group_key, collective_instance_key)
                    with ops.control_dependencies([bcast_send]):
                      return array_ops.identity(initial_value)
                  else:
                    return collective_ops.broadcast_recv(
                        initial_value.shape, initial_value.dtype, group_size,
                        group_key, collective_instance_key)
                return initial_value
          else:
            # Give replicas meaningful distinct names:
            var0name = value_list[0].name.split(":")[0]
            # We append a / to variable names created on replicas with id > 0 to
            # ensure that we ignore the name scope and instead use the given
            # name as the absolute name of the variable.
            kwargs["name"] = "%s/replica_%d/" % (var0name, i)

            # Variables on non-first replica get initial values from the
            # variables created on the first device of each worker.
            def _overridden_initial_value_fn(device=d, index=i):
              assert index > 0
              with ops.device(device):
                if context.executing_eagerly():
                  return array_ops.identity(value_list[0].value())
                else:
                  return array_ops.identity(value_list[0].initial_value)

          kwargs["initial_value"] = _overridden_initial_value_fn
          with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
            # Don't record operations (e.g. other variable reads) during
            # variable creation.
            with tape.stop_recording():
              v = next_creator(*args, **kwargs)

          if i == 0:
            actual_var_name = v.name.split(":")[0]
            assert unique_var_name == actual_var_name, "%r vs %r" % (
                unique_var_name, actual_var_name)
          assert not isinstance(v, values.DistributedVariable)
          value_list.append(v)
      return value_list

    # pylint: disable=protected-access
    return mirrored_strategy._create_mirrored_variable(
        self._container_strategy(), device_map, logical_device,
        _real_mirrored_creator, *args, **kwargs)

  def _make_input_context(self):
    if self._cluster_spec is None:
      input_pipeline_id = 0
    else:
      input_pipeline_id = multi_worker_util.id_in_cluster(
          self._cluster_spec, self._task_type, self._task_id)
    input_context = distribute_lib.InputContext(
        num_input_pipelines=self._num_workers,
        input_pipeline_id=input_pipeline_id,
        num_replicas_in_sync=self._num_replicas_in_sync)
    return input_context

  def _experimental_distribute_dataset(self, dataset):
    input_context = self._make_input_context()
    return input_lib.get_distributed_dataset(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync,
        input_context=input_context)

  def _make_dataset_iterator(self, dataset):
    """Distributes the dataset to each local GPU."""
    input_context = self._make_input_context()
    return input_lib.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync,
        input_context=input_context)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    """Distributes the input function to each local GPU."""
    input_context = self._make_input_context()
    return input_lib.InputFunctionIterator(input_fn, self._input_workers,
                                           [input_context],
                                           self._container_strategy())

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    """Configures the object.

    Args:
      session_config: a `tf.compat.v1.ConfigProto`
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type, such as "worker".
      task_id: the current task id.

    Raises:
      ValueError: if `task_type` is not in the `cluster_spec`.
    """
    if cluster_spec:
      # Use the num_gpus_per_worker recorded in constructor since _configure
      # doesn't take num_gpus.
      cluster_resolver = SimpleClusterResolver(
          cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
          task_type=task_type,
          task_id=task_id,
          num_accelerators={"GPU": self._num_gpus_per_worker},
          rpc_layer=self._rpc_layer)
      self._initialize_multi_worker(cluster_resolver)
      assert isinstance(self._get_cross_device_ops(),
                        cross_device_ops_lib.CollectiveAllReduce)

    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

  def _update_config_proto(self, config_proto):
    updated_config = copy.deepcopy(config_proto)
    # Enable the scoped allocator optimization for CollectiveOps.  This
    # optimization converts many small all-reduces into fewer larger
    # all-reduces.
    rewrite_options = updated_config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    # We turn on ScopedAllocator only for CollectiveReduce op, i.e. enable_op =
    # ["CollectiveReduce"].  Since we can't assign to a repeated proto field, we
    # clear and then append.
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append("CollectiveReduce")

    if ((self._communication ==
         cross_device_ops_lib.CollectiveCommunication.NCCL) and
        self._num_gpus_per_worker > 0):
      updated_config.experimental.collective_nccl = True

    if not self._cluster_spec:
      return updated_config

    assert self._task_type
    assert self._task_id is not None

    # Collective group leader is needed for collective ops to coordinate
    # workers.
    updated_config.experimental.collective_group_leader = (
        multi_worker_util.collective_leader(self._cluster_spec, self._task_type,
                                            self._task_id))

    # The device filters prevent communication between workers.
    del updated_config.device_filters[:]
    updated_config.device_filters.append(
        "/job:%s/task:%d" % (self._task_type, self._task_id))

    return updated_config

  def _reduce_to(self, reduce_op, value, destinations):
    if (isinstance(value, values.Mirrored) and
        reduce_op == reduce_util.ReduceOp.MEAN):
      return value
    assert not isinstance(value, values.Mirrored)

    if (isinstance(value, values.DistributedValues) and
        len(self.worker_devices) == 1):
      value = value.values[0]

    # When there are multiple workers, we need to reduce across workers using
    # collective ops.
    if (not isinstance(value, values.DistributedValues) and
        self._num_workers == 1):
      # This function handles reducing values that are not PerReplica or
      # Mirrored values. For example, the same value could be present on all
      # replicas in which case `value` would be a single value or value could
      # be 0.
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, self._device_map, value, destinations)
    return self._get_cross_device_ops().reduce(
        reduce_op, value, destinations=destinations)

  @property
  def experimental_between_graph(self):
    return True

  @property
  def experimental_should_init(self):
    return True

  @property
  def should_checkpoint(self):
    return self._is_chief

  @property
  def should_save_summary(self):
    return self._is_chief

  @property
  def _num_replicas_in_sync(self):
    return len(self.worker_devices) * self._num_workers

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True
