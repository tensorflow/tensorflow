"""Class HybridStrategy implementing tf.distribute.Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.core.protobuf import rewriter_config_pb2
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
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import tape
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.eager import context

_LOCAL_CPU = "/device:CPU:0"

@tf_export("distribute.experimental.HybridStrategy", v1=[])
class HybridStrategy(distribute_lib.Strategy):
  def __init__(
        self,
        communication=cross_device_ops_lib.CollectiveCommunication.AUTO,
        cluster_resolver=None):
    if cluster_resolver is None:
      cluster_resolver = TFConfigClusterResolver()
    if not cluster_resolver.cluster_spec():
      raise ValueError("Cluster spec must be non-empty in `cluster_resolver`.")
    extended = HybridStrategyExtended(
      self,
      communication=communication,
      cluster_resolver=cluster_resolver)
    super(HybridStrategy, self).__init__(extended)

    '''
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "HybridStrategy")
    distribute_lib.distribution_strategy_replica_gauge.get_cell("num_ps").set(
        len(self.extended.parameter_devices))
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended._num_gpus_per_worker)
    '''


class HybridStrategyExtended(distribute_lib.StrategyExtendedV1):
  def __init__(self,
               container_strategy,
               communication,
               cluster_resolver):
    super(HybridStrategyExtended, self).__init__(container_strategy)
    assert isinstance(
      communication,
      cross_device_ops_lib.CollectiveCommunication)
    self._communication = communication
    self._initialize_strategy(cluster_resolver)
    assert isinstance(self._get_cross_device_ops(),
                     cross_device_ops_lib.CollectiveAllReduce)

  def _initialize_strategy(self, cluster_resolver):
    assert cluster_resolver.cluster_spec().as_dict() is not None
    self._initialize_multi_worker(cluster_resolver)

  def _initialize_multi_worker(self, cluster_resolver):
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_resolver.cluster_spec())
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id

    if task_type is None or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`.")
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id

    self._num_workers = multi_worker_util.worker_count(cluster_spec, 'worker')
    if not self._num_workers:
      raise ValueError("No `worker`, `chief` or `evaluator` tasks can be found "
                       "in `cluster_spec`.")

    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)

    worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._host_input_device = numpy_dataset.SingleDevice(worker_device + "/device:CPU:0")
    self._default_device = worker_device

    # below codes are needed in eager mode
    '''
    if (ops.executing_eagerly_outside_functions() and
          not getattr(self, "_local_or_standalone_client_mode", False)):
      context.context().configure_collective_ops(
        collective_leader=multi_worker_util.collective_leader(
          cluster_spec, task_type, task_id),
        scoped_allocator_enabled_ops=("CollectiveReduce",),
        device_filters=("/job:%s/task:%d" % (task_type, task_id),))
      self._collective_ops_configured = True
    '''

    '''
    # Starting a std server in eager mode and in independent worker mode.
    if (context.executing_eagerly() and
          not getattr(self, "_std_server_started", False) and
          not getattr(self, "_local_or_standalone_client_mode", False)):
      # Checking _local_or_standalone_client_mode as well because we should not
      # create the std server in standalone client mode.
      config_proto = config_pb2.ConfigProto()
      config_proto = self._update_config_proto(config_proto)

      if hasattr(cluster_resolver, "port"):
        port = cluster_resolver.port
      else:
        port = 0
      server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_spec.as_cluster_def(),
        default_session_config=config_proto,
        job_name=task_type,
        task_index=task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        port=port)
      context.context().enable_collective_ops(server_def)
      self._std_server_started = True
      # The `ensure_initialized` is needed before calling
      # `context.context().devices()`.
      context.context().ensure_initialized()
      logging.info(
        "Enabled multi-worker collective ops with available devices: %r",
        context.context().devices())
    '''

    # TODO(yuefengz): The `num_gpus` is only for this particular task. It
    # assumes all workers have the same number of GPUs. We should remove this
    # assumption by querying all tasks for their numbers of GPUs.
    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    if isinstance(cluster_resolver, TFConfigClusterResolver):
      num_gpus = context.num_gpus()
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)

    self._num_gpus_per_worker = num_gpus

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

    self._collective_keys = cross_device_utils.CollectiveKeys()
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
      num_workers=self._num_workers,
      num_gpus_per_worker=num_gpus,
      collective_keys=self._collective_keys,
      communication=self._communication)

    self._ps_cross_device_ops = cross_device_ops_lib.ReductionToOneDevice(reduce_to_device="/device:CPU:0")
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()

    logging.info(
      "Multi-worker ParameterServerStrategy with "
      "cluster_spec = %r, task_type = %r, task_id = %r, "
      "num_ps_replicas = %r, is_chief = %r, device_map  = %r, "
      "variable_device = %r", cluster_spec.as_dict(), task_type, task_id,
      num_ps_replicas, self._is_chief, self._device_map,
      self._variable_device)

  def _warn_nccl_no_gpu(self):
    if ((self._communication ==
         cross_device_ops_lib.CollectiveCommunication.NCCL) and
        self._num_gpus_per_worker == 0):
      logging.warning("Enabled NCCL communication but no GPUs detected/"
                      "specified.")

  def _validate_colocate_with_variable(self, colocate_with_variable):
    values.validate_colocate(colocate_with_variable, self)

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
        split_batch_by=self._num_replicas_in_sync,)
        # input_context=input_context)

  def _make_dataset_iterator(self, dataset):
    input_context = self._make_input_context()
    return input_lib.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync,)
        # input_context=input_context)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    """Distributes the dataset to each local GPU."""
    input_context = self._make_input_context()
    return input_lib.InputFunctionIterator(input_fn, self._input_workers,
                                           [input_context],
                                           self._container_strategy())

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, self._input_host_device, session)

  def _experimental_distribute_datasets_from_function(self, dataset_fn):
    input_context = self._make_input_context()
    return input_lib.get_distributed_datasets_from_function(
      dataset_fn,
      self._input_workers,
      [input_context],
      self._container_strategy())

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
    return self._get_cross_device_ops().broadcast(tensor, destinations)

  def _get_cross_device_ops(self):
    return self._cross_device_ops

  def _allow_variable_partition(self):
    return not context.executing_eagerly()

  def _get_variable_creator_initial_value(self,
                                          replica_id,
                                          device,
                                          primary_var,
                                          **kwargs):
    """Return the initial value for variables on a replica."""
    if replica_id == 0:
      assert device is not None
      assert primary_var is None

      def initial_value_fn():  # pylint: disable=g-missing-docstring
        # Only the first device participates in the broadcast of initial values.
        group_key = self._collective_keys.get_group_key([device])
        group_size = self._num_workers
        collective_instance_key = (
          self._collective_keys.get_variable_instance_key())

        with ops.device(device):
          initial_value = kwargs["initial_value"]
          if callable(initial_value):
            initial_value = initial_value()
          assert not callable(initial_value)
          initial_value = ops.convert_to_tensor(
            initial_value, dtype=kwargs.get("dtype", None))

          if self._num_workers > 1:
            if self._is_chief:
              bcast_send = collective_ops.broadcast_send(
                initial_value, initial_value.shape, initial_value.dtype,
                group_size, group_key, collective_instance_key)
              with ops.control_dependencies([bcast_send]):
                return array_ops.identity(initial_value)
            else:
              return collective_ops.broadcast_recv(initial_value.shape,
                                                   initial_value.dtype,
                                                   group_size, group_key,
                                                   collective_instance_key)
          return initial_value

      return initial_value_fn
    else:
      assert primary_var is not None
      assert device is not None
      assert kwargs is not None

      def initial_value_fn():
        if context.executing_eagerly() or ops.inside_function():
          init_value = primary_var.value()
          return array_ops.identity(init_value)
        else:
          with ops.device(device):
            init_value = primary_var.initial_value
            return array_ops.identity(init_value)

      return initial_value_fn

  # TODO(yuefengz): Not all ops in device_setter.STANDARD_PS_OPS will go through
  # this creator, such as "MutableHashTable".
  def _create_variable(self, next_creator, *args, **kwargs):
    name = kwargs["name"]
    if (name and ('embedding' in name )):
      # ps case
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
              if v in l:
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

    else:
      """Create a mirrored variable. See `DistributionStrategy.scope`."""
      #  Mirrored Case
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

      def _real_mirrored_creator(devices, *args, **kwargs):  # pylint: disable=g-missing-docstring
        value_list = []
        for i, d in enumerate(devices):
          with ops.device(d):
            kwargs["initial_value"] = self._get_variable_creator_initial_value(
              replica_id=i,
              device=d,
              primary_var=value_list[0] if value_list else None,
              **kwargs)
            if i > 0:
              # Give replicas meaningful distinct names:
              var0name = value_list[0].name.split(":")[0]
              # We append a / to variable names created on replicas with id > 0 to
              # ensure that we ignore the name scope and instead use the given
              # name as the absolute name of the variable.
              kwargs["name"] = "%s/replica_%d/" % (var0name, i)
            with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
              # Don't record operations (e.g. other variable reads) during
              # variable creation.
              with tape.stop_recording():
                v = next_creator(*args, **kwargs)
            assert not isinstance(v, values.DistributedVariable)
            value_list.append(v)
        return value_list

      return values.create_mirrored_variable(
        self._container_strategy(), device_map, logical_device,
            _real_mirrored_creator, values.MirroredVariable,
            values.SyncOnReadVariable, *args, **kwargs)

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
    if isinstance(destinations, str):
      return self._ps_cross_device_ops.reduce( reduce_op, value, destinations=destinations)
      
    if isinstance(destinations, values.AggregatingVariable):
      return self._ps_cross_device_ops.reduce(
        reduce_op, value, destinations=destinations)

    if (isinstance(value, values.Mirrored) and
          reduce_op == reduce_util.ReduceOp.MEAN):
      return value

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
      # be 0.d
      return cross_device_ops_lib.reduce_non_distributed_value(
        reduce_op, self._device_map, value, destinations)
    return self._get_cross_device_ops().reduce(
      reduce_op, value, destinations=destinations)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs):
    # return self._ps_cross_device_ops.batch_reduce(reduce_op, value_destination_pairs)
    v_d_pairs_with_all_reduce = []
    indices_with_all_reduce = []
    v_d_pairs_with_ps_cross_device = []
    indices_with_ps_cross_device = []
    results = [None] * len(value_destination_pairs)

    for i, v_d_pair in enumerate(value_destination_pairs):
      if isinstance(v_d_pair[1], values.AggregatingVariable):
        v_d_pairs_with_ps_cross_device.append(v_d_pair)
        indices_with_ps_cross_device.append(i)
      else:
        v_d_pairs_with_all_reduce.append(v_d_pair)
        indices_with_all_reduce.append(i)

    if v_d_pairs_with_all_reduce:
      results_with_all_reduce = self._get_cross_device_ops().batch_reduce(reduce_op, v_d_pairs_with_all_reduce)
      for value, index in zip(results_with_all_reduce, indices_with_all_reduce):
        results[index] = value

    if v_d_pairs_with_ps_cross_device:
      results_with_ps_cross_device = self._ps_cross_device_ops.batch_reduce(reduce_op, v_d_pairs_with_ps_cross_device)
      for value, index in zip(results_with_ps_cross_device, indices_with_ps_cross_device):
        results[index] = value
    print('results : ', results)
    return results

  def _select_single_value(self, structured):
    """Select any single value in `structured`."""

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
    if isinstance(var, values.DistributedVariable):
      #  Mirroed case
      updates = []
      for i, (d, v) in enumerate(zip(var.devices, var.values)):
        name = "update_%d" % i
        with ops.device(d), distribute_lib.UpdateContext(d), ops.name_scope(name):
          # If args and kwargs are not mirrored, the value is returned as is.
          updates.append(fn(v,
                            *values.select_device_mirrored(d, args),
                            **values.select_device_mirrored(d, kwargs)))
      return values.update_regroup(self, self._device_map, updates, group)

    # PS case
    if isinstance(var, values.AggregatingVariable):
      var = var.get()
    if not isinstance(var, resource_variable_ops.BaseResourceVariable):
      raise ValueError(
          "You can not update `var` %r. It must be a Variable." % var)
    with ops.colocate_with(var), distribute_lib.UpdateContext(var.device):
      result = fn(var, *self._select_single_value(args),
                  **self._select_single_value(kwargs))
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  # TODO(yuefengz): does it need to call _select_single_value?
  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    if isinstance(colocate_with, tuple):
      # Mirrored
      updates = []
      for i, d in enumerate(colocate_with):
        name = "update_%d" % i
        with ops.device(d), distribute_lib.UpdateContext(i), ops.name_scope(name):
          updates.append(fn(*values.select_device_mirrored(d, args),
                            **values.select_device_mirrored(d, kwargs)))
      return values.update_regroup(self, self._device_map, updates, group)

    # PS
    with ops.device(
        colocate_with.device), distribute_lib.UpdateContext(colocate_with):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  def _local_results(self, val):
    if isinstance(val, values.DistributedValues):
      return val.values
    return (val,)

  def value_container(self, val):
    if (hasattr(val, "_aggregating_container") and
        not isinstance(val, values.AggregatingVariable)):
      wrapper = val._aggregating_container()  # pylint: disable=protected-access
      if wrapper is not None:
        return wrapper
    return values.value_container(val)

  def read_var(self, var):
    # No need to distinguish between normal variables and replica-local
    # variables.
    if isinstance(var, values.SyncOnReadVariable):
      return var._get_cross_replica()  # pylint: disable=protected-access
    if isinstance(var, values.Mirrored):
      return array_ops.identity(var.get())
    return array_ops.identity(var)

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    """Configures the strategy class with `cluser_spec`.

    The strategy object will be re-initialized if `cluster_spec` is passed to
    `configure` but was not passed when instantiating the strategy.

    Args:
      session_config: Session config object.
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

    if (not ops.executing_eagerly_outside_functions() and
        self._communication ==
        cross_device_ops_lib.CollectiveCommunication.NCCL):
      updated_config.experimental.collective_nccl = True

    if not self._cluster_spec:
      updated_config.isolate_session_state = True
      return updated_config

    updated_config.isolate_session_state = False

    assert self._task_type
    assert self._task_id is not None

    # Collective group leader is needed for collective ops to coordinate
    # workers.
    updated_config.experimental.collective_group_leader = (
        multi_worker_util.collective_leader(self._cluster_spec, self._task_type,
                                            self._task_id))

    # The device filters prevent communication between workers.
    del updated_config.device_filters[:]
    if self._task_type in ["chief", "worker"]:
      updated_config.device_filters.extend(
          ["/job:%s/task:%d" % (self._task_type, self._task_id), "/job:ps"])
    elif self._task_type == "evaluator":
      updated_config.device_filters.append(
          "/job:%s/task:%d" % (self._task_type, self._task_id))

    return updated_config

  def _in_multi_worker_mode(self):
    """Whether this strategy indicates working in multi-worker settings."""
    # With a PS job, PS strategy should always be considered as in multi
    # worker mode.
    return True

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
    return True

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

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True
