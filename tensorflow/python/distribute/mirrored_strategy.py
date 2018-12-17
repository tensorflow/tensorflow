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
"""Class MirroredStrategy implementing DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import functools
import threading

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import coordinator
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# TODO(josh11b): Replace asserts in this file with if ...: raise ...


@contextlib.contextmanager
def _enter_graph(g, eager, creator_stack=None):
  """Context manager for selecting a graph and maybe eager mode."""
  if eager:
    with g.as_default(), context.eager_mode():
      if creator_stack is not None:
        g._variable_creator_stack = creator_stack  # pylint: disable=protected-access
      yield
  else:
    with g.as_default():
      if creator_stack is not None:
        g._variable_creator_stack = creator_stack  # pylint: disable=protected-access
      yield


def _cpu_device(device):
  cpu_device = tf_device.DeviceSpec.from_string(device)
  cpu_device.merge_from(tf_device.DeviceSpec(device_type="CPU", device_index=0))
  return cpu_device.to_string()


class _RequestedStop(Exception):  # pylint: disable=g-bad-exception-name
  pass


# _call_for_each_replica is not a member of MirroredStrategy so that it is
# not allowed to use anything specific to MirroredStrategy and thus
# can be shared with other distribution strategies.


# TODO(yuefengz): maybe create a common class for those who need to call this
# _call_for_each_replica.
def _call_for_each_replica(distribution, fn, args, kwargs):
  """Run `fn` in separate threads, once per replica/worker device.

  Args:
    distribution: the DistributionStrategy object.
    fn: function to run (will be run once per device, each in its own thread).
    args: positional arguments for `fn`
    kwargs: keyword arguments for `fn`.

  Returns:
    Merged return value of `fn` across all replicas.

  Raises:
    RuntimeError: If fn() calls get_replica_context().merge_call() a different
        number of times from the available devices.
  """
  # TODO(josh11b): Add this option once we add synchronization to variable
  # creation. Until then, this is pretty unsafe to use.
  run_concurrently = False
  if not context.executing_eagerly():
    # Needed for per-thread device, etc. contexts in graph mode.
    ops.get_default_graph().switch_to_thread_local()

  coord = coordinator.Coordinator(clean_stop_exception_types=(_RequestedStop,))

  shared_variable_store = {}

  # TODO(isaprykin): Create these threads once instead of during every run()
  # call.
  threads = []
  for index, d in enumerate(distribution.extended.worker_devices):
    variable_creator_fn = shared_variable_creator.make_fn(
        shared_variable_store, index)
    t = MirroredExtended._MirroredReplicaThread(  # pylint: disable=protected-access
        distribution, coord, d, variable_creator_fn, fn,
        *values.select_device(d, args), **values.select_device(d, kwargs))
    threads.append(t)

  for t in threads:
    t.start()

  # When `fn` starts `should_run` event is set on _MirroredReplicaThread
  # (`MRT`) threads. The execution waits until
  # `MRT.has_paused` is set, which indicates that either `fn` is
  # complete or a `get_replica_context().merge_call()` is called.  If `fn` is
  # complete, then `MRT.done` is set to True.  Otherwise, arguments
  # of `get_replica_context().merge_call` from all paused threads are grouped
  # and the `merge_fn` is performed.  Results of the
  # `get_replica_context().merge_call` are then set to `MRT.merge_result`.
  # Each such `get_replica_context().merge_call` call returns the
  # `MRT.merge_result` for that thread when `MRT.should_run` event
  # is reset again. Execution of `fn` resumes.

  try:
    with coord.stop_on_exception():
      all_done = False
      while not all_done and not coord.should_stop():
        done = []
        if run_concurrently:
          for t in threads:
            t.should_run.set()
          for t in threads:
            t.has_paused.wait()
            t.has_paused.clear()
            if coord.should_stop():
              return None
            done.append(t.done)
        else:
          for t in threads:
            t.should_run.set()
            t.has_paused.wait()
            t.has_paused.clear()
            if coord.should_stop():
              return None
            done.append(t.done)
        if coord.should_stop():
          return None
        all_done = all(done)
        if not all_done:
          if any(done):
            raise RuntimeError("Some replicas made a different number of "
                               "replica_context().merge_call() calls.")
          # get_replica_context().merge_call() case
          merge_args = values.regroup({t.device: t.merge_args for t in threads})
          merge_kwargs = values.regroup(
              {t.device: t.merge_kwargs for t in threads})
          # We capture the name_scope of the MRT when we call merge_fn
          # to ensure that if we have opened a name scope in the MRT,
          # it will be respected when executing the merge function. We only
          # capture the name_scope from the first MRT and assume it is
          # the same for all other MRTs.
          mtt_captured_name_scope = threads[0].captured_name_scope
          # Capture and merge the control dependencies from all the threads.
          mtt_captured_control_deps = set()
          for t in threads:
            mtt_captured_control_deps.update(t.captured_control_deps)
          with ops.name_scope(mtt_captured_name_scope),\
              ops.control_dependencies(mtt_captured_control_deps):
            merge_result = threads[0].merge_fn(distribution, *merge_args,
                                               **merge_kwargs)
          for t in threads:
            t.merge_result = values.select_device(t.device, merge_result)
  finally:
    for t in threads:
      t.should_run.set()
    coord.join(threads)

  return values.regroup({t.device: t.main_result for t in threads})


def _create_mirrored_variable(devices, real_mirrored_creator, *args, **kwargs):  # pylint: disable=g-missing-docstring
  # Figure out what collections this variable should be added to.
  # We'll add the MirroredVariable to those collections instead.
  collections = kwargs.pop("collections", None)
  if collections is None:
    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  kwargs["collections"] = []

  # Get synchronization value
  synchronization = kwargs.get("synchronization",
                               variable_scope.VariableSynchronization.ON_WRITE)
  if synchronization == variable_scope.VariableSynchronization.NONE:
    raise ValueError("`NONE` variable synchronization mode is not "
                     "supported with `Mirrored` distribution strategy. Please"
                     " change the `synchronization` for variable: " +
                     kwargs["name"])
  elif synchronization == variable_scope.VariableSynchronization.ON_READ:
    # Variables that are to be synced on read are replica local.
    is_replica_local = True
    kwargs["trainable"] = False
  elif (synchronization == variable_scope.VariableSynchronization.ON_WRITE or
        synchronization == variable_scope.VariableSynchronization.AUTO):
    # `AUTO` synchronization for `MirroredStrategy` is `ON_WRITE`.
    is_replica_local = False
  else:
    raise ValueError("Invalid variable synchronization mode: " +
                     synchronization + " for variable: " + kwargs["name"])

  # Get aggregation value
  aggregation = kwargs.pop("aggregation",
                           variable_scope.VariableAggregation.NONE)
  if aggregation not in (
      variable_scope.VariableAggregation.NONE,
      variable_scope.VariableAggregation.SUM,
      variable_scope.VariableAggregation.MEAN,
      variable_scope.VariableAggregation.ONLY_FIRST_REPLICA
  ):
    raise ValueError("Invalid variable aggregation mode: " + aggregation +
                     " for variable: " + kwargs["name"])

  # Ignore user-specified caching device, not needed for mirrored variables.
  kwargs.pop("caching_device", None)

  # TODO(josh11b,apassos): It would be better if variable initialization
  # was never recorded on the tape instead of having to do this manually
  # here.
  with tape.stop_recording():
    index = real_mirrored_creator(devices, *args, **kwargs)

    if is_replica_local:
      result = values.ReplicaLocalVariable(
          index, index[devices[0]], aggregation)
    else:
      result = values.MirroredVariable(index, index[devices[0]], aggregation)

  # Add the wrapped variable to the requested collections.
  # The handling of eager mode and the global step matches
  # ResourceVariable._init_from_args().
  if not context.executing_eagerly():
    g = ops.get_default_graph()
    # If "trainable" is True, next_creator() will add the member variables
    # to the TRAINABLE_VARIABLES collection, so we manually remove
    # them and replace with the MirroredVariable. We can't set
    # "trainable" to False for next_creator() since that causes functions
    # like implicit_gradients to skip those variables.
    if kwargs.get("trainable", True):
      collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
      l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
      for v in index.values():
        if v in l:
          l.remove(v)
    g.add_to_collections(collections, result)
  elif ops.GraphKeys.GLOBAL_STEP in collections:
    ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, result)

  return result


def _is_device_list_local(devices):
  """Checks whether the devices list is for local or multi-worker.

  Args:
    devices: a list of device strings, either local for remote devices.

  Returns:
    a boolean indicating whether these device strings are for local or for
    remote.

  Raises:
    ValueError: if device strings are not consistent.
  """
  all_local = None
  for d in devices:
    d_spec = tf_device.DeviceSpec().parse_from_string(d)
    is_local = d_spec.job in (None, "localhost")

    if all_local is None:  # Determine all_local from first device.
      all_local = is_local

    if all_local:
      if not is_local:
        raise ValueError("Local device string cannot have job specified other "
                         "than 'localhost'")
    else:
      if is_local:
        raise ValueError("Remote device string must have job specified.")
      if d_spec.task is None:
        raise ValueError("Remote device string must have task specified.")
  return all_local


def _cluster_spec_to_device_list(cluster_spec, num_gpus_per_worker):
  """Returns a device list given a cluster spec."""
  cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
  devices = []
  for task_type in ("chief", "worker"):
    for task_id in range(len(cluster_spec.as_dict().get(task_type, []))):
      if num_gpus_per_worker is 0:
        devices.append("/job:%s/task:%d" % (task_type, task_id))
      else:
        devices.extend([
            "/job:%s/task:%d/device:GPU:%i" % (task_type, task_id, gpu_id)
            for gpu_id in range(num_gpus_per_worker)
        ])
  return devices


def _group_device_list(devices):
  """Groups the devices list by task_type and task_id.

  Args:
    devices: a list of device strings for remote devices.

  Returns:
    a dict of list of device strings mapping from task_type to a list of devices
    for the task_type in the asceding order of task_id.
  """
  assert not _is_device_list_local(devices)
  device_dict = {}

  for d in devices:
    d_spec = tf_device.DeviceSpec().parse_from_string(d)

    # Create an entry for the task_type.
    if d_spec.job not in device_dict:
      device_dict[d_spec.job] = []

    # Fill the device list for task_type until it covers the task_id.
    while len(device_dict[d_spec.job]) <= d_spec.task:
      device_dict[d_spec.job].append([])

    device_dict[d_spec.job][d_spec.task].append(d)

  return device_dict


def _infer_num_gpus_per_worker(devices):
  """Infers the number of GPUs on each worker.

  Currently to make multi-worker cross device ops work, we need all workers to
  have the same number of GPUs.

  Args:
    devices: a list of device strings, can be either local devices or remote
      devices.

  Returns:
    number of GPUs per worker.

  Raises:
    ValueError if workers have different number of GPUs or GPU indices are not
    consecutive and starting from 0.
  """
  if _is_device_list_local(devices):
    return len([d for d in devices if "GPU" in d.upper()])
  else:
    device_dict = _group_device_list(devices)
    num_gpus = None
    for _, devices_in_task in device_dict.items():
      for device_in_task in devices_in_task:
        if num_gpus is None:
          num_gpus = len([d for d in device_in_task if "GPU" in d.upper()])

        # Verify other workers have the same number of GPUs.
        elif (
            num_gpus != len([d for d in device_in_task if "GPU" in d.upper()])):
          raise ValueError("All workers should have the same number of GPUs.")

        for d in device_in_task:
          d_spec = tf_device.DeviceSpec().parse_from_string(d)
          if (d_spec.device_type.upper() == "GPU" and
              d_spec.device_index >= num_gpus):
            raise ValueError("Device_index on a worker should be consecutive "
                             "and start from 0.")
    return num_gpus


def all_local_devices(num_gpus=None):
  if num_gpus is None:
    num_gpus = context.num_gpus()
  return (tuple("/device:GPU:%d" % i for i in range(num_gpus)) or
          ("/device:CPU:0",))


@tf_export("distribute.MirroredStrategy")
class MirroredStrategy(distribute_lib.DistributionStrategy):
  """Mirrors vars to distribute across multiple devices and machines.

  This strategy uses one replica per device and sync replication for its
  multi-GPU version.

  The multi-worker version will be added in the fture.

  Args:
    devices: a list of device strings.
    cross_device_ops: optional, a descedant of `CrossDeviceOps`. If this is not
      set, nccl will be use by default.
  """

  def __init__(self, devices=None, cross_device_ops=None):
    extended = MirroredExtended(
        self, devices=devices, cross_device_ops=cross_device_ops)
    super(MirroredStrategy, self).__init__(extended)


class MirroredExtended(distribute_lib.DistributionStrategyExtended):
  """Implementation of MirroredStrategy."""

  def __init__(self, container_strategy, devices=None, cross_device_ops=None):
    super(MirroredExtended, self).__init__(container_strategy)
    if devices is None:
      devices = all_local_devices()
    if not devices:
      raise ValueError("Got an empty `devices` list. Please make sure the "
                       "`devices` you pass in is not empty.")
    self._cross_device_ops = cross_device_ops
    self._initialize_strategy(devices)

  def _initialize_strategy(self, devices):
    # The _initialize_strategy method is intended to be used by distribute
    # coordinator as well.
    if _is_device_list_local(devices):
      self._initialize_local(devices)
    else:
      self._initialize_multi_worker(devices)

  def _initialize_local(self, devices):
    """Initializes the object for local training."""
    self._local_mode = True
    assert devices, "Must specify at least one device."
    assert len(set(devices)) == len(devices), (
        "No duplicates allowed in `devices` argument.")
    # TODO(josh11b): Require at least 2 devices?
    self._devices = tuple(device_util.resolve(d) for d in devices)
    self._canonical_device_set = set(self._devices)
    self._device_index = values.PerReplica(
        {d: i for i, d in enumerate(devices)})

    self._inferred_cross_device_ops = cross_device_ops_lib.choose_the_best(
        devices)

  def _initialize_multi_worker(self, devices):
    """Initializes the object for multi-worker training."""
    self._local_mode = False

    assert devices, "Must specify at least one device."
    assert len(set(devices)) == len(devices), (
        "No duplicates allowed in `devices` argument.")
    # TODO(josh11b): Require at least 2 devices?
    self._devices = tuple(device_util.resolve(d) for d in devices)
    self._canonical_device_set = set(self._devices)
    self._device_index = values.PerReplica(
        {d: i for i, d in enumerate(devices)})

    device_dict = _group_device_list(devices)
    self._workers = []
    self._worker_devices = []
    for job in ["chief", "worker"]:
      for task in range(len(device_dict.get(job, []))):
        worker = "/job:%s/task:%d" % (job, task)
        self._workers.append(worker)
        self._worker_devices.append((worker, device_dict[job][task]))

    # Setting `_default_device` will add a device scope in the
    # distribution.scope. We set the default device to the first worker. When
    # users specify device under distribution.scope by
    #   with tf.device("/cpu:0"):
    #     ...
    # their ops will end up on the cpu device of its first worker, e.g.
    # "/job:worker/task:0/device:CPU:0". Note this is not used in replica mode.
    self._default_device = self._workers[0]

    self._inferred_cross_device_ops = cross_device_ops_lib.MultiWorkerAllReduce(
        self._workers, _infer_num_gpus_per_worker(self._devices))

  def _create_variable(self, next_creator, *args, **kwargs):
    """Create a mirrored variable. See `DistributionStrategy.scope`."""
    colocate_with = kwargs.pop("colocate_with", None)
    devices = self._get_devices_from(colocate_with)

    def _real_mirrored_creator(devices, *args, **kwargs):  # pylint: disable=g-missing-docstring
      index = {}
      for i, d in enumerate(devices):
        with ops.init_scope(), ops.device(d):
          if i > 0:
            # Give replicas meaningful distinct names:
            var0name = index[devices[0]].name.split(":")[0]
            # We append a / to variable names created on replicas with id > 0 to
            # ensure that we ignore the name scope and instead use the given
            # name as the absolute name of the variable.
            kwargs["name"] = "%s/replica_%d/" % (var0name, i)
            # Initialize replicas with the same value:
            def initial_value_fn(device=d):
              if context.executing_eagerly():
                init_value = index[devices[0]].value()
                return array_ops.identity(init_value)
              else:
                with ops.device(device):
                  init_value = index[devices[0]].initial_value
                  return array_ops.identity(init_value)
            kwargs["initial_value"] = initial_value_fn
          with context.context().device_policy(context.DEVICE_PLACEMENT_SILENT):
            # Don't record operations (e.g. other variable reads) during
            # variable creation.
            with tape.stop_recording():
              v = next_creator(*args, **kwargs)
          assert not isinstance(v, values.DistributedVariable)
          index[d] = v
      return index

    return _create_mirrored_variable(devices, _real_mirrored_creator, *args,
                                     **kwargs)

  def _distribute_dataset(self, dataset_fn):
    if self._local_mode:
      return values.PerReplicaDataset(
          self._call_dataset_fn(dataset_fn), self._devices)
    else:
      return values.MultiWorkerDataset(
          functools.partial(self._call_dataset_fn, dataset_fn),
          self._worker_devices,
          auto_shard=False)

  def _make_dataset_iterator(self, dataset):
    if self._local_mode:
      worker = device_util.canonicalize("/device:CPU:0")
      worker_device_pairs = [(worker, self._devices)]
    else:
      worker_device_pairs = self._worker_devices

    return values.DatasetIterator(dataset, worker_device_pairs,
                                  self._num_replicas_in_sync)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    input_contexts = []
    if self._local_mode:
      num_workers = 1
      worker = device_util.canonicalize("/device:CPU:0")
      worker_device_pairs = [(worker, self._devices)]
    else:
      num_workers = len(self._worker_devices)
      worker_device_pairs = self._worker_devices

    for i in range(num_workers):
      input_contexts.append(distribute_lib.InputContext(
          num_input_pipelines=num_workers,
          input_pipeline_id=i,
          num_replicas_in_sync=self._num_replicas_in_sync))
    return values.InputFunctionIterator(
        input_fn, worker_device_pairs, input_contexts)

  # TODO(priyag): Deal with OutOfRange errors once b/111349762 is fixed.
  def _experimental_run_steps_on_iterator(self, fn, iterator, iterations,
                                          initial_loop_values=None):
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)

    ctx = values.MultiStepContext()
    def body(i, *args):
      """A wrapper around `fn` to create the while loop body."""
      del args
      fn_inputs = iterator.get_next()
      if not isinstance(fn_inputs, tuple):
        fn_inputs = (fn_inputs,)
      fn_result = fn(ctx, fn_inputs)
      for (name, output) in ctx.last_step_outputs.items():
        # Convert all outputs to tensors, potentially from `DistributedValues`.
        ctx.last_step_outputs[name] = self._unwrap(output)
      flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
      with ops.control_dependencies([fn_result]):
        return [i + 1] + flat_last_step_outputs

    # We capture the control_flow_context at this point, before we run `fn`
    # inside a while_loop. This is useful in cases where we might need to exit
    # these contexts and get back to the outer context to do some things, for
    # e.g. create an op which should be evaluated only once at the end of the
    # loop on the host. One such usage is in creating metrics' value op.
    self._outer_control_flow_context = (
        ops.get_default_graph()._get_control_flow_context())  # pylint: disable=protected-access

    cond = lambda i, *args: i < iterations
    i = constant_op.constant(0)
    loop_result = control_flow_ops.while_loop(
        cond, body, [i] + initial_loop_values, name="",
        parallel_iterations=1, back_prop=False, swap_memory=False,
        return_same_structure=True)
    del self._outer_control_flow_context

    ctx.run_op = control_flow_ops.group(loop_result)

    # Convert the last_step_outputs from a list to the original dict structure
    # of last_step_outputs.
    last_step_tensor_outputs = loop_result[1:]
    last_step_tensor_outputs_dict = nest.pack_sequence_as(
        ctx.last_step_outputs, last_step_tensor_outputs)

    for name, reduce_op in ctx._last_step_outputs_reduce_ops.items():  # pylint: disable=protected-access
      output = last_step_tensor_outputs_dict[name]
      # For outputs that have already been reduced, wrap them in a Mirrored
      # container, else in a PerReplica container.
      if reduce_op is None:
        last_step_tensor_outputs_dict[name] = values.regroup(
            {d: t for d, t in zip(self._devices, output)}, values.PerReplica)
      else:
        assert len(output) == 1
        last_step_tensor_outputs_dict[name] = output[0]

    ctx._set_last_step_outputs(last_step_tensor_outputs_dict)  # pylint: disable=protected-access
    return ctx

  def _broadcast_to(self, tensor, destinations):
    # This is both a fast path for Python constants, and a way to delay
    # converting Python values to a tensor until we know what type it
    # should be converted to. Otherwise we have trouble with:
    #   global_step.assign_add(1)
    # since the `1` gets broadcast as an int32 but global_step is int64.
    if isinstance(tensor, (float, int)):
      return tensor
    # TODO(josh11b): In eager mode, use one thread per device, or async mode.
    return self._get_cross_device_ops().broadcast(
        tensor, destinations or self._devices)

  def _call_for_each_replica(self, fn, args, kwargs):
    return _call_for_each_replica(self._container_strategy(), fn, args, kwargs)

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    del task_type, task_id

    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

    if cluster_spec:
      # TODO(yuefengz): remove the following code once cluster_resolver is
      # added.
      num_gpus_per_worker = _infer_num_gpus_per_worker(self._devices)
      multi_worker_devices = _cluster_spec_to_device_list(
          cluster_spec, num_gpus_per_worker)
      self._initialize_multi_worker(multi_worker_devices)

  def _update_config_proto(self, config_proto):
    updated_config = copy.deepcopy(config_proto)
    updated_config.isolate_session_state = True
    return updated_config

  def _get_cross_device_ops(self):
    return self._cross_device_ops or self._inferred_cross_device_ops

  def _reduce_to(self, reduce_op, value, destinations):
    if (isinstance(value, values.Mirrored) and
        reduce_op == reduce_util.ReduceOp.MEAN):
      return value
    assert not isinstance(value, values.Mirrored)
    if not isinstance(value, values.DistributedValues):
      # This function handles reducing values that are not PerReplica or
      # Mirrored values. For example, the same value could be present on all
      # replicas in which case `value` would be a single value or value could
      # be 0.
      return cross_device_ops_lib.reduce_non_distributed_value(
          self, reduce_op, value, destinations)
    return self._get_cross_device_ops().reduce(
        reduce_op, value, destinations=destinations)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs):
    return self._get_cross_device_ops().batch_reduce(reduce_op,
                                                     value_destination_pairs)

  def _update(self, var, fn, args, kwargs, group):
    # TODO(josh11b): In eager mode, use one thread per device.
    assert isinstance(var, values.DistributedVariable)
    updates = {}
    for d, v in var._index.items():  # pylint: disable=protected-access
      name = "update_%d" % self._device_index.get(d)
      with ops.device(d), distribute_lib.UpdateContext(d), ops.name_scope(name):
        # If args and kwargs are not mirrored, the value is returned as is.
        updates[d] = fn(v,
                        *values.select_device_mirrored(d, args),
                        **values.select_device_mirrored(d, kwargs))
    return values.update_regroup(self, updates, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    assert isinstance(colocate_with, tuple)
    # TODO(josh11b): In eager mode, use one thread per device.
    updates = {}
    for d in colocate_with:
      name = "update_%d" % self._device_index.get(d)
      with ops.device(d), distribute_lib.UpdateContext(d), ops.name_scope(name):
        updates[d] = fn(*values.select_device_mirrored(d, args),
                        **values.select_device_mirrored(d, kwargs))
    return values.update_regroup(self, updates, group)

  def read_var(self, replica_local_var):
    """Read the aggregate value of a replica-local variable."""
    if isinstance(replica_local_var, values.ReplicaLocalVariable):
      return replica_local_var._get_cross_replica()  # pylint: disable=protected-access
    assert isinstance(replica_local_var, values.Mirrored)
    return array_ops.identity(replica_local_var.get())

  def _unwrap(self, val):
    if isinstance(val, values.DistributedValues):
      # Return in a deterministic order.
      if set(val.devices) == self._canonical_device_set:
        return tuple(val.get(device=d) for d in self._devices)
      return tuple(val.get(device=d) for d in sorted(val.devices))
    return (val,)

  def value_container(self, val):
    return values.value_container(val)

  @property
  def _num_replicas_in_sync(self):
    return len(self._devices)

  @property
  def worker_devices(self):
    return self._devices

  @property
  def parameter_devices(self):
    return self._devices

  @property
  def experimental_between_graph(self):
    return False

  @property
  def experimental_should_init(self):
    return True

  @property
  def should_checkpoint(self):
    return True

  @property
  def should_save_summary(self):
    return True

  def non_slot_devices(self, var_list):
    del var_list
    return tuple(self._devices)

  def _get_devices_from(self, colocate_with=None):
    if colocate_with is None:
      return self._devices
    else:
      return cross_device_ops_lib.get_devices_from(colocate_with)

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    return True

  class _MirroredReplicaThread(threading.Thread):
    """A thread that runs() a function on a device."""

    def __init__(self, dist, coord, device, variable_creator_fn, fn, *args,
                 **kwargs):
      super(MirroredExtended._MirroredReplicaThread, self).__init__()  # pylint: disable=protected-access
      self.coord = coord
      self.distribution = dist
      self.device = device
      self.replica_id = dist.extended.worker_devices.index(device)
      self.variable_creator_fn = variable_creator_fn
      # State needed to run and return the results of `fn`.
      self.main_fn = fn
      self.main_args = args
      self.main_kwargs = kwargs
      self.main_result = None
      self.done = False
      # State needed to run the next merge_call() (if any) requested via
      # ReplicaContext.
      self.merge_fn = None
      self.merge_args = None
      self.merge_kwargs = None
      self.merge_result = None
      self.captured_name_scope = None
      # We use a thread.Event for the main thread to signal when this
      # thread should start running (`should_run`), and another for
      # this thread to transfer control back to the main thread
      # (`has_paused`, either when it gets to a
      # `get_replica_context().merge_call` or when `fn` returns). In
      # either case the event starts cleared, is signaled by calling
      # set(). The receiving thread waits for the signal by calling
      # wait() and then immediately clearing the event using clear().
      self.should_run = threading.Event()
      self.has_paused = threading.Event()
      # These fields have to do with inheriting various contexts from the
      # parent thread:
      ctx = context.context()
      self.in_eager = ctx.executing_eagerly()
      # pylint: disable=protected-access
      if not ctx._context_handle:
        ctx._initialize_handle_and_devices()
      self.context_device_policy = (
          pywrap_tensorflow.TFE_ContextGetDevicePlacementPolicy(
              ctx._context_handle))
      self.graph = ops.get_default_graph()
      with ops.init_scope():
        self._init_in_eager = context.executing_eagerly()
        self._init_graph = ops.get_default_graph()

      self._variable_creator_stack = self.graph._variable_creator_stack[:]
      self._captured_var_scope = variable_scope.get_variable_scope()
      # Adding a "/" at end lets us re-enter this scope later.
      self._name_scope = self.graph.get_name_scope()
      if self._name_scope:
        self._name_scope += "/"
      if self.replica_id > 0:
        if not self._name_scope:
          self._name_scope = ""
        self._name_scope += "replica_%d/" % self.replica_id

    def run(self):
      # pylint: disable=protected-access
      self.should_run.wait()
      self.should_run.clear()
      try:
        if self.coord.should_stop():
          return
        with self.coord.stop_on_exception(), \
            _enter_graph(self._init_graph, self._init_in_eager), \
            _enter_graph(self.graph, self.in_eager,
                         self._variable_creator_stack), \
            context.context().device_policy(self.context_device_policy), \
            MirroredReplicaContext(self.distribution, constant_op.constant(
                self.replica_id, dtypes.int32)), \
            ops.device(self.device), \
            ops.name_scope(self._name_scope), \
            variable_scope.variable_scope(
                self._captured_var_scope, reuse=self.replica_id > 0), \
            variable_scope.variable_creator_scope(self.variable_creator_fn):
          self.main_result = self.main_fn(*self.main_args, **self.main_kwargs)
          self.done = True
      finally:
        self.has_paused.set()


class MirroredReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext used in MirroredStrategy.call_for_each_replica().

  Opened in `_MirroredReplicaThread`, to allow the user to invoke
  `MirroredStrategy`'s specific implementation of `merge_call()`,
  which works by delegating the function and its arguments to
  the main thread (the one that invoked
  `MirroredStrategy.call_for_each_replica()`).
  """

  def _merge_call(self, fn, args, kwargs):
    """Delegate to the main thread to actually perform merge_call()."""
    t = threading.current_thread()  # a _MirroredReplicaThread
    t.merge_fn = fn
    t.merge_args = args
    t.merge_kwargs = kwargs
    t.captured_name_scope = t.graph.get_name_scope()
    # Adding a "/" at end lets us re-enter this scope later.
    if t.captured_name_scope:
      t.captured_name_scope += "/"

    t.captured_control_deps = t.graph._current_control_dependencies()  # pylint: disable=protected-access
    t.has_paused.set()
    t.should_run.wait()
    t.should_run.clear()
    if t.coord.should_stop():
      raise _RequestedStop()
    return t.merge_result

  @property
  def devices(self):
    distribute_lib.require_replica_context(self)
    replica_id = tensor_util.constant_value(self._replica_id_in_sync_group)
    return [self._distribution_strategy.extended.worker_devices[replica_id]]
