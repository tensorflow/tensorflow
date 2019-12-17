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
"""Class MirroredStrategy implementing tf.distribute.Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import functools
import threading
import weakref

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import tape
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
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
  cpu_device = cpu_device.replace(device_type="CPU", device_index=0)
  return cpu_device.to_string()


class _RequestedStop(Exception):  # pylint: disable=g-bad-exception-name
  pass


# _call_for_each_replica is not a member of MirroredStrategy so that it is
# not allowed to use anything specific to MirroredStrategy and thus
# can be shared with other distribution strategies.


# TODO(yuefengz): maybe create a common class for those who need to call this
# _call_for_each_replica.
def _call_for_each_replica(distribution, device_map, fn, args, kwargs):
  """Run `fn` in separate threads, once per replica/worker device.

  Args:
    distribution: the DistributionStrategy object.
    device_map: the DeviceMap with the devices to run `fn` on.
    fn: function to run (will be run once per replica, each in its own thread).
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

  # TODO(isaprykin): Create these threads once instead of during every call.
  threads = []
  for index in range(device_map.num_replicas_in_graph):
    variable_creator_fn = shared_variable_creator.make_fn(
        shared_variable_store, index)
    t = _MirroredReplicaThread(
        distribution, coord, index, device_map, variable_creator_fn, fn,
        values.select_replica(index, args),
        values.select_replica(index, kwargs))
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
          merge_args = values.regroup(
              device_map, tuple(t.merge_args for t in threads))
          merge_kwargs = values.regroup(
              device_map, tuple(t.merge_kwargs for t in threads))
          # We capture the name_scope of the MRT when we call merge_fn
          # to ensure that if we have opened a name scope in the MRT,
          # it will be respected when executing the merge function. We only
          # capture the name_scope from the first MRT and assume it is
          # the same for all other MRTs.
          mtt_captured_name_scope = threads[0].captured_name_scope
          mtt_captured_var_scope = threads[0].captured_var_scope
          # Capture and merge the control dependencies from all the threads.
          mtt_captured_control_deps = set()
          for t in threads:
            mtt_captured_control_deps.update(t.captured_control_deps)
          with ops.name_scope(mtt_captured_name_scope),\
              ops.control_dependencies(mtt_captured_control_deps), \
              variable_scope.variable_scope(mtt_captured_var_scope):
            merge_result = threads[0].merge_fn(distribution, *merge_args,
                                               **merge_kwargs)
          for r, t in enumerate(threads):
            t.merge_result = values.select_replica(r, merge_result)
  finally:
    for t in threads:
      t.should_run.set()
    coord.join(threads)

  return values.regroup(device_map, tuple(t.main_result for t in threads))


def _is_device_list_single_worker(devices):
  """Checks whether the devices list is for single or multi-worker.

  Args:
    devices: a list of device strings or tf.config.LogicalDevice objects, for
      either local or for remote devices.

  Returns:
    a boolean indicating whether these device strings are for local or for
    remote.

  Raises:
    ValueError: if device strings are not consistent.
  """
  specs = []
  for d in devices:
    name = d.name if isinstance(d, context.LogicalDevice) else d
    specs.append(tf_device.DeviceSpec.from_string(name))
  num_workers = len({(d.job, d.task, d.replica) for d in specs})
  all_local = all(d.job in (None, "localhost") for d in specs)
  any_local = any(d.job in (None, "localhost") for d in specs)

  if any_local and not all_local:
    raise ValueError("Local device string cannot have job specified other "
                     "than 'localhost'")

  if num_workers == 1 and not all_local:
    if any(d.task is None for d in specs):
      raise ValueError("Remote device string must have task specified.")

  return num_workers == 1


def _cluster_spec_to_device_list(cluster_spec, num_gpus_per_worker):
  """Returns a device list given a cluster spec."""
  cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
  devices = []
  for task_type in ("chief", "worker"):
    for task_id in range(len(cluster_spec.as_dict().get(task_type, []))):
      if num_gpus_per_worker == 0:
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
  assert not _is_device_list_single_worker(devices)
  device_dict = {}

  for d in devices:
    d_spec = tf_device.DeviceSpec.from_string(d)

    # Create an entry for the task_type.
    if d_spec.job not in device_dict:
      device_dict[d_spec.job] = []

    # Fill the device list for task_type until it covers the task_id.
    while len(device_dict[d_spec.job]) <= d_spec.task:
      device_dict[d_spec.job].append([])

    device_dict[d_spec.job][d_spec.task].append(d)

  return device_dict


def _is_gpu_device(device):
  return tf_device.DeviceSpec.from_string(device).device_type == "GPU"


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
  if _is_device_list_single_worker(devices):
    return sum(1 for d in devices if _is_gpu_device(d))
  else:
    device_dict = _group_device_list(devices)
    num_gpus = None
    for _, devices_in_task in device_dict.items():
      for device_in_task in devices_in_task:
        if num_gpus is None:
          num_gpus = sum(1 for d in device_in_task if _is_gpu_device(d))

        # Verify other workers have the same number of GPUs.
        elif num_gpus != sum(1 for d in device_in_task if _is_gpu_device(d)):
          raise ValueError("All workers should have the same number of GPUs.")

        for d in device_in_task:
          d_spec = tf_device.DeviceSpec.from_string(d)
          if (d_spec.device_type == "GPU" and
              d_spec.device_index >= num_gpus):
            raise ValueError("GPU `device_index` on a worker should be "
                             "consecutive and start from 0.")
    return num_gpus


def all_local_devices(num_gpus=None):
  devices = config.list_logical_devices("GPU")
  if num_gpus is not None:
    devices = devices[:num_gpus]
  return devices or config.list_logical_devices("CPU")


def all_devices():
  devices = []
  tfconfig = TFConfigClusterResolver()
  if tfconfig.cluster_spec().as_dict():
    devices = _cluster_spec_to_device_list(tfconfig.cluster_spec(),
                                           context.num_gpus())
  return devices if devices else all_local_devices()


@tf_export("distribute.MirroredStrategy", v1=[])  # pylint: disable=g-classes-have-attributes
class MirroredStrategy(distribute_lib.Strategy):
  """Synchronous training across multiple replicas on one machine.

  This strategy is typically used for training on one
  machine with multiple GPUs. For TPUs, use
  `tf.distribute.experimental.TPUStrategy`. To use `MirroredStrategy` with
  multiple workers, please refer to
  `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

  For example, a variable created under a `MirroredStrategy` is a
  `MirroredVariable`. If no devices are specified in the constructor argument of
  the strategy then it will use all the available GPUs. If no GPUs are found, it
  will use the available CPUs. Note that TensorFlow treats all CPUs on a
  machine as a single device, and uses threads internally for parallelism.

  >>> strategy = tf.distribute.MirroredStrategy()
  >>> with strategy.scope():
  ...   x = tf.Variable(1.)
  >>> x
  MirroredVariable:{
      0 /job:localhost/replica:0/task:0/device:CPU:0: <tf.Variable ...
      shape=() dtype=float32, numpy=1.0>
    }

  While using distribution strategies, all the variable creation should be done
  within the strategy's scope. This will replicate the variables across all the
  replicas and keep them in sync using an all-reduce algorithm.

  Variables created inside a `MirroredStrategy` which is wrapped with a
  `tf.function` are still `MirroredVariables`.

  >>> x = []
  >>> @tf.function  # Wrap the function with tf.function.
  ... def create_variable():
  ...   if not x:
  ...     x.append(tf.Variable(1.))
  >>> strategy = tf.distribute.MirroredStrategy()
  >>> with strategy.scope():
  ...   create_variable()
  ...   print (x[0])
  MirroredVariable:{
      0 /job:localhost/replica:0/task:0/device:CPU:0: <tf.Variable ...
      shape=() dtype=float32, numpy=1.0>
    }

  `experimental_distribute_dataset` can be used to distribute the dataset across
  the replicas when writing your own training loop. If you are using `.fit` and
  `.compile` methods available in `tf.keras`, then `tf.keras` will handle the
  distribution for you.

  For example:

  ```python
  my_strategy = tf.distribute.MirroredStrategy()
  with my_strategy.scope():
    @tf.function
    def distribute_train_epoch(dataset):
      def replica_fn(input):
        # process input and return result
        return result

      total_result = 0
      for x in dataset:
        per_replica_result = my_strategy.experimental_run_v2(replica_fn,
                                                             args=(x,))
        total_result += my_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_result, axis=None)
      return total_result

    dist_dataset = my_strategy.experimental_distribute_dataset(dataset)
    for _ in range(EPOCHS):
      train_result = distribute_train_epoch(dist_dataset)
  ```

  Args:
    devices: a list of device strings such as `['/gpu:0', '/gpu:1']`.  If
      `None`, all available GPUs are used. If no GPUs are found, CPU is used.
    cross_device_ops: optional, a descedant of `CrossDeviceOps`. If this is not
      set, `NcclAllReduce()` will be used by default.  One would customize this
      if NCCL isn't available or if a special implementation that exploits
      the particular hardware is available.
  """

  def __init__(self, devices=None, cross_device_ops=None):
    extended = MirroredExtended(
        self, devices=devices, cross_device_ops=cross_device_ops)
    super(MirroredStrategy, self).__init__(extended)
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "MirroredStrategy")


@tf_export(v1=["distribute.MirroredStrategy"])
class MirroredStrategyV1(distribute_lib.StrategyV1):  # pylint: disable=g-missing-docstring

  __doc__ = MirroredStrategy.__doc__

  def __init__(self, devices=None, cross_device_ops=None):
    extended = MirroredExtended(
        self, devices=devices, cross_device_ops=cross_device_ops)
    super(MirroredStrategyV1, self).__init__(extended)
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set(
        "MirroredStrategy")


# TODO(josh11b): Switch to V2 when we no longer need to support tf.compat.v1.
class MirroredExtended(distribute_lib.StrategyExtendedV1):
  """Implementation of MirroredStrategy."""

  def __init__(self, container_strategy, devices=None, cross_device_ops=None):
    super(MirroredExtended, self).__init__(container_strategy)
    if context.executing_eagerly():
      if devices and not _is_device_list_single_worker(devices):
        raise RuntimeError("In-graph multi-worker training with "
                           "`MirroredStrategy` is not supported in eager mode.")
      else:
        if TFConfigClusterResolver().cluster_spec().as_dict():
          # if you are executing in eager mode, only the single machine code
          # path is supported.
          logging.info("Initializing local devices since in-graph multi-worker "
                       "training with `MirroredStrategy` is not supported in "
                       "eager mode. TF_CONFIG will be ignored when "
                       "when initializing `MirroredStrategy`.")
        devices = devices or all_local_devices()
    else:
      devices = devices or all_devices()

    assert devices, ("Got an empty `devices` list and unable to recognize "
                     "any local devices.")
    self._cross_device_ops = cross_device_ops
    self._initialize_strategy(devices)
    self._cfer_fn_cache = weakref.WeakKeyDictionary()

    # TODO(b/128995245): Enable last partial batch support in graph mode.
    if ops.executing_eagerly_outside_functions():
      self.experimental_enable_get_next_as_optional = True

  def _initialize_strategy(self, devices):
    # The _initialize_strategy method is intended to be used by distribute
    # coordinator as well.
    assert devices, "Must specify at least one device."
    devices = tuple(device_util.resolve(d) for d in devices)
    assert len(set(devices)) == len(devices), (
        "No duplicates allowed in `devices` argument: %s" % (devices,))
    if _is_device_list_single_worker(devices):
      self._initialize_single_worker(devices)
    else:
      self._initialize_multi_worker(devices)

  def _initialize_single_worker(self, devices):
    """Initializes the object for single-worker training."""
    self._device_map = values.ReplicaDeviceMap(devices)
    self._input_workers = input_lib.InputWorkers(self._device_map)
    self._inferred_cross_device_ops = None if self._cross_device_ops else (
        cross_device_ops_lib.choose_the_best(devices))
    self._host_input_device = numpy_dataset.SingleDevice(
        self._input_workers.worker_devices[0])
    self._is_multi_worker_training = False
    logging.info("Using MirroredStrategy with devices %r", devices)
    device_spec = tf_device.DeviceSpec.from_string(
        self._input_workers.worker_devices[0])
    # Ensures when we enter strategy.scope() we use the correct default device
    if device_spec.job is not None and device_spec.job != "localhost":
      self._default_device = "/job:%s/replica:%d/task:%d" % (
          device_spec.job, device_spec.replica, device_spec.task)

  def _initialize_multi_worker(self, devices):
    """Initializes the object for multi-worker training."""
    device_dict = _group_device_list(devices)
    workers = []
    worker_devices = []
    for job in ("chief", "worker"):
      for task in range(len(device_dict.get(job, []))):
        worker = "/job:%s/task:%d" % (job, task)
        workers.append(worker)
        worker_devices.append((worker, device_dict[job][task]))

    # Setting `_default_device` will add a device scope in the
    # distribution.scope. We set the default device to the first worker. When
    # users specify device under distribution.scope by
    #   with tf.device("/cpu:0"):
    #     ...
    # their ops will end up on the cpu device of its first worker, e.g.
    # "/job:worker/task:0/device:CPU:0". Note this is not used in replica mode.
    self._default_device = workers[0]
    self._host_input_device = numpy_dataset.SingleDevice(workers[0])

    self._device_map = values.ReplicaDeviceMap(devices)
    self._input_workers = input_lib.InputWorkers(
        self._device_map, worker_devices)
    self._is_multi_worker_training = True

    if len(workers) > 1:
      if not isinstance(self._cross_device_ops,
                        cross_device_ops_lib.MultiWorkerAllReduce):
        raise ValueError(
            "In-graph multi-worker training with `MirroredStrategy` is not "
            "supported.")
      self._inferred_cross_device_ops = self._cross_device_ops
    else:
      # TODO(yuefengz): make `choose_the_best` work with device strings
      # containing job names.
      self._inferred_cross_device_ops = cross_device_ops_lib.NcclAllReduce()

    logging.info("Using MirroredStrategy with remote devices %r", devices)

  def _get_variable_creator_initial_value(self,
                                          replica_id,
                                          device,
                                          primary_var,
                                          **kwargs):
    """Return the initial value for variables on a replica."""
    if replica_id == 0:
      return kwargs["initial_value"]
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

  def _create_variable(self, next_creator, *args, **kwargs):
    """Create a mirrored variable. See `DistributionStrategy.scope`."""
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

  def _validate_colocate_with_variable(self, colocate_with_variable):
    values.validate_colocate_distributed_variable(colocate_with_variable, self)

  def _make_dataset_iterator(self, dataset):
    return input_lib.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    input_contexts = []
    num_workers = self._input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(distribute_lib.InputContext(
          num_input_pipelines=num_workers,
          input_pipeline_id=i,
          num_replicas_in_sync=self._num_replicas_in_sync))
    return input_lib.InputFunctionIterator(input_fn, self._input_workers,
                                           input_contexts,
                                           self._container_strategy())

  def _experimental_distribute_dataset(self, dataset):
    return input_lib.get_distributed_dataset(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync)

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, self._host_input_device, session)

  def _experimental_distribute_datasets_from_function(self, dataset_fn):
    input_contexts = []
    num_workers = self._input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(distribute_lib.InputContext(
          num_input_pipelines=num_workers,
          input_pipeline_id=i,
          num_replicas_in_sync=self._num_replicas_in_sync))

    return input_lib.get_distributed_datasets_from_function(
        dataset_fn,
        self._input_workers,
        input_contexts,
        self._container_strategy())

  # TODO(priyag): Deal with OutOfRange errors once b/111349762 is fixed.
  def _experimental_run_steps_on_iterator(self, fn, iterator, iterations,
                                          initial_loop_values=None):
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)

    ctx = input_lib.MultiStepContext()
    def body(i, *args):
      """A wrapper around `fn` to create the while loop body."""
      del args
      fn_result = fn(ctx, iterator.get_next())
      for (name, output) in ctx.last_step_outputs.items():
        # Convert all outputs to tensors, potentially from `DistributedValues`.
        ctx.last_step_outputs[name] = self._local_results(output)
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
        last_step_tensor_outputs_dict[name] = values.regroup(self._device_map,
                                                             output)
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
    if not destinations:
      # TODO(josh11b): Use current logical device instead of 0 here.
      destinations = values.LogicalDeviceSpec(
          device_map=self._device_map, logical_device=0)
    return self._get_cross_device_ops().broadcast(tensor, destinations)

  def _call_for_each_replica(self, fn, args, kwargs):
    if isinstance(fn, def_function.Function):
      wrapped = self._cfer_fn_cache.get(fn)
      if wrapped is None:
        # We need to wrap fn such that it triggers _call_for_each_replica inside
        # the tf.function.
        wrapped = fn._clone(  # pylint: disable=protected-access
            python_function=functools.partial(self._call_for_each_replica,
                                              fn.python_function))
        self._cfer_fn_cache[fn] = wrapped
      return wrapped(args, kwargs)

    if context.executing_eagerly():
      logging.log_first_n(logging.WARN, "Using %s eagerly has significant "
                          "overhead currently. We will be working on improving "
                          "this in the future, but for now please wrap "
                          "`call_for_each_replica` or `experimental_run` or "
                          "`experimental_run_v2` inside a tf.function to get "
                          "the best performance." %
                          self._container_strategy().__class__.__name__, 5)
    else:
      # When a tf.function is wrapped to trigger _call_for_each_replica (see
      # the other branch above), AutoGraph stops conversion at
      # _call_for_each_replica itself (TF library functions are whitelisted).
      # This makes suresure that the Python function that originally passed to
      # the tf.function is still converted.
      fn = autograph.tf_convert(fn, ag_ctx.control_status_ctx())

    return _call_for_each_replica(self._container_strategy(), self._device_map,
                                  fn, args, kwargs)

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
      num_gpus_per_worker = _infer_num_gpus_per_worker(
          self._device_map.all_devices)
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
          reduce_op, self._device_map, value, destinations)
    return self._get_cross_device_ops().reduce(
        reduce_op, value, destinations=destinations)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs):
    return self._get_cross_device_ops().batch_reduce(reduce_op,
                                                     value_destination_pairs)

  def _update(self, var, fn, args, kwargs, group):
    # TODO(josh11b): In eager mode, use one thread per device.
    assert isinstance(var, values.DistributedVariable)
    updates = []
    for i, (d, v) in enumerate(zip(var.devices, var.values)):
      name = "update_%d" % i
      with ops.device(d), distribute_lib.UpdateContext(i), ops.name_scope(name):
        # If args and kwargs are not mirrored, the value is returned as is.
        updates.append(fn(v,
                          *values.select_device_mirrored(d, args),
                          **values.select_device_mirrored(d, kwargs)))
    return values.update_regroup(self, self._device_map, updates, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    assert isinstance(colocate_with, tuple)
    # TODO(josh11b): In eager mode, use one thread per device.
    updates = []
    for i, d in enumerate(colocate_with):
      name = "update_%d" % i
      with ops.device(d), distribute_lib.UpdateContext(i), ops.name_scope(name):
        updates.append(fn(*values.select_device_mirrored(d, args),
                          **values.select_device_mirrored(d, kwargs)))
    return values.update_regroup(self, self._device_map, updates, group)

  def read_var(self, replica_local_var):
    """Read the aggregate value of a replica-local variable."""
    if isinstance(replica_local_var, values.SyncOnReadVariable):
      return replica_local_var._get_cross_replica()  # pylint: disable=protected-access
    assert isinstance(replica_local_var, values.Mirrored)
    return array_ops.identity(replica_local_var.get())

  def _local_results(self, val):
    if isinstance(val, values.DistributedValues):
      return val.values
    return (val,)

  def value_container(self, val):
    return values.value_container(val)

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
    return self._device_map.all_devices

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
    # TODO(josh11b): Should this be the last logical device instead?
    return self._device_map.logical_to_actual_devices(0)

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True

  def _in_multi_worker_mode(self):
    """Whether this strategy indicates working in multi-worker settings."""
    return False


class _MirroredReplicaThread(threading.Thread):
  """A thread that runs() a function on a device."""

  def __init__(self, dist, coord, replica_id, device_map, variable_creator_fn,
               fn, args, kwargs):
    super(_MirroredReplicaThread, self).__init__()
    self.coord = coord
    self.distribution = dist
    self.device_map = device_map
    self.replica_id = replica_id
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
    self.captured_var_scope = None
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
    context.ensure_initialized()
    ctx = context.context()
    self.in_eager = ctx.executing_eagerly()
    self.record_thread_local_summary_state()
    self.record_thread_local_eager_context_state()
    self.context_device_policy = (
        pywrap_tensorflow.TFE_ContextGetDevicePlacementPolicy(
            ctx._context_handle))  # pylint: disable=protected-access
    self.graph = ops.get_default_graph()
    with ops.init_scope():
      self._init_in_eager = context.executing_eagerly()
      self._init_graph = ops.get_default_graph()
    self._variable_creator_stack = self.graph._variable_creator_stack[:]   # pylint: disable=protected-access
    self._var_scope = variable_scope.get_variable_scope()
    # Adding a "/" at end lets us re-enter this scope later.
    self._name_scope = self.graph.get_name_scope()
    if self._name_scope:
      self._name_scope += "/"
    if self.replica_id > 0:
      if not self._name_scope:
        self._name_scope = ""
      self._name_scope += "replica_%d/" % self.replica_id

  def run(self):
    self.should_run.wait()
    self.should_run.clear()
    try:
      if self.coord.should_stop():
        return
      self.restore_thread_local_summary_state()
      self.restore_thread_local_eager_context_state()
      # TODO(josh11b): Use current logical device instead of 0 here.
      with self.coord.stop_on_exception(), \
          _enter_graph(self._init_graph, self._init_in_eager), \
          _enter_graph(self.graph, self.in_eager,
                       self._variable_creator_stack), \
          context.device_policy(self.context_device_policy), \
          MirroredReplicaContext(self.distribution, constant_op.constant(
              self.replica_id, dtypes.int32)), \
          ops.device(self.device_map.logical_to_actual_devices(0)[
              self.replica_id]), \
          ops.name_scope(self._name_scope), \
          variable_scope.variable_scope(
              self._var_scope, reuse=self.replica_id > 0), \
          variable_scope.variable_creator_scope(self.variable_creator_fn):
        self.main_result = self.main_fn(*self.main_args, **self.main_kwargs)
        self.done = True
    finally:
      self.has_paused.set()

  def record_thread_local_summary_state(self):
    """Record the thread local summary state in self."""
    # TODO(slebedev): is this still relevant? the referenced bug is closed.
    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    self._summary_step = summary_state.step
    self._summary_writer = summary_state.writer
    self._summary_recording = summary_state.is_recording
    self._summary_recording_distribution_strategy = (
        summary_state.is_recording_distribution_strategy)

  def restore_thread_local_summary_state(self):
    """Restore thread local summary state from self."""
    # TODO(slebedev): is this still relevant? the referenced bug is closed.
    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    summary_state.step = self._summary_step
    summary_state.writer = self._summary_writer
    summary_state.is_recording = self._summary_recording
    summary_state.is_recording_distribution_strategy = (
        self._summary_recording_distribution_strategy)

  def record_thread_local_eager_context_state(self):
    ctx = context.context()
    eager_context_state = ctx._thread_local_data  # pylint: disable=protected-access
    self._eager_context_op_callbacks = eager_context_state.op_callbacks
    # TODO(b/125892694): record other fields in EagerContext.

  def restore_thread_local_eager_context_state(self):
    ctx = context.context()
    eager_context_state = ctx._thread_local_data  # pylint: disable=protected-access
    eager_context_state.op_callbacks = self._eager_context_op_callbacks
    # TODO(b/125892694): record other fields in EagerContext.


class MirroredReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext used in MirroredStrategy.extended.call_for_each_replica().

  Opened in `_MirroredReplicaThread`, to allow the user to invoke
  `MirroredStrategy`'s specific implementation of `merge_call()`,
  which works by delegating the function and its arguments to
  the main thread (the one that invoked
  `MirroredStrategy.extended.call_for_each_replica()`).
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

    t.captured_var_scope = variable_scope.get_variable_scope()
    t.captured_control_deps = t.graph._current_control_dependencies()  # pylint: disable=protected-access

    # NOTE(priyag): Throw an error if there is a merge call in the middle of a
    # `fn` passed to call_for_each_replica which changes the graph being used
    # while calling `fn`. This can happen when the `fn` is decorated with
    # `tf.function` and there is a merge_call in `fn`. This breaks because each
    # thread tries to create a distinct tf.function. Each tf.function creation
    # takes a lock, and so if there is a merge call in the middle, the lock is
    # never released and subsequent replica threads cannot proceed to define
    # their own functions. Checking for the graph being the same is one way for
    # us to check this didn't happen.
    if ops.get_default_graph() != t.graph:
      raise RuntimeError(
          "`merge_call` called while defining a new graph or a tf.function. "
          "This can often happen if the function `fn` passed to "
          "`strategy.experimental_run()` is decorated with "
          "`@tf.function` (or contains a nested `@tf.function`), and `fn` "
          "contains a synchronization point, such as aggregating gradients. "
          "This behavior is not yet supported. Instead, please wrap the entire "
          "call `strategy.experimental_run(fn)` in a `@tf.function`, and avoid "
          "nested `tf.function`s that may potentially cross a synchronization "
          "boundary.")

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
    return [self._strategy.extended.worker_devices_by_replica[replica_id]]
