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

import copy

from tensorflow.python import tf2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# TODO(josh11b): Replace asserts in this file with if ...: raise ...


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
    raise ValueError("Local device should have only 'localhost' in the job "
                     "field in device string. "
                     "E.g. 'job:localhost' in "
                     "/job:localhost/replica:0/task:0/device:CPU:0"
                     "Devices cannot have mixed list of device strings "
                     "containing both localhost and other job types such as "
                     "worker, ps etc. ")

  if num_workers == 1 and not all_local:
    if any(d.task is None for d in specs):
      raise ValueError("Remote device string must have task specified."
                       "E.g. 'task:0' in "
                       "/job:worker/replica:0/task:0/device:CPU:0")

  return num_workers == 1


def _cluster_spec_to_device_list(cluster_spec, num_gpus_per_worker):
  """Returns a device list given a cluster spec."""
  cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
  devices = []
  for task_type in ("chief", "worker"):
    for task_id in range(len(cluster_spec.as_dict().get(task_type, []))):
      if num_gpus_per_worker == 0:
        devices.append("/job:%s/task:%d/device:CPU:0" % (task_type, task_id))
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
    for the task_type in the ascending order of task_id.
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
  `tf.distribute.TPUStrategy`. To use `MirroredStrategy` with multiple workers,
  please refer to `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

  For example, a variable created under a `MirroredStrategy` is a
  `MirroredVariable`. If no devices are specified in the constructor argument of
  the strategy then it will use all the available GPUs. If no GPUs are found, it
  will use the available CPUs. Note that TensorFlow treats all CPUs on a
  machine as a single device, and uses threads internally for parallelism.

  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> with strategy.scope():
  ...   x = tf.Variable(1.)
  >>> x
  MirroredVariable:{
    0: <tf.Variable ... shape=() dtype=float32, numpy=1.0>,
    1: <tf.Variable ... shape=() dtype=float32, numpy=1.0>
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
  ...   return x[0]
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> with strategy.scope():
  ...   _ = create_variable()
  ...   print(x[0])
  MirroredVariable:{
    0: <tf.Variable ... shape=() dtype=float32, numpy=1.0>,
    1: <tf.Variable ... shape=() dtype=float32, numpy=1.0>
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
        per_replica_result = my_strategy.run(replica_fn, args=(x,))
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

  # Only set this in tests.
  _collective_key_base = 0

  def __init__(self, devices=None, cross_device_ops=None):
    extended = MirroredExtended(
        self, devices=devices, cross_device_ops=cross_device_ops)
    super(MirroredStrategy, self).__init__(extended)
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "MirroredStrategy")


@tf_export(v1=["distribute.MirroredStrategy"])
class MirroredStrategyV1(distribute_lib.StrategyV1):  # pylint: disable=g-missing-docstring

  __doc__ = MirroredStrategy.__doc__

  # Only set this in tests.
  _collective_key_base = 0

  def __init__(self, devices=None, cross_device_ops=None):
    extended = MirroredExtended(
        self, devices=devices, cross_device_ops=cross_device_ops)
    super(MirroredStrategyV1, self).__init__(extended)
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set(
        "MirroredStrategy")


# TODO(josh11b): Switch to V2 when we no longer need to support tf.compat.v1.
class MirroredExtended(distribute_lib.StrategyExtendedV1):
  """Implementation of MirroredStrategy."""

  # If this is set to True, use NCCL collective ops instead of NCCL cross device
  # ops.
  _prefer_collective_ops = False

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
    self._collective_ops_in_use = False
    self._collective_key_base = container_strategy._collective_key_base
    self._communication_options = collective_util.Options(
        implementation=collective_util.CommunicationImplementation.NCCL)
    self._initialize_strategy(devices)

    # TODO(b/128995245): Enable last partial batch support in graph mode.
    if ops.executing_eagerly_outside_functions():
      self.experimental_enable_get_next_as_optional = True

    # Flag to turn on VariablePolicy.
    self._use_var_policy = False

  def _use_merge_call(self):
    # We currently only disable merge_call when XLA is used to compile the `fn`
    # passed to `strategy.run` and all devices are GPU.
    return not control_flow_util.GraphOrParentsInXlaContext(
        ops.get_default_graph()) or not all(
            [_is_gpu_device(d) for d in self._devices])

  def _initialize_strategy(self, devices):
    # The _initialize_strategy method is intended to be used by distribute
    # coordinator as well.
    assert devices, "Must specify at least one device."
    devices = tuple(device_util.resolve(d) for d in devices)
    assert len(set(devices)) == len(devices), (
        "No duplicates allowed in `devices` argument: %s" % (devices,))
    if _is_device_list_single_worker(devices):
      self._initialize_single_worker(devices)
      self._collective_ops = self._make_collective_ops_with_fallbacks()
      if self._prefer_collective_ops and (
          isinstance(self._cross_device_ops, cross_device_ops_lib.NcclAllReduce)
          or isinstance(self._inferred_cross_device_ops,
                        cross_device_ops_lib.NcclAllReduce)):
        self._collective_ops_in_use = True
        self._inferred_cross_device_ops = None
      logging.info("Using MirroredStrategy with devices %r", devices)
    else:
      self._initialize_multi_worker(devices)

  def _make_collective_ops_with_fallbacks(self):
    self._collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=1 + self._collective_key_base)

    # Use ReductionToOneDevice() if mixed devices are used.
    if any("cpu" in d.lower() for d in self._devices) and any(
        "gpu" in d.lower() for d in self._devices):
      return cross_device_ops_lib.ReductionToOneDevice()

    if all("cpu" in d.lower() for d in self._devices):
      # Use RING collective ops if all devices are CPU.
      self._communication_options = collective_util.Options(
          implementation=collective_util.CommunicationImplementation.RING)

    else:
      physical_gpus = context.context().list_physical_devices(device_type="GPU")
      logical_gpus = context.context().list_logical_devices(device_type="GPU")
      # Use RING collective ops if virtual devices are used.
      if len(physical_gpus) != len(logical_gpus):
        self._communication_options = collective_util.Options(
            implementation=collective_util.CommunicationImplementation.RING)

    # If all devices are physical GPU, use NCCL implementation.
    return cross_device_ops_lib.CollectiveAllReduce(
        devices=self._devices,
        group_size=len(self._devices),
        options=self._communication_options,
        collective_keys=self._collective_keys)

  def _initialize_single_worker(self, devices):
    """Initializes the object for single-worker training."""
    self._devices = tuple(device_util.canonicalize(d) for d in devices)
    self._input_workers_devices = (
        (device_util.canonicalize("/device:CPU:0", devices[0]), devices),)

    self._inferred_cross_device_ops = None if self._cross_device_ops else (
        cross_device_ops_lib.select_cross_device_ops(devices))
    self._host_input_device = numpy_dataset.SingleDevice(
        self._input_workers_devices[0][0])
    self._is_multi_worker_training = False
    device_spec = tf_device.DeviceSpec.from_string(
        self._input_workers_devices[0][0])
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

    self._devices = tuple(devices)
    self._input_workers_devices = worker_devices
    self._is_multi_worker_training = True

    if len(workers) > 1:
      # Grandfather usage in the legacy tests if they're configured properly.
      if (not isinstance(self._cross_device_ops,
                         cross_device_ops_lib.ReductionToOneDevice) or
          self._cross_device_ops._num_between_graph_workers > 1):  # pylint: disable=protected-access
        raise ValueError(
            "In-graph multi-worker training with `MirroredStrategy` is not "
            "supported.")
      self._inferred_cross_device_ops = self._cross_device_ops
    else:
      # TODO(yuefengz): make `select_cross_device_ops` work with device strings
      # containing job names.
      self._inferred_cross_device_ops = cross_device_ops_lib.NcclAllReduce()

    logging.info("Using MirroredStrategy with remote devices %r", devices)

  def _input_workers_with_options(self, options=None):
    if not options:
      return input_lib.InputWorkers(self._input_workers_devices)
    if (options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      if options.experimental_place_dataset_on_device:
        self._input_workers_devices = (
            tuple(
                (device_util.canonicalize(d, d), (d,)) for d in self._devices))
      else:
        self._input_workers_devices = (
            tuple((device_util.canonicalize("/device:CPU:0", d), (d,))
                  for d in self._devices))
      return input_lib.InputWorkers(self._input_workers_devices)
    else:
      if not options.experimental_fetch_to_device:
        return input_lib.InputWorkers([
            (host_device, (host_device,) * len(compute_devices))
            for host_device, compute_devices in self._input_workers_devices
        ])
      else:
        return input_lib.InputWorkers(self._input_workers_devices)

  @property
  def _input_workers(self):
    return self._input_workers_with_options()

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

  def _create_variable(self, next_creator, **kwargs):
    """Create a mirrored variable. See `DistributionStrategy.scope`."""
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      devices = self._devices
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(**kwargs)
    else:
      devices = colocate_with._devices  # pylint: disable=protected-access

    def _real_mirrored_creator(**kwargs):  # pylint: disable=g-missing-docstring
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
              v = next_creator(**kwargs)
          assert not isinstance(v, values.DistributedVariable)
          value_list.append(v)
      return value_list

    return distribute_utils.create_mirrored_variable(
        self._container_strategy(), _real_mirrored_creator,
        distribute_utils.VARIABLE_CLASS_MAPPING,
        distribute_utils.VARIABLE_POLICY_MAPPING, **kwargs)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    distribute_utils.validate_colocate_distributed_variable(
        colocate_with_variable, self)

  def _make_dataset_iterator(self, dataset):
    return input_lib_v1.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync)

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
    return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers,
                                              input_contexts,
                                              self._container_strategy())

  def _experimental_distribute_dataset(self, dataset, options):
    if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`distribute_datasets_from_function`."
      )
    return input_util.get_distributed_dataset(
        dataset,
        self._input_workers_with_options(options),
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync,
        options=options)

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, self._host_input_device, session)

  def _distribute_datasets_from_function(self, dataset_fn, options):
    input_workers = self._input_workers_with_options(options)
    input_contexts = []
    num_workers = input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(distribute_lib.InputContext(
          num_input_pipelines=num_workers,
          input_pipeline_id=i,
          num_replicas_in_sync=self._num_replicas_in_sync))

    return input_util.get_distributed_datasets_from_function(
        dataset_fn, input_workers, input_contexts, self._container_strategy(),
        options)

  def _experimental_distribute_values_from_function(self, value_fn):
    per_replica_values = []
    for replica_id in range(self._num_replicas_in_sync):
      per_replica_values.append(value_fn(
          distribute_lib.ValueContext(replica_id,
                                      self._num_replicas_in_sync)))
    return distribute_utils.regroup(per_replica_values, always_wrap=True)

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
        last_step_tensor_outputs_dict[name] = distribute_utils.regroup(output)
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
      destinations = self._devices
    return self._get_cross_device_ops(tensor).broadcast(tensor, destinations)

  def _call_for_each_replica(self, fn, args, kwargs):
    return mirrored_run.call_for_each_replica(
        self._container_strategy(), fn, args, kwargs)

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

  def _get_cross_device_ops(self, value):
    if not self._use_merge_call():
      return self._collective_ops

    if self._collective_ops_in_use:
      if isinstance(value, values.DistributedValues):
        value_int32 = True in {
            dtypes.as_dtype(v.dtype) == dtypes.int32 for v in value.values
        }
      else:
        value_int32 = dtypes.as_dtype(value.dtype) == dtypes.int32
      if value_int32:
        return cross_device_ops_lib.ReductionToOneDevice()
      else:
        return self._collective_ops

    return self._cross_device_ops or self._inferred_cross_device_ops

  def _gather_to_implementation(self, value, destinations, axis, options):
    if not isinstance(value, values.DistributedValues):
      # ReductionToOneDevice._gather accepts DistributedValues only.
      return value
    return self._get_cross_device_ops(value)._gather(  # pylint: disable=protected-access
        value,
        destinations=destinations,
        axis=axis,
        options=self._communication_options.merge(options))

  def _reduce_to(self, reduce_op, value, destinations, options):
    if (distribute_utils.is_mirrored(value) and
        reduce_op == reduce_util.ReduceOp.MEAN):
      return value
    assert not distribute_utils.is_mirrored(value)
    def get_values(value):
      if not isinstance(value, values.DistributedValues):
        # This function handles reducing values that are not PerReplica or
        # Mirrored values. For example, the same value could be present on all
        # replicas in which case `value` would be a single value or value could
        # be 0.
        return cross_device_ops_lib.reduce_non_distributed_value(
            reduce_op, value, destinations, self._num_replicas_in_sync)
      if self._use_merge_call() and self._collective_ops_in_use and ((
          not cross_device_ops_lib._devices_match(value, destinations) or  # pylint: disable=protected-access
          any("cpu" in d.lower()
              for d in cross_device_ops_lib.get_devices_from(destinations)))):
        return cross_device_ops_lib.ReductionToOneDevice().reduce(
            reduce_op, value, destinations)
      return self._get_cross_device_ops(value).reduce(
          reduce_op,
          value,
          destinations=destinations,
          options=self._communication_options.merge(options))

    return nest.map_structure(get_values, value)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs, options):
    cross_device_ops = None
    for value, _ in value_destination_pairs:
      if cross_device_ops is None:
        cross_device_ops = self._get_cross_device_ops(value)
      elif cross_device_ops is not self._get_cross_device_ops(value):
        raise ValueError("Inputs to batch_reduce_to must be either all on "
                         "the host or all on the compute devices.")
    return cross_device_ops.batch_reduce(
        reduce_op,
        value_destination_pairs,
        options=self._communication_options.merge(options))

  def _update(self, var, fn, args, kwargs, group):
    # TODO(josh11b): In eager mode, use one thread per device.
    assert isinstance(var, values.DistributedVariable)
    updates = []
    for i, v in enumerate(var.values):
      name = "update_%d" % i
      with ops.device(v.device), \
           distribute_lib.UpdateContext(i), \
           ops.name_scope(name):
        # If args and kwargs are not mirrored, the value is returned as is.
        updates.append(
            fn(v, *distribute_utils.select_replica(i, args),
               **distribute_utils.select_replica(i, kwargs)))
    return distribute_utils.update_regroup(self, updates, group)

  def _replica_ctx_all_reduce(self, reduce_op, value, options=None):
    """Implements `StrategyExtendedV2._replica_ctx_all_reduce`."""
    # This implementation avoids using `merge_call` and just launches collective
    # ops in one replica.
    if options is None:
      options = collective_util.Options()

    if context.executing_eagerly() or (
        not tf2.enabled()) or self._use_merge_call():
      # In eager mode, falls back to the default implementation that uses
      # `merge_call`. Replica functions are running sequentially in eager mode,
      # and due to the blocking nature of collective ops, execution will hang if
      # collective ops are to be launched sequentially.
      return super()._replica_ctx_all_reduce(reduce_op, value, options)

    replica_context = distribution_strategy_context.get_replica_context()
    assert replica_context, (
        "`StrategyExtended._replica_ctx_all_reduce` must be called in a "
        "replica context")
    return self._get_cross_device_ops(value)._all_reduce(  # pylint: disable=protected-access
        reduce_op,
        value,
        replica_context._replica_id,  # pylint: disable=protected-access
        options)

  def _replica_ctx_update(self, var, fn, args, kwargs, group):
    if self._use_merge_call():
      return super()._replica_ctx_update(var, fn, args, kwargs, group)

    replica_context = distribution_strategy_context.get_replica_context()
    assert replica_context
    replica_id = values_util.get_current_replica_id_as_int()
    name = "update_%d" % replica_id

    if isinstance(var, values.DistributedVariable):
      var = var._get_replica(replica_id)  # pylint: disable=protected-access

    with ops.device(var.device), ops.name_scope(name):
      result = fn(var, *args, **kwargs)
    return result

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    assert isinstance(colocate_with, tuple)
    # TODO(josh11b): In eager mode, use one thread per device.
    updates = []
    for i, d in enumerate(colocate_with):
      name = "update_%d" % i
      with ops.device(d), distribute_lib.UpdateContext(i), ops.name_scope(name):
        updates.append(
            fn(*distribute_utils.select_replica(i, args),
               **distribute_utils.select_replica(i, kwargs)))
    return distribute_utils.update_regroup(self, updates, group)

  def read_var(self, replica_local_var):
    """Read the aggregate value of a replica-local variable."""
    # pylint: disable=protected-access
    if distribute_utils.is_sync_on_read(replica_local_var):
      return replica_local_var._get_cross_replica()
    assert distribute_utils.is_mirrored(replica_local_var)
    return array_ops.identity(replica_local_var._get())
    # pylint: enable=protected-access

  def value_container(self, val):
    return distribute_utils.value_container(val)

  @property
  def _num_replicas_in_sync(self):
    return len(self._devices)

  @property
  def worker_devices(self):
    return self._devices

  @property
  def worker_devices_by_replica(self):
    return [[d] for d in self._devices]

  @property
  def parameter_devices(self):
    return self.worker_devices

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
    return self._devices

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

  def _get_local_replica_id(self, replica_id_in_sync_group):
    return replica_id_in_sync_group

  def _get_replica_id_in_sync_group(self, replica_id):
    return replica_id
