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
"""TPU Distribution Strategy.

This is experimental.  It's not ready for general use.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.contrib.tpu.python.tpu import training_loop
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import device_util
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.util import nest


_TPU_INITIALIZE_SYSTEM_COLLECTION = "TPU_STRATEGY_INITIALIZE"


def get_tpu_system_metadata(tpu_cluster_resolver):
  """Retrieves TPU system metadata given a TPUClusterResolver."""
  master = tpu_cluster_resolver.master()

  # pylint: disable=protected-access
  cluster_spec = tpu_cluster_resolver.cluster_spec()
  cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
  tpu_system_metadata = (
      tpu_system_metadata_lib._query_tpu_system_metadata(
          master,
          cluster_def=cluster_def,
          query_topology=False))

  return tpu_system_metadata


# TODO(jhseu): Deduplicate with MirroredStrategy?
def _create_tpu_mirrored_variable(devices, real_mirrored_creator, *args,
                                  **kwargs):  # pylint: disable=g-missing-docstring
  # Figure out what collections this variable should be added to.
  # We'll add the TPUMirroredVariable to those collections instead.
  collections = kwargs.pop("collections", None)
  if collections is None:
    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  kwargs["collections"] = []

  # TODO(jhseu): Should we have different behavior for different
  # synchronization settings?

  # Get aggregation value
  # TODO(jhseu): Support aggregation in a replica context.
  aggregation = kwargs.pop("aggregation", vs.VariableAggregation.NONE)
  if aggregation not in [
      vs.VariableAggregation.NONE,
      vs.VariableAggregation.SUM,
      vs.VariableAggregation.MEAN,
      vs.VariableAggregation.ONLY_FIRST_REPLICA,
  ]:
    raise ValueError("Invalid variable aggregation mode: {} for variable: {}"
                     .format(aggregation, kwargs["name"]))

  # Ignore user-specified caching device, not needed for mirrored variables.
  kwargs.pop("caching_device", None)

  # TODO(josh11b,apassos): It would be better if variable initialization
  # was never recorded on the tape instead of having to do this manually
  # here.
  with tape.stop_recording():
    index = real_mirrored_creator(devices, *args, **kwargs)
    result = values.TPUMirroredVariable(index, index[devices[0]], aggregation)

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
        l.remove(v)
    g.add_to_collections(collections, result)
  return result


class TPUStrategy(distribute_lib.DistributionStrategy):
  """TPU distribution strategy implementation."""

  def __init__(self, tpu_cluster_resolver, steps_per_run, num_cores=None):
    """Initializes the TPUStrategy object.

    Args:
      tpu_cluster_resolver: A tf.contrib.cluster_resolver.TPUClusterResolver,
          which provides information about the TPU cluster.
      steps_per_run: Number of steps to run on device before returning to the
          host. Note that this can have side-effects on performance, hooks,
          metrics, summaries etc.
          This parameter is only used when Distribution Strategy is used with
          estimator or keras.
      num_cores: Number of cores to use on the TPU. If None specified, then
          auto-detect the cores and topology of the TPU system.
    """
    super(TPUStrategy, self).__init__(TPUExtended(
        self, tpu_cluster_resolver, steps_per_run, num_cores))

  @property
  def steps_per_run(self):
    """DEPRECATED: use .extended.steps_per_run instead."""
    return self._extended.steps_per_run


class TPUExtended(distribute_lib.DistributionStrategyExtended):
  """Implementation of TPUStrategy."""

  def __init__(self, container_strategy, tpu_cluster_resolver, steps_per_run,
               num_cores=None):
    super(TPUExtended, self).__init__(container_strategy)
    self._tpu_cluster_resolver = tpu_cluster_resolver
    self._tpu_metadata = get_tpu_system_metadata(self._tpu_cluster_resolver)
    # TODO(sourabhbajaj): Change this from num_cores to metadata_override
    self._num_cores_override = num_cores

    # TODO(jhseu): Switch to DeviceAssignment to support pods and model
    # parallelism.
    device_map = {d.name: i for i, d in enumerate(self._tpu_metadata.devices)
                  if "device:TPU:" in d.name}
    self._device_index = values.PerReplica(device_map)
    self._host_device = self.get_host_cpu_device(0)
    self._tpu_devices = sorted(device_map.keys())
    # Only create variables for the number of replicas we're running.
    self._tpu_devices = self._tpu_devices[:self._num_replicas_in_sync]

    # TODO(sourabhbajaj): Remove this once performance of running one step
    # at a time is comparable to multiple steps.
    self.steps_per_run = steps_per_run

    self._require_static_shapes = True

  def _get_enqueue_op_per_host(self, host_id, multi_worker_iterator,
                               input_shapes, iterations):
    """Create an enqueue op for a single host identified using host_id.

    The while_loop op returned will run `iterations` times and in each run
    enqueue batches for each shard.

    Args:
      host_id: integer, id of the host to run the enqueue ops on.
      multi_worker_iterator: MultiWorkerDataIterator to read the input data.
      input_shapes: shape of inputs to be enqueue on the queue. This is same as
        the value of `nest.flatten(iterator.output_shapes)`.
      iterations: integer, number of iterations to be run; determines the
        number of batches to be enqueued.

    Returns:
      while_loop_op running `iterations` times; in each run we enqueue a batch
      on the infeed queue from the host with id `host_id` for each device shard.
    """
    host = self.get_host_cpu_device(host_id)
    # TODO(sourabhbajaj): Possibly make changes to MultiWorkerDataset
    # to work with TPU Prefetch so clean up this code.
    iterator = (
        multi_worker_iterator.get_iterator(self.get_host(host_id))._iterator)  # pylint: disable=protected-access

    def _infeed_enqueue_ops_fn():
      """Enqueue ops for one iteration."""
      control_deps = []
      sharded_inputs = []
      enqueue_ops = []

      with ops.device(host):
        for _ in range(self.num_replicas_per_host):
          # Use control dependencies to ensure a deterministic ordering.
          with ops.control_dependencies(control_deps):
            inputs = nest.flatten(iterator.get_next())
            control_deps.extend(inputs)
            sharded_inputs.append(inputs)

      for core_id, shard_input in enumerate(sharded_inputs):
        enqueue_ops.append(
            tpu_ops.infeed_enqueue_tuple(
                inputs=shard_input,
                shapes=input_shapes,
                device_ordinal=core_id))
      return enqueue_ops

    def enqueue_ops_loop_body(i):
      """Callable for the loop body of the while_loop instantiated below."""
      with ops.control_dependencies(_infeed_enqueue_ops_fn()):
        return i + 1

    with ops.device(host):
      enqueue_op_per_host = control_flow_ops.while_loop(
          lambda i: i < iterations,
          enqueue_ops_loop_body,
          [constant_op.constant(0)],
          parallel_iterations=1)

    return enqueue_op_per_host

  def _make_dataset_iterator(self, dataset):
    """Make iterators for each of the TPU hosts."""

    worker_devices = [
        (self.get_host(hid), [self.get_host_cpu_device(hid)])
        for hid in range(self.num_hosts)
    ]
    return values.DatasetIterator(dataset, worker_devices,
                                  self._num_replicas_in_sync)

  def _distribute_dataset(self, dataset_fn):
    worker_devices = [
        (self.get_host(hid), [self.get_host_cpu_device(hid)])
        for hid in range(self.num_hosts)
    ]
    return values.MultiWorkerDataset(
        functools.partial(self._call_dataset_fn, dataset_fn), worker_devices)

  # TODO(priyag): Deal with OutOfRange errors once b/111349762 is fixed.
  # TODO(sourabhbajaj): Remove the initial_loop_values parameter when we have
  # a mechanism to infer the outputs of `fn`. Pending b/110550782.
  def _experimental_run_steps_on_iterator(
      self, fn, multi_worker_iterator, iterations, initial_loop_values=None):
    output_shapes = multi_worker_iterator.output_shapes
    shapes = nest.flatten(output_shapes)
    if any([not s.is_fully_defined() for s in shapes]):
      raise ValueError(
          "TPU currently requires fully defined shapes. Either use "
          "set_shape() on the input tensors or use "
          "dataset.batch(..., drop_remainder=True).")
    types = nest.flatten(multi_worker_iterator.output_types)

    enqueue_ops = [
        self._get_enqueue_op_per_host(host_id, multi_worker_iterator, shapes,
                                      iterations)
        for host_id in range(self.num_hosts)]

    def dequeue_fn():
      dequeued = tpu_ops.infeed_dequeue_tuple(dtypes=types, shapes=shapes)
      return nest.pack_sequence_as(output_shapes, dequeued)

    # Wrap `fn` for repeat.
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)
    ctx = values.MultiStepContext()
    def run_fn(*args, **kwargs):
      """Single step on the TPU device."""
      del args, kwargs
      fn_inputs = dequeue_fn()
      if not isinstance(fn_inputs, tuple):
        fn_inputs = (fn_inputs,)
      fn_result = fn(ctx, fn_inputs)
      flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
      if flat_last_step_outputs:
        with ops.control_dependencies([fn_result]):
          return [array_ops.identity(f) for f in flat_last_step_outputs]
      else:
        return fn_result

    # TODO(sourabhbajaj): The input to while loop should be based on the output
    # type of the step_fn
    def iterate_on_tpu():
      return training_loop.repeat(iterations, run_fn, initial_loop_values)

    # We capture the control_flow_context at this point, before we run `fn`
    # inside a while_loop and TPU replicate context. This is useful in cases
    # where we might need to exit these contexts and get back to the outer
    # context to do some things, for e.g. create an op which should be
    # evaluated only once at the end of the loop on the host. One such usage
    # is in creating metrics' value op.
    self._outer_control_flow_context = (
        ops.get_default_graph()._get_control_flow_context())  # pylint: disable=protected-access

    replicate_inputs = [[]] * self._num_replicas_in_sync
    replicate_outputs = tpu.replicate(iterate_on_tpu, replicate_inputs)
    del self._outer_control_flow_context
    ctx.run_op = control_flow_ops.group(replicate_outputs, enqueue_ops)

    # Filter out any ops from the outputs, typically this would be the case
    # when there were no tensor outputs.
    last_step_tensor_outputs = [x for x in replicate_outputs
                                if not isinstance(x, ops.Operation)]

    # Outputs are currently of the structure (grouped by device)
    # [[output0_device0, output1_device0, output2_device0],
    #  [output0_device1, output1_device1, output2_device1]]
    # Convert this to the following structure instead: (grouped by output)
    # [[output0_device0, output0_device1],
    #  [output1_device0, output1_device1],
    #  [output2_device0, output2_device1]]
    last_step_tensor_outputs = [list(x) for x in zip(*last_step_tensor_outputs)]

    # Convert replicate_outputs to the original dict structure of
    # last_step_outputs.
    last_step_tensor_outputs_dict = nest.pack_sequence_as(
        ctx.last_step_outputs, last_step_tensor_outputs)

    for name, reduce_op in ctx._last_step_outputs_reduce_ops.items():  # pylint: disable=protected-access
      output = last_step_tensor_outputs_dict[name]
      # For outputs that have already been reduced, take the first value
      # from the list as each value should be the same. Else return the full
      # list of values.
      # TODO(josh11b): If reduce_op is NONE, we should return a PerReplica
      # value.
      if reduce_op is not None:
        # TODO(priyag): Should this return the element or a list with 1 element
        last_step_tensor_outputs_dict[name] = output[0]
    ctx._set_last_step_outputs(last_step_tensor_outputs_dict)  # pylint: disable=protected-access

    return ctx

  def _call_for_each_replica(self, fn, args, kwargs):
    # TODO(jhseu): Consider making it so call_for_each_replica implies that
    # we're in a tpu.rewrite(), and update TPUMirroredVariable accordingly.
    with _TPUReplicaContext(self._container_strategy()):
      return fn(*args, **kwargs)

  def _initialize(self):
    if context.executing_eagerly():
      # TODO(priyag): Add appopriate call here when eager is supported for TPUs.
      raise NotImplementedError("Eager mode not supported in TPUStrategy.")
    else:
      # TODO(jhseu): We need this hack because DistributionStrategies must be
      # pickleable for copy.deepcopy(). Remove when initialize_system goes away.
      graph = ops.get_default_graph()
      tpu_init = graph.get_collection(_TPU_INITIALIZE_SYSTEM_COLLECTION)
      if tpu_init:
        return tpu_init
      graph.add_to_collection(_TPU_INITIALIZE_SYSTEM_COLLECTION,
                              tpu.initialize_system())
      return graph.get_collection(_TPU_INITIALIZE_SYSTEM_COLLECTION)

  def _finalize(self):
    if context.executing_eagerly():
      # TODO(priyag): Add appopriate call here when eager is supported for TPUs.
      raise NotImplementedError("Eager mode not supported in TPUStrategy.")
    else:
      return [tpu.shutdown_system()]

  def _get_devices_from(self, colocate_with=None):
    # TODO(jhseu): Change this when we support model parallelism.
    return self._tpu_devices

  def _create_variable(self, next_creator, *args, **kwargs):
    """Create a TPUMirroredVariable. See `DistributionStrategy.scope`."""
    colocate_with = kwargs.pop("colocate_with", None)
    devices = self._get_devices_from(colocate_with)

    def _real_mirrored_creator(devices, *args, **kwargs):  # pylint: disable=g-missing-docstring
      index = {}
      for i, d in enumerate(devices):
        with ops.device(d):
          if i > 0:
            # Give replicas meaningful distinct names:
            var0name = index[devices[0]].name.split(":")[0]
            # We append a / to variable names created on replicas with id > 0 to
            # ensure that we ignore the name scope and instead use the given
            # name as the absolute name of the variable.
            kwargs["name"] = "%s/replica_%d/" % (var0name, i)
            # Initialize replicas with the same value:
            if context.executing_eagerly():
              kwargs["initial_value"] = array_ops.identity(
                  index[devices[0]].value())
            else:
              def initial_value_fn(device=d):
                with ops.device(device):
                  return array_ops.identity(index[devices[0]].initial_value)
              kwargs["initial_value"] = initial_value_fn
          with context.context().device_policy(context.DEVICE_PLACEMENT_SILENT):
            v = next_creator(*args, **kwargs)
          assert not isinstance(v, values.TPUMirroredVariable)
          index[d] = v
      return index

    return _create_tpu_mirrored_variable(devices, _real_mirrored_creator, *args,
                                         **kwargs)

  def _reduce_to(self, reduce_op, value, destinations):
    if values._enclosing_tpu_context() is not None:  # pylint: disable=protected-access
      if reduce_op == reduce_util.ReduceOp.MEAN:
        # TODO(jhseu):  Revisit once we support model-parallelism.
        value *= (1. / self._num_replicas_in_sync)
      elif reduce_op != reduce_util.ReduceOp.SUM:
        raise NotImplementedError(
            "Currently only support sum & mean in TPUStrategy.")
      return tpu_ops.cross_replica_sum(value)

    # Validate that the destination is same as the host device
    # Note we don't do this when in replicate context as the reduction is
    # performed on the TPU device itself.
    devices = cross_device_ops_lib.get_devices_from(destinations)
    if len(devices) == 1:
      assert device_util.canonicalize(devices[0]) == device_util.canonicalize(
          self._host_device)
    else:
      raise ValueError("Multiple devices are not supported for TPUStrategy")

    output = math_ops.add_n(value)
    if reduce_op == reduce_util.ReduceOp.MEAN:
      return output * (1. / len(value))
    return output

  def _update(self, var, fn, args, kwargs, group):
    assert isinstance(var, values.TPUMirroredVariable)
    if values._enclosing_tpu_context() is not None:  # pylint: disable=protected-access
      if group:
        return fn(var, *args, **kwargs)
      else:
        return [fn(var, *args, **kwargs)]

    # Otherwise, we revert to MirroredStrategy behavior and update each variable
    # directly.
    updates = {}
    for d, v in var._index.items():  # pylint: disable=protected-access
      name = "update_%d" % self._device_index.get(d)
      with ops.device(d), distribute_lib.UpdateContext(d), ops.name_scope(name):
        # If args and kwargs are not mirrored, the value is returned as is.
        updates[d] = fn(v,
                        *values.select_device_mirrored(d, args),
                        **values.select_device_mirrored(d, kwargs))
    return values.update_regroup(self, updates, group)

  def read_var(self, var):
    assert isinstance(var, values.TPUMirroredVariable)
    return var.read_value()

  def _unwrap(self, val):
    if isinstance(val, values.DistributedValues):
      # Return in a deterministic order.
      return [val.get(device=d) for d in sorted(val.devices)]
    elif isinstance(val, list):
      # TODO(josh11b): We need to remove this case; per device values should
      # be represented using a PerReplica wrapper instead of a list with
      # one entry per device.
      return val
    return [val]

  def value_container(self, value):
    return value

  def _broadcast_to(self, tensor, destinations):
    del destinations
    return tensor

  @property
  def num_hosts(self):
    return self._tpu_metadata.num_hosts

  @property
  def num_replicas_per_host(self):
    return self._tpu_metadata.num_of_cores_per_host

  @property
  def _num_replicas_in_sync(self):
    return self._num_cores_override or self._tpu_metadata.num_cores

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

  @property
  def worker_devices(self):
    return self._tpu_devices

  @property
  def parameter_devices(self):
    return self._tpu_devices

  def non_slot_devices(self, var_list):
    return self._host_device

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    del colocate_with
    with ops.device(self._host_device), distribute_lib.UpdateContext(
        self._host_device):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._unwrap, result)

  def get_host(self, host_id):
    if self._tpu_cluster_resolver.get_master() in ("", "local"):
      return "/replica:0/task:0"
    job_name = self._tpu_cluster_resolver.get_job_name() or "tpu_worker"
    return "/job:%s/task:%d" % (job_name, host_id)

  def get_host_cpu_device(self, host_id):
    return self.get_host(host_id) + "/device:CPU:0"

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    del cluster_spec, task_type, task_id
    if session_config:
      session_config.isolate_session_state = True
      cluster_spec = self._tpu_cluster_resolver.cluster_spec()
      if cluster_spec:
        session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())


class _TPUReplicaContext(distribute_lib.ReplicaContext):
  """Replication Context class for TPU Strategy."""

  # TODO(sourabhbajaj): Call for each tower should be updating this.
  def __init__(self, distribution_strategy):
    distribute_lib.ReplicaContext.__init__(
        self,
        distribution_strategy,
        # TODO(b/118385803): properly initialize replica_id, instead of always 0
        replica_id_in_sync_group=constant_op.constant(0, dtypes.int32))

  @property
  def device(self):
    raise RuntimeError("Use .devices instead")

  @property
  def devices(self):
    distribute_lib.require_replica_context(self)
    ds = self._distribution_strategy
    replica_id = tensor_util.constant_value(self._replica_id_in_sync_group)
    return [ds.worker_devices[replica_id]]
