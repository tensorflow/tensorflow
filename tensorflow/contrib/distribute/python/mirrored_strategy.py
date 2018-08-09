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
import threading

from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
from tensorflow.contrib.distribute.python import shared_variable_creator
from tensorflow.contrib.distribute.python import values
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import coordinator
from tensorflow.python.training import device_util
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.util import nest


# TODO(josh11b): Replace asserts in this file with if ...: raise ...


@contextlib.contextmanager
def _enter_graph(g):
  if context.executing_eagerly():
    with g.as_default(), context.eager_mode():
      yield
  else:
    with g.as_default():
      yield


def _cpu_device(device):
  cpu_device = tf_device.DeviceSpec.from_string(device)
  cpu_device.merge_from(tf_device.DeviceSpec(device_type="CPU", device_index=0))
  return cpu_device.to_string()


class _RequestedStop(Exception):
  pass


# Make _call_for_each_tower and _reduce_non_distributed_value not members of
# MirroredStrategy so that they are generally not allowed to use anything
# specific to MirroredStrategy and thus can be shared with other distribution
# strategies.


# TODO(yuefengz): maybe create a common class for those who need to call this
# _call_for_each_tower.
def _call_for_each_tower(distribution, fn, *args, **kwargs):
  """Run `fn` in separate threads, once per tower/worker device.

  Args:
    distribution: the DistributionStrategy object.
    fn: function to run (will be run once per device, each in its own thread).
    *args: positional arguments for `fn`
    **kwargs: keyword arguments for `fn`.
        `"run_concurrently"`: Boolean indicating whether executions of `fn`
           can be run concurrently (under eager execution only), defaults to
           `True`.

  Returns:
    Merged return value of `fn` across all towers.

  Raises:
    RuntimeError: If fn() calls get_tower_context().merge_call() a different
        number of times from the available devices.
  """
  run_concurrently = kwargs.pop("run_concurrently", True)
  if not context.executing_eagerly():
    # Lots of TF library code isn't thread-safe in graph mode, and
    # there is little to be gained by turning on multithreading when
    # constructing a graph.
    run_concurrently = False
    # Needed for per-thread device, etc. contexts in graph mode.
    ops.get_default_graph().switch_to_thread_local()
  elif run_concurrently is None:
    run_concurrently = True

  coord = coordinator.Coordinator(clean_stop_exception_types=(_RequestedStop,))

  shared_variable_store = {}

  # TODO(isaprykin): Create these threads once instead of during every run()
  # call.
  threads = []
  for index, d in enumerate(distribution.worker_devices):
    variable_creator_fn = shared_variable_creator.make_fn(
        shared_variable_store, index)
    t = MirroredStrategy._MirroredTowerThread(  # pylint: disable=protected-access
        distribution, coord, d, variable_creator_fn, fn,
        *values.select_device(d, args), **values.select_device(d, kwargs))
    threads.append(t)

  for t in threads:
    t.start()

  # When `fn` starts `should_run` event is set on _MirroredTowerThread
  # (`MTT`) threads. The execution waits until
  # `MTT.has_paused` is set, which indicates that either `fn` is
  # complete or a `get_tower_context().merge_call()` is called.  If `fn` is
  # complete, then `MTT.done` is set to True.  Otherwise, arguments
  # of `get_tower_context().merge_call` from all paused threads are grouped
  # and the `merge_fn` is performed.  Results of the
  # `get_tower_context().merge_call` are then set to `MTT.merge_result`.
  # Each such `get_tower_context().merge_call` call returns the
  # `MTT.merge_result` for that thread when `MTT.should_run` event
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
            raise RuntimeError("Some towers made a different number of "
                               "tower_context().merge_call() calls.")
          # get_tower_context().merge_call() case
          merge_args = values.regroup({t.device: t.merge_args for t in threads})
          merge_kwargs = values.regroup(
              {t.device: t.merge_kwargs for t in threads})
          # We capture the name_scope of the MTT when we call merge_fn
          # to ensure that if we have opened a name scope in the MTT,
          # it will be respected when executing the merge function. We only
          # capture the name_scope from the first MTT and assume it is
          # the same for all other MTTs.
          mtt_captured_name_scope = threads[0].captured_name_scope
          with ops.name_scope(mtt_captured_name_scope):
            merge_result = threads[0].merge_fn(distribution, *merge_args,
                                               **merge_kwargs)
          for t in threads:
            t.merge_result = values.select_device(t.device, merge_result)
  finally:
    for t in threads:
      t.should_run.set()
    coord.join(threads)

  return values.regroup({t.device: t.main_result for t in threads})


def _reduce_non_distributed_value(distribution, aggregation, value,
                                  destinations):
  """Reduce a non-DistributedValue `value` to `destinations`."""
  if isinstance(value, values.DistributedValues):
    raise ValueError("You are passing a `DistributedValue` to "
                     "`_reduce_non_distributed_value`, which is not allowed.")

  # If the same value is present on all towers then the PerDevice value will
  # be a single value. We also handle the case when `value` is a single value
  # and equal to 0.
  if value == 0:
    return 0
  # If the aggregation type is MEAN, then this essentially means that the same
  # value should be on all destinations.
  if aggregation == variable_scope.VariableAggregation.MEAN:
    return distribution.broadcast(value, destinations)

  cross_tower_ops_lib.validate_destinations(destinations)
  # We do not support an aggregation type of SUM if the value is the same across
  # all towers. We call this as part of assign functions for MirroredVariables
  # and summing up identical values across towers is not clearly defined.
  if (len(distribution.worker_devices) != 1 or
      not cross_tower_ops_lib.check_destinations(destinations)):
    raise ValueError("A non-DistributedValues value cannot be reduced with the "
                     "given aggregation.")
  # TODO(anjalisridhar): Moves these methods to a device utility file?
  devices = cross_tower_ops_lib.get_devices_from(destinations)
  if len(devices) == 1:
    with ops.device(devices[0]):
      return array_ops.identity(value)
  else:
    value_updates = {}
    for d in devices:
      with ops.device(d):
        value_updates[d] = array_ops.identity(value)
    return values.Mirrored(value_updates)


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
    # Variables that are to be synced on read are tower local.
    is_tower_local = True
    kwargs["trainable"] = False
  elif (synchronization == variable_scope.VariableSynchronization.ON_WRITE or
        synchronization == variable_scope.VariableSynchronization.AUTO):
    # `AUTO` synchronization for `MirroredStrategy` is `ON_WRITE`.
    is_tower_local = False
  else:
    raise ValueError("Invalid variable synchronization mode: " +
                     synchronization + " for variable: " + kwargs["name"])

  # Get aggregation value
  aggregation = kwargs.pop("aggregation",
                           variable_scope.VariableAggregation.NONE)
  if aggregation not in [
      variable_scope.VariableAggregation.NONE,
      variable_scope.VariableAggregation.SUM,
      variable_scope.VariableAggregation.MEAN
  ]:
    raise ValueError("Invalid variable aggregation mode: " + aggregation +
                     " for variable: " + kwargs["name"])

  # Ignore user-specified caching device, not needed for mirrored variables.
  kwargs.pop("caching_device", None)

  # TODO(josh11b,apassos): It would be better if variable initialization
  # was never recorded on the tape instead of having to do this manually
  # here.
  with tape.stop_recording():
    index = real_mirrored_creator(devices, *args, **kwargs)

    if is_tower_local:
      result = values.TowerLocalVariable(index, index[devices[0]], aggregation)
    else:
      result = values.MirroredVariable(index, index[devices[0]], aggregation)

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


class MirroredStrategy(distribute_lib.DistributionStrategy):
  """Mirrors vars to distribute across multiple devices on a single machine.

  This strategy uses one tower per device and sync replication.
  """

  def __init__(self,
               devices=None,
               num_gpus=None,
               cross_tower_ops=None,
               prefetch_on_device=None):
    super(MirroredStrategy, self).__init__()
    # Convert `num_gpus` into `devices`, shouldn't specify both.
    if devices is None:
      if num_gpus is None:
        num_gpus = context.num_gpus()
      devices = ["/device:GPU:%d" % d for d in range(num_gpus)]
    elif num_gpus is not None:
      raise ValueError("Must only specify one of `devices` and `num_gpus`.")

    assert devices, "Must specify at least one device."
    assert len(set(devices)) == len(devices), (
        "No duplicates allowed in `devices` argument.")
    # TODO(josh11b): Require at least 2 devices?
    self._devices = [device_util.resolve(d) for d in devices]
    self._canonical_device_set = set(self._devices)
    self._device_index = values.PerDevice(
        dict((d, i) for i, d in enumerate(devices)))
    self._cross_tower_ops = cross_tower_ops
    self._prefetch_on_device = prefetch_on_device
    # TODO(yuefengz): consider setting the default device.

  def _create_variable(self, next_creator, *args, **kwargs):
    """Create a mirrored variable. See `DistributionStrategy.scope`."""
    colocate_with = kwargs.pop("colocate_with", None)
    devices = self._get_devices_from(colocate_with)

    def _real_mirrored_creator(devices, *args, **kwargs):  # pylint: disable=g-missing-docstring
      index = {}
      for i, d in enumerate(devices):
        with ops.device(d):
          if i > 0:
            # Give replicas meaningful distinct names:
            var0name = index[devices[0]].name.split(":")[0]
            # We append a / to variable names created on towers with id > 0 to
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
          assert not isinstance(v, values.DistributedVariable)
          index[d] = v
      return index

    return _create_mirrored_variable(devices, _real_mirrored_creator, *args,
                                     **kwargs)

  def distribute_dataset(self, dataset_fn):
    return values.PerDeviceDataset(
        self._call_dataset_fn(dataset_fn), self._devices,
        self._prefetch_on_device)

  # TODO(priyag): Deal with OutOfRange errors once b/111349762 is fixed.
  def _run_steps_on_dataset(self, fn, iterator, iterations,
                            initial_loop_values=None):
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)

    ctx = values.MultiStepContext()
    def body(i, *args):
      """A wrapper around `fn` to create the while loop body."""
      del args
      fn_result = fn(ctx, iterator.get_next())
      for (name, output) in ctx.last_step_outputs.items():
        # Convert all outputs to tensors, potentially from `DistributedValues`.
        ctx.last_step_outputs[name] = self.unwrap(output)
      flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
      with ops.control_dependencies([fn_result]):
        return [i + 1] + flat_last_step_outputs

    cond = lambda i, *args: i < iterations
    i = constant_op.constant(0)
    loop_result = control_flow_ops.while_loop(
        cond, body, [i] + initial_loop_values, name="",
        parallel_iterations=1, back_prop=False, swap_memory=False,
        return_same_structure=True)

    ctx.run_op = control_flow_ops.group(loop_result)

    # Convert the last_step_outputs from a list to the original dict structure
    # of last_step_outputs.
    last_step_tensor_outputs = loop_result[1:]
    last_step_tensor_outputs_dict = nest.pack_sequence_as(
        ctx.last_step_outputs, last_step_tensor_outputs)

    for (name, aggregation) in ctx._last_step_outputs_aggregations.items():  # pylint: disable=protected-access
      output = last_step_tensor_outputs_dict[name]
      # For outputs that have already been aggregated, wrap them in a Mirrored
      # container, else in a PerDevice container.
      if aggregation is variables_lib.VariableAggregation.NONE:
        last_step_tensor_outputs_dict[name] = values.regroup(
            {d: t for d, t in zip(self._devices, output)}, values.PerDevice)
      else:
        assert len(output) == 1
        last_step_tensor_outputs_dict[name] = output[0]

    ctx._set_last_step_outputs(last_step_tensor_outputs_dict)  # pylint: disable=protected-access
    return ctx

  def _broadcast(self, tensor, destinations):
    # TODO(josh11b): In eager mode, use one thread per device, or async mode.
    return self._get_cross_tower_ops().broadcast(tensor, destinations or
                                                 self._devices)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    return _call_for_each_tower(self, fn, *args, **kwargs)

  def map(self, map_over, fn, *args, **kwargs):
    # TODO(josh11b): In eager mode, use one thread per device.
    index = {}
    for i, m in enumerate(map_over):
      d = self._devices[i % len(self._devices)]
      with ops.device(d):
        l = index.get(d, [])
        l.append(fn(m,
                    *values.select_device_mirrored(d, args),
                    **values.select_device_mirrored(d, kwargs)))
        index[d] = l
    # TODO(josh11b): Need a values.regroup equivalent that handles MapOutput
    # in addition to PerDevice data.
    return values.PerDevice({k: values.MapOutput(v) for k, v in index.items()})

  def configure(self, session_config=None):
    if self._cross_tower_ops is None:
      self._cross_tower_ops = cross_tower_ops_lib.choose_the_best(
          self._devices, session_config=session_config)

  def _get_cross_tower_ops(self):
    if self._cross_tower_ops is None:
      self._cross_tower_ops = (
          cross_tower_ops_lib.ReductionToOneDeviceCrossTowerOps())
    return self._cross_tower_ops

  def _reduce(self, aggregation, value, destinations):
    assert not isinstance(value, values.Mirrored)
    if not isinstance(value, values.DistributedValues):
      # This function handles reducing values that are not PerDevice or Mirrored
      # values. For example, the same value could be present on all towers in
      # which case `value` would be a single value or value could be 0.
      return _reduce_non_distributed_value(self, aggregation, value,
                                           destinations)
    return self._get_cross_tower_ops().reduce(
        aggregation, value, destinations=destinations)

  def _batch_reduce(self, aggregation, value_destination_pairs):
    return self._get_cross_tower_ops().batch_reduce(aggregation,
                                                    value_destination_pairs)

  def _update(self, var, fn, *args, **kwargs):
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
    return values.regroup(updates, values.Mirrored)

  def _update_non_slot(self, colocate_with, fn, *args, **kwargs):
    assert isinstance(colocate_with, list)
    # TODO(josh11b): In eager mode, use one thread per device.
    updates = {}
    for d in colocate_with:
      name = "update_%d" % self._device_index.get(d)
      with ops.device(d), distribute_lib.UpdateContext(d), ops.name_scope(name):
        updates[d] = fn(*values.select_device_mirrored(d, args),
                        **values.select_device_mirrored(d, kwargs))
    return values.regroup(updates, values.Mirrored)

  def read_var(self, tower_local_var):
    """Read the aggregate value of a tower-local variable."""
    if isinstance(tower_local_var, values.TowerLocalVariable):
      return tower_local_var._get_cross_tower()  # pylint: disable=protected-access
    assert isinstance(tower_local_var, values.Mirrored)
    return array_ops.identity(tower_local_var.get())

  def _unwrap(self, val):
    if isinstance(val, values.DistributedValues):
      # Return in a deterministic order.
      if set(val.devices) == self._canonical_device_set:
        return [val.get(device=d) for d in self._devices]
      return [val.get(device=d) for d in sorted(val.devices)]
    return [val]

  def value_container(self, val):
    return values.value_container(val)

  @property
  def is_single_tower(self):
    return len(self._devices) == 1

  @property
  def num_towers(self):
    return len(self._devices)

  def _worker_device_index(self):
    return self._device_index

  @property
  def worker_devices(self):
    # Make a copy to prevent users from accidentally mutating our copy.
    return list(self._devices)

  @property
  def parameter_devices(self):
    return list(self._devices)

  def non_slot_devices(self, var_list):
    del var_list
    return list(self._devices)

  def _get_devices_from(self, colocate_with=None):
    if colocate_with is None:
      return self._devices
    else:
      return cross_tower_ops_lib.get_devices_from(colocate_with)

  class _MirroredTowerThread(threading.Thread):
    """A thread that runs() a function on a device."""

    def __init__(self, dist, coord, device, variable_creator_fn, fn, *args,
                 **kwargs):
      super(MirroredStrategy._MirroredTowerThread, self).__init__()  # pylint: disable=protected-access
      self.coord = coord
      self.distribution = dist
      self.device = device
      self.tower_id = dist.worker_devices.index(device)
      self.variable_creator_fn = variable_creator_fn
      # State needed to run and return the results of `fn`.
      self.main_fn = fn
      self.main_args = args
      self.main_kwargs = kwargs
      self.main_result = None
      self.done = False
      # State needed to run the next merge_call() (if any) requested via
      # TowerContext.
      self.merge_fn = None
      self.merge_args = None
      self.merge_kwargs = None
      self.merge_result = None
      self.captured_name_scope = None
      # We use a thread.Event for the main thread to signal when this
      # thread should start running (`should_run`), and another for
      # this thread to transfer control back to the main thread
      # (`has_paused`, either when it gets to a
      # `get_tower_context().merge_call` or when `fn` returns). In
      # either case the event starts cleared, is signaled by calling
      # set(). The receiving thread waits for the signal by calling
      # wait() and then immediately clearing the event using clear().
      self.should_run = threading.Event()
      self.has_paused = threading.Event()
      # These fields have to do with inheriting various contexts from the
      # parent thread:
      # pylint: disable=protected-access
      self.context_mode = context.context()._eager_context.mode
      if not context.context()._context_handle:
        context.context()._initialize_handle_and_devices()
      self.context_device_policy = (
          pywrap_tensorflow.TFE_ContextGetDevicePlacementPolicy(
              context.context()._context_handle))
      self.graph = ops.get_default_graph()
      self._variable_creator_stack = self.graph._variable_creator_stack[:]
      self._captured_var_scope = variable_scope.get_variable_scope()
      # Adding a "/" at end lets us re-enter this scope later.
      self._name_scope = self.graph.get_name_scope()
      if self._name_scope:
        self._name_scope += "/"
      if self.tower_id > 0:
        if not self._name_scope:
          self._name_scope = ""
        self._name_scope += "tower_%d/" % self.tower_id

    def run(self):
      # pylint: disable=protected-access
      self.graph._variable_creator_stack = self._variable_creator_stack
      self.should_run.wait()
      self.should_run.clear()
      try:
        if self.coord.should_stop():
          return
        with self.coord.stop_on_exception(), \
            context.context()._mode(self.context_mode), \
            context.context().device_policy(self.context_device_policy), \
            _enter_graph(self.graph), \
            MirroredTowerContext(self.distribution, self.tower_id), \
            ops.device(self.device), \
            ops.name_scope(self._name_scope), \
            variable_scope.variable_scope(
                self._captured_var_scope, reuse=self.tower_id > 0), \
            variable_scope.variable_creator_scope(self.variable_creator_fn):
          self.main_result = self.main_fn(*self.main_args, **self.main_kwargs)
          self.done = True
      finally:
        self.has_paused.set()


class MirroredTowerContext(distribute_lib.TowerContext):
  """TowerContext used in MirroredStrategy.call_for_each_tower().

  Opened in `_MirroredTowerThread`, to allow the user to invoke
  `MirroredStrategy`'s specific implementation of `merge_call()`,
  which works by delegating the function and its arguments to
  the main thread (the one that invoked
  `MirroredStrategy.call_for_each_tower()`).
  """

  def _merge_call(self, fn, *args, **kwargs):
    """Delegate to the main thread to actually perform merge_call()."""
    t = threading.current_thread()  # a _MirroredTowerThread
    t.merge_fn = fn
    t.merge_args = args
    t.merge_kwargs = kwargs
    t.captured_name_scope = t.graph.get_name_scope()
    # Adding a "/" at end lets us re-enter this scope later.
    if t.captured_name_scope:
      t.captured_name_scope += "/"
    t.has_paused.set()
    t.should_run.wait()
    t.should_run.clear()
    if t.coord.should_stop():
      raise _RequestedStop()
    return t.merge_result

  @property
  def device(self):
    distribute_lib.require_tower_context(self)
    return self._distribution_strategy.worker_devices[self._tower_id]
