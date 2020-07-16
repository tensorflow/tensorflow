# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Class MirroredFunctionStrategy implementing tf.distribute.Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest


_replica_index = threading.local()
_replica_id_key = object()


def _replica_id_tensor():
  return ops.get_default_graph().capture_call_time_value(
      closure=lambda: constant_op.constant(_replica_index.current),
      spec=tensor_spec.TensorSpec((), dtypes.int32),
      key=_replica_id_key)


def _in_run():
  return (hasattr(_replica_index, "current") and
          _replica_index.current is not None)


def _outside_run_graph():
  if hasattr(_replica_index, "graph_outside_run"):
    return _replica_index.graph_outside_run
  else:
    return None


class MirroredFunctionStrategy(distribute_lib.Strategy):
  """Mirrors vars to distribute across multiple devices and machines.

  This strategy uses one replica per device and sync replication for its
  multi-GPU version. Unlike `tf.distribute.MirroredStrategy`, it creates a
  function for a single replica, and calls that function repeatedly instead of
  recording the operations for each replica separately.
  """

  def __init__(self, devices=None):
    """Create an instance of `MirroredFunctionStrategy`.

    Args:
      devices: a list of device strings.  If `None`, all available GPUs are
        used. If no GPUs are found, CPU is used.
    """
    extended = MirroredFunctionExtended(self, devices)
    super(MirroredFunctionStrategy, self).__init__(extended)


# TODO(josh11b): Switch to V2 when we no longer need to support tf.compat.v1.
class MirroredFunctionExtended(distribute_lib.StrategyExtendedV1):
  """Implementation of MirroredFunctionStrategy."""

  def __init__(self, container_strategy, devices):
    super(MirroredFunctionExtended, self).__init__(container_strategy)
    if devices is None:
      devices = mirrored_strategy.all_devices()
    if not devices:
      raise ValueError("Got an empty `devices` list. Please make sure the "
                       "`devices` you pass in is not empty.")
    device_tuple = tuple(device_util.resolve(d) for d in devices)
    assert len(set(device_tuple)) == len(device_tuple), (
        "No duplicates allowed in `devices` argument: %s" % (devices,))
    self._devices = device_tuple
    self._retrace_functions_for_each_device = False

  def _call_for_each_replica(self, fn, args, kwargs):
    # For now, `fn` must be an @tf.function.
    # TODO(josh11b): Relax this restriction?  Main problem is if
    # (a) executing eagerly, (b) `fn` not @tf.function, and
    # (c) executed frequently.
    assert isinstance(fn, def_function.Function)

    if _outside_run_graph() is not None:
      # Nested case, should just use outer function's context for things like
      # the current replica index.
      # TODO(josh11b): Test this case!
      with MirroredFunctionReplicaContext(self._container_strategy()):
        results = fn(*nest.map_structure(_unwrap_tensors, args),
                     **nest.map_structure(_unwrap_tensors, kwargs))
        return nest.map_structure(_wrap_tensors, results)

    _replica_index.graph_outside_run = ops.get_default_graph()
    return_values = []

    try:
      with MirroredFunctionReplicaContext(self._container_strategy()):
        for index, device in enumerate(self._devices):
          _replica_index.current = index
          with ops.device(device):
            if context.executing_eagerly():
              # NOTE: These functions need to execute concurrently if they
              # use a collective op. This is a particular concern with eager
              # execution.
              with context.execution_mode(context.ASYNC):
                return_values.append(
                    fn(*distribute_utils.select_replica(index, args),
                       **distribute_utils.select_replica(index, kwargs)))
            else:
              return_values.append(
                  fn(*distribute_utils.select_replica(index, args),
                     **distribute_utils.select_replica(index, kwargs)))
    finally:
      _replica_index.graph_outside_run = None
      _replica_index.current = None

    return distribute_utils.regroup(return_values)

  def _local_results(self, val):
    if isinstance(val, values.DistributedValues):
      return val.values
    return (val,)


class FnMergedValue(object):

  def __init__(self, value):
    self._value = value


def _wrap_tensors(maybe_tensor):
  if isinstance(maybe_tensor, ops.Tensor):  # TODO(josh11b): or composite tensor?
    return FnMergedValue(maybe_tensor)
  return maybe_tensor


def _unwrap_tensors(maybe_wrapped):
  if isinstance(maybe_wrapped, FnMergedValue):
    return maybe_wrapped._value  # pylint: disable=protected-access
  return maybe_wrapped


class MirroredFunctionReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext used in MirroredFunctionStrategy."""

  def __init__(self, strategy):
    distribute_lib.ReplicaContext.__init__(self, strategy, None)

  @property
  def _replica_id_in_sync_group(self):
    return _replica_id_tensor()

  @_replica_id_in_sync_group.setter
  def _replica_id_in_sync_group(self, value):
    assert value is None

  def _merge_call(self, merge_fn, args, kwargs):
    # We wrap all args/kwargs with tensor values in a class that prevents them
    # for being used by anything other than MirroredFunctionStrategy APIs that
    # have been specifically written to recognize the wrapper and unwrap the
    # values (such as extended.reduce_to/update).

    # TODO(josh11b): Should these set expand_composites=True?
    args = nest.map_structure(_wrap_tensors, args)
    kwargs = nest.map_structure(_wrap_tensors, kwargs)
    # pylint: disable=protected-access
    distribution_strategy_context._push_per_thread_mode(
        distribution_strategy_context._CrossReplicaThreadMode(self._strategy))
    try:
      results = merge_fn(self._strategy, *args, **kwargs)
    finally:
      distribution_strategy_context._pop_per_thread_mode()
    # pylint: enable=protected-access
    return nest.map_structure(_unwrap_tensors, results)

  @property
  def devices(self):
    raise RuntimeError("Can't get the devices for the current replica.")
