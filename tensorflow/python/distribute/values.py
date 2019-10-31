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
"""Various classes representing distributed values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import weakref
import six

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import saver
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest


def _devices_match(d1, d2):
  return device_util.canonicalize(d1) == device_util.canonicalize(d2)


class DeviceMap(object):
  """A mapping of replicas & logical device ids to devices."""

  @property
  def all_devices(self):
    """Returns a tuple of strings with all devices in this DeviceMap."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  @property
  def devices_by_replica(self):
    """Returns a tuple `t` where `t[replica]` is the devices for `replica`."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  @property
  def num_logical_devices(self):
    """Count of the number of devices each replica may be defined across."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  @property
  def num_replicas_in_graph(self):
    """Number of replicas defined in this graph."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  def logical_device_from_values(self, values):
    """Returns the logical device index `values` is on."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  def logical_to_actual_devices(self, logical_device_id):
    """Returns sequence of `num_replicas_in_graph` devices."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  def select_for_current_replica(self, values, replica_context):
    """Select the element of `values` for the current replica."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  def replica_for_device(self, device):
    """Return the replica id containing `device`."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  def select_for_device(self, values, device):
    """Select the element of `values` to access from `device`."""
    raise NotImplementedError("Required for DeviceMap implementations.")

  def is_device_in_replica(self, device, replica_id):
    """Returns whether `device` is a member of replica `replica_id`."""
    raise NotImplementedError("Required for DeviceMap implementations.")


class SingleDeviceMap(DeviceMap):
  """A device map for 1 non-computation device.

  Use `SingleDeviceMap` when the device does not correspond to some replica of
  the computation. For computation devices, use `ReplicaDeviceMap` below (even
  if there is only a single device in the map).
  """

  def __init__(self, device):
    """Initialize a `SingleDeviceMap`.

    Args:
      device: A string device.
    """
    assert isinstance(device, six.string_types)
    self._device = device_util.canonicalize(device)
    self._devices = (self._device,)

  @property
  def all_devices(self):
    return self._devices

  @property
  def devices_by_replica(self):
    raise ValueError("SingleDeviceMap not indexed by replicas")

  @property
  def num_logical_devices(self):
    return 1

  @property
  def num_replicas_in_graph(self):
    return 1

  def logical_device_from_values(self, values):
    del values
    return 0

  def logical_to_actual_devices(self, logical_device_id):
    assert logical_device_id == 0
    return self._devices

  def select_for_current_replica(self, values, replica_context):
    assert len(values) == 1
    del replica_context
    return values[0]

  def replica_for_device(self, device):
    raise ValueError("SingleDeviceMap not indexed by replicas")

  def select_for_device(self, values, device):
    assert len(values) == 1
    if self._device != device:
      raise ValueError("Device %s not found in %s (current device %s)" %
                       (device, self._devices, device_util.current()))
    return values[0]

  def is_device_in_replica(self, device, replica_id):
    raise ValueError("SingleDeviceMap not indexed by replicas")

  def __repr__(self):
    return "%s(%r)" % (self.__class__.__name__, self._device)


class ReplicaDeviceMap(DeviceMap):
  """A device map for 1 device per replica."""

  def __init__(self, devices):
    """Initialize a `ReplicaDeviceMap`.

    Args:
      devices: `devices[i]` is the string device for replica `i`.
    """
    self._devices = tuple(device_util.canonicalize(d) for d in devices)
    if len(set(self._devices)) != len(self._devices):
      raise ValueError("Duplicate devices in %s, after canonicalization: %s" %
                       (devices, self._devices))
    self._device_to_replica = {d: r for r, d in enumerate(self._devices)}

  @property
  def all_devices(self):
    return self._devices

  @property
  def devices_by_replica(self):
    return ((d,) for d in self._devices)

  @property
  def num_logical_devices(self):
    return 1

  @property
  def num_replicas_in_graph(self):
    return len(self._devices)

  def logical_device_from_values(self, values):
    del values
    return 0

  def logical_to_actual_devices(self, logical_device_id):
    assert logical_device_id == 0
    return self._devices

  def select_for_current_replica(self, values, replica_context):
    assert len(values) == len(self._devices)
    replica_id = replica_context.replica_id_in_sync_group
    if not isinstance(replica_id, int):
      replica_id = tensor_util.constant_value(replica_id)
    if replica_id is None:
      replica_id = 0
    return values[replica_id]

  def replica_for_device(self, device):
    return self._device_to_replica.get(device)

  def select_for_device(self, values, device):
    assert len(values) == len(self._devices)
    replica_id = self._device_to_replica.get(device)
    if replica_id is None:
      raise ValueError("Device %s not found in %s (current device %s)" %
                       (device, self._devices, device_util.current()))
    return values[replica_id]

  def is_device_in_replica(self, device, replica_id):
    return _devices_match(device, self._devices[replica_id])

  def __str__(self):
    return "[%s]" % (", ".join(self._devices))

  def __repr__(self):
    return "%s([%s])" % (self.__class__.__name__,
                         ", ".join(repr(d) for d in self._devices))


LogicalDeviceSpec = collections.namedtuple(
    "LogicalDeviceSpec", ("device_map", "logical_device"))


class WorkerDeviceMap(DeviceMap):
  """A device map for one value per worker."""

  def __init__(self, devices, num_replicas_per_worker):
    """Initialize a `WorkerDeviceMap`.

    Args:
      devices: `devices[i]` is the string device for worker `i` in in-graph
        relication case; devices is single-element list for its corresponding
        worker in between-graph case.
      num_replicas_per_worker: number of replicas per worker, useful in in-graph
        replication case.
    """
    self._devices = tuple(device_util.canonicalize(d) for d in devices)
    if len(set(self._devices)) != len(self._devices):
      raise ValueError("Duplicate devices in %s, after canonicalization: %s" %
                       (devices, self._devices))
    self._num_replicas_per_worker = num_replicas_per_worker

  @property
  def all_devices(self):
    return self._devices

  @property
  def devices_by_replica(self):
    raise ValueError("`WorkerDeviceMap` is not indexed by replicas")

  @property
  def num_logical_devices(self):
    return 1

  @property
  def num_replicas_in_graph(self):
    return len(self._devices)

  def logical_device_from_values(self, values):
    del values
    return 0

  def logical_to_actual_devices(self, logical_device_id):
    assert logical_device_id == 0
    return self._devices

  def select_for_current_replica(self, values, replica_context):
    return values[replica_context.replica_id_in_sync_group //
                  self._num_replicas_per_worker]

  def replica_for_device(self, device):
    raise ValueError("`WorkerDeviceMap` not indexed by replicas")

  def select_for_device(self, values, device):
    # TODO(yuefengz): this should map from any device to the value on its
    # corresponding worker.
    return values[self._devices.index(device_util.canonicalize(device))]

  def is_device_in_replica(self, device, replica_id):
    raise ValueError("WorkerDeviceMap not indexed by replicas")

  def __repr__(self):
    return "%s(%r, num_replicas_per_worker=%d)" % (
        self.__class__.__name__, self._devices, self._num_replicas_per_worker)


class DistributedValues(object):
  """Holds a map from replica to values. Either PerReplica or Mirrored."""

  def __init__(self, device_map, values, logical_device=None):
    assert isinstance(device_map, DeviceMap)
    self._device_map = device_map
    self._values = tuple(values)
    if logical_device is None:
      logical_device = device_map.logical_device_from_values(self._values)
    self._logical_device = logical_device

  # TODO(josh11b): Split this into two functions, one with device, one without.
  def get(self, device=None):
    """Returns the value for the current device or raises a ValueError."""
    if device is None:
      replica_context = distribution_strategy_context.get_replica_context()
      if replica_context:
        return self._device_map.select_for_current_replica(
            self._values, replica_context)
      else:
        device = distribute_lib.get_update_device()
        if device is None:
          return self._get_cross_replica()
    device = device_util.canonicalize(device)
    return self._device_map.select_for_device(self._values, device)

  @property
  def primary(self):
    """Returns a representative component."""
    return self._values[0]

  @property
  def devices(self):
    return self._device_map.logical_to_actual_devices(self._logical_device)

  @property
  def logical_device(self):
    return self._logical_device

  @property
  def device_map(self):
    return self._device_map

  # TODO(josh11b): Replace experimental_local_results with this?
  @property
  def values(self):
    return self._values

  @property
  def is_tensor_like(self):
    return all(tensor_util.is_tensor(v) for v in self._values)

  def __str__(self):
    devices = self.devices
    assert len(self._values) == len(devices)
    debug_str = ",\n".join("  %d %s: %s" % (i, devices[i], self._values[i])
                           for i in range(len(devices)))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_str)

  def __repr__(self):
    devices = self.devices
    assert len(self._values) == len(devices)
    debug_repr = ",\n".join("  %d %s: %r" % (i, devices[i], self._values[i])
                            for i in range(len(devices)))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_repr)


# NOTE(josh11b,apassos): It would be great if we could inspect the values this was
# initialized with and use that to generate the overloaded operators here.
# Unfortunately, Python's rules for special methods don't allow this, see
# https://docs.python.org/3/reference/datamodel.html#special-method-names
# "if a class defines a method named __getitem__(), and x is an instance of
# this class, then x[i] is roughly equivalent to type(x).__getitem__(x, i)."
# In particular, these special methods don't go through __getattr__, and
# it will only use those methods if they are defined in the class, not the
# object.
class DistributedDelegate(DistributedValues):
  """A map from device to values; acts as the same type as the values."""

  def __getattr__(self, name):
    # The '_use_resource_variables' and the attrs starts with '_self' are used
    # for restoring the saved_model proto, and '_attribute_sentinel' is used for
    # Layer tracking. At the point these attrs are queried, the variable has not
    # been initialized. Thus it should not query those of the underlying
    # components.
    if name.startswith("_self_") or name in (
        "_use_resource_variables", "_attribute_sentinel",
        "_distributed_container"):
      return super(DistributedDelegate, self).__getattr__(name)

    # TODO(priyag): This needs to be made robust against pitfalls from mix use
    # __getattr__ and @property. See b/120402273.
    return getattr(self.get(), name)

  def _get_as_operand(self):
    """Returns the value for operations for the current device.

    Some implementations, e.g. `TPUMirroredVariable`, are not able to return the
    value type within a replica context. They can, however, return a value that
    can be used by the operations below.
    """
    return self.get()

  # pylint: disable=multiple-statements
  def __add__(self, o): return self._get_as_operand() + o
  def __radd__(self, o): return o + self._get_as_operand()
  def __sub__(self, o): return self._get_as_operand() - o
  def __rsub__(self, o): return o - self._get_as_operand()
  def __mul__(self, o): return self._get_as_operand() * o
  def __rmul__(self, o): return o * self._get_as_operand()
  def __truediv__(self, o): return self._get_as_operand() / o
  def __rtruediv__(self, o): return o / self._get_as_operand()

  def __floordiv__(self, o):
    return self._get_as_operand() // o

  def __rfloordiv__(self, o): return o // self._get_as_operand()
  def __mod__(self, o): return self._get_as_operand() % o
  def __rmod__(self, o): return o % self._get_as_operand()
  def __lt__(self, o): return self._get_as_operand() < o
  def __le__(self, o): return self._get_as_operand() <= o
  def __gt__(self, o): return self._get_as_operand() > o
  def __ge__(self, o): return self._get_as_operand() >= o
  def __and__(self, o): return self._get_as_operand() & o
  def __rand__(self, o): return o & self._get_as_operand()
  def __or__(self, o): return self._get_as_operand() | o
  def __ror__(self, o): return o | self._get_as_operand()
  def __xor__(self, o): return self._get_as_operand() ^ o
  def __rxor__(self, o): return o ^ self._get_as_operand()
  def __getitem__(self, o): return self._get_as_operand()[o]
  def __pow__(self, o, modulo=None):
    return pow(self._get_as_operand(), o, modulo)
  def __rpow__(self, o): return pow(o, self._get_as_operand())
  def __invert__(self): return ~self._get_as_operand()
  def __neg__(self): return -self._get_as_operand()
  def __abs__(self): return abs(self._get_as_operand())

  def __div__(self, o):
    try:
      return self._get_as_operand().__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    try:
      return self._get_as_operand().__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    try:
      return self._get_as_operand().__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    try:
      return self._get_as_operand().__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  # TODO(josh11b): Even more operator overloads.


class PerReplica(DistributedValues, composite_tensor.CompositeTensor):
  """Holds a map from replica to unsynchronized values."""

  @property
  def _type_spec(self):
    value_specs = nest.map_structure(type_spec.type_spec_from_value,
                                     self._values)
    return PerReplicaSpec(value_specs, self._device_map, self._logical_device)


class PerReplicaSpec(type_spec.TypeSpec):
  """Type specification for a `PerReplica`."""

  __slots__ = ["_value_specs", "_device_map", "_logical_device"]

  value_type = property(lambda self: PerReplica)

  def __init__(self, value_specs, device_map, logical_device):
    if isinstance(device_map, tuple):
      device_map = self._deserialize_device_map(device_map)
    self._value_specs = tuple(value_specs)
    self._device_map = device_map
    self._logical_device = logical_device

  def _serialize(self):
    device_map = self._serialize_device_map(self._device_map)
    return (self._value_specs, device_map, self._logical_device)

  @property
  def _component_specs(self):
    return self._value_specs

  def _to_components(self, value):
    replica_context = distribution_strategy_context.get_replica_context()
    if replica_context is not None and replica_context.num_replicas_in_sync > 1:
      raise ValueError(
          "Flattening a PerReplica to components is not supported in replica "
          "context.")
    return value._values  # pylint: disable=protected-access

  def _from_components(self, tensor_list):
    return PerReplica(self._device_map, tensor_list,
                      logical_device=self._logical_device)

  @staticmethod
  def _serialize_device_map(device_map):
    if isinstance(device_map, SingleDeviceMap):
      return ("single", device_map.all_devices[0])
    elif isinstance(device_map, ReplicaDeviceMap):
      return ("replica", device_map.all_devices)
    elif isinstance(device_map, WorkerDeviceMap):
      return ("worker", device_map.all_devices,
              device_map.num_replicas_per_worker)
    else:
      raise ValueError("PerReplicaSpec does not support device_map type %s"
                       % type(device_map).__name__)

  @staticmethod
  def _deserialize_device_map(device_map_info):
    device_map_type = device_map_info[0]
    device_map_args = device_map_info[1:]
    if device_map_type == "single":
      return SingleDeviceMap(*device_map_args)
    elif device_map_type == "replica":
      return ReplicaDeviceMap(*device_map_args)
    elif device_map_type == "worker":
      return WorkerDeviceMap(*device_map_args)
    else:
      raise ValueError("Unexpected value in state tuple")


# Note that unlike PerReplica, Mirrored values inherit from
# DistributedDelegate and so can be used directly in cross-replica mode.
# TODO(tomhennigan) Should this extend CompositeTensor?
class Mirrored(DistributedDelegate):
  """Holds a map from replica to values which are kept in sync."""

  def _get_cross_replica(self):
    device = device_util.canonicalize(device_util.current())
    replica_id = self._device_map.replica_for_device(device)
    if replica_id is None:
      return self.primary
    return self._values[replica_id]

  def _as_graph_element(self):
    obj = self.get()
    conv_fn = getattr(obj, "_as_graph_element", None)
    if conv_fn and callable(conv_fn):
      return conv_fn()
    return obj


def _assign_on_device(device, variable, tensor):
  with ops.device(device):
    return variable.assign(tensor)


def _assign_add_on_device(device, variable, tensor):
  with ops.device(device):
    return variable.assign_add(tensor)


def _assign_sub_on_device(device, variable, tensor):
  with ops.device(device):
    return variable.assign_sub(tensor)


def _assert_strategy(strategy):
  if not distribution_strategy_context.has_strategy():
    raise RuntimeError(
        'Need to be inside "with strategy.scope()" for %s' %
        (strategy,))
  current_strategy = distribution_strategy_context.get_strategy()
  if current_strategy is not strategy:
    raise RuntimeError(
        "Mixing different tf.distribute.Strategy objects: %s is not %s" %
        (current_strategy, strategy))


@contextlib.contextmanager
def _enter_or_assert_strategy(strategy):
  if not distribution_strategy_context.has_strategy():
    with strategy.scope():
      yield
  else:
    _assert_strategy(strategy)
    yield


DistributedVarOp = collections.namedtuple(
    "DistributedVarOp", ["name", "graph", "traceback", "type"])


class DistributedVariable(DistributedDelegate, variables_lib.Variable):
  """Holds a map from replica to variables."""
  # TODO(josh11b): Support changing the set of variables if e.g. if new
  # devices are joining or a device is to leave.

  def __init__(self, strategy, device_map, values, logical_device=None):
    self._distribute_strategy = strategy
    super(DistributedVariable, self).__init__(
        device_map, values, logical_device=logical_device)
    self._common_name = self.primary.name.split(":")[0]
    # Use a weakref to make it easy to map from the contained values
    # to the container without introducing a reference cycle.
    for v in values:
      v._distributed_container = weakref.ref(self)  # pylint: disable=protected-access
    # tf.keras keeps track of variables initialized using this attribute. When
    # tf.keras gets the default session, it initializes all uninitialized vars.
    # We need to make _keras_initialized a member of DistributedVariable because
    # without this it will use `__getattr__` which will delegate to a component
    # variable.
    self._keras_initialized = False
    # Typically, a `DistributedVariable`'s initializer is composed of the
    # initializers of the components variables. However, in some cases, such as
    # when restoring from a checkpoint, we may set the _initializer_op
    # property on the entire `DistributedVariable`.
    self._initializer_op = None

  def is_initialized(self, name=None):
    """Identifies if all the component variables are initialized.

    Args:
      name: Name of the final `logical_and` op.

    Returns:
      The op that evaluates to True or False depending on if all the
      component variables are initialized.
    """
    result = self.primary.is_initialized()
    # We iterate through the list of values except the last one to allow us to
    # name the final `logical_and` op the same name that is passed by the user
    # to the `is_initialized` op. For distributed variables, the
    # `is_initialized` op is a `logical_and` op.
    for v in self._values[1:-1]:
      result = math_ops.logical_and(result, v.is_initialized())
    result = math_ops.logical_and(result, self._values[-1].is_initialized(),
                                  name=name)
    return result

  @property
  def initializer(self):
    if self._initializer_op:
      init_op = self._initializer_op
    else:
      # return grouped ops of all the var initializations of component values of
      # the mirrored variable
      init_op = control_flow_ops.group(tuple(
          v.initializer for v in self._values))
    return init_op

  def _get_closest(self):
    """Return member in the same replica if possible, else the primary."""
    replica_context = distribution_strategy_context.get_replica_context()
    if replica_context:
      return self._device_map.select_for_current_replica(
          self._values, replica_context)
    device = distribute_lib.get_update_device()
    if device is None:
      device = device_util.canonicalize(device_util.current())
    replica_id = self._device_map.replica_for_device(device)
    if replica_id is None:
      return self.primary
    return self._values[replica_id]

  def initialized_value(self):
    return self._get_closest().initialized_value()

  @property
  def initial_value(self):
    return self._get_closest().initial_value

  @property
  def graph(self):
    return self.primary.graph

  @property
  def _shared_name(self):
    return self._common_name

  @property
  def _unique_id(self):
    return self.primary._unique_id   # pylint: disable=protected-access

  @property
  def _graph_key(self):
    """Lets Optimizers know which graph this variable is from."""
    return self.primary._graph_key  # pylint: disable=protected-access

  @property
  def name(self):
    return self.primary.name

  @property
  def dtype(self):
    return self.primary.dtype

  @property
  def shape(self):
    return self.primary.shape

  @property
  def synchronization(self):
    return self.primary.synchronization

  @property
  def handle(self):
    device = None
    replica_context = distribution_strategy_context.get_replica_context()
    if replica_context is None:
      device = distribute_lib.get_update_device()
      if device is None:
        raise ValueError("`handle` is not available outside the replica context"
                         " or a `tf.distribute.Strategy.update()` call.")
    return self.get(device=device).handle

  def eval(self, session=None):
    return self._get_closest().eval(session)

  @property
  def _save_slice_info(self):
    return self.primary._save_slice_info  # pylint: disable=protected-access

  def _get_save_slice_info(self):
    return self.primary._get_save_slice_info()  # pylint: disable=protected-access

  def _set_save_slice_info(self, save_slice_info):
    for v in self._values:
      v._set_save_slice_info(save_slice_info)  # pylint: disable=protected-access

  @property
  def device(self):
    return self._get_closest().device

  @property
  def trainable(self):
    return self.primary.trainable

  @property
  def distribute_strategy(self):
    return self._distribute_strategy

  def get_shape(self):
    return self.primary.get_shape()

  def to_proto(self, export_scope=None):
    return self.primary.to_proto(export_scope=export_scope)

  @property
  def op(self):
    # We want cross-replica code that does some var.op.X calls
    # to work (even if the current device isn't in self.devices), but
    # other uses of var.op in a cross-replica context to fail.
    if distribution_strategy_context.in_cross_replica_context():
      return DistributedVarOp(self.primary.op.name,
                              self.primary.op.graph,
                              self.primary.op.traceback,
                              self.primary.op.type)
    return self.get().op

  @property
  def _in_graph_mode(self):
    return self.primary._in_graph_mode   # pylint: disable=protected-access

  def read_value(self):
    with _enter_or_assert_strategy(self._distribute_strategy):
      return array_ops.identity(self.get())

  def value(self):
    return self._get_closest().value()

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def _clone_with_new_values(self, new_values):
    raise NotImplementedError("Must be implemented in descendents.")


ops.register_dense_tensor_like_type(DistributedVariable)


@contextlib.contextmanager
def _maybe_enter_graph(tensor):
  # Note: might have an eager tensor but not be executing eagerly when
  # building functions.
  if (context.executing_eagerly() or isinstance(tensor, ops.EagerTensor)
      or ops.has_default_graph()):
    yield
  else:
    with tensor.graph.as_default():
      yield


def _make_raw_assign_fn(raw_assign_fn):  # pylint: disable=missing-docstring
  def assign_fn(var, value, use_locking=False, name=None, read_value=True):  # pylint: disable=missing-docstring
    del use_locking  # Unused.

    with _maybe_enter_graph(var.handle):
      op = raw_assign_fn(
          var.handle, ops.convert_to_tensor(value, dtype=var.dtype), name=name)

      with ops.control_dependencies([op]):
        return var._read_variable_op() if read_value else op  # pylint: disable=protected-access
  return assign_fn


class TPUVariableMixin(object):
  """Mixin for TPU variables."""

  def __init__(self, *args, **kwargs):
    super(TPUVariableMixin, self).__init__(*args, **kwargs)

    # Handle ID is needed for `get_replicated_var_handle` to cache the variables
    # correctly since in eager mode different variables can have the same name.
    if ops.executing_eagerly_outside_functions():
      self._handle_id = self._common_name + "_" + str(id(self.primary))
    else:
      self._handle_id = self._common_name

  def __getattr__(self, name):
    if _enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).__getattr__(name)
    else:
      raise AttributeError(
          "'{}' not accessible within a TPU context.".format(name))

  def get(self, device=None):
    if (_enclosing_tpu_context() is None) or (device is not None):
      return super(TPUVariableMixin, self).get(device=device)
    else:
      raise NotImplementedError(
          "`TPUVariableMixin.get()` is not supported within a TPU context.")

  def _get_as_operand(self):
    return self.read_value()

  def _get_closest(self):
    if _enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self)._get_closest()
    else:
      return self.primary

  def numpy(self):
    if context.executing_eagerly():
      return self.read_value().numpy()
    else:
      raise NotImplementedError(
          "numpy() is only available when eager execution is enabled.")

  def _is_mirrored(self):
    raise NotImplementedError(
        "`TPUVariableMixin._is_mirrored()` must be implemented by subclasses.")

  @property
  def handle(self):
    # If we're in a tpu.rewrite(), return the replicated handle.
    tpu_context = _enclosing_tpu_context()
    if tpu_context is None:
      return self._get_closest().handle
    else:
      return tpu_context.get_replicated_var_handle(self._handle_id,
                                                   self._values,
                                                   self._device_map,
                                                   self._is_mirrored())

  @property
  def device(self):
    return self.handle.device

  def _read_variable_op(self):
    if self.trainable:
      tape.variable_accessed(self)
    return gen_resource_variable_ops.read_variable_op(self.handle, self.dtype)

  def read_value(self):
    if _enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).read_value()
    else:
      return self._read_variable_op()

  @property
  def constraint(self):
    return self.primary.constraint

  def _as_graph_element(self):
    if _enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self)._as_graph_element()  # pylint: disable=protected-access
    else:
      return None

  @property
  def op(self):
    return DistributedVarOp(
        self.primary.op.name, self.primary.op.graph, self.primary.op.traceback,
        self.primary.op.type)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if _enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self)._dense_var_to_tensor(
          dtype=dtype, name=name, as_ref=as_ref)
    # pylint: enable=protected-access
    elif dtype is not None and dtype != self.dtype:
      return math_ops.cast(self.read_value(), dtype)
    else:
      return self.handle if as_ref else self.read_value()


def _validate_colocate_extended(v, extended):
  variable_strategy = v._distribute_strategy  # pylint: disable=protected-access
  if variable_strategy.extended is not extended:
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not %s created in scope: %s" %
        (v, variable_strategy))


def validate_colocate_distributed_variable(v, extended):
  if not isinstance(v, DistributedVariable):
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not: %r" % (v,))
  _validate_colocate_extended(v, extended)


def validate_colocate(v, extended):
  if not hasattr(v, "_distribute_strategy"):
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not: %r" % (v,))
  _validate_colocate_extended(v, extended)


def _apply_aggregation(strategy, value, aggregation, destinations):
  if aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
    return strategy.extended.broadcast_to(
        strategy.experimental_local_results(value)[0],
        destinations=destinations)
  reduce_op = reduce_util.ReduceOp.from_variable_aggregation(aggregation)
  return strategy.extended.reduce_to(reduce_op, value, destinations)

_aggregation_error_msg = (
    "You must specify an aggregation method to update a "
    "{variable_type} in Replica Context. You can do so by passing "
    "an explicit value for argument `aggregation` to tf.Variable(..)."
    "e.g. `tf.Variable(..., aggregation=tf.VariableAggregation.SUM)`"
    "`tf.VariableAggregation` lists the possible aggregation methods."
    "This is required because {variable_type} should always be "
    "kept in sync. When updating them or assigning to them in a "
    "replica context, we automatically try to aggregate the values "
    "before updating the variable. For this aggregation, we need to "
    "know the aggregation method. "
    "Another alternative is to not try to update such "
    "{variable_type} in replica context, but in cross replica "
    "context. You can enter cross replica context by calling "
    "`tf.distribute.get_replica_context().merge_call(merge_fn, ..)`."
    "Inside `merge_fn`, you can then update the {variable_type} "
    "using `tf.distribute.StrategyExtended.update()`.")


class _MirroredSaveable(saver.BaseSaverBuilder.ResourceVariableSaveable):
  """Class for defining how to restore a MirroredVariable."""

  def __init__(self, mirrored_variable, primary_variable, name):
    self._mirrored_variable = mirrored_variable
    super(_MirroredSaveable, self).__init__(primary_variable, "", name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return control_flow_ops.group(tuple(
        _assign_on_device(v.device, v, tensor)
        for v in self._mirrored_variable.values))


def create_mirrored_variable(  # pylint: disable=missing-docstring
    strategy, device_map, logical_device, real_mirrored_creator, mirrored_cls,
    sync_on_read_cls, *args, **kwargs):
  # Figure out what collections this variable should be added to.
  # We'll add the MirroredVariable to those collections instead.
  var_collections = kwargs.pop("collections", None)
  if var_collections is None:
    var_collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  kwargs["collections"] = []

  synchronization = kwargs.get(
      "synchronization", vs.VariableSynchronization.ON_WRITE)

  if synchronization == vs.VariableSynchronization.NONE:
    raise ValueError(
        "`NONE` variable synchronization mode is not supported with `Mirrored` "
        "distribution strategy. Please change the `synchronization` for "
        "variable: " + str(kwargs["name"]))
  elif synchronization == vs.VariableSynchronization.ON_READ:
    is_sync_on_read = True
  elif synchronization in (
      vs.VariableSynchronization.ON_WRITE,
      vs.VariableSynchronization.AUTO):
    # `AUTO` synchronization defaults to `ON_WRITE`.
    is_sync_on_read = False
  else:
    raise ValueError(
        "Invalid variable synchronization mode: %s for variable: %s" %
        (synchronization, kwargs["name"]))

  aggregation = kwargs.pop("aggregation", vs.VariableAggregation.NONE)

  if aggregation not in (
      vs.VariableAggregation.NONE,
      vs.VariableAggregation.SUM,
      vs.VariableAggregation.MEAN,
      vs.VariableAggregation.ONLY_FIRST_REPLICA):
    raise ValueError(
        "Invalid variable aggregation mode: %s for variable: %s" %
        (aggregation, kwargs["name"]))

  # Ignore user-specified caching device, not needed for mirrored variables.
  kwargs.pop("caching_device", None)

  # TODO(josh11b,apassos): It would be better if variable initialization
  # was never recorded on the tape instead of having to do this manually
  # here.
  with tape.stop_recording():
    devices = device_map.logical_to_actual_devices(logical_device)
    value_list = real_mirrored_creator(devices, *args, **kwargs)

    var_cls = sync_on_read_cls if is_sync_on_read else mirrored_cls

    result = var_cls(
        strategy, device_map, value_list, aggregation,
        logical_device=logical_device)

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
      var_collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
      l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
      for value in value_list:
        for i, trainable_variable in enumerate(l):
          if value is trainable_variable:
            del l[i]
            break

    g.add_to_collections(var_collections, result)
  elif ops.GraphKeys.GLOBAL_STEP in var_collections:
    ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, result)

  return result


class MirroredVariable(DistributedVariable, Mirrored):
  """Holds a map from replica to variables whose values are kept in sync."""

  def __init__(
      self, strategy, device_map, values, aggregation, logical_device=None):
    super(MirroredVariable, self).__init__(
        strategy, device_map, values, logical_device=logical_device)
    self._aggregation = aggregation

  # The arguments to update() are automatically unwrapped so the update()
  # function would normally see regular variables, not MirroredVariables.
  # However, the update function can still operate on wrapped MirroredVariables
  # through object members, captured arguments, etc. This is more likely in an
  # update_non_slot() function (like OptimizerV2._finish), which can
  # update several non-slot variables in one call.
  def _assign_func(self, *args, **kwargs):
    with _enter_or_assert_strategy(self._distribute_strategy):
      f = kwargs.pop("f")
      if distribution_strategy_context.in_cross_replica_context():
        update_device = distribute_lib.get_update_device()
        if update_device is not None:
          # We are calling an assign function on the mirrored variable in an
          # update context.
          v = self.get(device=update_device)
          return f(v, *args, **kwargs)

        # We are calling assign on the mirrored variable in cross replica
        # context, use `strategy.extended.update()` to update the variable.
        return self._distribute_strategy.extended.update(
            self, f, args=args, kwargs=kwargs)
      else:
        _assert_replica_context(self._distribute_strategy)
        # We are calling an assign function on the mirrored variable in replica
        # context.
        # We reduce the value we want to assign/add/sub. More details about how
        # we handle the different use cases can be found in the _reduce method.
        # We call the function on each of the mirrored variables with the
        # reduced value.
        if self._aggregation == vs.VariableAggregation.NONE:
          raise ValueError(_aggregation_error_msg.format(
              variable_type="MirroredVariable"))

        def merge_fn(strategy, value, *other_args, **other_kwargs):
          v = _apply_aggregation(strategy, value, self._aggregation, self)
          return strategy.extended.update(
              self, f, args=(v,) + other_args, kwargs=other_kwargs)

        return distribution_strategy_context.get_replica_context().merge_call(
            merge_fn, args=args, kwargs=kwargs)

  def assign_sub(self, *args, **kwargs):
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return self._assign_func(f=assign_sub_fn, *args, **kwargs)

  def assign_add(self, *args, **kwargs):
    assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
    return self._assign_func(f=assign_add_fn, *args, **kwargs)

  def assign(self, *args, **kwargs):
    assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
    return self._assign_func(f=assign_fn, *args, **kwargs)

  @property
  def aggregation(self):
    return self._aggregation

  def _get_cross_replica(self):
    device = device_util.canonicalize(device_util.current())
    replica_id = self._device_map.replica_for_device(device)
    if replica_id is None:
      return array_ops.identity(self.primary)
    return array_ops.identity(self._values[replica_id])

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.in_cross_replica_context():
      return self.primary._as_graph_element()
    return self.get()._as_graph_element()

  def _gather_saveables_for_checkpoint(self):
    """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    MirroredVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """
    def _saveable_factory(name=self._common_name):
      return _MirroredSaveable(self, self.primary, name)
    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # Try to avoid assignments to and other mutations of MirroredVariable
    # state except through a DistributionStrategy.extended.update() call.
    assert not as_ref
    return ops.convert_to_tensor(
        self.get(), dtype=dtype, name=name, as_ref=as_ref)

  def _clone_with_new_values(self, new_values):
    return type(self)(self._distribute_strategy, self._device_map, new_values,
                      self._aggregation, logical_device=self._logical_device)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_mirrored(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(MirroredVariable,
                                        _tensor_conversion_mirrored)

def _tensor_conversion_mirrored_val(value, dtype=None, name=None, as_ref=False):
  return ops.convert_to_tensor(
      value.get(), dtype=dtype, name=name, as_ref=as_ref)

ops.register_tensor_conversion_function(Mirrored,
                                        _tensor_conversion_mirrored_val)

def _enclosing_tpu_context():
  """Returns the XLAControlFlowContext, which exists inside a tpu.rewrite()."""
  graph = ops.get_default_graph()
  while graph is not None:
    # pylint: disable=protected-access
    context_ = graph._get_control_flow_context()
    # pylint: enable=protected-access
    while context_ is not None:
      if isinstance(context_, control_flow_ops.XLAControlFlowContext):
        return context_
      context_ = context_.outer_context
    # This may be a FuncGraph due to defuns or v2 control flow. We need to
    # find the original graph with the XLAControlFlowContext.
    graph = getattr(graph, "outer_graph", None)
  return None


def is_distributed_variable(v):
  """Determine if a variable is ds variable or TPU mirrored variable."""
  return isinstance(v, DistributedVariable)


class TPUMirroredVariable(TPUVariableMixin, MirroredVariable):
  """Holds a map from replica to TPU variables whose values are kept in sync."""

  def _assign_func(self, *args, **kwargs):
    with _enter_or_assert_strategy(self._distribute_strategy):
      if (distribution_strategy_context.in_cross_replica_context()
          and (_enclosing_tpu_context() is not None)):
        f = kwargs.pop("f")
        return self._distribute_strategy.extended.update(
            self, f, args=args, kwargs=kwargs)
      else:
        return MirroredVariable._assign_func(self, *args, **kwargs)

  def assign_sub(self, *args, **kwargs):
    assign_sub_fn = _make_raw_assign_fn(
        gen_resource_variable_ops.assign_sub_variable_op)
    return self._assign_func(f=assign_sub_fn, *args, **kwargs)

  def assign_add(self, *args, **kwargs):
    assign_add_fn = _make_raw_assign_fn(
        gen_resource_variable_ops.assign_add_variable_op)
    return self._assign_func(f=assign_add_fn, *args, **kwargs)

  def assign(self, *args, **kwargs):
    assign_fn = _make_raw_assign_fn(
        gen_resource_variable_ops.assign_variable_op)
    return self._assign_func(f=assign_fn, *args, **kwargs)

  def _is_mirrored(self):
    return True


class _SyncOnReadSaveable(saver.BaseSaverBuilder.SaveableObject):
  """Class for defining how to restore a SyncOnReadVariable."""

  def __init__(self, sync_on_read_variable, name):
    self._sync_on_read_variable = sync_on_read_variable
    # We use a callable so that we don't have to evaluate this expression
    # in the case where we are trying to restore instead of save.
    def tensor():
      strategy = sync_on_read_variable._distribute_strategy  # pylint: disable=protected-access
      return strategy.extended.read_var(sync_on_read_variable)

    spec = saver.BaseSaverBuilder.SaveSpec(
        tensor=tensor,
        slice_spec="",
        name=name,
        dtype=sync_on_read_variable.dtype,
        device=sync_on_read_variable.primary.device)
    super(_SyncOnReadSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    # To preserve the sum across save and restore, we have to divide the
    # total across all devices when restoring a variable that was summed
    # when saving.
    tensor, = restored_tensors
    if self._sync_on_read_variable.aggregation == vs.VariableAggregation.SUM:
      tensor = math_ops.cast(tensor / len(self._sync_on_read_variable.devices),
                             self._sync_on_read_variable.dtype)
    return control_flow_ops.group(
        tuple(
            _assign_on_device(v.device, v, tensor)
            for v in self._sync_on_read_variable.values))


def _assert_replica_context(strategy):
  replica_context = distribution_strategy_context.get_replica_context()
  if not replica_context:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")
  if replica_context.strategy is not strategy:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")


class SyncOnReadVariable(DistributedVariable):
  """Holds a map from replica to variables whose values are reduced on save."""

  def __init__(
      self, strategy, device_map, values, aggregation, logical_device=None):
    self._aggregation = aggregation
    super(SyncOnReadVariable, self).__init__(
        strategy, device_map, values, logical_device=logical_device)

  def assign_sub(self, *args, **kwargs):
    with _enter_or_assert_strategy(self._distribute_strategy):
      if distribution_strategy_context.in_cross_replica_context():
        if self._aggregation == vs.VariableAggregation.SUM:
          raise ValueError(
              "SyncOnReadVariable does not support `assign_sub` in "
              "cross-replica context when aggregation is set to "
              "`tf.VariableAggregation.SUM`.")
        return control_flow_ops.group(tuple(
            _assign_sub_on_device(v.device, v, args[0]) for v in self._values))
      else:
        return self.get().assign_sub(*args, **kwargs)

  def assign_add(self, *args, **kwargs):
    with _enter_or_assert_strategy(self._distribute_strategy):
      if distribution_strategy_context.in_cross_replica_context():
        if self._aggregation == vs.VariableAggregation.SUM:
          raise ValueError(
              "SyncOnReadVariable does not support `assign_add` in "
              "cross-replica context when aggregation is set to "
              "`tf.VariableAggregation.SUM`.")
        return control_flow_ops.group(tuple(
            _assign_add_on_device(v.device, v, args[0]) for v in self._values))
      else:
        return self.get().assign_add(*args, **kwargs)

  def assign(self, *args, **kwargs):
    with _enter_or_assert_strategy(self._distribute_strategy):
      if distribution_strategy_context.in_cross_replica_context():
        # To preserve the sum across save and restore, we have to divide the
        # total across all devices when restoring a variable that was summed
        # when saving.
        tensor = args[0]
        if self._aggregation == vs.VariableAggregation.SUM:
          tensor = math_ops.cast(tensor / len(self.devices), self.dtype)
        return control_flow_ops.group(tuple(
            _assign_on_device(v.device, v, tensor) for v in self._values))
      else:
        return self.get().assign(*args, **kwargs)

  @property
  def aggregation(self):
    return self._aggregation

  def _get_cross_replica(self):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      return self.primary

    with _enter_or_assert_strategy(self._distribute_strategy):
      return self._distribute_strategy.reduce(
          reduce_util.ReduceOp.from_variable_aggregation(self.aggregation),
          self, axis=None)

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.in_cross_replica_context():
      return self._get_cross_replica()
    return self.get()._as_graph_element()

  def _gather_saveables_for_checkpoint(self):
    """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    `SyncOnReadVariable`s.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """
    def _saveable_factory(name=self._common_name):
      return _SyncOnReadSaveable(self, name)
    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    return ops.convert_to_tensor(
        self.get(), dtype=dtype, name=name, as_ref=as_ref)

  def _clone_with_new_values(self, new_values):
    return type(self)(self._distribute_strategy, self._device_map, new_values,
                      self._aggregation, logical_device=self._logical_device)


# Register a conversion function for SyncOnReadVariable which allows as_ref to
# be true.
def _tensor_conversion_sync_on_read(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(SyncOnReadVariable,
                                        _tensor_conversion_sync_on_read)


class TPUSyncOnReadVariable(TPUVariableMixin, SyncOnReadVariable):
  """Holds a map from replica to variables whose values are reduced on save."""

  def assign_sub(self, *args, **kwargs):
    if _enclosing_tpu_context() is None:
      return SyncOnReadVariable.assign_sub(self, *args, **kwargs)
    else:
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(
              self, *args, **kwargs)

  def assign_add(self, *args, **kwargs):
    if _enclosing_tpu_context() is None:
      return SyncOnReadVariable.assign_add(self, *args, **kwargs)
    else:
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(
              self, *args, **kwargs)

  def assign(self, *args, **kwargs):
    if _enclosing_tpu_context() is None:
      return SyncOnReadVariable.assign(self, *args, **kwargs)
    else:
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_variable_op)(self, *args, **kwargs)

  def _is_mirrored(self):
    return False


def regroup(device_map, values, wrap_class=PerReplica):
  """Makes a nest per-replica into a nest of PerReplica/Mirrored values."""
  assert isinstance(device_map, DeviceMap)
  assert len(values) == device_map.num_replicas_in_graph
  v0 = values[0]

  if isinstance(v0, list):
    for v in values[1:]:
      assert isinstance(v, list)
      assert len(v) == len(v0), ("len(v) == %d, len(v0) == %d, v: %s, v0: %s" %
                                 (len(v), len(v0), v, v0))
    return [regroup(device_map, tuple(v[i] for v in values), wrap_class)
            for i in range(len(v0))]

  if isinstance(v0, tuple):
    for v in values[1:]:
      assert isinstance(v, tuple)
      assert len(v) == len(v0)
    regrouped_tuple = tuple(
        regroup(device_map, tuple(v[i] for v in values), wrap_class)
        for i in range(len(v0)))
    if hasattr(v0, "_fields"):
      # This tuple is in fact a namedtuple! Create a new namedtuple instance
      # and initialize it with the regrouped values:
      assert hasattr(type(v0), "_make")
      return type(v0)._make(regrouped_tuple)
    else:
      return regrouped_tuple

  if isinstance(v0, dict):
    v0keys = set(v0.keys())
    for v in values[1:]:
      assert isinstance(v, dict), ("v[0]: %r  v[i]: %r" % (v0, v))
      assert set(v.keys()) == v0keys, ("v[0].keys: %s  v[i].keys: %s" %
                                       (v0keys, set(v.keys())))
    return {key: regroup(device_map, tuple(v[key] for v in values), wrap_class)
            for key in v0keys}

  # If exactly the same object across all devices, return it unwrapped.
  same_id = True
  for v in values[1:]:
    if v is not v0:
      same_id = False
      break
  # Consider three cases where same_id is true:
  # * If v0 is a DistributedVariable (a MirroredVariable or
  #   SyncOnReadVariable, and same_id means it is the same across all
  #   devices), we want to return it. We check DistributedVariable
  #   specifically since it can look like it has a
  #   _distributed_container member since its members do.
  # * If v0 is a member of a distributed variable, in which case
  #   hasattr(v0, "_distributed_container") is true, we want to
  #   return the DistributedVariable that contains it using the
  #   _distributed_container logic below. This case can trigger
  #   same_id when there is only one device.
  # * In any other situation, same_id means we return v0.
  if same_id and (isinstance(v0, DistributedVariable) or
                  not hasattr(v0, "_distributed_container")):
    return v0

  # Detect the case where each device has a parallel component of the
  # same MirroredVariable (or SyncOnReadVariable). In this case we
  # want to return the containing MirroredVariable, after a bunch of
  # sanity checking. In particular, each component should have the
  # same container, and the devices of the variables should match the
  # keys of the per-replica dictionary.
  if hasattr(v0, "_distributed_container"):
    # pylint: disable=protected-access
    assert not isinstance(v0, MirroredVariable), (
        "ids = %s, values = %s" % ([id(v) for v in values], values))
    assert device_map.is_device_in_replica(v0.device, 0), (
        "v0.device = %s, device_map = %s" % (v0.device, device_map))
    distributed_container = v0._distributed_container()
    assert distributed_container is not None
    for r, v in enumerate(values[1:]):
      assert device_map.is_device_in_replica(v.device, r + 1), (
          "v.device = %s, r = %d, device_map = %s" %
          (v.device, r + 1, device_map))
      assert distributed_container is v._distributed_container()
    return distributed_container
  # pylint: enable=protected-access

  return wrap_class(device_map, values)


def select_replica(replica_id, structured):
  """Specialize a nest of regular & per-replica values for one replica."""
  def _get(x):
    # `DistributedValues` would be sliced according to replica unless it is a
    # `DistributedVariable` because `DistributedVariable` can be handled
    # directly in the replica context.
    if (isinstance(x, DistributedVariable) or
        not isinstance(x, DistributedValues)):
      return x
    else:
      return x.values[replica_id]

  return nest.map_structure(_get, structured)


def select_device_mirrored(device, structured):
  """Specialize a nest of regular & mirrored values for one device."""
  def _get_mirrored(x):
    if isinstance(x, DistributedValues):
      if not isinstance(x, Mirrored):
        raise TypeError(
            "Expected value to be mirrored across replicas: %s in %s." %
            (x, structured))
      return x.get(device)
    else:
      return x

  return nest.map_structure(_get_mirrored, structured)


def update_regroup(extended, device_map, updates, group):
  """Regroup for an update, with dependencies to ensure all updates execute."""
  if not group:
    regrouped = regroup(device_map, updates, Mirrored)
    return nest.map_structure(extended._local_results, regrouped)  # pylint: disable=protected-access

  def _make_grouped_mirrored(device_map, values):
    """Convert per-replica list `values` into Mirrored type with grouping."""
    if len(values) == 1:
      return Mirrored(device_map, values)

    # Make sure we run all updates. Without this, something like
    # session.run(extended.update(...)) may only update one replica.
    g = control_flow_ops.group(values)

    # If values is just ops, the grouping is enough. Everything in values
    # should have the same type, since we expect every replica to be performing
    # the same computation.
    if not all(tensor_util.is_tensor(v) for v in values):
      return g

    # Otherwise we need tensors with the same values as `values`, but
    # that have a dependency on `g`.
    devices = device_map.logical_to_actual_devices(
        device_map.logical_device_from_values(values))
    assert len(values) == len(devices)
    with_dep = []
    for v, d in zip(values, devices):
      with ops.device(d), ops.control_dependencies([g]):
        with_dep.append(array_ops.identity(v))

    return Mirrored(device_map, with_dep)

  return regroup(device_map, updates, _make_grouped_mirrored)


def value_container(val):
  """Returns the container that this per-replica `value` belongs to.

  Args:
    val: A value returned by `call_for_each_replica()` or a variable
      created in `scope()`.

  Returns:
    A container that `value` belongs to.
    If value does not belong to any container (including the case of
    container having been destroyed), returns the value itself.
  """
  if (hasattr(val, "_distributed_container") and
      # DistributedVariable has _distributed_container defined
      # but we don't want to return it.
      not isinstance(val, DistributedVariable)):
    container = val._distributed_container()  # pylint: disable=protected-access
    if container is not None:
      return container
  return val


class AggregatingVariable(variables_lib.Variable):
  """A wrapper around a variable that aggregates updates across replicas."""

  def __init__(self, strategy, v, aggregation):
    self._distribute_strategy = strategy
    self._v = v
    # NOTE: We don't use "_distributed_container" here because we don't want
    # to trigger that code path in regroup().
    v._aggregating_container = weakref.ref(self)  # pylint: disable=protected-access
    self._aggregation = aggregation

  def get(self):
    return self._v

  @property
  def distribute_strategy(self):
    return self._distribute_strategy

  def __getattr__(self, name):
    return getattr(self._v, name)

  def _assign_func(self, *args, **kwargs):
    with _enter_or_assert_strategy(self._distribute_strategy):
      f = kwargs.pop("f")
      if distribution_strategy_context.in_cross_replica_context():
        update_device = distribute_lib.get_update_device()
        if update_device is not None:
          # We are calling an assign function in an update context.
          return f(self._v, *args, **kwargs)

        # We are calling an assign function in cross replica context, wrap it in
        # an update call.
        return self._distribute_strategy.extended.update(
            self, f, args=args, kwargs=kwargs)
      else:
        replica_context = distribution_strategy_context.get_replica_context()
        assert replica_context
        # We are calling an assign function in replica context.
        # We reduce the value we want to assign/add/sub. More details about how
        # we handle the different use cases can be found in the _reduce method.
        # We call the function with the reduced value.
        if self._aggregation == vs.VariableAggregation.NONE:
          raise ValueError(_aggregation_error_msg.format(
              variable_type="AggregatingVariable"))

        def merge_fn(strategy, value, *other_args, **other_kwargs):
          v = _apply_aggregation(strategy, value, self._aggregation, self)
          return strategy.extended.update(
              self, f, args=(v,) + other_args, kwargs=other_kwargs)

        return replica_context.merge_call(merge_fn, args=args, kwargs=kwargs)

  def assign_sub(self, *args, **kwargs):
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return self._assign_func(f=assign_sub_fn, *args, **kwargs)

  def assign_add(self, *args, **kwargs):
    assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
    return self._assign_func(f=assign_add_fn, *args, **kwargs)

  def assign(self, *args, **kwargs):
    assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
    return self._assign_func(f=assign_fn, *args, **kwargs)

  @property
  def initializer(self):
    return self._v.initializer

  def initialized_value(self):
    return self._v.initialized_value()

  @property
  def initial_value(self):
    return self._v.initial_value

  @property
  def op(self):
    return self._v.op

  def read_value(self):
    return self._v.read_value()

  def eval(self, session=None):
    return self._v.eval(session)

  @property
  def graph(self):
    return self._v.graph

  @property
  def device(self):
    return self._v.device

  @property
  def shape(self):
    return self._v.shape

  @property
  def aggregation(self):
    return self._aggregation

  @property
  def name(self):
    return self._v.name

  @property
  def dtype(self):
    return self._v.dtype

  # TODO(josh11b): Test saving & restoring.
  def _gather_saveables_for_checkpoint(self):
    return {trackable.VARIABLE_VALUE_KEY: self._v}

  # pylint: disable=multiple-statements
  def __add__(self, o): return self._v + o
  def __radd__(self, o): return o + self._v
  def __sub__(self, o): return self._v - o
  def __rsub__(self, o): return o - self._v
  def __mul__(self, o): return self._v * o
  def __rmul__(self, o): return o * self._v
  def __truediv__(self, o): return self._v / o
  def __rtruediv__(self, o): return o / self._v
  def __floordiv__(self, o): return self._v // o
  def __rfloordiv__(self, o): return o // self._v
  def __mod__(self, o): return self._v % o
  def __rmod__(self, o): return o % self._v
  def __lt__(self, o): return self._v < o
  def __le__(self, o): return self._v <= o
  def __gt__(self, o): return self._v > o
  def __ge__(self, o): return self._v >= o
  def __and__(self, o): return self._v & o
  def __rand__(self, o): return o & self._v
  def __or__(self, o): return self._v | o
  def __ror__(self, o): return o | self._v
  def __xor__(self, o): return self._v ^ o
  def __rxor__(self, o): return o ^ self._v
  def __getitem__(self, o): return self._v[o]
  def __pow__(self, o, modulo=None): return pow(self._v, o, modulo)
  def __rpow__(self, o): return pow(o, self._v)
  def __invert__(self): return ~self._v
  def __neg__(self): return -self._v
  def __abs__(self): return abs(self._v)

  def __div__(self, o):
    try:
      return self._v.__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    try:
      return self._v.__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    try:
      return self._v.__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    try:
      return self._v.__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __str__(self):
    return str(self._v)

  def __repr__(self):
    return repr(self._v)

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_aggregate(var, dtype=None, name=None, as_ref=False):
  return ops.convert_to_tensor(var.get(), dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(
    AggregatingVariable, _tensor_conversion_aggregate)
ops.register_dense_tensor_like_type(AggregatingVariable)
