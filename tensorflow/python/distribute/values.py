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

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import saver
from tensorflow.python.training.checkpointable import base as checkpointable
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


class DistributedValues(object):
  """Holds a map from device to values. Either PerReplica or Mirrored."""

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

  # TODO(josh11b): Replace unwrap with this?
  @property
  def values(self):
    return self._values

  @property
  def is_tensor_like(self):
    for v in self._values:
      if not tensor_util.is_tensor(v):
        return False
    return True

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
    # TODO(priyag): This needs to be made robust against pitfalls from mix use
    # __getattr__ and @property. See b/120402273.
    return getattr(self.get(), name)

  # pylint: disable=multiple-statements
  def __add__(self, o): return self.get() + o
  def __radd__(self, o): return o + self.get()
  def __sub__(self, o): return self.get() - o
  def __rsub__(self, o): return o - self.get()
  def __mul__(self, o): return self.get() * o
  def __rmul__(self, o): return o * self.get()
  def __truediv__(self, o): return self.get() / o
  def __rtruediv__(self, o): return o / self.get()

  def __floordiv__(self, o):
    return self.get() // o

  def __rfloordiv__(self, o): return o // self.get()
  def __mod__(self, o): return self.get() % o
  def __rmod__(self, o): return o % self.get()
  def __lt__(self, o): return self.get() < o
  def __le__(self, o): return self.get() <= o
  def __gt__(self, o): return self.get() > o
  def __ge__(self, o): return self.get() >= o
  def __and__(self, o): return self.get() & o
  def __rand__(self, o): return o & self.get()
  def __or__(self, o): return self.get() | o
  def __ror__(self, o): return o | self.get()
  def __xor__(self, o): return self.get() ^ o
  def __rxor__(self, o): return o ^ self.get()
  def __getitem__(self, o): return self.get()[o]
  def __pow__(self, o, modulo=None): return pow(self.get(), o, modulo)
  def __rpow__(self, o): return pow(o, self.get())
  def __invert__(self): return ~self.get()
  def __neg__(self): return -self.get()
  def __abs__(self): return abs(self.get())

  def __div__(self, o):
    try:
      return self.get().__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    try:
      return self.get().__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    try:
      return self.get().__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    try:
      return self.get().__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  # TODO(josh11b): Even more operator overloads.


class PerReplica(DistributedValues):
  """Holds a map from device to unsynchronized values."""
  pass


# Note that unlike PerReplica, Mirrored values inherit from
# DistributedDelegate and so can be used directly in cross-replica mode.
class Mirrored(DistributedDelegate):
  """Holds a map from device to values which are kept in sync."""

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
    return variable.assign(array_ops.identity(tensor))


def _assert_strategy(strategy):
  if not distribution_strategy_context.has_distribution_strategy():
    raise RuntimeError(
        'Need to be inside "with strategy.scope()" for %s' %
        (strategy,))
  current_strategy = distribution_strategy_context.get_distribution_strategy()
  if current_strategy is not strategy:
    raise RuntimeError(
        "Mixing different tf.distribute.Strategy objects: %s is not %s" %
        (current_strategy, strategy))


DistributedVarOp = collections.namedtuple(
    "DistributedVarOp", ["name", "graph", "type"])


class DistributedVariable(DistributedDelegate):
  """Holds a map from device to variables."""
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
  def name(self):
    return self.primary.name

  @property
  def dtype(self):
    return self.primary.dtype

  @property
  def shape(self):
    return self.primary.shape

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
                              self.primary.op.type)
    return self.get().op

  @property
  def _in_graph_mode(self):
    return self.primary._in_graph_mode   # pylint: disable=protected-access

  def read_value(self):
    return self._distribute_strategy.extended.read_var(self)

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass


ops.register_dense_tensor_like_type(DistributedVariable)


def _apply_aggregation(strategy, value, aggregation, destinations):
  if aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
    return strategy.broadcast(strategy.unwrap(value)[0],
                              destinations=destinations)
  reduce_op = reduce_util.ReduceOp.from_variable_aggregation(aggregation)
  return strategy.extended.reduce_to(reduce_op, value, destinations)


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


class MirroredVariable(DistributedVariable, Mirrored,
                       checkpointable.CheckpointableBase):
  """Holds a map from device to variables whose values are kept in sync."""

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
    _assert_strategy(self._distribute_strategy)
    f = kwargs.pop("f")
    if distribution_strategy_context.in_cross_replica_context():
      update_device = distribute_lib.get_update_device()
      if update_device is not None:
        # We are calling an assign function on the mirrored variable in an
        # update context.
        v = self.get(device=update_device)
        return f(v, *args, **kwargs)

      # We are calling assign on the mirrored variable in cross replica context,
      # use `strategy.update()` to update the variable.
      return self._distribute_strategy.update(self, f, *args, **kwargs)
    else:
      _assert_replica_context(self._distribute_strategy)
      # We are calling an assign function on the mirrored variable in replica
      # context.
      # We reduce the value we want to assign/add/sub. More details about how we
      # handle the different use cases can be found in the _reduce method.
      # We call the function on each of the mirrored variables with the reduced
      # value.
      if self._aggregation == vs.VariableAggregation.NONE:
        raise ValueError("You must specify an aggregation method to update a "
                         "MirroredVariable in Replica Context.")

      def merge_fn(strategy, value, *other_args, **other_kwargs):
        v = _apply_aggregation(strategy, value, self._aggregation, self)
        return strategy.update(self, f, v, *other_args, **other_kwargs)

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
    """Overrides CheckpointableBase method.

    This allows both name-based and object-based save and restore of
    MirroredVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """
    def _saveable_factory(name=self._common_name):
      return _MirroredSaveable(self, self.primary, name)
    return {checkpointable.VARIABLE_VALUE_KEY: _saveable_factory}


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_mirrored(var, dtype=None, name=None, as_ref=False):
  # Try to avoid assignments to and other mutations of MirroredVariable
  # state except through a DistributionStrategy.update() call.
  assert not as_ref
  return ops.internal_convert_to_tensor(
      var.get(), dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(MirroredVariable,
                                        _tensor_conversion_mirrored)


def _enclosing_tpu_context():
  # pylint: disable=protected-access
  tpu_context = ops.get_default_graph()._get_control_flow_context()
  # pylint: enable=protected-access
  while tpu_context is not None and not isinstance(
      tpu_context, control_flow_ops.XLAControlFlowContext):
    tpu_context = tpu_context.outer_context
  return tpu_context


# TODO(jhseu): Deduplicate code. We copy code because we don't want to
# inherit from DistributedDelegate. DistributedDelegate will not work in a
# tpu.replicate() because it assumes that you're in a device context where you
# can operate on a single version of the variable, but a tpu.replicate()
# operates on all variables and is replicated during a rewrite pass.
class TPUMirroredVariable(checkpointable.CheckpointableBase):
  """Holds a map from device to TPU variables whose values are kept in sync."""

  def __init__(
      self, strategy, device_map, values, aggregation, logical_device=None):
    assert isinstance(device_map, DeviceMap)
    self._distribute_strategy = strategy
    self._device_map = device_map
    self._values = tuple(values)
    if logical_device is None:
      logical_device = device_map.logical_device_from_values(self._values)
    self._logical_device = logical_device

    # Use a weakref to make it easy to map from the contained values
    # to the container without introducing a reference cycle.
    for v in self._values:
      v._mirrored_container = weakref.ref(self)  # pylint: disable=protected-access
    self._common_name = self.primary.name.split(":")[0]
    self._aggregation = aggregation
    # Needed for GradientTape
    self._trainable = self.primary.trainable
    # Typically like `DistributedVariable`, a `TPUMirroredVariable`'s
    # initializer is composed of the initializers of the components variables.
    # However, in some cases, such as when restoring from a checkpoint, we may
    # set the _initializer_op property on the entire `TPUMirroredVariable`.
    self._initializer_op = None

  def _get(self, device=None):
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

  # TODO(josh11b): Replace unwrap with this?
  @property
  def values(self):
    return self._values

  @property
  def distribute_strategy(self):
    return self._distribute_strategy

  # pylint: disable=multiple-statements
  def __add__(self, o): return self.read_value() + o
  def __radd__(self, o): return o + self.read_value()
  def __sub__(self, o): return self.read_value() - o
  def __rsub__(self, o): return o - self.read_value()
  def __mul__(self, o): return self.read_value() * o
  def __rmul__(self, o): return o * self.read_value()
  def __truediv__(self, o): return self.read_value() / o
  def __rtruediv__(self, o): return o / self.read_value()
  def __floordiv__(self, o): return self.read_value() // o
  def __rfloordiv__(self, o): return o // self.read_value()
  def __mod__(self, o): return self.read_value() % o
  def __rmod__(self, o): return o % self.read_value()
  def __lt__(self, o): return self.read_value() < o
  def __le__(self, o): return self.read_value() <= o
  def __gt__(self, o): return self.read_value() > o
  def __ge__(self, o): return self.read_value() >= o
  def __and__(self, o): return self.read_value() & o
  def __rand__(self, o): return o & self.read_value()
  def __or__(self, o): return self.read_value() | o
  def __ror__(self, o): return o | self.read_value()
  def __xor__(self, o): return self.read_value() ^ o
  def __rxor__(self, o): return o ^ self.read_value()
  def __getitem__(self, o): return self.read_value()[o]
  def __pow__(self, o, modulo=None): return pow(self.read_value(), o, modulo)
  def __rpow__(self, o): return pow(o, self.read_value())
  def __invert__(self): return ~self.read_value()
  def __neg__(self): return -self.read_value()
  def __abs__(self): return abs(self.read_value())

  def __div__(self, o):
    try:
      return self.read_value().__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    try:
      return self.read_value().__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    try:
      return self.read_value().__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    try:
      return self.read_value().__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __str__(self):
    devices = self.devices
    debug_str = ",\n".join("  %d %s: %s" % (i, devices[i], self._values[i])
                           for i in range(len(devices)))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_str)

  def __repr__(self):
    devices = self.devices
    debug_repr = ",\n".join("  %d %s: %r" % (i, devices[i], self._values[i])
                            for i in range(len(devices)))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_repr)

  @property
  def handle(self):
    # If we're in a tpu.rewrite(), return the replicated handle.
    tpu_context = _enclosing_tpu_context()
    if tpu_context is not None:
      return tpu_context.get_replicated_var_handle(
          self._common_name, self._values)

    device = distribute_lib.get_update_device()
    if device is None:
      return self.primary.handle
    return self._get(device=device).handle

  @property
  def device(self):
    return self._get().device

  def eval(self, session=None):
    return self.primary.eval(session)

  # The arguments to update() are automatically unwrapped so the update()
  # function would normally see regular variables, not MirroredVariables.
  # However, the update function can still operate on wrapped MirroredVariables
  # through object members, captured arguments, etc. This is more likely in an
  # update_non_slot() function (like OptimizerV2._finish), which can
  # update several non-slot variables in one call.
  def _assign_func(self, *args, **kwargs):
    _assert_strategy(self._distribute_strategy)
    f = kwargs.pop("f")
    if distribution_strategy_context.in_cross_replica_context():
      if _enclosing_tpu_context() is not None:
        return self._distribute_strategy.update(self, f, *args, **kwargs)

      update_device = distribute_lib.get_update_device()
      # We are calling update on the mirrored variable in cross replica context.
      if update_device is not None:
        # We are calling an assign function on the mirrored variable in cross
        # replica context.
        v = self._get(device=update_device)
        return f(v, *args, **kwargs)

      return self._distribute_strategy.update(self, f, *args, **kwargs)
    else:
      _assert_replica_context(self._distribute_strategy)
      # We are calling an assign function on the mirrored variable in replica
      # context.
      # We reduce the value we want to assign/add/sub. More details about how we
      # handle the different use cases can be found in the _reduce method.
      # We call the function on each of the mirrored variables with the reduced
      # value.
      if self._aggregation == vs.VariableAggregation.NONE:
        raise ValueError("You must specify an aggregation method to update a "
                         "TPUMirroredVariable in Replica Context.")

      def merge_fn(strategy, value, *other_args, **other_kwargs):
        v = _apply_aggregation(strategy, value, self._aggregation, self)
        return strategy.update(self, f, v, *other_args, **other_kwargs)

      return distribution_strategy_context.get_replica_context().merge_call(
          merge_fn, args=args, kwargs=kwargs)

  @contextlib.contextmanager
  def _handle_graph(self, handle):
    # Note: might have an eager tensor but not be executing eagerly when
    # building functions.
    if (context.executing_eagerly() or isinstance(handle, ops.EagerTensor)
        or ops.has_default_graph()):
      yield
    else:
      with handle.graph.as_default():
        yield

  @property
  def trainable(self):
    return self._trainable

  def _read_variable_op(self, parent_op=None):
    if self.trainable:
      tape.variable_accessed(self)
    if parent_op is not None:
      with ops.control_dependencies([parent_op]):
        return gen_resource_variable_ops.read_variable_op(
            self.handle, self.dtype)

    return gen_resource_variable_ops.read_variable_op(
        self.handle, self.dtype)

  def read_value(self):
    return self._read_variable_op()

  def assign_sub(self, *args, **kwargs):
    def assign_sub_fn(var, delta, **kw):
      name = kw.pop("name", None)
      read_value = kw.pop("read_value", True)
      with self._handle_graph(var.handle):
        op = gen_resource_variable_ops.assign_sub_variable_op(
            var.handle, ops.convert_to_tensor(delta, dtype=self.dtype),
            name=name)
      if read_value:
        return self._read_variable_op(parent_op=op)
      return op

    return self._assign_func(f=assign_sub_fn, *args, **kwargs)

  def assign_add(self, *args, **kwargs):
    def assign_add_fn(var, delta, **kw):
      name = kw.pop("name", None)
      read_value = kw.pop("read_value", True)
      with self._handle_graph(var.handle):
        op = gen_resource_variable_ops.assign_add_variable_op(
            var.handle, ops.convert_to_tensor(delta, dtype=self.dtype),
            name=name)
      if read_value:
        return self._read_variable_op(parent_op=op)
      return op

    return self._assign_func(f=assign_add_fn, *args, **kwargs)

  def assign(self, *args, **kwargs):
    def assign_fn(var, value, **kw):
      name = kw.pop("name", None)
      read_value = kw.pop("read_value", True)
      with self._handle_graph(var.handle):
        op = gen_resource_variable_ops.assign_variable_op(
            var.handle, ops.convert_to_tensor(value, dtype=self.dtype),
            name=name)
      if read_value:
        return self._read_variable_op(parent_op=op)
      return op

    return self._assign_func(f=assign_fn, *args, **kwargs)

  @property
  def aggregation(self):
    return self._aggregation

  @property
  def constraint(self):
    return None

  @property
  def initializer(self):
    if self._initializer_op:
      init_op = self._initializer_op
    else:
      init_op = control_flow_ops.group(tuple(
          v.initializer for v in self._values))
    return init_op

  @property
  def graph(self):
    return self.primary.graph

  @property
  def _shared_name(self):
    return self._common_name

  @property
  def _unique_id(self):
    return self.primary._unique_id  # pylint: disable=protected-access

  @property
  def name(self):
    return self.primary.name

  @property
  def dtype(self):
    return self.primary.dtype

  @property
  def shape(self):
    return self.primary.shape

  def get_shape(self):
    return self.primary.get_shape()

  def to_proto(self, export_scope=None):
    return self.primary.to_proto(export_scope=export_scope)

  def _get_cross_replica(self):
    device = device_util.canonicalize(device_util.current())
    replica = self._device_map.replica_for_device(device)
    if replica is None:
      return self.primary
    return self._values[replica]

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.in_cross_replica_context():
      return self.primary._as_graph_element()
    return self._read_variable_op()

  def _gather_saveables_for_checkpoint(self):
    """Overrides CheckpointableBase method.

    This allows both name-based and object-based save and restore of
    MirroredVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """
    def _saveable_factory(name=self._common_name):
      return _MirroredSaveable(self, self.primary, name)
    return {checkpointable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  # Needed to pass ResourceVariable checks.
  @property
  def op(self):
    return self.primary.op

  # pylint: disable=protected-access
  @property
  def _save_slice_info(self):
    return self.primary._save_slice_info

  def _get_save_slice_info(self):
    return self.primary._get_save_slice_info()

  def _set_save_slice_info(self, save_slice_info):
    return self.primary._set_save_slice_info(save_slice_info)
  # pylint: enable=protected-access

  @property
  def _in_graph_mode(self):
    return self.primary._in_graph_mode   # pylint: disable=protected-access

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if _enclosing_tpu_context() is None:
      return self._get()._dense_var_to_tensor(dtype, name, as_ref)
    # pylint: enable=protected-access
    if dtype is not None and dtype != self.dtype:
      return math_ops.cast(self.read_value(), dtype)
    if as_ref:
      return self.handle
    else:
      return self.read_value()

  def is_initialized(self, name=None):
    """Identifies if all the component variables are initialized.

    Args:
      name: Name of the final `logical_and` op.

    Returns:
      The op that evaluates to True or False depending on if all the
      component variables are initialized.
    """
    # TODO(jhseu): Do we need TPU context implementation?

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


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_tpu_mirrored(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(TPUMirroredVariable,
                                        _tensor_conversion_tpu_mirrored)
ops.register_dense_tensor_like_type(TPUMirroredVariable)


class _ReplicaLocalSaveable(saver.BaseSaverBuilder.SaveableObject):
  """Class for defining how to restore a ReplicaLocalVariable."""

  def __init__(self, replica_local_variable, name):
    self._replica_local_variable = replica_local_variable
    # We use a callable so that we don't have to evaluate this expression
    # in the case where we are trying to restore instead of save.
    def tensor():
      strategy = replica_local_variable.distribute_strategy
      return strategy.extended.read_var(replica_local_variable)

    spec = saver.BaseSaverBuilder.SaveSpec(
        tensor=tensor,
        slice_spec="",
        name=name,
        dtype=replica_local_variable.dtype)
    super(_ReplicaLocalSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return self._replica_local_variable.assign(tensor)


def _assert_replica_context(strategy):
  replica_context = distribution_strategy_context.get_replica_context()
  if not replica_context:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")
  if replica_context.strategy is not strategy:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")


class ReplicaLocalVariable(DistributedVariable, PerReplica,
                           checkpointable.CheckpointableBase):
  """Holds a map from device to variables whose values are reduced on save."""

  def __init__(
      self, strategy, device_map, values, aggregation, logical_device=None):
    self._aggregation = aggregation
    super(ReplicaLocalVariable, self).__init__(
        strategy, device_map, values, logical_device=logical_device)

  def assign_sub(self, *args, **kwargs):
    _assert_replica_context(self._distribute_strategy)
    return self.get().assign_sub(*args, **kwargs)

  def assign_add(self, *args, **kwargs):
    _assert_replica_context(self._distribute_strategy)
    return self.get().assign_add(*args, **kwargs)

  def assign(self, *args, **kwargs):
    if distribution_strategy_context.in_cross_replica_context():
      # To preserve the sum across save and restore, we have to divide the
      # total across all devices when restoring a variable that was summed
      # when saving.
      tensor = args[0]
      if self._aggregation == vs.VariableAggregation.SUM:
        tensor *= 1. / len(self.devices)
      return control_flow_ops.group(tuple(
          _assign_on_device(v.device, v, tensor) for v in self._values))
    else:
      _assert_replica_context(self._distribute_strategy)
      return self.get().assign(*args, **kwargs)

  @property
  def aggregation(self):
    return self._aggregation

  def _get_cross_replica(self):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      return self.primary
    # TODO(josh11b): Use a strategy-specific method.
    total = math_ops.add_n(self._values)
    if self._aggregation == vs.VariableAggregation.MEAN:
      return total * (1./ len(self._values))
    return total

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.in_cross_replica_context():
      return self._get_cross_replica()
    return self.get()._as_graph_element()

  def _gather_saveables_for_checkpoint(self):
    """Overrides CheckpointableBase method.

    This allows both name-based and object-based save and restore of
    ReplicaLocalVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """
    def _saveable_factory(name=self._common_name):
      return _ReplicaLocalSaveable(self, name)
    return {checkpointable.VARIABLE_VALUE_KEY: _saveable_factory}


# Register a conversion function for ReplicaLocalVariable which allows as_ref to
# be true.
def _tensor_conversion_replica_local(var, dtype=None, name=None, as_ref=False):
  return ops.internal_convert_to_tensor(
      var.get(), dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(ReplicaLocalVariable,
                                        _tensor_conversion_replica_local)


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
  #   ReplicaLocalVariable, and same_id means it is the same across all
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
  # same MirroredVariable (or ReplicaLocalVariable). In this case we
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
    return x.values[replica_id] if isinstance(x, DistributedValues) else x

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
  # TODO(josh11b): Replace "Mirrored" here with a function that does the following
  # so we can avoid all these nest operations.
  regrouped = regroup(device_map, updates, Mirrored)
  if not group:
    return nest.map_structure(extended._unwrap, regrouped)  # pylint: disable=protected-access
  grouped_flat = []
  for u in nest.flatten(regrouped):
    if isinstance(u, DistributedValues):
      g = extended._group(u)  # pylint: disable=protected-access
      if u.is_tensor_like:
        # Make sure we run all updates. Without this, something like
        # session.run(extended.update(...)) may only update one replica.
        values = []
        for d in u.devices:
          with ops.device(d), ops.control_dependencies([g]):
            values.append(array_ops.identity(u.get(d)))
        g = Mirrored(u.device_map, values)
    else:
      g = u
    grouped_flat.append(g)
  return nest.pack_sequence_as(regrouped, grouped_flat)


class InputWorkers(object):
  """A 1-to-many mapping from input worker devices to compute devices."""

  def __init__(self, device_map, worker_device_pairs=None, logical_device=0):
    """Initialize an `InputWorkers` object.

    Args:
      device_map: A `DeviceMap` with the computation devices fed by the
        input workers.
      worker_device_pairs: A sequence of pairs:
        `(input device, a tuple of compute devices fed by that input device)`.
      logical_device: The logical device of `device_map` to feed.
    """
    self._device_map = device_map
    self._logical_device = logical_device
    if worker_device_pairs is None:
      worker_device_pairs = ((
          device_util.canonicalize("/device:CPU:0"),
          device_map.logical_to_actual_devices(logical_device)),)
    self._input_worker_devices = tuple(d for d, _ in worker_device_pairs)
    self._fed_devices = tuple(tuple(device_util.canonicalize(d) for d in f)
                              for _, f in worker_device_pairs)
    flattened = tuple(d for l in self._fed_devices for d in l)
    assert (flattened ==
            device_map.logical_to_actual_devices(logical_device)), (
                "flattened: %s logical device %d: %s" %
                (flattened, logical_device,
                 device_map.logical_to_actual_devices(logical_device)))

  @property
  def device_map(self):
    return self._device_map

  @property
  def logical_device(self):
    return self._logical_device

  @property
  def num_workers(self):
    return len(self._input_worker_devices)

  @property
  def worker_devices(self):
    return self._input_worker_devices

  def compute_devices_for_worker(self, worker_index):
    return self._fed_devices[worker_index]

  def __repr__(self):
    devices = self.worker_devices
    debug_repr = ",\n".join("  %d %s: %s" %
                            (i, devices[i], self._fed_devices[i])
                            for i in range(len(devices)))
    return "%s:{\n%s\n  device_map: %s}" % (
        self.__class__.__name__, debug_repr, self._device_map)


class PerReplicaDataIterator(object):
  """An iterator (like `tf.data.Iterator`) into a `PerReplicaDataset`."""

  def __init__(self, iterator, input_workers, worker_index, prefetch_on_device):
    assert isinstance(input_workers, InputWorkers)
    self._iterator = iterator
    self._input_workers = input_workers
    self._worker_index = worker_index
    self._prefetch_on_device = prefetch_on_device

  @property
  def initializer(self):
    return self._iterator.initializer

  def get_next_as_list(self, name=None):
    """Scatter the input across devices."""
    if self._prefetch_on_device:
      data_list = self._iterator.get_next()
    else:
      batch = self._iterator.get_next(name=name)
      data_list = []
      def get_ith(i):
        return lambda x: x[i]

      devices = self._input_workers.compute_devices_for_worker(
          self._worker_index)
      for i, d in enumerate(devices):
        v = nest.map_structure(get_ith(i), batch)
        if context.executing_eagerly():
          with ops.device(d):
            v = nest.map_structure(array_ops.identity, v)
        data_list.append(v)

    return data_list

  def get_next(self, name=None):
    assert self._input_workers.num_workers == 1
    data_list = self.get_next_as_list(name)
    return regroup(self._input_workers.device_map, data_list)

  @property
  def output_classes(self):
    return self._iterator.output_classes

  @property
  def output_shapes(self):
    return self._iterator.output_shapes

  @property
  def output_types(self):
    return self._iterator.output_types


class PerReplicaDataset(object):
  """Like `tf.data.Dataset` split devices, producing `PerReplica` data."""

  def __init__(self, dataset, input_workers, worker_index,
               prefetch_on_device=None):
    assert isinstance(input_workers, InputWorkers)
    assert worker_index is not None
    assert worker_index is not True
    assert worker_index is not False
    self._input_workers = input_workers
    self._worker_index = worker_index

    # Default to using prefetching, unless specified.
    self._prefetch_on_device = prefetch_on_device
    if self._prefetch_on_device is None:
      self._prefetch_on_device = True

    self._dataset = dataset
    if not self._prefetch_on_device:
      # TODO(priyag): If dropping remainder is not appropriate, find another
      # approach to distributing the dataset when not possible to divide evenly.
      # Possibly not an issue when we start using PartitionedDataset.
      num_replicas = len(
          self._input_workers.compute_devices_for_worker(self._worker_index))
      self._dataset = self._dataset.batch(num_replicas, drop_remainder=True)
    else:
      self._replica_devices = self._input_workers.compute_devices_for_worker(
          self._worker_index)

  def make_one_shot_iterator(self):
    """Get a one time use iterator for the distributed PerReplicaDataset."""
    # Graph mode with one shot iterator is disabled.
    if not context.executing_eagerly():
      raise ValueError("Cannot create a one shot iterator. Please use "
                       "`make_initializable_iterator()` instead.")
    if self._prefetch_on_device:
      dataset_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          self._dataset, self._replica_devices)
    else:
      dataset_iterator = dataset_ops.make_one_shot_iterator(self._dataset)
    return PerReplicaDataIterator(
        dataset_iterator,
        self._input_workers,
        self._worker_index,
        prefetch_on_device=self._prefetch_on_device)

  def make_initializable_iterator(self):
    """Get an initializable iterator for the distributed PerReplicaDataset."""
    # Eager mode generates already initialized iterators. Hence we cannot create
    # an initializable iterator.
    if context.executing_eagerly():
      raise ValueError("Cannot create initializable iterator in Eager mode. "
                       "Please use `make_one_shot_iterator` instead.")
    if self._prefetch_on_device:
      dataset_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          self._dataset, self._replica_devices)
    else:
      dataset_iterator = dataset_ops.make_initializable_iterator(self._dataset)
    return PerReplicaDataIterator(
        dataset_iterator, self._input_workers, self._worker_index,
        prefetch_on_device=self._prefetch_on_device)


class MultiWorkerDataIterator(object):
  """An iterator (like `tf.data.Iterator`) into a `MultiWorkerDataset`."""

  def __init__(self, iterators, input_workers):
    """Initialize the `MultiWorkerDataIterator` object.

    Args:
      iterators: a list of worker, iterator pairs.
      input_workers: an `InputWorkers` object.

    Raises:
      ValueError: if iterators and input_workers are not compatible.
    """
    assert isinstance(input_workers, InputWorkers)
    workers = tuple(d for d, _ in iterators)
    if workers != input_workers.worker_devices:
      raise ValueError("iterators and input_workers are not compatible. "
                       "iterator workers: %r input_workers devices: %r" %
                       (workers, input_workers.worker_devices))
    self._iterators = tuple(i for _, i in iterators)
    self._input_workers = input_workers

  @property
  def initializer(self):
    return control_flow_ops.group(
        tuple(iterator.initializer for iterator in self._iterators))

  def get_iterator(self, worker):
    for i, w in enumerate(self._input_workers.worker_devices):
      if worker == w:
        return self._iterators[i]
    return None

  @property
  def output_shapes(self):
    return self._iterators[0].output_shapes

  @property
  def output_types(self):
    return self._iterators[0].output_types

  def get_next(self, name=None):
    """Scatter the input across hosts and devices."""
    replicas = []
    for worker, iterator in zip(self._input_workers.worker_devices,
                                self._iterators):
      if name is not None:
        d = tf_device.DeviceSpec.from_string(worker)
        new_name = "%s_%s_%d" % (name, d.job, d.task)
      else:
        new_name = None
      with ops.device(worker):
        data_per_worker = iterator.get_next_as_list(name=new_name)
        # Append to replicas to get a flat list of values indexed by replica.
        replicas.extend(data_per_worker)

    return regroup(self._input_workers.device_map, replicas)


class MultiWorkerDataset(object):
  """Like a `tf.data.Dataset` that distributes data to different workers.

  Each worker gets one shard of the input dataset. This currently does not work
  in eager mode.
  """

  def __init__(self, dataset_fn, input_workers, prefetch_on_device=None,
               auto_shard=False):
    """Initialize the MultiWorkerDataset object.

    Args:
      dataset_fn: a function or a list of functions that returns a
        `tf.data.Dataset`.
      input_workers: an `InputWorkers` object.
      prefetch_on_device: whether to prefetch to devices.
      auto_shard: whether to auto-shard the dataset.
    """
    assert isinstance(input_workers, InputWorkers)
    if isinstance(dataset_fn, (list, tuple)):
      if len(dataset_fn) != input_workers.num_workers:
        raise ValueError("If `dataset_fn` is a list, it must have one entry "
                         "per worker")
    # TODO(rohanj): b/120673685 to track re-enabling auto sharding.
    if auto_shard:
      raise ValueError("Currently autosharding is not supported.")
    self._input_workers = input_workers
    self._datasets = []
    # TODO(yuefengz, priyag): support different set of jobs for input
    # processing.
    for i, worker in enumerate(input_workers.worker_devices):
      with ops.device(worker):
        if isinstance(dataset_fn, (list, tuple)):
          worker_input = dataset_fn[i]()
        else:
          worker_input = dataset_fn()
        dataset = PerReplicaDataset(worker_input, input_workers, i,
                                    prefetch_on_device=prefetch_on_device)
        self._datasets.append((worker, dataset))

  def make_one_shot_iterator(self):
    iterators = []
    for worker, dataset in self._datasets:
      with ops.device(worker):
        iterators.append((worker, dataset_ops.make_one_shot_iterator(dataset)))
    return MultiWorkerDataIterator(iterators, self._input_workers)

  def make_initializable_iterator(self):
    iterators = []
    for worker, dataset in self._datasets:
      with ops.device(worker):
        iterators.append(
            (worker, dataset_ops.make_initializable_iterator(dataset)))
    return MultiWorkerDataIterator(iterators, self._input_workers)


class InputIterator(object):
  """An input iterator, intended to be passed to `DistributionStrategy.run`."""

  def get_next(self):
    """Returns the next inputs for all replicas."""
    raise NotImplementedError("must be implemented in descendants")

  def initialize(self):
    """Initialize the underlying input dataset, when applicable.

    In eager mode, this will create a new iterator and return it.
    In graph mode, this will initialize the same underlying iterator(s).

    Users are required to call this if
    - This iterator was returned from a call to `make_input_fn_iterator` with an
      input function that returns a dataset.
    - Or this iterator was returned from a call to `make_dataset_iterator`.

    Returns:
      A list of initialization ops to be executed.
    """
    raise NotImplementedError("must be implemented in descendants")


class InputIteratorImpl(InputIterator):
  """Common implementation for all input iterators."""

  def __init__(self, input_workers, iterators):
    assert isinstance(input_workers, InputWorkers)
    if not input_workers.worker_devices:
      raise ValueError("Should have at least one worker for input iterator.")

    self._iterators = iterators
    self._input_workers = input_workers

  def get_next(self, name=None):
    """Returns the next input from the iterator for all replicas."""
    replicas = []
    for i, worker in enumerate(self._input_workers.worker_devices):
      if name is not None:
        d = tf_device.DeviceSpec.from_string(worker)
        new_name = "%s_%s_%d" % (name, d.job, d.task)
      else:
        new_name = None
      with ops.device(worker):
        # Make `replicas` a flat list of values across all replicas.
        replicas.extend(self._iterators[i].get_next_as_list(new_name))

    return regroup(self._input_workers.device_map, replicas)

  def initialize(self):
    """Initialze underlying iterators.

    Returns:
      A list of any initializer ops that should be run.
    """
    init_ops = []
    for it in self._iterators:
      init_ops.extend(it.initialize())
    return init_ops

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_classes(self):
    return self._iterators[0].output_classes

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_shapes(self):
    return self._iterators[0].output_shapes

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_types(self):
    return self._iterators[0].output_types

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  def get_iterator(self, worker):
    for i, w in enumerate(self._input_workers.worker_devices):
      if worker == w:
        return self._iterators[i]
    return None


class InputFunctionIterator(InputIteratorImpl):
  """Iterator created from input function."""

  def __init__(self, input_fn, input_workers, input_contexts):
    """Make an iterator for input provided via an input function.

    Currently implements PER_WORKER mode, in which the `input_fn` is called
    once on each worker.

    TODO(priyag): Add other replication modes.
    TODO(priyag): Allow taking input function that returns a callable that
    returns nest of tensors.

    Args:
      input_fn: Input function that returns a `tf.data.Dataset` object.
      input_workers: an `InputWorkers` object.
      input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `input_fn`. Length and order should match worker order in
        `worker_device_pairs`.
    """
    assert isinstance(input_workers, InputWorkers)
    if input_workers.num_workers != len(input_contexts):
      raise ValueError(
          "Number of input workers (%d) is not same as number of "
          "input_contexts (%d)" %
          (input_workers.num_workers, len(input_contexts)))

    iterators = []
    for i, ctx in enumerate(input_contexts):
      worker = input_workers.worker_devices[i]
      with ops.device(worker):
        result = input_fn(ctx)
        if not isinstance(result, dataset_ops.DatasetV2):
          raise ValueError("input_fn must return a tf.data.Dataset.")
        devices = input_workers.compute_devices_for_worker(i)
        iterator = _SingleWorkerDatasetIterator(result, worker, devices)
        iterators.append(iterator)

    super(InputFunctionIterator, self).__init__(input_workers, iterators)


class DatasetIterator(InputIteratorImpl):
  """Iterator created from input dataset."""

  def __init__(self, dataset, input_workers, split_batch_by=None):
    """Make an iterator for the dataset on given devices.

    If `split_batch_by` is not None, we "split" each batch of the
    dataset by `split_batch_by` value. To achieve this, we first unbatch the
    input dataset and then rebatch it with the per replica batch size that is
    calculated using `global_batch_size // split_batch_by`.
    The currently supported datasets are as follows:
    `dataset.batch()` is the last operation on the dataset OR
    `dataset.apply(map_and_batch)` is the last operation on the dataset OR
    `dataset.batch().prefetch()` are the last 2 operations on the dataset OR
    `dataset.apply(map_and_batch).prefetch()` are the last 2 operations.

    TODO(priyag): Support multi worker / host cases properly by cloning
    and sharding the dataset on each worker. Current setup will only work in
    some cases, such as in-graph multi worker GPU case. If the input pipeline
    has random shuffling (with a different seed on each worker), each worker
    will see random input from the same overall dataset in each step. Otherwise,
    each worker will see the same input in each step.

    Args:
      dataset: `tf.data.Dataset` that will be used as the input source.
      input_workers: an `InputWorkers` object.
      split_batch_by: Optional integer. If present, we "split" each batch of the
        dataset by `split_batch_by` value.
    """
    assert isinstance(input_workers, InputWorkers)
    if split_batch_by:
      dataset = _split_dataset_batch(dataset, split_batch_by)

    iterators = []
    for i, worker in enumerate(input_workers.worker_devices):
      with ops.device(worker):
        worker_devices = input_workers.compute_devices_for_worker(i)
        cloned_dataset = dataset
        if not context.executing_eagerly():
          cloned_dataset = input_ops._clone_dataset(dataset)  # pylint: disable=protected-access
        iterator = _SingleWorkerDatasetIterator(cloned_dataset, worker,
                                                worker_devices)
        iterators.append(iterator)

    super(DatasetIterator, self).__init__(input_workers, iterators)


class _SingleWorkerDatasetIterator(object):
  """Iterator for a single `tf.data.Dataset`."""

  def __init__(self, dataset, worker, devices):
    """Create iterator for the `dataset` to fetch data to worker's `devices` .

    `MultiDeviceIterator` is used to prefetch input to the devices on the
    given worker.

    Args:
      dataset: A `tf.data.Dataset` instance.
      worker: Worker on which ops should be created.
      devices: Distribute data from `dataset` to these devices.
    """
    self._dataset = dataset
    self._worker = worker
    self._devices = devices
    self._make_iterator()

  def _make_iterator(self):
    """Make appropriate iterator on the dataset."""
    with ops.device(self._worker):
      self._iterator = multi_device_iterator_ops.MultiDeviceIterator(
          self._dataset, self._devices)

  def get_next_as_list(self, name=None):
    """Get next element from the underlying iterator."""
    del name
    with ops.device(self._worker):
      data_list = self._iterator.get_next()
      return data_list

  def initialize(self):
    """Initialze underlying iterator.

    In eager execution, this simply recreates the underlying iterator.
    In graph execution, it returns the initializer ops for the underlying
    iterator.

    Returns:
      A list of any initializer ops that should be run.
    """
    if context.executing_eagerly():
      self._make_iterator()
      return []
    else:
      return [self._iterator.initializer]

  @property
  def output_classes(self):
    return self._iterator.output_classes

  @property
  def output_shapes(self):
    return self._iterator.output_shapes

  @property
  def output_types(self):
    return self._iterator.output_types


def _split_dataset_batch(dataset, split_batch_by):
  """Divide a batch-ed dataset's batches into smaller batches."""
  # TODO(sourabhbajaj): Remove this in lieu of distributed datasets
  # pylint: disable=protected-access
  def _get_batch_dataset(d):
    """Get the underlying batch dataset from the dataset object."""
    if isinstance(d, dataset_ops.DatasetV1Adapter):
      d = d._dataset

    if isinstance(d, (dataset_ops.BatchDataset, batching._MapAndBatchDataset)):
      return d
    elif isinstance(d, dataset_ops.PrefetchDataset):
      return _get_batch_dataset(d._input_dataset)
    raise ValueError(
        "Unable to get batched dataset from the input dataset. `batch` "
        "`map_and_batch` need to be the last operations on the dataset. "
        "The batch operations can be followed by a prefetch.")

  batched_dataset = _get_batch_dataset(dataset)
  if isinstance(batched_dataset, dataset_ops.BatchDataset):
    batch_size = batched_dataset._batch_size
    drop_remainder = batched_dataset._drop_remainder
  elif isinstance(batched_dataset, batching._MapAndBatchDataset):
    batch_size = batched_dataset._batch_size_t
    drop_remainder = batched_dataset._drop_remainder_t

  prefetch_buffer = None
  if isinstance(dataset, dataset_ops.PrefetchDataset):
    prefetch_buffer = dataset._buffer_size
  # pylint: enable=protected-access

  if tensor_util.is_tensor(batch_size):
    batch_size = tensor_util.constant_value(batch_size)

  if tensor_util.is_tensor(drop_remainder):
    drop_remainder = tensor_util.constant_value(drop_remainder)

  if batch_size % split_batch_by:
    raise ValueError(
        "Batch size %s cannot be sharded evenly across replicas %s" % (
            batch_size, split_batch_by))
  new_batch_size = batch_size // split_batch_by

  dataset = dataset.apply(batching.unbatch())
  dataset = dataset.batch(new_batch_size, drop_remainder=drop_remainder)
  if prefetch_buffer is not None:
    dataset = dataset.prefetch(prefetch_buffer)
  return dataset


class MultiStepContext(object):
  """A context object that can be used to capture things when running steps.

  This context object is useful when running multiple steps at a time using the
  `experimental_run_steps_on_iterator` API. For e.g. it allows the user's step
  function to specify which outputs to emit at what frequency. Currently it
  supports capturing output from the last step, as well as capturing non tensor
  outputs.  In the future it will be augmented to support other use cases such
  as output each N steps.
  """

  def __init__(self):
    """Initialize an output context.

    Returns:
      A context object.
    """
    self._last_step_outputs = {}
    self._last_step_outputs_reduce_ops = {}
    self._non_tensor_outputs = {}

  @property
  def last_step_outputs(self):
    """A dictionary consisting of outputs to be captured on last step.

    Keys in the dictionary are names of tensors to be captured, as specified
    when `set_last_step_output` is called.
    Values in the dictionary are the tensors themselves. If
    `set_last_step_output` was called with a `reduce_op` for this output,
    then the value is the reduced value.

    Returns:
      A dictionary with last step outputs.
    """
    return self._last_step_outputs

  def _set_last_step_outputs(self, outputs):
    """Replace the entire dictionary of last step outputs."""
    if not isinstance(outputs, dict):
      raise ValueError("Need a dictionary to set last_step_outputs.")
    self._last_step_outputs = outputs

  def set_last_step_output(self, name, output, reduce_op=None):
    """Set `output` with `name` to be outputted from the last step.

    Args:
      name: String, name to identify the output. Doesn't need to match tensor
        name.
      output: The tensors that should be outputted with `name`. See below for
        actual types supported.
      reduce_op: Reduction method to use to reduce outputs from multiple
        replicas. Required if `set_last_step_output` is called in a replica
        context. Optional in cross_replica_context.
        When present, the outputs from all the replicas are reduced using the
        current distribution strategy's `reduce` method. Hence, the type of
        `output` must be what's supported by the corresponding `reduce` method.
        For e.g. if using MirroredStrategy and reduction is set, output
        must be a `PerReplica` value.
        The reduce method is also recorded in a dictionary
        `_last_step_outputs_reduce_ops` for later interpreting of the
        outputs as already reduced or not.
    """
    if distribution_strategy_context.in_cross_replica_context():
      self._last_step_outputs_reduce_ops[name] = reduce_op
      if reduce_op is None:
        self._last_step_outputs[name] = output
      else:
        distribution = distribution_strategy_context.get_distribution_strategy()
        self._last_step_outputs[name] = distribution.reduce(reduce_op, output)
    else:
      assert reduce_op is not None
      def merge_fn(distribution, value):
        self._last_step_outputs[name] = distribution.reduce(reduce_op, value)
        # Setting this inside the `merge_fn` because all replicas share the same
        # context object, so it's more robust to set it only once (even if all
        # the replicas are trying to set the same value).
        self._last_step_outputs_reduce_ops[name] = reduce_op

      distribution_strategy_context.get_replica_context().merge_call(
          merge_fn, args=(output,))

  @property
  def non_tensor_outputs(self):
    """A dictionary consisting of any non tensor outputs to be captured."""
    return self._non_tensor_outputs

  def set_non_tensor_output(self, name, output):
    """Set `output` with `name` to be captured as a non tensor output."""
    if distribution_strategy_context.in_cross_replica_context():
      self._non_tensor_outputs[name] = output
    else:
      def merge_fn(distribution, value):
        # NOTE(priyag): For non tensor outputs, we simply return all the values
        # in a list as reduction doesn't make sense on non tensors.
        self._non_tensor_outputs[name] = distribution.unwrap(value)
      distribution_strategy_context.get_replica_context().merge_call(
          merge_fn, args=(output,))


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


# TODO(josh11b): Descend from Variable.
class AggregatingVariable(checkpointable.CheckpointableBase):
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
    _assert_strategy(self._distribute_strategy)
    f = kwargs.pop("f")
    if distribution_strategy_context.in_cross_replica_context():
      update_device = distribute_lib.get_update_device()
      if update_device is not None:
        # We are calling an assign function in an update context.
        return f(self._v, *args, **kwargs)

      # We are calling an assign function in cross replica context, wrap it in
      # an update call.
      return self._distribute_strategy.update(self, f, *args, **kwargs)
    else:
      replica_context = distribution_strategy_context.get_replica_context()
      assert replica_context
      # We are calling an assign function in replica context.
      # We reduce the value we want to assign/add/sub. More details about how we
      # handle the different use cases can be found in the _reduce method.
      # We call the function with the reduced value.
      if self._aggregation == vs.VariableAggregation.NONE:
        raise ValueError("You must specify an aggregation method to update a "
                         "a variable in replica context.")

      def merge_fn(strategy, value, *other_args, **other_kwargs):
        v = _apply_aggregation(strategy, value, self._aggregation, self)
        return strategy.update(self, f, v, *other_args, **other_kwargs)

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
    return {checkpointable.VARIABLE_VALUE_KEY: self._v}

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
  return ops.internal_convert_to_tensor(
      var.get(), dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(
    AggregatingVariable, _tensor_conversion_aggregate)
ops.register_dense_tensor_like_type(AggregatingVariable)
