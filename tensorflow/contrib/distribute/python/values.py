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
"""Various classes representing distributed values.

See go/tf-distribution-strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import weakref
import six

from tensorflow.contrib.distribute.python import input_ops
from tensorflow.contrib.distribute.python import prefetching_ops_v2
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
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import device_util
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.training import saver
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest


# pylint: disable=line-too-long
# TODO(josh11b): Should device values be strings or DeviceSpec objects?
# Not sure DeviceSpec objects are usable as a dict key.
class DistributedValues(object):
  """Holds a map from device to values. Either PerDevice or Mirrored."""

  def __init__(self, index):
    self._index = {device_util.canonicalize(key): value
                   for key, value in six.iteritems(index)}

  def get(self, device=None):
    """Returns the value for the current device or raises a ValueError."""
    if device is None:
      tower_context = distribution_strategy_context.get_tower_context()
      if tower_context:
        device = tower_context.device
      else:
        device = distribute_lib.get_update_device()
        if device is None:
          return self._get_cross_tower()
    device = device_util.canonicalize(device)
    try:
      return self._index[device]
    except KeyError as e:
      six.raise_from(
          ValueError("Device %s not found in %s (current device %s)" %
                     (device, self._index.keys(), device_util.current())), e)

  def on_device(self, device):
    device = device_util.canonicalize(device)
    return device in self._index

  @property
  def devices(self):
    return list(self._index.keys())

  @property
  def is_tensor_like(self):
    for v in self._index.values():
      if not tensor_util.is_tensor(v):
        return False
    return True

  def __str__(self):
    return "%s:%s" % (self.__class__.__name__, self._index)

  def __repr__(self):
    return "%s(%r)" % (self.__class__.__name__, self._index)

  # TODO(josh11b): Possibly make an accessor for _index for use by
  # DistributionStrategy implementations.


class DistributedDelegate(DistributedValues):
  """A map from device to values; acts as the same type as the values."""

  def __init__(self, index):
    super(DistributedDelegate, self).__init__(index)

  def __getattr__(self, name):
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
  def __floordiv__(self, o): return self.get() // o
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


class PerDevice(DistributedValues):
  """Holds a map from device to unsynchronized values."""
  pass


# Note that unlike PerDevice, Mirrored values inherit from
# DistributedDelegate and so can be used directly in cross-tower mode.
class Mirrored(DistributedDelegate):
  """Holds a map from device to values which are kept in sync."""

  def _get_cross_tower(self):
    device = device_util.canonicalize(device_util.current())
    if device in self._index:
      return self._index[device]
    return list(self._index.values())[0]

  def _as_graph_element(self):
    obj = self.get()
    # pylint: disable=protected-access
    conv_fn = getattr(obj, "_as_graph_element", None)
    if conv_fn and callable(conv_fn):
      return conv_fn()
    return obj


def _assign_on_device(device, variable, tensor):
  with ops.device(device):
    return variable.assign(array_ops.identity(tensor))


DistributedVarOp = collections.namedtuple(
    "DistributedVarOp", ["name", "graph", "type"])


class DistributedVariable(DistributedDelegate):
  """Holds a map from device to variables."""
  # TODO(josh11b): Support changing the set of variables if e.g. if new
  # devices are joining or a device is to leave.

  def __init__(self, index):
    # Child class must set self._primary_var before calling
    # super(...).__init__(index).
    self._common_name = self._primary_var.name.split(":")[0]
    # Use a weakref to make it easy to map from the contained values
    # to the container without introducing a reference cycle.
    for v in six.itervalues(index):
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
    super(DistributedVariable, self).__init__(index)

  def is_initialized(self, name=None):
    """Identifies if all the component variables are initialized.

    Args:
      name: Name of the final `logical_and` op.

    Returns:
      The op that evaluates to True or False depending on if all the
      component variables are initialized.
    """
    # We have to cast the self._index.values() to a `list` because when we
    # use `model_to_estimator` to run tf.keras models, self._index.values() is
    # of type `dict_values` and not `list`.
    values_list = list(self._index.values())
    result = values_list[0].is_initialized()
    # We iterate through the list of values except the last one to allow us to
    # name the final `logical_and` op the same name that is passed by the user
    # to the `is_initialized` op. For distributed variables, the
    # `is_initialized` op is a `logical_and` op.
    for v in values_list[1:-1]:
      result = math_ops.logical_and(result, v.is_initialized())
    result = math_ops.logical_and(result, values_list[-1].is_initialized(),
                                  name=name)
    return result

  @property
  def initializer(self):
    if self._initializer_op:
      init_op = self._initializer_op
    else:
      # return grouped ops of all the var initializations of component values of
      # the mirrored variable
      init_op = control_flow_ops.group(
          [v.initializer for v in self._index.values()])
    return init_op

  @property
  def graph(self):
    return self._primary_var.graph

  @property
  def _shared_name(self):
    return self._common_name

  @property
  def _unique_id(self):
    return self._primary_var._unique_id   # pylint: disable=protected-access

  @property
  def name(self):
    return self._primary_var.name

  @property
  def dtype(self):
    return self._primary_var.dtype

  @property
  def shape(self):
    return self._primary_var.shape

  def get_shape(self):
    return self._primary_var.get_shape()

  def to_proto(self, export_scope=None):
    return self._primary_var.to_proto(export_scope=export_scope)

  @property
  def op(self):
    # We want cross-tower code that does some var.op.X calls
    # to work (even if the current device isn't in self.devices), but
    # other uses of var.op in a cross-tower context to fail.
    if distribution_strategy_context.get_cross_tower_context():
      return DistributedVarOp(self._primary_var.op.name,
                              self._primary_var.op.graph,
                              self._primary_var.op.type)
    return self.get().op

  @property
  def _in_graph_mode(self):
    return self._primary_var._in_graph_mode   # pylint: disable=protected-access

  def read_value(self):
    return distribution_strategy_context.get_distribution_strategy().read_var(
        self)

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass


ops.register_dense_tensor_like_type(DistributedVariable)


class _MirroredSaveable(saver.BaseSaverBuilder.ResourceVariableSaveable):
  """Class for defining how to restore a MirroredVariable."""

  def __init__(self, mirrored_variable, primary_variable, name):
    self._mirrored_variable = mirrored_variable
    super(_MirroredSaveable, self).__init__(primary_variable, "", name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return control_flow_ops.group([
        _assign_on_device(d, v, tensor)
        for d, v in six.iteritems(self._mirrored_variable._index)])  # pylint: disable=protected-access


class MirroredVariable(DistributedVariable, Mirrored,
                       checkpointable.CheckpointableBase):
  """Holds a map from device to variables whose values are kept in sync."""

  def __init__(self, index, primary_var, aggregation):
    self._primary_var = primary_var
    self._aggregation = aggregation
    super(MirroredVariable, self).__init__(index)

  # The arguments to update() are automatically unwrapped so the update()
  # function would normally see regular variables, not MirroredVariables.
  # However, the update function can still operate on wrapped MirroredVariables
  # through object members, captured arguments, etc. This is more likely in an
  # update_non_slot() function (like OptimizerV2._finish), which can
  # update several non-slot variables in one call.
  def _assign_func(self, *args, **kwargs):
    f = kwargs.pop("f")
    if distribution_strategy_context.get_cross_tower_context():
      update_device = distribute_lib.get_update_device()
      if update_device is not None:
        # We are calling an assign function on the mirrored variable in an
        # update context.
        v = self.get(device=update_device)
        return f(v, *args, **kwargs)

      # We are calling assign on the mirrored variable in cross tower context,
      # use update to update the variable.
      strategy = distribution_strategy_context.get_distribution_strategy()
      return strategy.update(self, f, *args, **kwargs)
    else:
      _assert_tower_context()
      # We are calling an assign function on the mirrored variable in tower
      # context.
      # We reduce the value we want to assign/add/sub. More details about how we
      # handle the different use cases can be found in the _reduce method.
      # We call the function on each of the mirrored variables with the reduced
      # value.
      if self._aggregation == vs.VariableAggregation.NONE:
        raise ValueError("You must specify an aggregation method to update a "
                         "MirroredVariable in Tower Context.")

      def merge_fn(strategy, value, *other_args, **other_kwargs):
        return strategy.update(
            self, f,
            strategy.reduce(
                aggregation=self._aggregation, value=value, destinations=self),
            *other_args, **other_kwargs)

      return distribution_strategy_context.get_tower_context().merge_call(
          merge_fn, *args, **kwargs)

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

  def _get_cross_tower(self):
    device = device_util.canonicalize(device_util.current())
    if device in self._index:
      return array_ops.identity(self._index[device])
    return array_ops.identity(self._primary_var)

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.get_cross_tower_context():
      return self._primary_var._as_graph_element()
    return self.get()._as_graph_element()

  def _gather_saveables_for_checkpoint(self):
    """Overrides CheckpointableBase method.

    This allows both name-based and object-based save and restore of
    MirroredVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """
    def _saveable_factory(name=self._common_name):
      return _MirroredSaveable(self, self._primary_var, name)
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

  def __init__(self, index, primary_var, aggregation):
    # Use a weakref to make it easy to map from the contained values
    # to the container without introducing a reference cycle.
    for v in six.itervalues(index):
      v._mirrored_container = weakref.ref(self)  # pylint: disable=protected-access
    self._index = {device_util.canonicalize(key): value
                   for key, value in six.iteritems(index)}
    self._primary_var = primary_var
    self._common_name = self._primary_var.name.split(":")[0]
    self._aggregation = aggregation
    # Needed for GradientTape
    self._trainable = self._primary_var.trainable
    # Typically like `DistributedVariable`, a `TPUMirroredVariable`'s
    # initializer is composed of the initializers of the components variables.
    # However, in some cases, such as when restoring from a checkpoint, we may
    # set the _initializer_op property on the entire `TPUMirroredVariable`.
    self._initializer_op = None

  def _get(self, device=None):
    """Returns the value for the current device or raises a ValueError."""
    if device is None:
      tower_context = distribution_strategy_context.get_tower_context()
      if tower_context:
        device = tower_context.device
      else:
        device = distribute_lib.get_update_device()
        if device is None:
          return self._get_cross_tower()
    device = device_util.canonicalize(device)
    try:
      return self._index[device]
    except KeyError as e:
      six.raise_from(
          ValueError("Device %s not found in %s (current device %s)" %
                     (device, self._index.keys(), device_util.current())), e)

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

  @property
  def handle(self):
    # If we're in a tpu.rewrite(), return the replicated handle.
    tpu_context = _enclosing_tpu_context()
    if tpu_context is not None:
      return tpu_context.get_replicated_var_handle(
          self._common_name, nest.flatten(self._index))

    device = distribute_lib.get_update_device()
    if device is None:
      return self._primary_var.handle
    device = device_util.canonicalize(device)
    try:
      return self._index[device].handle
    except KeyError as e:
      six.raise_from(
          ValueError("Device %s not found in %s (current device %s)" %
                     (device, self._index.keys(), device_util.current())), e)

  @property
  def device(self):
    return self._get().device

  # The arguments to update() are automatically unwrapped so the update()
  # function would normally see regular variables, not MirroredVariables.
  # However, the update function can still operate on wrapped MirroredVariables
  # through object members, captured arguments, etc. This is more likely in an
  # update_non_slot() function (like OptimizerV2._finish), which can
  # update several non-slot variables in one call.
  def _assign_func(self, *args, **kwargs):
    if distribution_strategy_context.get_distribution_strategy().__class__.__name__ != "TPUStrategy":
      raise ValueError("You may only assign to a TPUMirroredVariable within a "
                       "TPUStrategy.")
    f = kwargs.pop("f")
    if distribution_strategy_context.get_cross_tower_context():
      if _enclosing_tpu_context() is not None:
        return distribution_strategy_context.get_distribution_strategy().update(
            self, f, *args, **kwargs)

      update_device = distribute_lib.get_update_device()
      # We are calling update on the mirrored variable in cross tower context.
      if update_device is not None:
        # We are calling an assign function on the mirrored variable in cross
        # tower context.
        v = self._get(device=update_device)
        return f(v, *args, **kwargs)

      return distribution_strategy_context.get_distribution_strategy().update(
          self, f, *args, **kwargs)
    else:
      _assert_tower_context()
      # We are calling an assign function on the mirrored variable in tower
      # context.
      # We reduce the value we want to assign/add/sub. More details about how we
      # handle the different use cases can be found in the _reduce method.
      # We call the function on each of the mirrored variables with the reduced
      # value.
      if self._aggregation == vs.VariableAggregation.NONE:
        raise ValueError("You must specify an aggregation method to update a "
                         "TPUMirroredVariable in Tower Context.")

      def merge_fn(strategy, value, *other_args, **other_kwargs):
        return strategy.update(
            self, f,
            strategy.reduce(
                aggregation=self._aggregation, value=value, destinations=self),
            *other_args, **other_kwargs)

      return distribution_strategy_context.get_tower_context().merge_call(
          merge_fn, *args, **kwargs)

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
      init_op = control_flow_ops.group(
          [v.initializer for v in self._index.values()])
    return init_op

  @property
  def graph(self):
    return self._primary_var.graph

  @property
  def _shared_name(self):
    return self._common_name

  @property
  def _unique_id(self):
    return self._primary_var._unique_id  # pylint: disable=protected-access

  @property
  def name(self):
    return self._primary_var.name

  @property
  def dtype(self):
    return self._primary_var.dtype

  @property
  def shape(self):
    return self._primary_var.shape

  def get_shape(self):
    return self._primary_var.get_shape()

  def to_proto(self, export_scope=None):
    return self._primary_var.to_proto(export_scope=export_scope)

  def _get_cross_tower(self):
    device = device_util.canonicalize(device_util.current())
    if device in self._index:
      return self._index[device]
    return self._primary_var

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.get_cross_tower_context():
      return self._primary_var._as_graph_element()
    return self._read_variable_op()

  def _gather_saveables_for_checkpoint(self):
    """Overrides CheckpointableBase method.

    This allows both name-based and object-based save and restore of
    MirroredVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """
    def _saveable_factory(name=self._common_name):
      return _MirroredSaveable(self, self._primary_var, name)
    return {checkpointable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  # Needed to pass ResourceVariable checks.
  @property
  def op(self):
    return self._primary_var.op

  @property
  def _in_graph_mode(self):
    return self._primary_var._in_graph_mode   # pylint: disable=protected-access

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if _enclosing_tpu_context() is None:
      return self._get()._dense_var_to_tensor(dtype, name, as_ref)
    # pylint: enable=protected-access
    if dtype is not None and dtype != self.dtype:
      raise NotImplementedError
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

    # We have to cast the self._index.values() to a `list` because when we
    # use `model_to_estimator` to run tf.keras models, self._index.values() is
    # of type `dict_values` and not `list`.
    values_list = nest.flatten(self._index)
    result = values_list[0].is_initialized()
    # We iterate through the list of values except the last one to allow us to
    # name the final `logical_and` op the same name that is passed by the user
    # to the `is_initialized` op. For distributed variables, the
    # `is_initialized` op is a `logical_and` op.
    for v in values_list[1:-1]:
      result = math_ops.logical_and(result, v.is_initialized())
    result = math_ops.logical_and(result, values_list[-1].is_initialized(),
                                  name=name)
    return result


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_tpu_mirrored(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(TPUMirroredVariable,
                                        _tensor_conversion_tpu_mirrored)
ops.register_dense_tensor_like_type(TPUMirroredVariable)


class _TowerLocalSaveable(saver.BaseSaverBuilder.SaveableObject):
  """Class for defining how to restore a TowerLocalVariable."""

  def __init__(self, tower_local_variable, name):
    self._tower_local_variable = tower_local_variable
    # We use a callable so that we don't have to evaluate this expression
    # in the case where we are trying to restore instead of save.
    def tensor():
      return distribution_strategy_context.get_distribution_strategy().read_var(
          tower_local_variable)
    spec = saver.BaseSaverBuilder.SaveSpec(
        tensor=tensor,
        slice_spec="",
        name=name,
        dtype=tower_local_variable.dtype)
    super(_TowerLocalSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return self._tower_local_variable.assign(tensor)


def _assert_tower_context():
  if not distribution_strategy_context.get_tower_context():
    raise RuntimeError(
        "Tower-local variables may only be assigned in a tower context.")


class TowerLocalVariable(DistributedVariable, PerDevice,
                         checkpointable.CheckpointableBase):
  """Holds a map from device to variables whose values are reduced on save."""

  def __init__(self, index, primary_var, aggregation):
    self._primary_var = primary_var
    self._aggregation = aggregation
    super(TowerLocalVariable, self).__init__(index)

  def assign_sub(self, *args, **kwargs):
    _assert_tower_context()
    return self.get().assign_sub(*args, **kwargs)

  def assign_add(self, *args, **kwargs):
    _assert_tower_context()
    return self.get().assign_add(*args, **kwargs)

  def assign(self, *args, **kwargs):
    if distribution_strategy_context.get_cross_tower_context():
      # To preserve the sum across save and restore, we have to divide the
      # total across all devices when restoring a variable that was summed
      # when saving.
      tensor = args[0]
      if self._aggregation == vs.VariableAggregation.SUM:
        tensor *= 1. / len(self.devices)
      return control_flow_ops.group(
          [_assign_on_device(d, v, tensor)
           for d, v in six.iteritems(self._index)])
    else:
      _assert_tower_context()
      return self.get().assign(*args, **kwargs)

  @property
  def aggregation(self):
    return self._aggregation

  def _get_cross_tower(self):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_TOWER:
      return self._primary_var
    all_components = tuple(self._index.values())
    # TODO(josh11b): Use a strategy-specific method.
    total = math_ops.add_n(all_components)
    if self._aggregation == vs.VariableAggregation.MEAN:
      return total * (1./ len(all_components))
    return total

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.get_cross_tower_context():
      return self._get_cross_tower()
    return self.get()._as_graph_element()

  def _gather_saveables_for_checkpoint(self):
    """Overrides CheckpointableBase method.

    This allows both name-based and object-based save and restore of
    TowerLocalVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """
    def _saveable_factory(name=self._common_name):
      return _TowerLocalSaveable(self, name)
    return {checkpointable.VARIABLE_VALUE_KEY: _saveable_factory}


# Register a conversion function for TowerLocalVariable which allows as_ref to
# be true.
def _tensor_conversion_tower_local(var, dtype=None, name=None, as_ref=False):
  return ops.internal_convert_to_tensor(
      var.get(), dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(TowerLocalVariable,
                                        _tensor_conversion_tower_local)


def _devices_match(d1, d2):
  return device_util.canonicalize(d1) == device_util.canonicalize(d2)


def regroup(per_device, wrap_class=PerDevice):
  """Makes device->nest map into a nest of PerDevice/Mirrored values."""
  items = list(per_device.items())
  assert items
  v0 = items[0][1]  # First value

  if isinstance(v0, list):
    for _, v in items[1:]:
      assert isinstance(v, list)
      assert len(v) == len(v0), ("len(v) == %d, len(v0) == %d, v: %s, v0: %s" %
                                 (len(v), len(v0), v, v0))
    return [regroup({k: v[i] for k, v in items}, wrap_class)
            for i in range(len(v0))]

  if isinstance(v0, tuple):
    for _, v in items[1:]:
      assert isinstance(v, tuple)
      assert len(v) == len(v0)
    regrouped_tuple = tuple(regroup({k: v[i] for k, v in items}, wrap_class)
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
    for _, v in items[1:]:
      assert isinstance(v, dict)
      assert set(v.keys()) == v0keys
    return {key: regroup({k: v[key] for k, v in items}, wrap_class)
            for key in v0keys}

  # If exactly the same object across all devices, return it unwrapped.
  same_id = True
  for _, v in items[1:]:
    if v is not v0:
      same_id = False
      break
  # Consider three cases where same_id is true:
  # * If v0 is a DistributedVariable (a MirroredVariable or
  #   TowerLocalVariable, and same_id means it is the same across all
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
  # same MirroredVariable (or TowerLocalVariable). In this case we
  # want to return the containing MirroredVariable, after a bunch of
  # sanity checking. In particular, each component should have the
  # same container, and the devices of the variables should match the
  # keys of the per-device dictionary.
  if hasattr(v0, "_distributed_container"):
    # pylint: disable=protected-access
    assert not isinstance(v0, MirroredVariable), (
        "ids = %s, items = %s" % ([id(v[1]) for v in items], items))
    assert _devices_match(v0.device, items[0][0]), (
        "v0.device = %s, items = %s" % (v0.device, items))
    distributed_container = v0._distributed_container()
    assert distributed_container is not None
    for d, v in items[1:]:
      assert _devices_match(v.device, d), (
          "v.device = %s, d = %s, items = %s" % (v.device, d, items))
      assert distributed_container is v._distributed_container()
    return distributed_container
  # pylint: enable=protected-access

  return wrap_class(per_device)


def select_device(device, structured):
  """Specialize a nest of regular & per-device values for one device."""
  def _get(x):
    return x.get(device) if isinstance(x, DistributedValues) else x

  return nest.map_structure(_get, structured)


def select_device_mirrored(device, structured):
  """Specialize a nest of regular & mirrored values for one device."""
  def _get_mirrored(x):
    if isinstance(x, DistributedValues):
      if not isinstance(x, Mirrored):
        raise TypeError(
            "Expected value to be mirrored across towers: %s in %s." %
            (x, structured))
      return x.get(device)
    else:
      return x

  return nest.map_structure(_get_mirrored, structured)


def update_regroup(strategy, updates, should_group):
  """Regroup for an update, with dependencies to ensure all updates execute."""
  regrouped = regroup(updates, Mirrored)
  if not should_group:
    return nest.map_structure(strategy.unwrap, regrouped)
  grouped_flat = []
  for u in nest.flatten(regrouped):
    if isinstance(u, DistributedValues):
      g = strategy.group(u)
      if u.is_tensor_like:
        # Make sure we run all updates. Without this, something like
        # session.run(strategy.update(...)) may only update one tower.
        index = {}
        for d in u.devices:
          with ops.device(d), ops.control_dependencies([g]):
            index[d] = array_ops.identity(u.get(d))
        g = Mirrored(index)
    else:
      g = u
    grouped_flat.append(g)
  return nest.pack_sequence_as(regrouped, grouped_flat)


class PerDeviceDataIterator(object):
  """An iterator (like `tf.data.Iterator`) into a `PerDeviceDataset`."""

  def __init__(self, iterator, devices, prefetch_on_device=None):
    self._iterator = iterator
    self._devices = devices
    self._prefetch_on_device = prefetch_on_device

  @property
  def initializer(self):
    return self._iterator.initializer

  def get_next(self, name=None):
    """Scatter the input across devices."""
    if self._prefetch_on_device:
      data_list = self._iterator.get_next(name=name)
      index = dict(zip(self._devices, data_list))
    else:
      batch = self._iterator.get_next(name=name)
      index = {}
      def get_ith(i):
        return lambda x: x[i]

      for i, d in enumerate(self._devices):
        index[d] = nest.map_structure(get_ith(i), batch)
        if context.executing_eagerly():
          with ops.device(d):
            index[d] = nest.map_structure(array_ops.identity, index[d])

    return regroup(index)


class PerDeviceDataset(object):
  """Like `tf.data.Dataset` split devices, producing `PerDevice` data."""

  def __init__(self, dataset, devices, prefetch_on_device=None):
    self._devices = devices

    # Default to using prefetching in graph mode, unless specified.
    # TODO(priyag): Enable prefetching in eager mode.
    self._prefetch_on_device = prefetch_on_device
    if self._prefetch_on_device is None:
      self._prefetch_on_device = not context.executing_eagerly()
    assert not (self._prefetch_on_device and context.executing_eagerly()), (
        "Prefetching is only supported in graph mode currently")

    if self._prefetch_on_device:
      self._dataset = dataset.apply(
          prefetching_ops_v2.prefetch_to_devices(self._devices))
    else:
      # TODO(priyag): If dropping remainder is not appropriate, find another
      # approach to distributing the dataset when not possible to divide evenly.
      # Possibly not an issue when we start using PartitionedDataset.
      self._dataset = dataset.batch(len(devices), drop_remainder=True)

  def make_one_shot_iterator(self):
    """Get a one time use iterator for the distributed PerDeviceDataset."""
    dataset_iterator = self._dataset.make_one_shot_iterator()
    return PerDeviceDataIterator(dataset_iterator, self._devices,
                                 self._prefetch_on_device)

  def make_initializable_iterator(self):
    """Get an initializable iterator for the distributed PerDeviceDataset."""
    dataset_iterator = self._dataset.make_initializable_iterator()
    return PerDeviceDataIterator(dataset_iterator, self._devices,
                                 self._prefetch_on_device)


class MultiWorkerDataIterator(object):
  """An iterator (like `tf.data.Iterator`) into a `MultiWorkerDataset`."""

  def __init__(self, iterators, worker_device_map):
    """Initialize the MultiWorkerDataIterator object.

    Args:
      iterators: a dict mapping from each worker to an iterator for
        that worker.
      worker_device_map: a dict mapping from each worker's devices to a list of
        devices that belong to this worker.

    Raises:
      ValueError: if iterators and worker_device_map are not compatible.
    """
    self._iterators = iterators
    self._worker_device_map = worker_device_map
    if set(self._iterators) != set(self._worker_device_map):
      raise ValueError("iterators and worker_device_map are not compatible.")

  @property
  def initializer(self):
    return control_flow_ops.group(
        [iterator.initializer for iterator in self._iterators.values()])

  def get_next(self, name=None):
    """Scatter the input across hosts and devices."""
    index = {}
    for worker, iterator in six.iteritems(self._iterators):
      if name is not None:
        d = tf_device.DeviceSpec.from_string(worker)
        new_name = "%s_%s_%d" % (name, d.job, d.task)
      else:
        new_name = None
      with ops.device(worker):
        data_per_worker = iterator.get_next(name=new_name)

      worker_devices = self._worker_device_map[worker]
      # Ungroup these per-device value so as to get a flat map from devices to
      # values.
      for d in worker_devices:
        v = select_device(d, data_per_worker)
        if d in index:
          raise ValueError("Duplicated devices in worker_device_map: %r" % v)
        index[d] = v

    return regroup(index)


class MultiWorkerDataset(object):
  """Like a `tf.data.Dataset` that distributes data to different workers.

  Each worker gets one shard of the input dataset. It is currently not working
  in
  eager mode.
  """

  def __init__(self, dataset_fn, worker_device_map, prefetch_on_device=None,
               auto_shard=False):
    """Initialize the MultiWorkerDataset object.

    Args:
      dataset_fn: a function that returns a `tf.data.Dataset`.
      worker_device_map: a dict mapping from each worker to a list of devices
        that belong to this worker.
      prefetch_on_device: whether to prefetch to devices.
      auto_shard: whether to auto-shard the dataset.
    """
    self._worker_device_map = worker_device_map
    self._datasets = {}
    # TODO(yuefengz, priyag): support different set of jobs for input
    # processing.
    for i, (worker, worker_devices) in enumerate(
        six.iteritems(worker_device_map)):
      with ops.device(worker):
        worker_input = dataset_fn()
        if auto_shard:
          worker_input = input_ops.auto_shard_dataset(
              worker_input, len(worker_device_map), i)
        self._datasets[worker] = PerDeviceDataset(
            worker_input, worker_devices, prefetch_on_device=prefetch_on_device)

  def make_one_shot_iterator(self):
    iterators = {}
    for worker, dataset in six.iteritems(self._datasets):
      with ops.device(worker):
        iterators[worker] = dataset.make_one_shot_iterator()
    return MultiWorkerDataIterator(iterators, self._worker_device_map)

  def make_initializable_iterator(self):
    iterators = {}
    for worker, dataset in six.iteritems(self._datasets):
      with ops.device(worker):
        iterators[worker] = dataset.make_initializable_iterator()
    return MultiWorkerDataIterator(iterators, self._worker_device_map)


class _PerKey(object):
  """Holds data associated by keys."""

  def __init__(self, *index):
    # pylint: disable=protected-access
    self._index = list(index)

  def get(self, iteration):
    return array_ops.gather(self._index, iteration)

  def get_shape(self):
    return self._index[-1][-1].get_shape()

  def get_dtype(self):
    return self._index[-1][-1].dtype

  def __str__(self):
    return "%s:%s" % (self.__class__.__name__, self._index)

  def __repr__(self):
    return "%s(%r)" % (self.__class__.__name__, self._index)


class PerIteration(_PerKey):
  """Holds input for multiple iterations at once."""

  def __init__(self, *index):
    # pylint: disable=protected-access
    super(PerIteration, self).__init__(*[batch._index for batch in index])


class Batches(_PerKey):
  pass


class MultiIterator(object):
  """Iterator that returns results of multiple get_next()s."""

  def __init__(self, dataset_iterator, iterations, batches_per_iteration):
    self._dataset_iterator = dataset_iterator
    self._iterations = iterations
    self._batches_per_iteration = batches_per_iteration

  def get_next(self, name=None):
    """Return PerIteration with `iterations x batches_per_iteration` inputs."""
    data = []
    for _ in range(self._batches_per_iteration):
      batch = []
      for _ in range(self._iterations):
        batch.append(self._dataset_iterator.get_next(name=name))
      data.append(batch)

    # Here is an example.  Suppose each get_next returns a tuple of two tensors.
    # For 3 `iterations` and 2 `batches_per_iteration`, the `data` is:
    # [[(a,z), (b,y), (c,x)], [(A,Z), (B,Y), (C,X)]]
    #
    # After the first `map_structure` it gets transformed to:
    #  [(Batches(a, A), Batches(z, Z)),
    #   (Batches(b, B), Batches(y, Y)),
    #   (Batches(c, C), Batches(x, X))]
    #
    # After the second `map_structure` it gets transformed to a tuple of:
    # (PerIteration([Batches(a, A), Batches(b, B), Batches(c, C)]),
    #  PerIteration([Batches(z, Z), Batches(y, Y), Batches(x, X)]))

    data = nest.map_structure(Batches, *data)
    data = nest.map_structure(PerIteration, *data)

    return data

  @property
  def initializer(self):
    return self._dataset_iterator.initializer


class PerIterationDataset(object):
  """A dataset that returns MultiIterators."""

  def __init__(self, dataset, iterations, batches_per_iteration):
    self._dataset = dataset
    self._iterations = iterations
    self._batches_per_iteration = batches_per_iteration

  def make_one_shot_iterator(self):
    iterator = self._dataset.make_one_shot_iterator()
    return MultiIterator(iterator, self._iterations,
                         self._batches_per_iteration)

  def make_initializable_iterator(self):
    iterator = self._dataset.make_initializable_iterator()
    return MultiIterator(iterator, self._iterations,
                         self._batches_per_iteration)


class MapOutput(object):
  """Map can result in multiple outputs per device."""

  def __init__(self, l):
    self._l = l

  def get(self):
    return self._l


class MultiStepContext(object):
  """A context object that can be used to capture things when running steps.

  This context object is useful when running multiple steps at a time using the
  `run_steps_on_dataset` API. For e.g. it allows the user's step function to
  specify which outputs to emit at what frequency. Currently it supports
  capturing output from the last step, as well as capturing non tensor outputs.
  In the future it will be augmented to support other use cases such as output
  each N steps.
  """

  def __init__(self):
    """Initializes an output context.

    Returns:
      A context object.
    """
    self._last_step_outputs = {}
    self._last_step_outputs_aggregations = {}
    self._non_tensor_outputs = {}

  @property
  def last_step_outputs(self):
    """A dictionary consisting of outputs to be captured on last step.

    Keys in the dictionary are names of tensors to be captured, as specified
    when `set_last_step_output` is called.
    Values in the dictionary are the tensors themselves. If
    `set_last_step_output` was called with an `aggregation` for this output,
    then the value is the aggregated value.

    Returns:
      A dictionary with last step outputs.
    """
    return self._last_step_outputs

  def _set_last_step_outputs(self, outputs):
    """Replace the entire dictionary of last step outputs."""
    if not isinstance(outputs, dict):
      raise ValueError("Need a dictionary to set last_step_outputs.")
    self._last_step_outputs = outputs

  def set_last_step_output(self, name, output,
                           aggregation=variables_lib.VariableAggregation.NONE):
    """Set `output` with `name` to be outputted from the last step.

    Args:
      name: String, name to identify the output. Doesn't need to match tensor
        name.
      output: The tensors that should be outputted with `name`. See below for
        actual types supported.
      aggregation: Aggregation method to use to aggregate outputs from multiple
        towers. Required if `set_last_step_output` is called in a tower context.
        Optional in cross_tower_context.
        When present, the outputs from all the towers are aggregated using the
        current distribution strategy's `reduce` method. Hence, the type of
        `output` must be what's supported by the corresponding `reduce` method.
        For e.g. if using MirroredStrategy and aggregation is set, output
        must be a `PerDevice` value.
        The aggregation method is also recorded in a dictionary
        `_last_step_outputs_aggregations` for later interpreting of the
        outputs as already reduced or not.

    """
    if distribution_strategy_context.get_cross_tower_context():
      self._last_step_outputs_aggregations[name] = aggregation
      if aggregation is variables_lib.VariableAggregation.NONE:
        self._last_step_outputs[name] = output
      else:
        distribution = distribution_strategy_context.get_distribution_strategy()
        self._last_step_outputs[name] = distribution.reduce(
            aggregation, output, destinations="/device:CPU:0")
    else:
      assert aggregation is not variables_lib.VariableAggregation.NONE
      def merge_fn(distribution, value):
        self._last_step_outputs[name] = distribution.reduce(
            aggregation, value, destinations="/device:CPU:0")
        # Setting this inside the `merge_fn` because all towers share the same
        # context object, so it's more robust to set it only once (even if all
        # the towers are trying to set the same value).
        self._last_step_outputs_aggregations[name] = aggregation

      distribution_strategy_context.get_tower_context().merge_call(
          merge_fn, output)

  @property
  def non_tensor_outputs(self):
    """A dictionary consisting of any non tensor outputs to be captured."""
    return self._non_tensor_outputs

  def set_non_tensor_output(self, name, output):
    """Set `output` with `name` to be captured as a non tensor output."""
    if distribution_strategy_context.get_cross_tower_context():
      self._non_tensor_outputs[name] = output
    else:
      def merge_fn(distribution, value):
        # NOTE(priyag): For non tensor outputs, we simply return all the values
        # in a list as aggregation doesn't make sense on non tensors.
        self._non_tensor_outputs[name] = distribution.unwrap(value)
      distribution_strategy_context.get_tower_context().merge_call(
          merge_fn, output)


def value_container(val):
  """Returns the container that this per-device `value` belongs to.

  Args:
    val: A value returned by `call_for_each_tower()` or a variable
      created in `scope()`.

  Returns:
    A container that `value` belongs to.
    If value does not belong to any container (including the case of
    container having been destroyed), returns the value itself.
  """
  # pylint: disable=protected-access
  if (hasattr(val, "_distributed_container") and
      # DistributedVariable has _distributed_container defined
      # but we don't want to return it.
      not isinstance(val, DistributedVariable)):
    container = val._distributed_container()
    # pylint: disable=protected-access
    if container is not None:
      return container
  return val


# TODO(josh11b): Descend from Variable.
class AggregatingVariable(checkpointable.CheckpointableBase):
  """A wrapper around a variable that aggregates updates across towers."""

  def __init__(self, v, aggregation):
    self._v = v
    # TODO(josh11b): Set v._distributed_container?
    # v._distributed_container = weakref.ref(self)  # pylint: disable=protected-access
    self._aggregation = aggregation

  def get(self):
    return self._v

  def __getattr__(self, name):
    return getattr(self._v, name)

  def _assign_func(self, *args, **kwargs):
    f = kwargs.pop("f")
    if distribution_strategy_context.get_cross_tower_context():
      update_device = distribute_lib.get_update_device()
      if update_device is not None:
        # We are calling an assign function in an update context.
        return f(self._v, *args, **kwargs)

      # We are calling an assign function in cross tower context, wrap it in an
      # update call.
      return distribution_strategy_context.get_distribution_strategy().update(
          self, f, *args, **kwargs)
    else:
      assert distribution_strategy_context.get_tower_context()
      # We are calling an assign function in tower context.
      # We reduce the value we want to assign/add/sub. More details about how we
      # handle the different use cases can be found in the _reduce method.
      # We call the function with the reduced value.
      if self._aggregation == vs.VariableAggregation.NONE:
        raise ValueError("You must specify an aggregation method to update a "
                         "a variable in Tower Context.")

      def merge_fn(strategy, value, *other_args, **other_kwargs):
        return strategy.update(
            self, f,
            strategy.reduce(
                aggregation=self._aggregation, value=value, destinations=self),
            *other_args, **other_kwargs)

      return distribution_strategy_context.get_tower_context().merge_call(
          merge_fn, *args, **kwargs)

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
