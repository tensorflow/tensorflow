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
import operator
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


# pylint: disable=line-too-long
# TODO(josh11b): Should device values be strings or DeviceSpec objects?
# Not sure DeviceSpec objects are usable as a dict key.
class DistributedValues(object):
  """Holds a map from device to values. Either PerReplica or Mirrored."""

  def __init__(self, index):
    self._index = {device_util.canonicalize(key): value
                   for key, value in six.iteritems(index)}

  def get(self, device=None):
    """Returns the value for the current device or raises a ValueError."""
    if device is None:
      replica_context = distribution_strategy_context.get_replica_context()
      if replica_context:
        # TODO(josh11b): support model parallelism better here
        device = replica_context.devices[0]
      else:
        device = distribute_lib.get_update_device()
        if device is None:
          return self._get_cross_replica()
    device = device_util.canonicalize(device)
    try:
      return self._index[device]
    except KeyError as e:
      six.raise_from(
          ValueError("Device %s not found in %s (current device %s)" %
                     (device, self._index.keys(), device_util.current())), e)

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


class PerReplica(DistributedValues):
  """Holds a map from device to unsynchronized values."""
  pass


# Note that unlike PerReplica, Mirrored values inherit from
# DistributedDelegate and so can be used directly in cross-replica mode.
class Mirrored(DistributedDelegate):
  """Holds a map from device to values which are kept in sync."""

  def _get_cross_replica(self):
    device = device_util.canonicalize(device_util.current())
    if device in self._index:
      return self._index[device]
    return list(self._index.values())[0]

  def _as_graph_element(self):
    obj = self.get()
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
    # We want cross-replica code that does some var.op.X calls
    # to work (even if the current device isn't in self.devices), but
    # other uses of var.op in a cross-replica context to fail.
    if distribution_strategy_context.get_cross_replica_context():
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
    if distribution_strategy_context.get_cross_replica_context():
      update_device = distribute_lib.get_update_device()
      if update_device is not None:
        # We are calling an assign function on the mirrored variable in an
        # update context.
        v = self.get(device=update_device)
        return f(v, *args, **kwargs)

      # We are calling assign on the mirrored variable in cross replica context,
      # use update to update the variable.
      strategy = distribution_strategy_context.get_distribution_strategy()
      return strategy.update(self, f, *args, **kwargs)
    else:
      _assert_replica_context()
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
    if device in self._index:
      return array_ops.identity(self._index[device])
    return array_ops.identity(self._primary_var)

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.get_cross_replica_context():
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
      replica_context = distribution_strategy_context.get_replica_context()
      if replica_context:
        # TODO(josh11b): support model parallelism better here
        device = replica_context.devices[0]
      else:
        device = distribute_lib.get_update_device()
        if device is None:
          return self._get_cross_replica()
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
    strategy = distribution_strategy_context.get_distribution_strategy()
    if strategy.__class__.__name__ != "TPUStrategy":
      raise ValueError("You may only assign to a TPUMirroredVariable within a "
                       "TPUStrategy.")
    f = kwargs.pop("f")
    if distribution_strategy_context.get_cross_replica_context():
      if _enclosing_tpu_context() is not None:
        return distribution_strategy_context.get_distribution_strategy().update(
            self, f, *args, **kwargs)

      update_device = distribute_lib.get_update_device()
      # We are calling update on the mirrored variable in cross replica context.
      if update_device is not None:
        # We are calling an assign function on the mirrored variable in cross
        # replica context.
        v = self._get(device=update_device)
        return f(v, *args, **kwargs)

      return distribution_strategy_context.get_distribution_strategy().update(
          self, f, *args, **kwargs)
    else:
      _assert_replica_context()
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

  def _get_cross_replica(self):
    device = device_util.canonicalize(device_util.current())
    if device in self._index:
      return self._index[device]
    return self._primary_var

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.get_cross_replica_context():
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

  # pylint: disable=protected-access
  @property
  def _save_slice_info(self):
    return self._primary_var._save_slice_info

  def _get_save_slice_info(self):
    return self._primary_var._get_save_slice_info()

  def _set_save_slice_info(self, save_slice_info):
    return self._primary_var._set_save_slice_info(save_slice_info)
  # pylint: enable=protected-access

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


class _ReplicaLocalSaveable(saver.BaseSaverBuilder.SaveableObject):
  """Class for defining how to restore a ReplicaLocalVariable."""

  def __init__(self, replica_local_variable, name):
    self._replica_local_variable = replica_local_variable
    # We use a callable so that we don't have to evaluate this expression
    # in the case where we are trying to restore instead of save.
    def tensor():
      return distribution_strategy_context.get_distribution_strategy().read_var(
          replica_local_variable)
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


def _assert_replica_context():
  if not distribution_strategy_context.get_replica_context():
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")


class ReplicaLocalVariable(DistributedVariable, PerReplica,
                           checkpointable.CheckpointableBase):
  """Holds a map from device to variables whose values are reduced on save."""

  def __init__(self, index, primary_var, aggregation):
    self._primary_var = primary_var
    self._aggregation = aggregation
    super(ReplicaLocalVariable, self).__init__(index)

  def assign_sub(self, *args, **kwargs):
    _assert_replica_context()
    return self.get().assign_sub(*args, **kwargs)

  def assign_add(self, *args, **kwargs):
    _assert_replica_context()
    return self.get().assign_add(*args, **kwargs)

  def assign(self, *args, **kwargs):
    if distribution_strategy_context.get_cross_replica_context():
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
      _assert_replica_context()
      return self.get().assign(*args, **kwargs)

  @property
  def aggregation(self):
    return self._aggregation

  def _get_cross_replica(self):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      return self._primary_var
    all_components = tuple(self._index.values())
    # TODO(josh11b): Use a strategy-specific method.
    total = math_ops.add_n(all_components)
    if self._aggregation == vs.VariableAggregation.MEAN:
      return total * (1./ len(all_components))
    return total

  def _as_graph_element(self):
    # pylint: disable=protected-access
    if distribution_strategy_context.get_cross_replica_context():
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


def _devices_match(d1, d2):
  return device_util.canonicalize(d1) == device_util.canonicalize(d2)


def regroup(per_replica, wrap_class=PerReplica):
  """Makes device->nest map into a nest of PerReplica/Mirrored values."""
  items = list(per_replica.items())
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

  return wrap_class(per_replica)


def select_device(device, structured):
  """Specialize a nest of regular & per-replica values for one device."""
  def _get(x):
    return x.get(device) if isinstance(x, DistributedValues) else x

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


def update_regroup(extended, updates, group):
  """Regroup for an update, with dependencies to ensure all updates execute."""
  regrouped = regroup(updates, Mirrored)
  if not group:
    return nest.map_structure(extended._unwrap, regrouped)  # pylint: disable=protected-access
  grouped_flat = []
  for u in nest.flatten(regrouped):
    if isinstance(u, DistributedValues):
      g = extended._group(u)  # pylint: disable=protected-access
      if u.is_tensor_like:
        # Make sure we run all updates. Without this, something like
        # session.run(extended.update(...)) may only update one replica.
        index = {}
        for d in u.devices:
          with ops.device(d), ops.control_dependencies([g]):
            index[d] = array_ops.identity(u.get(d))
        g = Mirrored(index)
    else:
      g = u
    grouped_flat.append(g)
  return nest.pack_sequence_as(regrouped, grouped_flat)


class PerReplicaDataIterator(object):
  """An iterator (like `tf.data.Iterator`) into a `PerReplicaDataset`."""

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
      data_list = self._iterator.get_next()
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

  def __init__(self, dataset, devices, prefetch_on_device=None):
    self._devices = devices

    # Default to using prefetching in graph mode, unless specified.
    # TODO(rohanj): Enable prefetching in eager mode.
    self._prefetch_on_device = prefetch_on_device
    if self._prefetch_on_device is None:
      self._prefetch_on_device = not context.executing_eagerly()
    assert not (self._prefetch_on_device and context.executing_eagerly()), (
        "Prefetching is only supported in graph mode currently")

    self._dataset = dataset
    if not self._prefetch_on_device:
      # TODO(priyag): If dropping remainder is not appropriate, find another
      # approach to distributing the dataset when not possible to divide evenly.
      # Possibly not an issue when we start using PartitionedDataset.
      self._dataset = dataset.batch(len(devices), drop_remainder=True)

  def make_one_shot_iterator(self):
    """Get a one time use iterator for the distributed PerReplicaDataset."""
    # Graph mode with one shot iterator is disabled.
    if not context.executing_eagerly():
      raise ValueError("Cannot create a one shot iterator. Please use "
                       "`make_initializable_iterator()` instead.")
    # Eager mode prefetching would error out in constructor. Only remaining
    # case is non-prefetching in eager mode. We delegate to
    # PerReplicaDataIterator to handle that case.
    dataset_iterator = self._dataset.make_one_shot_iterator()
    return PerReplicaDataIterator(
        dataset_iterator, self._devices, prefetch_on_device=False)

  def make_initializable_iterator(self):
    """Get an initializable iterator for the distributed PerReplicaDataset."""
    # Eager mode generates already initialized iterators. Hence we cannot create
    # an initializable iterator.
    if context.executing_eagerly():
      raise ValueError("Cannot create initializable iterator in Eager mode. "
                       "Please use `make_one_shot_iterator` instead.")
    if self._prefetch_on_device:
      dataset_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          self._dataset, self._devices)
    else:
      dataset_iterator = self._dataset.make_initializable_iterator()
    return PerReplicaDataIterator(
        dataset_iterator,
        self._devices,
        prefetch_on_device=self._prefetch_on_device)


class MultiWorkerDataIterator(object):
  """An iterator (like `tf.data.Iterator`) into a `MultiWorkerDataset`."""

  def __init__(self, iterators, worker_device_pairs):
    """Initialize the MultiWorkerDataIterator object.

    Args:
      iterators: a list of worker, iterator pairs.
      worker_device_pairs: a list of (worker's devices, a list of
        devices that belong to this worker) pairs.

    Raises:
      ValueError: if iterators and worker_device_pairs are not compatible.
    """
    if [d for d, _ in iterators] != [d for d, _ in worker_device_pairs]:
      raise ValueError("iterators and worker_device_pairs are not compatible.")
    self._workers = [d for d, _ in iterators]
    self._iterators = [i for _, i in iterators]
    self._worker_devices = [l for _, l in worker_device_pairs]

  @property
  def initializer(self):
    return control_flow_ops.group(
        [iterator.initializer for iterator in self._iterators])

  def get_iterator(self, worker):
    for i, w in enumerate(self._workers):
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
    index = {}
    worker_info = zip(self._workers, self._iterators, self._worker_devices)
    for worker, iterator, worker_devices in worker_info:
      if name is not None:
        d = tf_device.DeviceSpec.from_string(worker)
        new_name = "%s_%s_%d" % (name, d.job, d.task)
      else:
        new_name = None
      with ops.device(worker):
        data_per_worker = iterator.get_next(name=new_name)

      # Ungroup these per-replica value so as to get a flat map from devices to
      # values.
      for d in worker_devices:
        v = select_device(d, data_per_worker)
        if d in index:
          raise ValueError("Duplicated devices in worker_device_pairs: %r" % v)
        index[d] = v

    return regroup(index)


class MultiWorkerDataset(object):
  """Like a `tf.data.Dataset` that distributes data to different workers.

  Each worker gets one shard of the input dataset. This currently does not work
  in eager mode.
  """

  def __init__(self, dataset_fn, worker_device_pairs, prefetch_on_device=None,
               auto_shard=False):
    """Initialize the MultiWorkerDataset object.

    Args:
      dataset_fn: a function or a list of functions that returns a
        `tf.data.Dataset`.
      worker_device_pairs: a list of (worker, list of devices on that worker)
        pairs; it must have same length with `dataset_fn` if `dataset_fn` is a
        list.
      prefetch_on_device: whether to prefetch to devices.
      auto_shard: whether to auto-shard the dataset.
    """
    if isinstance(dataset_fn, list):
      if len(dataset_fn) != len(worker_device_pairs):
        raise ValueError("If `dataset_fn` is a list, it must have same length "
                         "as `worker_device_pairs`")
      if auto_shard:
        raise ValueError(
            "If `dataset_fn` is a list, `auto_shard` is not supported.")
    self._worker_device_pairs = worker_device_pairs
    self._datasets = []
    # TODO(yuefengz, priyag): support different set of jobs for input
    # processing.
    for i, (worker, worker_devices) in enumerate(worker_device_pairs):
      with ops.device(worker):
        if isinstance(dataset_fn, list):
          worker_input = dataset_fn[i]()
        else:
          worker_input = dataset_fn()
          if auto_shard:
            worker_input = input_ops.auto_shard_dataset(
                worker_input, len(worker_device_pairs), i)
        dataset = PerReplicaDataset(
            worker_input, worker_devices, prefetch_on_device=prefetch_on_device)
        self._datasets.append((worker, dataset))

  def make_one_shot_iterator(self):
    iterators = []
    for worker, dataset in self._datasets:
      with ops.device(worker):
        iterators.append((worker, dataset.make_one_shot_iterator()))
    return MultiWorkerDataIterator(iterators, self._worker_device_pairs)

  def make_initializable_iterator(self):
    iterators = []
    for worker, dataset in self._datasets:
      with ops.device(worker):
        iterators.append((worker, dataset.make_initializable_iterator()))
    return MultiWorkerDataIterator(iterators, self._worker_device_pairs)


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

  def __init__(self, worker_device_pairs, iterators):
    if not worker_device_pairs:
      raise ValueError("Should have at least one worker for input iterator.")

    self._iterators = iterators
    self._worker_device_pairs = worker_device_pairs
    self._is_eager = context.executing_eagerly()

  def get_next(self, name=None):
    """Returns the next input from the iterator for all replicas."""
    assert self._is_eager == context.executing_eagerly(), (
        "Iterator should be created and used in same execution mode.")

    index = {}
    for i, (worker, worker_devices) in enumerate(self._worker_device_pairs):
      if name is not None:
        d = tf_device.DeviceSpec.from_string(worker)
        new_name = "%s_%s_%d" % (name, d.job, d.task)
      else:
        new_name = None
      with ops.device(worker):
        data_per_worker = self._iterators[i].get_next(new_name)

      # Ungroup these per-replica value so as to get a flat map from devices to
      # values.
      for d in worker_devices:
        v = select_device(d, data_per_worker)
        if d in index:
          raise ValueError("Duplicated devices in worker_device_pairs: %r" % v)
        index[d] = v

    return regroup(index)

  def initialize(self):
    """Initialze underlying iterators.

    Returns:
      A list of any initializer ops that should be run.
    """
    assert self._is_eager == context.executing_eagerly(), (
        "Iterator should be created and used in same execution mode.")

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
    for i, (w, _) in enumerate(self._worker_device_pairs):
      if worker == w:
        return self._iterators[i]
    return None


class InputFunctionIterator(InputIteratorImpl):
  """Iterator created from input function."""

  def __init__(self, input_fn, worker_device_pairs, input_contexts):
    """Make an iterator for input provided via an input function.

    Currently implements PER_WORKER mode, in which the `input_fn` is called
    once on each worker.

    TODO(priyag): Add other replication modes.
    TODO(priyag): Allow taking input function that returns a callable that
    returns nest of tensors.

    Args:
      input_fn: Input function that returns a `tf.data.Dataset` object.
      worker_device_pairs: A list of (worker, list of devices on that worker)
        pairs.
      input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `input_fn`. Length and order should match worker order in
        `worker_device_pairs`.
    """
    if len(worker_device_pairs) != len(input_contexts):
      raise ValueError(
          "Number of worker_device_pairs (%d) is not same as number of"
          "input_contexts (%d)" % (
              len(worker_device_pairs), len(input_contexts)))

    iterators = []
    for (worker, devices), ctx in zip(worker_device_pairs, input_contexts):
      # TODO(priyag): We should probably explicitly specify CPU device on worker.
      with ops.device(worker):
        result = input_fn(ctx)
        if not isinstance(result, dataset_ops.DatasetV2):
          raise ValueError("input_fn must return a tf.data.Dataset.")
        iterator = _SingleWorkerDatasetIterator(result, worker, devices)
        iterators.append(iterator)

    super(InputFunctionIterator, self).__init__(
        worker_device_pairs, iterators)


class DatasetIterator(InputIteratorImpl):
  """Iterator created from input dataset."""

  def __init__(self, dataset, worker_device_pairs, split_batch_by=None):
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
      worker_device_pairs: A list of (worker, list of devices on that worker)
        pairs.
      split_batch_by: Optional integer. If present, we "split" each batch of the
        dataset by `split_batch_by` value.
    """
    if split_batch_by:
      dataset = _split_dataset_batch(dataset, split_batch_by)

    iterators = []
    for worker, worker_devices in worker_device_pairs:
      with ops.device(worker):
        iterator = _SingleWorkerDatasetIterator(dataset, worker, worker_devices)
        iterators.append(iterator)

    super(DatasetIterator, self).__init__(worker_device_pairs, iterators)


class _SingleWorkerDatasetIterator(object):
  """Iterator for a single `tf.data.Dataset`."""

  def __init__(self, dataset, worker, devices):
    """Create iterator for the `dataset` to fetch data to worker's `devices` .

    `MultiDeviceIterator` is used to prefetch input to the devices on the
    given worker. `MultiDeviceIterator` doesn't work in eager mode yet.

    Args:
      dataset: A `tf.data.Dataset` instance.
      worker: Worker on which ops should be created.
      devices: Distribute data from `dataset` to these devices.
    """
    self._dataset = dataset
    self._worker = worker
    self._devices = devices
    self._is_eager = context.executing_eagerly()
    self._make_iterator()

  def _make_iterator(self):
    """Make appropriate iterator on the dataset."""
    with ops.device(self._worker):
      if self._is_eager:
        # TODO(rohanj): Enable prefetching in eager mode.
        # TODO(priyag): Measure the performance of this approach vs calling
        # get_next on the original dataset N times.
        dataset = self._dataset.batch(len(self._devices), drop_remainder=True)
        iterator = dataset.make_one_shot_iterator()
      else:
        iterator = multi_device_iterator_ops.MultiDeviceIterator(
            self._dataset, self._devices)
    self._iterator = iterator

  def get_next(self, name=None):
    """Get next element from the underlying iterator."""
    with ops.device(self._worker):
      if self._is_eager:
        # Batched dataset case.
        batch = self._iterator.get_next(name=name)
        index = {}
        for i, d in enumerate(self._devices):
          index[d] = nest.map_structure(operator.itemgetter(i), batch)
          with ops.device(d):
            index[d] = nest.map_structure(array_ops.identity, index[d])
      else:
        # MultiDeviceIterator case.
        data_list = self._iterator.get_next()
        index = dict(zip(self._devices, data_list))

      return regroup(index)

  def initialize(self):
    """Initialze underlying iterator.

    In eager execution, this simply recreates the underlying iterator.
    In graph execution, it returns the initializer ops for the underlying
    iterator.

    Returns:
      A list of any initializer ops that should be run.
    """
    if self._is_eager:
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
  batch_size = batched_dataset._batch_size
  drop_remainder = batched_dataset._drop_remainder
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
  return dataset.batch(new_batch_size, drop_remainder=drop_remainder)


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
    if distribution_strategy_context.get_cross_replica_context():
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
    if distribution_strategy_context.get_cross_replica_context():
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

  def __init__(self, v, aggregation):
    self._v = v
    # NOTE: We don't use "_distributed_container" here because we don't want
    # to trigger that code path in regroup().
    v._aggregating_container = weakref.ref(self)  # pylint: disable=protected-access
    self._aggregation = aggregation

  def get(self):
    return self._v

  def __getattr__(self, name):
    return getattr(self._v, name)

  def _assign_func(self, *args, **kwargs):
    f = kwargs.pop("f")
    if distribution_strategy_context.get_cross_replica_context():
      update_device = distribute_lib.get_update_device()
      if update_device is not None:
        # We are calling an assign function in an update context.
        return f(self._v, *args, **kwargs)

      # We are calling an assign function in cross replica context, wrap it in
      # an update call.
      return distribution_strategy_context.get_distribution_strategy().update(
          self, f, *args, **kwargs)
    else:
      assert distribution_strategy_context.get_replica_context()
      # We are calling an assign function in replica context.
      # We reduce the value we want to assign/add/sub. More details about how we
      # handle the different use cases can be found in the _reduce method.
      # We call the function with the reduced value.
      if self._aggregation == vs.VariableAggregation.NONE:
        raise ValueError("You must specify an aggregation method to update a "
                         "a variable in Replica Context.")

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
