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

import copy
import weakref

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.types import core
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.types import trace


def _on_write_update_replica(var, update_fn, value, **kwargs):
  """Updates variables with ON_WRITE synchronization in replica context."""
  if var.aggregation == vs.VariableAggregation.NONE:
    return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access

  if not ds_context.get_strategy().extended._use_merge_call():  # pylint: disable=protected-access
    # Don't allow MEAN with non float dtype, since it may cause unexpected
    # precision loss. Python3 and NumPy automatically upcast integers to
    # float in division, but we should always preserve the type.
    if var.aggregation == vs.VariableAggregation.MEAN and (
        not var.dtype.is_floating) and tensor_util.is_tf_type(value):
      raise ValueError(
          "Cannot update non-float variables with "
          "tf.VariableAggregation.MEAN aggregation in replica context. "
          "Either change the variable dtype to float or update it in "
          "cross-replica context.")

    aggregated_value = apply_aggregation_replica_context(
        value, var.aggregation, var)
    values_util.mark_as_unsaveable()

    return ds_context.get_replica_context()._update(  # pylint: disable=protected-access
        var,
        update_fn,
        args=(aggregated_value,),
        kwargs=kwargs,
        group=True)

  else:

    def merge_fn(strategy, value, **kwargs):
      """Aggregate values and update all variables in cross replica context."""
      # Don't allow MEAN with non float dtype, since it may cause unexpected
      # precision loss. Python3 and NumPy automatically upcast integers to
      # float in division, but we should always preserve the type.
      #
      # Note that to be backward compatible we allow the case when the value
      # is *always* the same on each replica. I.E. value is not a
      # PerReplica. Refer to regroup() to see how values are grouped.
      if var.aggregation == vs.VariableAggregation.MEAN and (
          not var.dtype.is_floating) and isinstance(value, PerReplica):
        raise ValueError(
            "Cannot update non-float variables with "
            "tf.VariableAggregation.MEAN aggregation in replica context. "
            "Either change the variable dtype to float or update it in "
            "cross-replica context.")

      assert strategy == var.distribute_strategy
      v = values_util.apply_aggregation(strategy, value, var.aggregation, var)
      return var._update_cross_replica(update_fn, v, **kwargs)  # pylint: disable=protected-access

    return ds_context.get_replica_context().merge_call(
        merge_fn, args=(value,), kwargs=kwargs)


def apply_aggregation_replica_context(value, aggregation, destinations):
  """Aggregate `value` to `destinations` as specified by `aggregation`."""
  # if it is a python literal, return without aggregation
  if isinstance(value, DistributedValues):
    raise TypeError(
        "Cannot use DistributedValues to update variables in replica context.")
  if not tensor_util.is_tf_type(value):
    return value

  if aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
    # Switch to cross-replica context to broadcast
    def merge_fn(strategy, value):
      return strategy.extended.broadcast_to(
          strategy.experimental_local_results(value)[0],
          destinations=destinations)

    return ds_context.get_replica_context().merge_call(merge_fn, args=(value,))

  else:
    reduce_op = reduce_util.ReduceOp.from_variable_aggregation(aggregation)
    aggregated_value = ds_context.get_strategy(  # pylint: disable=protected-access
    ).extended._replica_ctx_all_reduce(reduce_op, value)
    return aggregated_value


class DistributedValues(ds_types.DistributedValues):
  """Base class for representing distributed values."""

  def __init__(self, values):
    """Should only be called by subclass __init__."""
    self._values = tuple(values)

  def _get(self):
    """Returns the value for the current device or raises a ValueError."""
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      return self._get_cross_replica()
    else:
      return self._values[replica_id]

  def _get_cross_replica(self):
    raise NotImplementedError(
        "DistributedValues._get_cross_replica should be implemented by "
        "sub-classes which support cross-replica accesses.")

  def _get_on_device_or_primary(self):
    """Returns value in same replica or device if possible, else the _primary."""
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      # Try to find a value on the current device.
      current_device = device_util.canonicalize(device_util.current())
      for value in self._values:
        if device_util.canonicalize(value.device) == current_device:
          return value
      return self._primary
    else:
      return self._values[replica_id]

  @property
  def _primary(self):
    """Returns a representative component."""
    return self._values[0]

  @property
  def _devices(self):
    return tuple(v.device for v in self._values)

  def __str__(self):
    debug_str = ",\n".join(
        "  %d: %s" % (i, v) for i, v in enumerate(self._values))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_str)

  def __repr__(self):
    debug_repr = ",\n".join(
        "  %d: %r" % (i, v) for i, v in enumerate(self._values))
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
    if name.startswith("_self_") or name in ("_use_resource_variables",
                                             "_attribute_sentinel",
                                             "_distributed_container"):
      return super(DistributedDelegate, self).__getattr__(name)

    # This allows copy.copy(DistributedDelegate). When copying an object,
    # copy.copy doesn't invoke its __init__ method, instead it makes a new
    # empty object, then copies the attributes over. copy.copy looks for
    # attributes like "__getstate__" in case the object implements its custom
    # copying. Since DistributedDelegate doesn't have those attributes defined,
    # __getattr__ will be invoked, which tries to access "_values" attributes,
    # but that doesn't exist either because this is an empty object, and again
    # __getattr__ is invoked, leading to an infinite recursion.
    if name == "_values":
      raise AttributeError()

    # TODO(priyag): This needs to be made robust against pitfalls from mix use
    # __getattr__ and @property. See b/120402273.
    return getattr(self._get(), name)

  @property
  def values(self):
    """Returns the per replica values."""
    return self._values

  def _get_as_operand(self):
    """Returns the value for operations for the current device.

    Some implementations, e.g. `TPUMirroredVariable`, are not able to return the
    value type within a replica context. They can, however, return a value that
    can be used by the operations below.
    """
    return self._get()

  # pylint: disable=multiple-statements
  def __add__(self, o):
    return self._get_as_operand() + o

  def __radd__(self, o):
    return o + self._get_as_operand()

  def __sub__(self, o):
    return self._get_as_operand() - o

  def __rsub__(self, o):
    return o - self._get_as_operand()

  def __mul__(self, o):
    return self._get_as_operand() * o

  def __rmul__(self, o):
    return o * self._get_as_operand()

  def __truediv__(self, o):
    return self._get_as_operand() / o

  def __rtruediv__(self, o):
    return o / self._get_as_operand()

  def __floordiv__(self, o):
    return self._get_as_operand() // o

  def __rfloordiv__(self, o):
    return o // self._get_as_operand()

  def __mod__(self, o):
    return self._get_as_operand() % o

  def __rmod__(self, o):
    return o % self._get_as_operand()

  def __lt__(self, o):
    return self._get_as_operand() < o

  def __le__(self, o):
    return self._get_as_operand() <= o

  def __gt__(self, o):
    return self._get_as_operand() > o

  def __ge__(self, o):
    return self._get_as_operand() >= o

  def __and__(self, o):
    return self._get_as_operand() & o

  def __rand__(self, o):
    return o & self._get_as_operand()

  def __or__(self, o):
    return self._get_as_operand() | o

  def __ror__(self, o):
    return o | self._get_as_operand()

  def __xor__(self, o):
    return self._get_as_operand() ^ o

  def __rxor__(self, o):
    return o ^ self._get_as_operand()

  def __getitem__(self, o):
    return self._get_as_operand()[o]

  def __pow__(self, o, modulo=None):
    return pow(self._get_as_operand(), o, modulo)

  def __rpow__(self, o):
    return pow(o, self._get_as_operand())

  def __invert__(self):
    return ~self._get_as_operand()

  def __neg__(self):
    return -self._get_as_operand()

  def __abs__(self):
    return abs(self._get_as_operand())

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


class PerReplica(DistributedValues, composite_tensor.CompositeTensor,
                 ds_types.PerReplica):
  """Holds a map from replica to unsynchronized values."""

  @property
  def _type_spec(self):
    return PerReplicaSpec(
        *(type_spec.type_spec_from_value(v) for v in self._values))

  @property
  def values(self):
    """Returns the per replica values."""
    return self._values


def _per_replica_to_tensor(var, dtype=None, name=None, as_ref=False):
  """Converts a `PerReplica` to a `Tensor`."""
  del name
  if dtype is not None and not dtype.is_compatible_with(var.dtype):
    raise ValueError(
        "Incompatible type conversion requested to type {!r} for variable "
        "of type {!r}".format(dtype.name, var.dtype.name))
  if as_ref:
    raise NotImplementedError(
        "PerReplica doesn't support being used as a reference.")
  if ds_context.in_cross_replica_context() or not ds_context.has_strategy():
    raise ValueError("It looks like you are using a PerReplica object while "
                     "not inside a replica context, which is not supported. "
                     "Try running your op or function inside a replica context "
                     "by using `strategy.run`")
  else:
    replica_id = values_util.get_current_replica_id_as_int()
    return var.values[replica_id]

# Register a conversion function to provide a useful error message when users
# try to use PerReplica values in the wrong contexts
ops.register_tensor_conversion_function(PerReplica, _per_replica_to_tensor)


class PerReplicaSpec(type_spec.TypeSpec):
  """Type specification for a `PerReplica`."""

  __slots__ = ["_value_specs"]

  value_type = property(lambda self: PerReplica)

  def __init__(self, *value_specs):
    self._value_specs = tuple(value_specs)

  def _serialize(self):
    return self._value_specs

  @property
  def _component_specs(self):
    return self._value_specs

  def _to_components(self, value):
    replica_context = ds_context.get_replica_context()
    if replica_context is not None and replica_context.num_replicas_in_sync > 1:
      raise ValueError(
          "Flattening a PerReplica to components is not supported in replica "
          "context.")
    return value._values  # pylint: disable=protected-access

  def _from_components(self, tensor_list):
    return PerReplica(tensor_list)


# Note that unlike PerReplica, Mirrored values inherit from
# DistributedDelegate and so can be used directly in cross-replica mode.
# TODO(tomhennigan) Should this extend CompositeTensor?
class Mirrored(DistributedDelegate, ds_types.Mirrored):
  """Holds a map from replica to values which are kept in sync."""

  def _get_cross_replica(self):
    return self._get_on_device_or_primary()

  def _as_graph_element(self):
    obj = self._get()
    conv_fn = getattr(obj, "_as_graph_element", None)
    if conv_fn and callable(conv_fn):
      return conv_fn()
    return obj


class DistributedVarOp(object):
  """A class that looks like `tf.Operation`."""

  def __init__(self, name, graph, traceback, typ):
    self.name = name
    self.graph = graph
    self.traceback = traceback
    self.type = typ

  def __eq__(self, o):
    if not isinstance(o, self.__class__):
      raise NotImplementedError
    return (self.name == o.name and self.graph == o.graph and
            self.traceback == o.traceback and self.type == o.type)

  def __hash__(self):
    return hash((self.name, self.graph, tuple(self.traceback), self.type))


# TODO(b/209081027): Remove this once Variable is a CompositeTensor.
class DistributedVariableTraceType(trace.TraceType):
  """TraceType of DistributedVariable objects."""

  def __init__(self, distributed_variable):
    self.distributed_variable = distributed_variable
    self.components = (tuple(distributed_variable.shape.as_list()),
                       distributed_variable.dtype)

  def is_subtype_of(self, other):
    return self == other

  def most_specific_common_supertype(self, others):
    return self if all(self == other for other in others) else None

  def _placeholder_value(self):
    return self.distributed_variable

  def __hash__(self) -> int:
    return hash(self.components)

  def __eq__(self, other) -> bool:
    if not isinstance(other, DistributedVariableTraceType):
      return False

    return self.components == other.components


class DistributedVariable(DistributedDelegate, variables_lib.Variable,
                          core.Tensor):
  """Holds a map from replica to variables."""

  def __init__(self, strategy, values, aggregation, var_policy=None):
    if (aggregation == variables_lib.VariableAggregation.MEAN and
        not values[0].dtype.is_floating):
      raise ValueError(
          "creating distributed tf.Variable with aggregation=MEAN and a "
          "non-floating dtype is not supported, please use a different "
          "aggregation or dtype")
    self._distribute_strategy = strategy
    self._aggregation = aggregation
    super(DistributedVariable, self).__init__(values)
    self._common_name = self._primary.name.split(":")[0]
    # Use a weakref to make it easy to map from the contained values
    # to the container without introducing a reference cycle.
    for v in values:
      v._distributed_container = weakref.ref(self)  # pylint: disable=protected-access

    # Packed variable is used to reduce the overhead of function execution.
    # For a DistributedVariable, only one variable handle is captured into a
    # function graph. It's only supported in eager mode.
    if ops.executing_eagerly_outside_functions() and getattr(
        strategy, "_enable_packed_variable_in_eager_mode", False):
      name = "%s/packed/" % self._common_name
      self._packed_var = packed.PackedDistributedVariable(values, name=name)
    else:
      self._packed_var = None

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
    # Set a VariablePolicy which decides how we replicate/aggregate the given
    # variable.
    self._policy = var_policy

  def __deepcopy__(self, memo):
    """Perform a deepcopy of the `DistributedVariable`.

    Unlike the deepcopy of a regular tf.Variable, this keeps the original
    strategy and devices of the `DistributedVariable`.  To avoid confusion
    with the behavior of deepcopy on a regular `Variable` (which does
    copy into new devices), we only allow a deepcopy of a `DistributedVariable`
    within its originating strategy scope.

    Args:
      memo: The memoization object for `deepcopy`.

    Returns:
      A deep copy of the current `DistributedVariable`.

    Raises:
      RuntimeError: If trying to deepcopy into a different strategy.
    """
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      new_values = []

      for value in self._values:
        with ops.device(value.device):
          new_values.append(copy.deepcopy(value, memo))

    copied_variable = type(self)(
        strategy=self._distribute_strategy,
        values=new_values,
        aggregation=self._aggregation,
        var_policy=copy.deepcopy(self._policy, memo))

    memo[id(self)] = copied_variable

    return copied_variable

  def _use_packed_variable(self):
    # Don't use packed variable when under a SaveContext to avoid explicit
    # device placement on variable consuming ops.
    return self._packed_var is not None and not save_context.in_save_context()

  def is_initialized(self, name=None):
    """Identifies if all the component variables are initialized.

    Args:
      name: Name of the final `logical_and` op.

    Returns:
      The op that evaluates to True or False depending on if all the
      component variables are initialized.
    """
    if values_util.is_saving_non_distributed():
      return self._primary.is_initialized()
    if self._use_packed_variable():
      return self._packed_var.is_initialized()
    result = self._primary.is_initialized()
    # We iterate through the list of values except the last one to allow us to
    # name the final `logical_and` op the same name that is passed by the user
    # to the `is_initialized` op. For distributed variables, the
    # `is_initialized` op is a `logical_and` op.
    for v in self._values[1:-1]:
      result = math_ops.logical_and(result, v.is_initialized())
    result = math_ops.logical_and(
        result, self._values[-1].is_initialized(), name=name)
    return result

  @property
  def initializer(self):
    if values_util.is_saving_non_distributed():
      return self._primary.initializer
    if self._initializer_op:
      init_op = self._initializer_op
    else:
      # return grouped ops of all the var initializations of component values of
      # the mirrored variable
      init_op = control_flow_ops.group(
          tuple(v.initializer for v in self._values))
    return init_op

  def initialized_value(self):
    return self._get_on_device_or_primary().initialized_value()

  @property
  def initial_value(self):
    return self._get_on_device_or_primary().initial_value

  @property
  def constraint(self):
    return self._primary.constraint

  @property
  def graph(self):
    return self._primary.graph

  @property
  def _shared_name(self):
    return self._common_name

  @property
  def _unique_id(self):
    return self._primary._unique_id  # pylint: disable=protected-access

  @property
  def _graph_key(self):
    """Lets Optimizers know which graph this variable is from."""
    return self._primary._graph_key  # pylint: disable=protected-access

  @property
  def name(self):
    return self._primary.name

  @property
  def dtype(self):
    return self._primary.dtype

  @property
  def shape(self):
    return self._primary.shape

  @property
  def synchronization(self):
    return self._primary.synchronization

  @property
  def aggregation(self):
    return self._aggregation

  @property
  def _packed_variable(self):
    if self._use_packed_variable():
      return self._packed_var
    return None

  @property
  def handle(self):
    if values_util.is_saving_non_distributed():
      return self._primary.handle
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      raise ValueError(
          "DistributedVariable.handle is not available outside the replica "
          "context or a `tf.distribute.Strategy.update()` call.")
    else:
      if self._use_packed_variable():
        return self._packed_var.handle
      return self._values[replica_id].handle

  def eval(self, session=None):
    return self._get_on_device_or_primary().eval(session)

  @property
  def _save_slice_info(self):
    return self._primary._save_slice_info  # pylint: disable=protected-access

  def _get_save_slice_info(self):
    return self._primary._get_save_slice_info()  # pylint: disable=protected-access

  def _set_save_slice_info(self, save_slice_info):
    for v in self._values:
      v._set_save_slice_info(save_slice_info)  # pylint: disable=protected-access

  @property
  def device(self):
    return self._get_on_device_or_primary().device

  @property
  def trainable(self):
    return self._primary.trainable

  @property
  def distribute_strategy(self):
    return self._distribute_strategy

  def get_shape(self):
    return self._primary.get_shape()

  def to_proto(self, export_scope=None):
    return self._primary.to_proto(export_scope=export_scope)

  @property
  def op(self):
    if values_util.is_saving_non_distributed():
      return self._primary.op
    # We want cross-replica code that does some var.op.X calls
    # to work (even if the current device isn't in self._devices), but
    # other uses of var.op in a cross-replica context to fail.
    if ds_context.in_cross_replica_context():
      return DistributedVarOp(self._primary.op.name, self._primary.op.graph,
                              self._primary.op.traceback, self._primary.op.type)
    return self._get().op

  @property
  def _in_graph_mode(self):
    return self._primary._in_graph_mode  # pylint: disable=protected-access

  def _get_replica(self, replica_id):
    """Returns the value on a device with the given replica_id."""
    if self._use_packed_variable():
      return self._packed_var.on_device(self._devices[replica_id])
    return self._values[replica_id]

  def _get(self):
    """Returns the value for the current device or raises a ValueError."""
    if values_util.is_saving_non_distributed():
      return self._primary
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      return self._get_cross_replica()
    else:
      return self._get_replica(replica_id)

  def _get_on_device_or_primary(self):
    """Returns value in same replica or device if possible, else the _primary."""
    if values_util.is_saving_non_distributed():
      return self._primary
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      # Try to find a value on the current device.
      current_device = device_util.canonicalize(device_util.current())
      for i, value in enumerate(self._values):
        if device_util.canonicalize(value.device) == current_device:
          return self._get_replica(i)
      return self._get_replica(0)
    else:
      return self._get_replica(replica_id)

  def read_value(self):
    if values_util.is_saving_non_distributed():
      return self._primary.read_value()
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return array_ops.identity(self._get())

  def value(self):
    if values_util.is_saving_non_distributed():
      return self._primary.value()
    if self._policy:
      return self._policy.value(self)
    return self._get_on_device_or_primary().value()

  def numpy(self):
    if context.executing_eagerly():
      return self.read_value().numpy()
    else:
      raise NotImplementedError("DistributedVariable.numpy() is only available "
                                "when eager execution is enabled.")

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign_sub(value, use_locking, name, read_value)
    if self._policy:
      return self._policy.assign_sub(
          self,
          value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    return values_util.on_write_assign_sub(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign_add(value, use_locking, name, read_value)
    if self._policy:
      return self._policy.assign_add(
          self,
          value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    return values_util.on_write_assign_add(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign(value, use_locking, name, read_value)
    if self._policy:
      return self._policy.assign(
          self,
          value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    return values_util.on_write_assign(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_sub(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_sub(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_sub(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_add(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_add(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_add(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_mul(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_mul(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_mul(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_div(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_div(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_div(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_min(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_min(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_max(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_max(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_update(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_update(
        self, sparse_delta, use_locking=use_locking, name=name)

  def __tf_tracing_type__(self, _):
    return DistributedVariableTraceType(self)

  def _gather_saveables_for_checkpoint(self):
    """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    DistributedVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """

    def _saveable_factory(name=self._common_name):
      return _DistributedVariableSaveable(self, self._primary, name)

    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _as_graph_element(self):
    if values_util.is_saving_non_distributed():
      return self._primary._as_graph_element()  # pylint: disable=protected-access
    if self._policy:
      return self._policy._as_graph_element(self)  # pylint: disable=protected-access

    raise NotImplementedError(
        "DistributedVariable._as_graph_element requires a valid "
        "VariablePolicy. Please set the policy via the `var_policy` argument "
        "in the constructor, or override this method in sub-classes which "
        "support cross-replica accesses.")

  def _get_cross_replica(self):
    if values_util.is_saving_non_distributed():
      return self._primary
    if self._policy:
      return self._policy._get_cross_replica(self)  # pylint: disable=protected-access

    raise NotImplementedError(
        "DistributedVariable._get_cross_replica requires a valid "
        "VariablePolicy. Please set the policy via the `var_policy` argument "
        "in the constructor, or override this method in sub-classes which "
        "support cross-replica accesses.")

  def _update_cross_replica(self, update_fn, value, **kwargs):
    """Applies updates across replicas.

    Args:
      update_fn: A callable to pass to `strategy.extended.update` to update the
        variable. It should has the same signature as `Variable.assign()`.
      value: value to be passed to `update_fn`.
      **kwargs: remaining arguments to `update_fn`.

    Returns:
      Updated variable or `tf.Operation`.
    """
    values_util.mark_as_unsaveable()
    return self.distribute_strategy.extended.update(
        self, update_fn, args=(value,), kwargs=kwargs, group=True)

  def _update_replica(self, update_fn, value, **kwargs):
    """Applies updates in one replica.

    Args:
      update_fn: A callable to update the variable. It should has the same
        signature as `Variable.assign()`.
      value: value to be passed to `update_fn`.
      **kwargs: remaining arguments to `update_fn`.

    Returns:
      Updated variable or `tf.Operation`.
    """
    if self._policy:
      return self._policy._update_replica(self, update_fn, value, **kwargs)  # pylint: disable=protected-access
    raise NotImplementedError(
        "DistributedVariable._update_replica requires a valid VariablePolicy. "
        "Please set the policy via the `var_policy` argument in the "
        "constructor, or override this method in sub-classes which support "
        "cross-replica accesses.")

  def _update(self, update_fn, value, **kwargs):
    """Applies updates depending on the context.

    The method calls `_update_replica` in replica context,
    `_update_cross_replica` in cross replica context, and `update_fn` in update
    context.

    If `read_value` is True, the method returns the updated Variable. If
    `read_value` is False, the method returns the update `tf.Operation`.

    Args:
      update_fn: A callable to pass to `strategy.extended.update` to update the
        variable. It should have the same signature as `Variable.assign()`.
      value: value to be passed to `update_fn`.
      **kwargs: keyword arguments to `update_fn`.

    Returns:
      Updated variable or `tf.Operation`.

    """
    if values_util.is_saving_non_distributed():
      return update_fn(self._primary, value, **kwargs)
    with ds_context.enter_or_assert_strategy(self.distribute_strategy):
      if ds_context.in_cross_replica_context():
        update_replica_id = distribute_lib.get_update_replica_id()
        if update_replica_id is not None:
          replica_value = self._get_replica(update_replica_id)
          return update_fn(replica_value, value, **kwargs)
        return self._update_cross_replica(update_fn, value, **kwargs)
      else:
        values_util.assert_replica_context(self.distribute_strategy)
        return self._update_replica(update_fn, value, **kwargs)

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    if values_util.is_saving_non_distributed():
      return ops.convert_to_tensor(
          self._primary, dtype=dtype, name=name, as_ref=as_ref)
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return ops.convert_to_tensor(
          self._get(), dtype=dtype, name=name, as_ref=as_ref)

  def _map_resources(self, save_options):
    """For implementing `Trackable`."""
    # Initialize for self._primary first, so that obj_map[self._primary] and
    # resource_map[self._primary.handle] contain mapped values.
    obj_map, resource_map = self._primary._map_resources(save_options)  # pylint:disable=protected-access
    for v in [v for v in self._values if v != self._primary]:

      if (save_options.experimental_variable_policy  # pylint:disable=protected-access
          ._expand_distributed_variables()):
        v_obj_map, v_resource_map = v._map_resources(save_options)  # pylint:disable=protected-access
        obj_map.update(v_obj_map)
        resource_map.update(v_resource_map)
      else:
        obj_map[v] = obj_map[self._primary]
        resource_map[v.handle] = resource_map[self._primary.handle]
    obj_map[self] = obj_map[self._primary]
    resource_map[self] = resource_map[self._primary.handle]
    if self._packed_var is not None:
      resource_map[self._packed_var.packed_handle] = resource_map[
          self._primary.handle]
    return obj_map, resource_map

  def _write_object_proto(self, proto, options):
    """Update a SavedObject proto for the caller.

    If a DistributedVariable object supports this method, it will be called when
    saving with a pre-built `SavedObject` proto representing the object, plus an
    instance of `SaveOptions`. This method is then free to modify that proto
    instance.

    `DistributedVariable` with `AUTO` or `ON_WRITE` synchronization optionally
    write out information about their components to the
    `experimental_distributed_variable_components` field of a
    `SavedVariable` (depending on the `SaveOptions` variable policy).

    Args:
      proto: A pre-built `SavedObject` proto for this object. It is assumed this
        will be a `SavedVariable` instance.
      options: A `SaveOptions` instance.
    """
    resource_variable_ops.write_object_proto_for_resource_variable(
        self, proto, options)
    if self._policy:
      if self._policy._is_mirrored():  # pylint: disable=protected-access
        self._policy._write_object_proto(self, proto, options)  # pylint: disable=protected-access

  @property
  def is_distributed_variable(self):
    return True

  def __tf_experimental_restore_capture__(
      self, concrete_function, internal_capture):
    concrete_function.graph.capture_distributed_variable(self, internal_capture)
    return self


# We extend from `saveable_object.SaveableObject` instead of
# `saveable_object_util.ResourceVariableSaveable` since we need to read the
# value of ONREAD variables when saving. `SaveableObject` provides a way to
# specify the function to run to get the value of the variable or tensor at
# saving time. We can use this for both ON_READ and ON_WRITE variables.
# TODO(b/164586507): Consolidate ON_WRITE and ON_READ saving/restoring logic
# if possible.
class _DistributedVariableSaveable(saveable_object.SaveableObject):
  """Class for defining how to restore a DistributedVariable."""

  def __init__(self, distributed_variable, primary_variable, name):
    self._distributed_variable = distributed_variable
    if not self._distributed_variable._policy:
      raise ValueError(
          "The VariablePolicy of the argument `distributed_variable` must be "
          "set to create a _DistributedVariableSaveable. Please set it via "
          "the `var_policy` argument in the constructor of DistributedVariable."
      )
    tensor, spec = distributed_variable._policy.get_saveable(
        distributed_variable, primary_variable, name)
    super(_DistributedVariableSaveable, self).__init__(tensor, spec, name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return self._distributed_variable._policy.get_restore_ops(  # pylint: disable=protected-access
        self._distributed_variable, tensor)


class _MirroredSaveable(saveable_object.SaveableObject):
  """Class for defining how to restore a MirroredVariable."""

  def __init__(self, mirrored_variable, primary_variable, name):
    self._mirrored_variable = mirrored_variable
    tensor, spec = values_util.get_on_write_saveable(self._mirrored_variable,
                                                     primary_variable, name)
    super(_MirroredSaveable, self).__init__(tensor, spec, name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return values_util.get_on_write_restore_ops(self._mirrored_variable, tensor)


class MirroredVariable(DistributedVariable, Mirrored):
  """Holds a map from replica to variables whose values are kept in sync."""

  def _update_replica(self, update_fn, value, **kwargs):
    return _on_write_update_replica(self, update_fn, value, **kwargs)

  def scatter_min(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(*args, **kwargs)
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_min", aggregation=self._aggregation))
    return super(MirroredVariable, self).scatter_min(*args, **kwargs)

  def scatter_max(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(*args, **kwargs)
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_max", aggregation=self._aggregation))
    return super(MirroredVariable, self).scatter_max(*args, **kwargs)

  def scatter_update(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(*args, **kwargs)
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_update", aggregation=self._aggregation))
    return super(MirroredVariable, self).scatter_update(*args, **kwargs)

  def _get_cross_replica(self):
    # Return identity, to avoid directly exposing the variable to the user and
    # allowing it to be modified by mistake.
    return array_ops.identity(Mirrored._get_cross_replica(self))

  def _as_graph_element(self):
    return self._get_on_device_or_primary()._as_graph_element()  # pylint: disable=protected-access

  def _gather_saveables_for_checkpoint(self):
    """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    MirroredVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """

    def _saveable_factory(name=self._common_name):
      return _MirroredSaveable(self, self._primary, name)

    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _write_object_proto(self, proto, options):
    """Update a SavedObject proto for the caller.

    If a DistributedVariable object supports this method, it will be called when
    saving with a pre-built `SavedObject` proto representing the object, plus an
    instance of `SaveOptions`. This method is then free to modify that proto
    instance.

    `DistributedVariable` with `AUTO` or `ON_WRITE` synchronization optionally
    write out information about their components to the
    `experimental_distributed_variable_components` field of a
    `SavedVariable` (depending on the `SaveOptions` variable policy).

    Args:
      proto: A pre-built `SavedObject` proto for this object. It is assumed this
        will be a `SavedVariable` instance.
      options: A `SaveOptions` instance.
    """
    super(MirroredVariable, self)._write_object_proto(proto, options)
    values_util.write_object_proto(self, proto, options)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # TODO(b/154017756): Make _dense_var_to_tensor consistent between ON_READ
    # and ON_WRITE.
    # Try to avoid assignments to and other mutations of MirroredVariable
    # state except through a DistributionStrategy.extended.update() or any of
    # the `assign*` and `scatter*` calls.
    if as_ref:
      # A TF 1.x case where the variable is a boolean variable and used like:
      # tf.cond(v, true_fn, false_fn).
      raise ValueError(
          "You may be using variable created under distribute strategy in TF "
          "1.x control flows. Try explicitly converting the variable to Tensor "
          "using variable.read_value(), or switch to TF 2.x.")
    return ops.convert_to_tensor(
        self._get(), dtype=dtype, name=name, as_ref=as_ref)


class _SyncOnReadSaveable(saveable_object.SaveableObject):
  """Class for defining how to restore a SyncOnReadVariable."""

  def __init__(self, sync_on_read_variable, name):
    self._sync_on_read_variable = sync_on_read_variable
    tensor, spec = values_util.get_on_read_saveable(
        sync_on_read_variable, sync_on_read_variable._primary, name)

    super(_SyncOnReadSaveable, self).__init__(tensor, spec, name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return values_util.get_on_read_restore_ops(
        self._sync_on_read_variable, tensor,
        self._sync_on_read_variable.aggregation)


class SyncOnReadVariable(DistributedVariable):
  """Holds a map from replica to variables whose values are reduced on save."""

  def _update_replica(self, update_fn, value, **kwargs):
    return update_fn(self._get_on_device_or_primary(), value, **kwargs)

  def _get(self):
    """Returns the value of SyncOnReadVariable based on surrounding context.

    If called under a non-default replica-context, returns the corresponding
    variable on that replica.
    If called under default replica-context or cross-replica context, returns
    the synced value.
    """
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return super(SyncOnReadVariable, self)._get()

  # TODO(b/154017756): Make assign behaivor in cross replica context consistent
  # with MirroredVariable.
  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign_sub(value, use_locking, name, read_value)
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if (ds_context.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_sub_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable,
                     self).assign_sub(value, use_locking, name, read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign_add(value, use_locking, name, read_value)
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if (ds_context.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_add_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable,
                     self).assign_add(value, use_locking, name, read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign(value, use_locking, name, read_value)
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if (ds_context.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable, self).assign(value, use_locking, name,
                                                      read_value)

  def _scatter_not_implemented(self, method):
    raise NotImplementedError(
        f"Variables with `synchronization=ON_READ` doesn't support `{method}`")

  def scatter_sub(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_sub(*args, **kwargs)
    self._scatter_not_implemented("scatter_sub")

  def scatter_add(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_add(*args, **kwargs)
    self._scatter_not_implemented("scatter_add")

  def scatter_mul(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_mul(*args, **kwargs)
    self._scatter_not_implemented("scatter_mul")

  def scatter_div(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_div(*args, **kwargs)
    self._scatter_not_implemented("scatter_div")

  def scatter_min(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(*args, **kwargs)
    self._scatter_not_implemented("scatter_min")

  def scatter_max(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(*args, **kwargs)
    self._scatter_not_implemented("scatter_max")

  def scatter_update(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(*args, **kwargs)
    self._scatter_not_implemented("scatter_update")

  def value(self):
    if ds_context.in_variable_sync_on_read_context():
      raise NotImplementedError(
          "call `variable.value()` inside variable_sync_on_read_context is not "
          "supported")
    if values_util.is_saving_non_distributed():
      return self._primary.value()
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if (ds_context.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
          return self._get_replica(0).value()
        return self._get_cross_replica()
      else:
        # _get_on_device_or_primary() returns a Variable.
        return self._get_on_device_or_primary().value()

  def read_value(self):
    if ds_context.in_variable_sync_on_read_context():
      raise NotImplementedError(
          "call `variable.read_value()` inside variable_sync_on_read_context is"
          " not supported")
    return super().read_value()

  def _get_cross_replica(self):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      # Consider returning a tensor value here to make the return value of
      # _get_cross_replica consistent.
      return self._get_replica(0)
    if self._aggregation == vs.VariableAggregation.SUM:
      values_util.mark_as_unsaveable()
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return self._distribute_strategy.reduce(
          reduce_util.ReduceOp.from_variable_aggregation(self._aggregation),
          self,
          axis=None)

  def _as_graph_element(self):
    if values_util.is_saving_non_distributed():
      return self._primary._as_graph_element()  # pylint: disable=protected-access
    # pylint: disable=protected-access
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if ds_context.in_cross_replica_context():
        return ops.convert_to_tensor(self._get_cross_replica())
    return self._get()._as_graph_element()

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
    """Converts a SyncOnReadVariable to a tensor."""
    if values_util.is_saving_non_distributed():
      return ops.convert_to_tensor(
          self._primary, dtype=dtype, name=name, as_ref=as_ref)
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      replica_context = ds_context.get_replica_context()
      if (replica_context is not None and
          ds_context.in_variable_sync_on_read_context()):
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
          return ops.convert_to_tensor(
              self._get_replica(0), dtype=dtype, name=name, as_ref=as_ref)
        if self._aggregation == vs.VariableAggregation.SUM:
          values_util.mark_as_unsaveable()
        # pylint: disable=protected-access
        reduced = (
            replica_context.strategy.extended._replica_ctx_all_reduce(
                reduce_util.ReduceOp.from_variable_aggregation(
                    self._aggregation),
                self._get().read_value()))
        return ops.convert_to_tensor(
            reduced, dtype=dtype, name=name, as_ref=as_ref)

      return ops.convert_to_tensor(
          self._get(), dtype=dtype, name=name, as_ref=as_ref)


# Register a conversion functions which reads the value of the variable,
# allowing instances of the class to be used as tensors.
# DistributedVariable
def _tensor_conversion_distributed_var(var,
                                       dtype=None,
                                       name=None,
                                       as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(DistributedVariable,
                                        _tensor_conversion_distributed_var)


# MirroredVariables
def _tensor_conversion_mirrored(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(MirroredVariable,
                                        _tensor_conversion_mirrored)


# Mirrored Values
def _tensor_conversion_mirrored_val(value, dtype=None, name=None, as_ref=False):
  return ops.convert_to_tensor(
      value._get(), dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(Mirrored,
                                        _tensor_conversion_mirrored_val)


# SyncOnReadVariables
def _tensor_conversion_sync_on_read(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(SyncOnReadVariable,
                                        _tensor_conversion_sync_on_read)


class VariablePolicy(object):
  """Policy defining synchronization and aggregation of a distributed variable.

  Given `synchronization` and `aggregation` parameters set on a `tf.Variable`
  during variable creation within `tf.distribute` scope, `tf.distribute` creates
  an appropriate policy object and assigns it to the distributed variable. All
  variable operations are delegated to the respective policy object.
  """

  def __init__(self, aggregation):
    self._aggregation = aggregation

  def value(self):
    raise NotImplementedError(
        "VariablePolicy.value should be overriden by sub-classes.")

  def _is_mirrored(self):
    raise NotImplementedError(
        "VariablePolicy._is_mirrored should be overriden by sub-classes.")

  def _as_graph_element(self, _):
    raise NotImplementedError(
        "VariablePolicy._as_graph_element should be overriden by sub-classes.")

  def _get_cross_replica(self, var):
    raise NotImplementedError(
        "VariablePolicy._get_cross_replica should be overriden by sub-classes.")

  def _update_replica(self, var, update_fn, value, **kwargs):
    raise NotImplementedError(
        "VariablePolicy._update_replica should be overriden by sub-classes.")


class OnReadPolicy(VariablePolicy):
  """Policy defined for `tf.VariableSynchronization.ON_READ` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.ON_READ` and `aggregation` is set to any of the
  values allowed by the `tf.VariableAggregation` enum such as `NONE`, `SUM`,
  `MEAN` or `ONLY_FIRST_REPLICA`when creating a `tf.Variable` in `tf.distribute`
  scope.
  """

  def _is_mirrored(self):
    return False

  def value(self, var):
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if (ds_context.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
          return var._get_replica(0).value()  # pylint: disable=protected-access
        return var._get_cross_replica()  # pylint: disable=protected-access
      else:
        return var._get_on_device_or_primary().value()  # pylint: disable=protected-access

  def _as_graph_element(self, var):
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if ds_context.in_cross_replica_context():
        return ops.convert_to_tensor(var._get_cross_replica())  # pylint: disable=protected-access
    return var._get()._as_graph_element()  # pylint: disable=protected-access

  def _get_cross_replica(self, var):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      return var._get_replica(0)  # pylint: disable=protected-access
    if self._aggregation == vs.VariableAggregation.SUM:
      values_util.mark_as_unsaveable()
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      return var.distribute_strategy.reduce(
          reduce_util.ReduceOp.from_variable_aggregation(self._aggregation),
          var,
          axis=None)

  def _update_replica(self, var, update_fn, value, **kwargs):
    return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access

  def _scatter_not_implemented(self, method):
    raise NotImplementedError(f"ON_READ variables doesn't support `{method}` "
                              "in cross replica context")

  def assign_sub(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    """Subtracts a value from this variable."""
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if (ds_context.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_sub_cross_replica(
            var, value, read_value=read_value)
      else:
        return values_util.on_write_assign_sub(
            var,
            value,
            use_locking=use_locking,
            name=name,
            read_value=read_value)

  def assign_add(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    """Adds a value to this variable."""
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if (ds_context.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_add_cross_replica(
            var, value, read_value=read_value)
      else:
        return values_util.on_write_assign_add(
            var,
            value,
            use_locking=use_locking,
            name=name,
            read_value=read_value)

  def assign(self, var, value, use_locking=False, name=None, read_value=True):
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if (ds_context.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_cross_replica(
            var, value, read_value=read_value)
      else:
        return values_util.on_write_assign(
            var,
            value,
            use_locking=use_locking,
            name=name,
            read_value=read_value)

  def scatter_sub(self, *args, **kwargs):
    del args, kwargs
    self._scatter_not_implemented("scatter_sub")

  def scatter_add(self, *args, **kwargs):
    del args, kwargs
    self._scatter_not_implemented("scatter_add")

  def scatter_mul(self, *args, **kwargs):
    del args, kwargs
    self._scatter_not_implemented("scatter_mul")

  def scatter_div(self, *args, **kwargs):
    del args, kwargs
    self._scatter_not_implemented("scatter_div")

  def scatter_min(self, *args, **kwargs):
    del args, kwargs
    self._scatter_not_implemented("scatter_min")

  def scatter_max(self, *args, **kwargs):
    del args, kwargs
    self._scatter_not_implemented("scatter_max")

  def scatter_update(self, *args, **kwargs):
    del args, kwargs
    self._scatter_not_implemented("scatter_update")

  def get_saveable(self, var, primary_var, name):
    """Create a saveable object for the given variable."""
    return values_util.get_on_read_saveable(var, primary_var, name)

  def get_restore_ops(self, var, tensor):
    """Restore the same value into all variables."""
    return values_util.get_on_read_restore_ops(var, tensor, self._aggregation)


class OnWritePolicy(VariablePolicy):
  """Policy defined for `tf.VariableSynchronization.ON_WRITE` synchronization.

  This policy is created when the following `synchronization` and `aggregation`
  parameters are specified when creating a `tf.Variable` in `tf.distribute`
  scope and `synchronization` is equal to `tf.VariableSynchronization.ON_WRITE`
  or `tf.VariableSynchronization.AUTO`.
  """

  def _is_mirrored(self):
    return True

  def value(self, var):
    return var._get_on_device_or_primary().value()  # pylint: disable=protected-access

  def _as_graph_element(self, var):
    return var._get_on_device_or_primary()._as_graph_element()  # pylint: disable=protected-access

  def _get_cross_replica(self, var):
    # Return identity, to avoid directly exposing the variable to the user and
    # allowing it to be modified by mistake.
    return array_ops.identity(var._get_on_device_or_primary())  # pylint: disable=protected-access

  def _update_replica(self, var, update_fn, value, **kwargs):
    if var.aggregation == variables_lib.VariableAggregation.NONE:
      return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access
    return _on_write_update_replica(var, update_fn, value, **kwargs)

  def assign(self, var, value, use_locking=False, name=None, read_value=True):
    return values_util.on_write_assign(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    return values_util.on_write_assign_add(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_sub(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    return values_util.on_write_assign_sub(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def scatter_sub(self, var, sparse_delta, use_locking=False, name=None):
    return values_util.scatter_sub(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_add(self, var, sparse_delta, use_locking=False, name=None):
    return values_util.scatter_add(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_mul(self, var, sparse_delta, use_locking=False, name=None):
    return values_util.scatter_mul(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_div(self, var, sparse_delta, use_locking=False, name=None):
    return values_util.scatter_div(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_min(self, var, sparse_delta, use_locking=False, name=None):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_min", aggregation=self._aggregation))
    return values_util.scatter_min(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_max(self, var, sparse_delta, use_locking=False, name=None):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_max", aggregation=self._aggregation))
    return values_util.scatter_max(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_update(self, var, sparse_delta, use_locking=False, name=None):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_update", aggregation=self._aggregation))
    return values_util.scatter_update(
        var, sparse_delta, use_locking=use_locking, name=name)

  def get_saveable(self, var, primary_var, name):
    """Saveable ops for AUTO variables."""
    return values_util.get_on_write_saveable(var, primary_var, name)

  def get_restore_ops(self, var, tensor):
    return values_util.get_on_write_restore_ops(var, tensor)

  def _write_object_proto(self, var, proto, options):
    """Update a SavedObject proto for the caller.

    If a DistributedVariable object supports this method, it will be called when
    saving with a pre-built `SavedObject` proto representing the object, plus an
    instance of `SaveOptions`. This method is then free to modify that proto
    instance.

    `DistributedVariable` with `AUTO` or `ON_WRITE` synchronization optionally
    write out information about their components to the
    `experimental_distributed_variable_components` field of a
    `SavedVariable` (depending on the `SaveOptions` variable policy).

    Args:
      var : A DistributedVariable object
      proto: A pre-built `SavedObject` proto for this object. It is assumed this
        will be a `SavedVariable` instance.
      options: A `SaveOptions` instance.
    """
    values_util.write_object_proto(var, proto, options)


class PerWorkerResource():
  """A per-worker CapturableResource class for non-ParameterServer strategy.

  Resources that populate `host_to_resources` should be instances of classes
  subclassing CapturableResource, although currently it's only used and tested
  for StaticHashTable with TPUStrategy.
  """

  def __init__(self, strategy, host_to_resources):
    distribute_lib.distribution_strategy_input_api_counter.get_cell(
        "PerWorkerResource", "TPUDistributedLookupTable").increase_by(1)
    self._strategy = strategy
    self._host_to_resources = host_to_resources

  def __getattribute__(self, name):
    if name not in ("__init__", "__getattribute__", "_host_to_resources",
                    "_strategy", "local_resource"):
      return getattr(self.local_resource(), name)
    return super(PerWorkerResource, self).__getattribute__(name)

  def local_resource(self):
    """Returns the resource on the local worker."""
    current_device = device_util.canonicalize(device_util.current())
    host_device = device_util.canonicalize(
        device_util.get_host_for_device(current_device))
    return self._host_to_resources.get(
        host_device,
        self._host_to_resources[next(iter(self._host_to_resources))])
