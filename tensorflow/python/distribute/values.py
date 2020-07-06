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


from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export


def _on_write_update_replica(var, update_fn, value, **kwargs):
  """Updates variables with ON_WRITE synchronization in replica context."""
  if var.aggregation == vs.VariableAggregation.NONE:
    return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access

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


@tf_export("distribute.DistributedValues", v1=[])
class DistributedValues(object):
  """Base class for representing distributed values.

  A subclass instance of `tf.distribute.DistributedValues` is created when
  creating variables within a distribution strategy, iterating a
  `tf.distribute.DistributedDataset` or through `tf.distribute.Strategy.run`.
  This base class should never be instantiated directly.
  `tf.distribute.DistributedValues` contains a value per replica. Depending on
  the subclass, the values could either be synced on update, synced on demand,
  or never synced.

  `tf.distribute.DistributedValues` can be reduced to obtain single value across
  replicas, as input into `tf.distribute.Strategy.run` or the per-replica values
  inspected using `tf.distribute.Strategy.experimental_local_results`.

  Example usage:

  1. Created from a `tf.distribute.DistributedDataset`:

  >>> strategy = tf.distribute.MirroredStrategy()
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)

  2. Returned by `run`:

  >>> strategy = tf.distribute.MirroredStrategy()
  >>> @tf.function
  ... def run():
  ...   ctx = tf.distribute.get_replica_context()
  ...   return ctx.replica_id_in_sync_group
  >>> distributed_values = strategy.run(run)

  3. As input into `run`:

  >>> strategy = tf.distribute.MirroredStrategy()
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> @tf.function
  ... def run(input):
  ...   return input + 1.0
  >>> updated_value = strategy.run(run, args=(distributed_values,))

  4. Reduce value:

  >>> strategy = tf.distribute.MirroredStrategy()
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> reduced_value = strategy.reduce(tf.distribute.ReduceOp.SUM,
  ...                                 distributed_values,
  ...                                 axis = 0)

  5. Inspect per replica values:

  >>> strategy = tf.distribute.MirroredStrategy()
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> per_replica_values = strategy.experimental_local_results(
  ...    distributed_values)
  >>> per_replica_values
  (<tf.Tensor: shape=(2,), dtype=float32,
   numpy=array([5., 6.], dtype=float32)>,)

  """

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
        "This method should be overridden by sub-classes which support cross-"
        "replica accesses.")

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


class PerReplica(DistributedValues, composite_tensor.CompositeTensor):
  """Holds a map from replica to unsynchronized values."""

  @property
  def _type_spec(self):
    return PerReplicaSpec(
        *(type_spec.type_spec_from_value(v) for v in self._values))

  @property
  def values(self):
    """Returns the per replica values."""
    return self._values


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
class Mirrored(DistributedDelegate):
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
    return hash((self.name, self.graph, self.traceback, self.type))


class DistributedVariable(DistributedDelegate, variables_lib.Variable,
                          core.Tensor):
  """Holds a map from replica to variables."""

  def __init__(self, strategy, values, aggregation, var_policy=None):
    self._distribute_strategy = strategy
    self._aggregation = aggregation
    super(DistributedVariable, self).__init__(values)
    self._common_name = self._primary.name.split(":")[0]

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
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      raise ValueError("`handle` is not available outside the replica context"
                       " or a `tf.distribute.Strategy.update()` call.")
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
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      return self._get_cross_replica()
    else:
      return self._get_replica(replica_id)

  def _get_on_device_or_primary(self):
    """Returns value in same replica or device if possible, else the _primary."""
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
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return array_ops.identity(self._get())

  def value(self):
    if self._policy:
      return self._policy.value(self)
    return self._get_on_device_or_primary().value()

  def numpy(self):
    if context.executing_eagerly():
      return self.read_value().numpy()
    else:
      raise NotImplementedError(
          "numpy() is only available when eager execution is enabled.")

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
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
    if self._policy:
      self._policy.scatter_sub(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_sub(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    if self._policy:
      self._policy.scatter_add(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_add(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    if self._policy:
      self._policy.scatter_mul(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_mul(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    if self._policy:
      self._policy.scatter_div(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_div(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    if self._policy:
      self._policy.scatter_min(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_min(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    if self._policy:
      self._policy.scatter_max(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_max(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    if self._policy:
      self._policy.scatter_update(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_update(
        self, sparse_delta, use_locking=use_locking, name=name)

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
    if self._policy:
      return self._policy._as_graph_element(self)  # pylint: disable=protected-access

    raise NotImplementedError("No policy set for calling _as_graph_element.")

  def _get_cross_replica(self):
    if self._policy:
      return self._policy._get_cross_replica(self)  # pylint: disable=protected-access

    raise NotImplementedError(
        "This method should be overridden by sub-classes which support cross-"
        "replica accesses.")

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
    raise NotImplementedError("should be implemented by subclass.")

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
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return ops.convert_to_tensor(
          self._get(), dtype=dtype, name=name, as_ref=as_ref)

  def _map_resources(self):
    """For implementing `Trackable`."""
    new_obj = resource_variable_ops.copy_to_graph_uninitialized(self._primary)
    obj_map, resource_map = {}, {}
    for v in self._values:
      obj_map[v] = new_obj
      resource_map[v.handle] = new_obj.handle
    obj_map[self] = new_obj
    resource_map[self] = new_obj.handle
    if self._packed_var is not None:
      resource_map[self._packed_var.packed_handle] = new_obj.handle
    return obj_map, resource_map


class _DistributedVariableSaveable(saveable_object.SaveableObject):
  """Class for defining how to restore a DistributedVariable."""

  def __init__(self, distributed_variable, primary_variable, name):
    self._distributed_variable = distributed_variable
    if not self._distributed_variable._policy:
      raise ValueError("VariablePolicy has not been set for the distributed "
                       "variable.")
    tensor, spec = distributed_variable._policy.get_saveable(
        distributed_variable, primary_variable, name)
    super(_DistributedVariableSaveable, self).__init__(tensor, spec, name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return self._distributed_variable._policy.get_restore_ops(  # pylint: disable=protected-access
        self._distributed_variable, tensor)


class _MirroredSaveable(saveable_object_util.ResourceVariableSaveable):
  """Class for defining how to restore a MirroredVariable."""

  def __init__(self, mirrored_variable, primary_variable, name):
    self._mirrored_variable = mirrored_variable
    super(_MirroredSaveable, self).__init__(primary_variable, "", name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    packed_var = self._mirrored_variable._packed_variable  # pylint: disable=protected-access
    if packed_var is not None:
      return control_flow_ops.group(
          tuple(
              values_util.assign_on_device(d, packed_var, tensor)
              for d in packed_var.devices))
    return control_flow_ops.group(
        tuple(
            values_util.assign_on_device(v.device, v, tensor)
            for v in self._mirrored_variable.values))


class MirroredVariable(DistributedVariable, Mirrored):
  """Holds a map from replica to variables whose values are kept in sync."""

  def _update_replica(self, update_fn, value, **kwargs):
    return _on_write_update_replica(self, update_fn, value, **kwargs)

  def scatter_min(self, *args, **kwargs):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(values_util.scatter_error_msg.format(
          op_name="scatter_min", aggregation=self._aggregation))
    return super(MirroredVariable, self).scatter_min(*args, **kwargs)

  def scatter_max(self, *args, **kwargs):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(values_util.scatter_error_msg.format(
          op_name="scatter_min", aggregation=self._aggregation))
    return super(MirroredVariable, self).scatter_max(*args, **kwargs)

  def scatter_update(self, *args, **kwargs):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(values_util.scatter_error_msg.format(
          op_name="scatter_min", aggregation=self._aggregation))
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

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # Try to avoid assignments to and other mutations of MirroredVariable
    # state except through a DistributionStrategy.extended.update() call.
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

    # We use a callable so that we don't have to evaluate this expression
    # in the case where we are trying to restore instead of save.
    def tensor():
      strategy = sync_on_read_variable._distribute_strategy  # pylint: disable=protected-access
      return strategy.extended.read_var(sync_on_read_variable)

    spec = saveable_object.SaveSpec(
        tensor=tensor,
        slice_spec="",
        name=name,
        dtype=sync_on_read_variable.dtype,
        device=sync_on_read_variable._primary.device)  # pylint: disable=protected-access

    super(_SyncOnReadSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    # To preserve the sum across save and restore, we have to divide the
    # total across all devices when restoring a variable that was summed
    # when saving.
    tensor, = restored_tensors
    if self._sync_on_read_variable.aggregation == vs.VariableAggregation.SUM:
      tensor = math_ops.cast(tensor / len(self._sync_on_read_variable._devices),  # pylint: disable=protected-access
                             self._sync_on_read_variable.dtype)
    return control_flow_ops.group(
        tuple(
            values_util.assign_on_device(v.device, v, tensor)
            for v in self._sync_on_read_variable.values))


class SyncOnReadVariable(DistributedVariable):
  """Holds a map from replica to variables whose values are reduced on save."""

  def _update_replica(self, update_fn, value, **kwargs):
    return update_fn(self._get_on_device_or_primary(), value, **kwargs)

  # TODO(b/154017756): Make assign behaivor in cross replica context consistent
  # with MirroredVariable.
  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if ds_context.in_cross_replica_context():
        return values_util.on_read_assign_sub_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable,
                     self).assign_sub(value, use_locking, name, read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if ds_context.in_cross_replica_context():
        return values_util.on_read_assign_add_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable,
                     self).assign_add(value, use_locking, name, read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if ds_context.in_cross_replica_context():
        return values_util.on_read_assign_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable,
                     self).assign(value, use_locking, name, read_value)

  def _scatter_not_implemented(self, method):
    raise NotImplementedError(
        "Variables with `synchronization=ON_READ` doesn't support `%s`" %
        method)

  def scatter_sub(self, *args, **kwargs):
    self._scatter_not_implemented("scatter_sub")

  def scatter_add(self, *args, **kwargs):
    self._scatter_not_implemented("scatter_add")

  def scatter_mul(self, *args, **kwargs):
    self._scatter_not_implemented("scatter_mul")

  def scatter_div(self, *args, **kwargs):
    self._scatter_not_implemented("scatter_div")

  def scatter_min(self, *args, **kwargs):
    self._scatter_not_implemented("scatter_min")

  def scatter_max(self, *args, **kwargs):
    self._scatter_not_implemented("scatter_max")

  def scatter_update(self, *args, **kwargs):
    self._scatter_not_implemented("scatter_update")

  def value(self):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if ds_context.in_cross_replica_context():
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
          return self._get_replica(0).value()
        return self._get_cross_replica()
      else:
        # _get_on_device_or_primary() returns a Variable.
        return self._get_on_device_or_primary().value()

  def _get_cross_replica(self):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      # Consider returning a tensor value here to make the return value of
      # _get_cross_replica consistent.
      return self._get_replica(0)

    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return self._distribute_strategy.reduce(
          reduce_util.ReduceOp.from_variable_aggregation(self._aggregation),
          self,
          axis=None)

  def _as_graph_element(self):
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
    """Converts a variable to a tensor."""
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return ops.convert_to_tensor(
          self._get(), dtype=dtype, name=name, as_ref=as_ref)


# Register a conversion functions which reads the value of the variable,
# allowing instances of the class to be used as tensors.
# DistributedVariable
def _tensor_conversion_distributed_var(var, dtype=None, name=None,
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
        "This method should be overridden by sub-classes.")

  def _is_mirrored(self):
    raise NotImplementedError(
        "This method should be overridden by sub-classes.")

  def _as_graph_element(self, _):
    raise NotImplementedError(
        "This method should be overridden by sub-classes.")

  def _get_cross_replica(self, var):
    raise NotImplementedError(
        "This method should be overridden by sub-classes.")

  def _update_replica(self, var, update_fn, value, **kwargs):
    raise NotImplementedError(
        "This method should be overridden by sub-classes.")


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
      if ds_context.in_cross_replica_context():
        return var._get_cross_replica()  # pylint: disable=protected-access
      else:
        return var._get_on_device_or_primary().value()  # pylint: disable=protected-access

  def _as_graph_element(self, var):
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if ds_context.in_cross_replica_context():
        return ops.convert_to_tensor(var._get_cross_replica())   # pylint: disable=protected-access
    return var._get()._as_graph_element()   # pylint: disable=protected-access

  def _get_cross_replica(self, var):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      return var._primary  # pylint: disable=protected-access

    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      return  var.distribute_strategy.reduce(
          reduce_util.ReduceOp.from_variable_aggregation(self._aggregation),
          var,
          axis=None)

  def _update_replica(self, var, update_fn, value, **kwargs):
    return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access

  def _scatter_not_implemented(self, method):
    raise NotImplementedError(
        "ON_READ variables doesn't support `%s` in cross replica context" %
        method)

  def assign_sub(self, var, value, use_locking=False, name=None,
                 read_value=True):
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if ds_context.in_cross_replica_context():
        return values_util.on_read_assign_sub_cross_replica(
            var, value, read_value=read_value)
      else:
        return values_util.on_write_assign_sub(
            var, value, use_locking=use_locking, name=name,
            read_value=read_value)

  def assign_add(self, var, value, use_locking=False, name=None,
                 read_value=True):
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if ds_context.in_cross_replica_context():
        return values_util.on_read_assign_add_cross_replica(
            var, value, read_value=read_value)
      else:
        return values_util.on_write_assign_add(
            var, value, use_locking=use_locking, name=name,
            read_value=read_value)

  def assign(self, var, value, use_locking=False, name=None, read_value=True):
    with ds_context.enter_or_assert_strategy(var.distribute_strategy):
      if ds_context.in_cross_replica_context():
        return values_util.on_read_assign_cross_replica(var, value,
                                                        read_value=read_value)
      else:
        return values_util.on_write_assign(var, value,
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
    # We use a callable so that we don't have to evaluate this expression
    # in the case where we are trying to restore instead of save.
    def tensor():
      strategy = var.distribute_strategy
      return strategy.extended.read_var(var)

    spec = saveable_object.SaveSpec(
        tensor=tensor,
        slice_spec="",
        name=name,
        dtype=var.dtype,
        device=primary_var.device)

    return tensor, [spec]

  def get_restore_ops(self, var, tensor):
    """Restore the same value into all variables."""
    # To preserve the sum across save and restore, we have to divide the
    # total across all devices when restoring a variable that was summed
    # when saving.
    if self._aggregation == vs.VariableAggregation.SUM:
      tensor = math_ops.cast(tensor / len(var._devices),  # pylint: disable=protected-access
                             var.dtype)
    return control_flow_ops.group(
        tuple(
            values_util.assign_on_device(v.device, v, tensor)
            for v in var.values))


class AutoPolicy(VariablePolicy):
  """Policy defined for `tf.VariableSynchronization.AUTO` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.AUTO` and `aggregation` is set to
  `tf.VariableAggregation.NONE` when creating a `tf.Variable` in `tf.distribute`
  scope.
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
    return array_ops.identity(Mirrored._get_cross_replica(var))  # pylint: disable=protected-access

  def _update_replica(self, var, update_fn, value, **kwargs):
    return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access

  def assign(self, var, value, use_locking=False, name=None, read_value=True):
    return values_util.on_write_assign(var, value, use_locking=use_locking,
                                       name=name, read_value=read_value)

  def assign_add(self, var, value, use_locking=False, name=None,
                 read_value=True):
    return values_util.on_write_assign_add(var, value, use_locking=use_locking,
                                           name=name, read_value=read_value)

  def assign_sub(self, var, value, use_locking=False, name=None,
                 read_value=True):
    return values_util.on_write_assign_sub(var, value, use_locking=use_locking,
                                           name=name, read_value=read_value)

  def scatter_sub(self, var, sparse_delta, use_locking=False, name=None):
    return values_util.scatter_sub(var, sparse_delta, use_locking=use_locking,
                                   name=name)

  def scatter_add(self, var, sparse_delta, use_locking=False, name=None):
    return values_util.scatter_add(var, sparse_delta, use_locking=use_locking,
                                   name=name)

  def scatter_mul(self, var, sparse_delta, use_locking=False, name=None):
    return values_util.scatter_mul(var, sparse_delta, use_locking=use_locking,
                                   name=name)

  def scatter_div(self, var, sparse_delta, use_locking=False, name=None):
    return values_util.scatter_div(var, sparse_delta, use_locking=use_locking,
                                   name=name)

  def scatter_min(self, var, sparse_delta, use_locking=False, name=None):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(values_util.scatter_error_msg.format(
          op_name="scatter_min", aggregation=self._aggregation))
    return values_util.scatter_min(var, sparse_delta, use_locking=use_locking,
                                   name=name)

  def scatter_max(self, var, sparse_delta, use_locking=False, name=None):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(values_util.scatter_error_msg.format(
          op_name="scatter_max", aggregation=self._aggregation))
    return values_util.scatter_max(var, sparse_delta, use_locking=use_locking,
                                   name=name)

  def scatter_update(self, var, sparse_delta, use_locking=False, name=None):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(values_util.scatter_error_msg.format(
          op_name="scatter_update", aggregation=self._aggregation))
    return values_util.scatter_update(var, sparse_delta,
                                      use_locking=use_locking,
                                      name=name)

  def get_saveable(self, var, primary_var, name):
    del var, name
    return primary_var, ""

  def get_restore_ops(self, var, tensor):
    return control_flow_ops.group(
        tuple(
            values_util.assign_on_device(v.device, v, tensor)
            for v in var.values))


class OnWritePolicy(AutoPolicy):
  """Policy defined for `tf.VariableSynchronization.ON_WRITE` synchronization.

  This policy is created when the following `synchronization` and
  `aggregation` parameters are specified when creating a `tf.Variable` in
  `tf.distribute` scope:
  * `synchronization` is equal to `tf.VariableSynchronization.AUTO` and
  aggregation can be any of the following `tf.VariableAggregation` enum
  values such as `SUM`, `MEAN` or `ONLY_FIRST_REPLICA`.
  * `synchronization` is equal to `tf.VariableSynchronization.ON_WRITE` and
  aggregation can be any of the following `tf.VariableAggregation` enum
  values such as `NONE`, `SUM`, `MEAN` or `ONLY_FIRST_REPLICA`.
  """

  def _update_replica(self, var, update_fn, value, **kwargs):
    return _on_write_update_replica(var, update_fn, value, **kwargs)


# Utility functions
# Return True if the Value is Mirrored or the Variable is replicated and kept in
# sync.
def _is_mirrored(val):
  if isinstance(val, DistributedVariable):
    if val._policy:  # pylint: disable=protected-access
      return val._policy._is_mirrored()  # pylint: disable=protected-access
  return isinstance(val, Mirrored)


def _is_sync_on_read(val):
  if isinstance(val, DistributedVariable):
    if val._policy:  # pylint: disable=protected-access
      return not val._policy._is_mirrored()  # pylint: disable=protected-access
  return not isinstance(val, Mirrored)
