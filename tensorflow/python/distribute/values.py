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

import weakref

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def _get_current_replica_id_as_int():
  """Returns the current replica ID as an integer, or `None`."""
  replica_context = ds_context.get_replica_context()
  if replica_context:
    replica_id = replica_context.replica_id_in_sync_group
    if not isinstance(replica_id, int):
      replica_id = tensor_util.constant_value(replica_id)
  else:
    replica_id = distribute_lib.get_update_replica_id()
  return replica_id


@tf_export("distribute.DistributedValues", v1=[])
class DistributedValues(object):
  """Base class for representing distributed values.

  A subclass instance of DistributedValues is created when creating variables
  within a distribution strategy, iterating a `tf.Dataset` or through
  `strategy.run`.  This base class should never be instantiated
  directly.  DistributedValues contains a value per replica.  Depending on
  the subclass, the values could either be synced on update, synced on demand,
  or never synced.

  DistributedValues can be reduced to obtain single value across replicas,
  as input into `run` or the per replica values inspected
  using `experimental_local_results`.

  Example usage:

  1. Created from Dataset:

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
    replica_id = _get_current_replica_id_as_int()
    if replica_id is None:
      return self._get_cross_replica()
    else:
      return self._values[replica_id]

  def _get_cross_replica(self):
    raise NotImplementedError(
        "This method should be overridden by sub-classes which support cross-"
        "replica accesses.")

  def _get_closest(self):
    """Returns value in same replica or device if possible, else the _primary."""
    replica_id = _get_current_replica_id_as_int()
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
    return self._get_closest()

  def _as_graph_element(self):
    obj = self._get()
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


class DistributedVariable(DistributedDelegate, variables_lib.Variable):
  """Holds a map from replica to variables."""

  # TODO(josh11b): Support changing the set of variables if e.g. if new
  # devices are joining or a device is to leave.

  def __init__(self, strategy, values, aggregation):
    self._distribute_strategy = strategy
    self._aggregation = aggregation
    super(DistributedVariable, self).__init__(values)
    self._common_name = self._primary.name.split(":")[0]
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
    return self._get_closest().initialized_value()

  @property
  def initial_value(self):
    return self._get_closest().initial_value

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
  def handle(self):
    replica_id = _get_current_replica_id_as_int()
    if replica_id is None:
      raise ValueError("`handle` is not available outside the replica context"
                       " or a `tf.distribute.Strategy.update()` call.")
    else:
      return self._values[replica_id].handle

  def eval(self, session=None):
    return self._get_closest().eval(session)

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
    return self._get_closest().device

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

  def read_value(self):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return array_ops.identity(self._get())

  def value(self):
    return self._get_closest().value()

  def numpy(self):
    if context.executing_eagerly():
      return self.read_value().numpy()
    else:
      raise NotImplementedError(
          "numpy() is only available when eager execution is enabled.")

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return self._update(
        update_fn=assign_sub_fn,
        value=value,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
    return self._update(
        update_fn=assign_add_fn,
        value=value,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
    return self._update(
        update_fn=assign_fn,
        value=value,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    scatter_sub_fn = lambda var, *a, **kw: var.scatter_sub(*a, **kw)
    return self._update(
        update_fn=scatter_sub_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    scatter_add_fn = lambda var, *a, **kw: var.scatter_add(*a, **kw)
    return self._update(
        update_fn=scatter_add_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    scatter_mul_fn = lambda var, *a, **kw: var.scatter_mul(*a, **kw)
    return self._update(
        update_fn=scatter_mul_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    scatter_div_fn = lambda var, *a, **kw: var.scatter_div(*a, **kw)
    return self._update(
        update_fn=scatter_div_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    scatter_min_fn = lambda var, *a, **kw: var.scatter_min(*a, **kw)
    return self._update(
        update_fn=scatter_min_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    scatter_max_fn = lambda var, *a, **kw: var.scatter_max(*a, **kw)
    return self._update(
        update_fn=scatter_max_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    scatter_update_fn = lambda var, *a, **kw: var.scatter_update(*a, **kw)
    return self._update(
        update_fn=scatter_update_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

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
          return update_fn(self._values[update_replica_id], value, **kwargs)
        return self._update_cross_replica(update_fn, value, **kwargs)
      else:
        _assert_replica_context(self.distribute_strategy)
        return self._update_replica(update_fn, value, **kwargs)

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass


ops.register_dense_tensor_like_type(DistributedVariable)


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


class _MirroredSaveable(saveable_object_util.ResourceVariableSaveable):
  """Class for defining how to restore a MirroredVariable."""

  def __init__(self, mirrored_variable, primary_variable, name):
    self._mirrored_variable = mirrored_variable
    super(_MirroredSaveable, self).__init__(primary_variable, "", name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return control_flow_ops.group(
        tuple(
            _assign_on_device(v.device, v, tensor)
            for v in self._mirrored_variable.values))


def create_mirrored_variable(  # pylint: disable=missing-docstring
    strategy, real_mirrored_creator, mirrored_cls, sync_on_read_cls, **kwargs):
  # Figure out what collections this variable should be added to.
  # We'll add the MirroredVariable to those collections instead.
  var_collections = kwargs.pop("collections", None)
  if var_collections is None:
    var_collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  kwargs["collections"] = []

  synchronization = kwargs.get("synchronization",
                               vs.VariableSynchronization.ON_WRITE)

  if synchronization == vs.VariableSynchronization.NONE:
    raise ValueError(
        "`NONE` variable synchronization mode is not supported with `Mirrored` "
        "distribution strategy. Please change the `synchronization` for "
        "variable: " + str(kwargs["name"]))
  elif synchronization == vs.VariableSynchronization.ON_READ:
    is_sync_on_read = True
  elif synchronization in (vs.VariableSynchronization.ON_WRITE,
                           vs.VariableSynchronization.AUTO):
    # `AUTO` synchronization defaults to `ON_WRITE`.
    is_sync_on_read = False
  else:
    raise ValueError(
        "Invalid variable synchronization mode: %s for variable: %s" %
        (synchronization, kwargs["name"]))

  aggregation = kwargs.pop("aggregation", vs.VariableAggregation.NONE)

  if aggregation not in (vs.VariableAggregation.NONE,
                         vs.VariableAggregation.SUM,
                         vs.VariableAggregation.MEAN,
                         vs.VariableAggregation.ONLY_FIRST_REPLICA):
    raise ValueError("Invalid variable aggregation mode: %s for variable: %s" %
                     (aggregation, kwargs["name"]))

  # Ignore user-specified caching device, not needed for mirrored variables.
  kwargs.pop("caching_device", None)

  # TODO(josh11b,apassos): It would be better if variable initialization
  # was never recorded on the tape instead of having to do this manually
  # here.
  with tape.stop_recording():
    value_list = real_mirrored_creator(**kwargs)
    var_cls = sync_on_read_cls if is_sync_on_read else mirrored_cls
    result = var_cls(strategy, value_list, aggregation)
    # Install the created DistributedVariable as _distributed_container property
    # of the underlying variables, to make it easy to map back to the container.
    for v in result.values:
      # Hold a strong reference to avoid the container from being GC-ed. After
      # v = v.assign(), the user code may no longer holds references to the
      # original container, since v.assign() returns a new DistributedVariable.
      v._distributed_container = result  # pylint: disable=protected-access

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

  def _update_replica(self, update_fn, value, **kwargs):
    if self.aggregation == vs.VariableAggregation.NONE:
      raise ValueError(
          _aggregation_error_msg.format(variable_type="MirroredVariable"))

    def merge_fn(strategy, value, **kwargs):
      """Aggregate values and update all variables in cross replica context."""
      # Don't allow MEAN with non float dtype, since it may cause unexpected
      # precision loss. Python3 and NumPy automatically upcast integers to
      # float in division, but we should always preserve the type.
      #
      # Note that to be backward compatible we allow the case when the value
      # is *always* the same on each replica. I.E. value is not a
      # PerReplica. Refer to regroup() to see how values are grouped.
      if self._aggregation == vs.VariableAggregation.MEAN and (
          not self.dtype.is_floating) and isinstance(value, PerReplica):
        raise ValueError(
            "Cannot update non-float variables with "
            "tf.VariableAggregation.MEAN aggregation in replica context. "
            "Either change the variable dtype to float or update it in "
            "cross-replica context.")

      assert strategy == self.distribute_strategy
      v = _apply_aggregation(strategy, value, self.aggregation, self)
      return self._update_cross_replica(update_fn, v, **kwargs)

    return ds_context.get_replica_context().merge_call(
        merge_fn, args=(value,), kwargs=kwargs)

  def scatter_min(self, *args, **kwargs):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError("scatter_min is only supported for mirrored "
                                "variable (variable created within certain "
                                "`tf.distribute.Strategy` scope) with NONE or "
                                "`ONLY_FIRST_REPLICA` aggregation, got: %s" %
                                self._aggregation)
    return super(MirroredVariable, self).scatter_min(*args, **kwargs)

  def scatter_max(self, *args, **kwargs):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError("scatter_max is only supported for mirrored "
                                "variable (variable created within certain "
                                "`tf.distribute.Strategy` scope) with NONE or "
                                "`ONLY_FIRST_REPLICA` aggregation, got: %s" %
                                self._aggregation)
    return super(MirroredVariable, self).scatter_max(*args, **kwargs)

  def scatter_update(self, *args, **kwargs):
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError("scatter_update is only supported for mirrored "
                                "variable (variable created within certain "
                                "`tf.distribute.Strategy` scope) with NONE or "
                                "`ONLY_FIRST_REPLICA` aggregation, got: %s" %
                                self._aggregation)
    return super(MirroredVariable, self).scatter_update(*args, **kwargs)

  def _get_cross_replica(self):
    # Return identity, to avoid directly exposing the variable to the user and
    # allowing it to be modified by mistake.
    return array_ops.identity(Mirrored._get_cross_replica(self))

  def _as_graph_element(self):
    return self._get_closest()._as_graph_element()  # pylint: disable=protected-access

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


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_mirrored(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(MirroredVariable,
                                        _tensor_conversion_mirrored)


def _tensor_conversion_mirrored_val(value, dtype=None, name=None, as_ref=False):
  return ops.convert_to_tensor(
      value._get(), dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(Mirrored,
                                        _tensor_conversion_mirrored_val)


def is_distributed_variable(v):
  """Determine if a variable is ds variable or TPU mirrored variable."""
  return isinstance(v, DistributedVariable)


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
            _assign_on_device(v.device, v, tensor)
            for v in self._sync_on_read_variable.values))


def _assert_replica_context(strategy):
  replica_context = ds_context.get_replica_context()
  if not replica_context:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")
  if replica_context.strategy is not strategy:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")


class SyncOnReadVariable(DistributedVariable):
  """Holds a map from replica to variables whose values are reduced on save."""

  def _update_replica(self, update_fn, value, **kwargs):
    return update_fn(self._get_closest(), value, **kwargs)

  # TODO(b/154017756): Make assign behaivor in cross replica context consistent
  # with MirroredVariable.
  def assign_sub(self, *args, **kwargs):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if ds_context.in_cross_replica_context():
        if self._aggregation == vs.VariableAggregation.SUM:
          raise ValueError(
              "SyncOnReadVariable does not support `assign_sub` in "
              "cross-replica context when aggregation is set to "
              "`tf.VariableAggregation.SUM`.")
        return control_flow_ops.group(
            tuple(
                _assign_sub_on_device(v.device, v, args[0])
                for v in self._values))
      else:
        return super(SyncOnReadVariable, self).assign_sub(*args, **kwargs)

  def assign_add(self, *args, **kwargs):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if ds_context.in_cross_replica_context():
        if self._aggregation == vs.VariableAggregation.SUM:
          raise ValueError(
              "SyncOnReadVariable does not support `assign_add` in "
              "cross-replica context when aggregation is set to "
              "`tf.VariableAggregation.SUM`.")
        return control_flow_ops.group(
            tuple(
                _assign_add_on_device(v.device, v, args[0])
                for v in self._values))
      else:
        return super(SyncOnReadVariable, self).assign_add(*args, **kwargs)

  def assign(self, *args, **kwargs):
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      if ds_context.in_cross_replica_context():
        # To preserve the sum across save and restore, we have to divide the
        # total across all devices when restoring a variable that was summed
        # when saving.
        tensor = args[0]
        if self._aggregation == vs.VariableAggregation.SUM:
          tensor = math_ops.cast(tensor / len(self._values), self.dtype)
        return control_flow_ops.group(
            tuple(_assign_on_device(v.device, v, tensor) for v in self._values))
      else:
        return super(SyncOnReadVariable, self).assign(*args, **kwargs)

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
        return self._get_cross_replica()
      else:
        # _get_closest() returns a Variable.
        return self._get_closest().value()

  def _get_cross_replica(self):
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      return self._primary

    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      return self._distribute_strategy.reduce(
          reduce_util.ReduceOp.from_variable_aggregation(self.aggregation),
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


# Register a conversion function for SyncOnReadVariable which allows as_ref to
# be true.
def _tensor_conversion_sync_on_read(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(SyncOnReadVariable,
                                        _tensor_conversion_sync_on_read)


def regroup(values, wrap_class=PerReplica, always_wrap=False):
  """Makes a nest per-replica into a nest of PerReplica/Mirrored values.

  Args:
    values: Values to regroup
    wrap_class: Class that `values` be wrapped in.
    always_wrap: Always wrap the `values` in `wrap_class` even if the values
        are the same except for DistributeVariable.
  Returns:
    Wrapped `values`.
  """
  v0 = values[0]

  if isinstance(v0, list):
    for v in values[1:]:
      assert isinstance(v, list)
      assert len(v) == len(v0), ("len(v) == %d, len(v0) == %d, v: %s, v0: %s" %
                                 (len(v), len(v0), v, v0))
    return [
        regroup(tuple(v[i] for v in values), wrap_class)
        for i in range(len(v0))
    ]

  if isinstance(v0, tuple):
    for v in values[1:]:
      assert isinstance(v, tuple)
      assert len(v) == len(v0)
    regrouped_tuple = tuple(
        regroup(tuple(v[i] for v in values), wrap_class)
        for i in range(len(v0)))
    if hasattr(v0, "_fields"):
      # This tuple is in fact a namedtuple! Create a new namedtuple instance
      # and initialize it with the regrouped values:
      assert hasattr(type(v0), "_make")
      return type(v0)._make(regrouped_tuple)
    else:
      return regrouped_tuple

  if isinstance(v0, dict):
    v0keys = v0.keys()
    for v in values[1:]:
      assert isinstance(v, dict), ("v[0]: %r  v[i]: %r" % (v0, v))
      assert set(v.keys()) == set(v0keys), ("v[0].keys: %s  v[i].keys: %s" %
                                            (set(v0keys), set(v.keys())))
    # Use the actual type in case it is a class inherited from a dict.
    return type(v0)({
        key: regroup(tuple(v[key] for v in values), wrap_class)
        for key in v0keys
    })

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
  if same_id and isinstance(v0, DistributedVariable):
    return v0
  # * If v0 is a member of a distributed variable, in which case
  #   hasattr(v0, "_distributed_container") is true, we want to
  #   return the DistributedVariable that contains it using the
  #   _distributed_container logic below. This case can trigger
  #   same_id when there is only one device.
  # * In any other situation, same_id means we return v0 unless `always_wrap` is
  #   true.
  if same_id and not always_wrap and not hasattr(v0, "_distributed_container"):
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
    distributed_container = v0._distributed_container
    assert distributed_container is not None
    for v in values[1:]:
      assert distributed_container is v._distributed_container
    return distributed_container
  # pylint: enable=protected-access

  return wrap_class(values)


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


def select_replica_mirrored(replica_id, structured):
  """Specialize a nest of regular & mirrored values for one replica."""

  def _get_mirrored(x):
    if isinstance(x, DistributedValues):
      if not isinstance(x, Mirrored):
        raise TypeError(
            "Expected value to be mirrored across replicas: %s in %s." %
            (x, structured))
      return x.values[replica_id]
    else:
      return x

  return nest.map_structure(_get_mirrored, structured)


def update_regroup(extended, updates, group):
  """Regroup for an update, with dependencies to ensure all updates execute."""
  if not group:
    regrouped = regroup(updates, Mirrored)
    return nest.map_structure(extended._local_results, regrouped)  # pylint: disable=protected-access

  def _make_grouped_mirrored(values):
    """Convert per-replica list `values` into Mirrored type with grouping."""
    if len(values) == 1:
      return Mirrored(values)

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
    with_dep = []
    for v in values:
      with ops.device(v.device), ops.control_dependencies([g]):
        with_dep.append(array_ops.identity(v))

    return Mirrored(with_dep)

  return regroup(updates, _make_grouped_mirrored)


def value_container(val):
  """Returns the container that this per-replica `value` belongs to.

  Args:
    val: A value returned by `call_for_each_replica()` or a variable created in
      `scope()`.

  Returns:
    A container that `value` belongs to.
    If value does not belong to any container (including the case of
    container having been destroyed), returns the value itself.
  """
  if (hasattr(val, "_distributed_container") and
      # DistributedVariable has _distributed_container defined
      # but we don't want to return it.
      not isinstance(val, DistributedVariable)):
    container = val._distributed_container  # pylint: disable=protected-access
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
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      f = kwargs.pop("f")
      if ds_context.in_cross_replica_context():
        if distribute_lib.get_update_replica_id() is not None:
          # We are calling an assign function in an update context.
          return f(self._v, *args, **kwargs)

        # We are calling an assign function in cross replica context, wrap it in
        # an update call.
        return self._distribute_strategy.extended.update(
            self, f, args=args, kwargs=kwargs)
      else:
        replica_context = ds_context.get_replica_context()
        assert replica_context
        # We are calling an assign function in replica context.
        # We reduce the value we want to assign/add/sub. More details about how
        # we handle the different use cases can be found in the _reduce method.
        # We call the function with the reduced value.
        if self._aggregation == vs.VariableAggregation.NONE:
          raise ValueError(
              _aggregation_error_msg.format(
                  variable_type="AggregatingVariable"))

        def merge_fn(strategy,
                     value,
                     use_locking=False,
                     name=None,
                     read_value=True):
          v = _apply_aggregation(strategy, value, self._aggregation, self)
          if name and isinstance(name, PerReplica):
            name = name.values[0]
          return strategy.extended.update(
              self,
              f,
              args=(v,),
              kwargs={
                  "use_locking": use_locking,
                  "name": name,
                  "read_value": read_value
              })
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
  def synchronization(self):
    return self._v.synchronization

  @property
  def name(self):
    return self._v.name

  @property
  def trainable(self):
    return self._v.trainable

  @property
  def dtype(self):
    return self._v.dtype

  # TODO(josh11b): Test saving & restoring.
  def _gather_saveables_for_checkpoint(self):
    return {trackable.VARIABLE_VALUE_KEY: self._v}

  # pylint: disable=multiple-statements
  def __add__(self, o):
    return self._v + o

  def __radd__(self, o):
    return o + self._v

  def __sub__(self, o):
    return self._v - o

  def __rsub__(self, o):
    return o - self._v

  def __mul__(self, o):
    return self._v * o

  def __rmul__(self, o):
    return o * self._v

  def __truediv__(self, o):
    return self._v / o

  def __rtruediv__(self, o):
    return o / self._v

  def __floordiv__(self, o):
    return self._v // o

  def __rfloordiv__(self, o):
    return o // self._v

  def __mod__(self, o):
    return self._v % o

  def __rmod__(self, o):
    return o % self._v

  def __lt__(self, o):
    return self._v < o

  def __le__(self, o):
    return self._v <= o

  def __gt__(self, o):
    return self._v > o

  def __ge__(self, o):
    return self._v >= o

  def __and__(self, o):
    return self._v & o

  def __rand__(self, o):
    return o & self._v

  def __or__(self, o):
    return self._v | o

  def __ror__(self, o):
    return o | self._v

  def __xor__(self, o):
    return self._v ^ o

  def __rxor__(self, o):
    return o ^ self._v

  def __getitem__(self, o):
    return self._v[o]

  def __pow__(self, o, modulo=None):
    return pow(self._v, o, modulo)

  def __rpow__(self, o):
    return pow(o, self._v)

  def __invert__(self):
    return ~self._v

  def __neg__(self):
    return -self._v

  def __abs__(self):
    return abs(self._v)

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

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    return ops.convert_to_tensor(self.get(), dtype=dtype, name=name,
                                 as_ref=as_ref)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_aggregate(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype, name, as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(AggregatingVariable,
                                        _tensor_conversion_aggregate)
ops.register_dense_tensor_like_type(AggregatingVariable)
