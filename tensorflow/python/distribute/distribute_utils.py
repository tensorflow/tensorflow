# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Class implementing utilities used by tf.distribute.Strategy."""

from collections import abc
import contextlib
import threading

from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


def regroup(values, wrap_class=values_lib.PerReplica, always_wrap=False):
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
        regroup(tuple(v[i] for v in values), wrap_class, always_wrap)
        for i in range(len(v0))
    ]

  if isinstance(v0, tuple):
    for v in values[1:]:
      assert isinstance(v, tuple)
      assert len(v) == len(v0)
    regrouped_tuple = tuple(
        regroup(tuple(v[i] for v in values), wrap_class, always_wrap)
        for i in range(len(v0)))
    if hasattr(v0, "_fields"):
      # This tuple is in fact a namedtuple! Create a new namedtuple instance
      # and initialize it with the regrouped values:
      assert hasattr(v0, "_make")
      return v0._make(regrouped_tuple)
    else:
      return regrouped_tuple

  if isinstance(v0, abc.Mapping):
    v0keys = v0.keys()
    for v in values[1:]:
      assert isinstance(v, abc.Mapping), ("v[0]: %r  v[i]: %r" % (v0, v))
      assert set(v.keys()) == set(v0keys), ("v[0].keys: %s  v[i].keys: %s" %
                                            (set(v0keys), set(v.keys())))
    # Use the actual type in case it is a class inherited from a dict.
    return type(v0)({
        key: regroup(tuple(v[key] for v in values),
                     wrap_class, always_wrap)
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
  if same_id and isinstance(v0, values_lib.DistributedVariable):
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
    assert not isinstance(v0, values_lib.MirroredVariable), (
        "ids = %s, values = %s" % ([id(v) for v in values], values))
    distributed_container = v0._distributed_container()
    assert distributed_container is not None
    for v in values[1:]:
      assert distributed_container is v._distributed_container()
    return distributed_container
  # pylint: enable=protected-access

  return wrap_class(values)


def select_replica(replica_id, structured):
  """Specialize a nest of regular & per-replica values for one replica."""

  def _get(x):
    # `DistributedValues` would be sliced according to replica unless it is a
    # `DistributedVariable` because `DistributedVariable` can be handled
    # directly in the replica context.
    if (isinstance(x, values_lib.DistributedVariable) or
        not isinstance(x, values_lib.DistributedValues)):
      return x
    else:
      return x.values[replica_id]

  return nest.map_structure(_get, structured)


def select_replica_mirrored(replica_id, structured):
  """Specialize a nest of regular & mirrored values for one replica."""
  assert_mirrored(structured)
  return select_replica(replica_id, structured)


def assert_mirrored(structured):
  """Raises if the structured is not composed of mirrored or regular values."""

  def _assert_mirrored(x):
    if isinstance(x, values_lib.DistributedValues) and not is_mirrored(x):
      raise TypeError(
          "Expected value to be mirrored across replicas: %s in %s." %
          (x, structured))

  nest.map_structure(_assert_mirrored, structured)


def update_regroup(extended, updates, group):
  """Regroup for an update, with dependencies to ensure all updates execute."""
  if not group:
    regrouped = regroup(updates, values_lib.Mirrored)
    return nest.map_structure(extended._local_results, regrouped)  # pylint: disable=protected-access

  def _make_grouped_mirrored(values):
    """Convert per-replica list `values` into Mirrored type with grouping."""
    if len(values) == 1:
      return values_lib.Mirrored(values)

    # Make sure we run all updates. Without this, something like
    # session.run(extended.update(...)) may only update one replica.
    g = control_flow_ops.group(values)

    # If values is just ops, the grouping is enough. Everything in values
    # should have the same type, since we expect every replica to be performing
    # the same computation.
    if not all(tensor_util.is_tf_type(v) for v in values):
      return g

    # Otherwise we need tensors with the same values as `values`, but
    # that have a dependency on `g`.
    with_dep = []
    for v in values:
      with ops.device(v.device), ops.control_dependencies([g]):
        with_dep.append(array_ops.identity(v))

    return values_lib.Mirrored(with_dep)

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
      not isinstance(val, values_lib.DistributedVariable)):
    container = val._distributed_container()  # pylint: disable=protected-access
    if container is not None:
      return container
  return val


def is_distributed_variable(v):
  """Determine if a variable is ds variable or TPU mirrored variable."""
  return isinstance(v, values_lib.DistributedVariable)


def is_distributed_table(v):
  """Determine if an object is a DistributedTable."""
  return v.__class__.__name__ in ("DistributedTable",
                                  "RestoredDistributedTable")


def _validate_colocate_extended(v, extended):
  variable_strategy = v._distribute_strategy  # pylint: disable=protected-access
  if variable_strategy.extended is not extended:
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not %s created in scope: %s" %
        (v, variable_strategy))


def validate_colocate_distributed_variable(v, extended):
  if not isinstance(v, values_lib.DistributedVariable):
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


# Variable creation function for sync strategies.
def _validate_synchronization(kwargs):
  """Validate that given synchronization value is valid."""
  synchronization = kwargs.get("synchronization",
                               vs.VariableSynchronization.AUTO)
  if synchronization == vs.VariableSynchronization.NONE:
    raise ValueError(
        "`NONE` variable synchronization mode is not supported with "
        "tf.distribute strategy. Please change the `synchronization` for "
        "variable: " + str(kwargs["name"]))
  if synchronization not in (vs.VariableSynchronization.ON_READ,
                             vs.VariableSynchronization.ON_WRITE,
                             vs.VariableSynchronization.AUTO):
    raise ValueError(
        "Invalid variable synchronization mode: %s for variable: %s" %
        (synchronization, kwargs["name"]))
  if synchronization == vs.VariableSynchronization.AUTO:
    return vs.VariableSynchronization.ON_WRITE
  return synchronization


def _validate_aggregation(kwargs):
  aggregation = kwargs.get("aggregation", vs.VariableAggregation.NONE)

  if aggregation not in (vs.VariableAggregation.NONE,
                         vs.VariableAggregation.SUM,
                         vs.VariableAggregation.MEAN,
                         vs.VariableAggregation.ONLY_FIRST_REPLICA):
    raise ValueError("Invalid variable aggregation mode: %s for variable: %s" %
                     (aggregation, kwargs["name"]))
  return aggregation


def create_mirrored_variable(strategy, real_mirrored_creator, class_mapping,
                             policy_mapping, **kwargs):
  """Create distributed variables with given synchronization and aggregation."""
  # Figure out what collections this variable should be added to.
  # We'll add the MirroredVariable to those collections instead.
  var_collections = kwargs.pop("collections", None)
  if var_collections is None:
    var_collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  kwargs["collections"] = []

  synchronization = _validate_synchronization(kwargs)
  # Update synchronization in kwargs in case it's AUTO, which is converted to
  # ON_WRITE.
  kwargs["synchronization"] = synchronization
  aggregation = _validate_aggregation(kwargs)
  use_var_policy = getattr(strategy.extended, "_use_var_policy", False)

  # Ignore user-specified caching device, not needed for mirrored variables.
  kwargs.pop("caching_device", None)

  # TODO(josh11b,apassos): It would be better if variable initialization
  # was never recorded on the tape instead of having to do this manually
  # here.
  with tape.stop_recording():
    value_list = real_mirrored_creator(**kwargs)
    # MirroredVariable is recreated during saved_model loading, and its
    # component variables (value_list) will have None initializer. We
    # set their initializers to no_op so that consumer like
    # `global_variables_initializer` wouldn't complain, as it groups all
    # variables' initializers thus all variables have to have initializers.
    for v in value_list:
      # pylint:disable=protected-access
      if hasattr(v, "_initializer_op") and v._initializer_op is None:
        v._initializer_op = control_flow_ops.no_op()
      # pylint:enable=protected-access
    if use_var_policy:
      var_policy_cls = policy_mapping.get(synchronization)
      var_policy = var_policy_cls(aggregation=aggregation)
      var_cls = class_mapping.get("VariableClass")
      result = var_cls(strategy, value_list, aggregation, var_policy=var_policy)
    else:
      var_cls = class_mapping.get(synchronization)
      result = var_cls(strategy, value_list, aggregation)

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


# Utility functions
# Return True if the Value is Mirrored or the Variable is replicated and kept in
# sync.
def is_mirrored(val):
  if isinstance(val, values_lib.DistributedVariable):
    if val._policy:  # pylint: disable=protected-access
      return val._policy._is_mirrored()  # pylint: disable=protected-access
  return isinstance(val, values_lib.Mirrored)


def is_sync_on_read(val):
  if isinstance(val, values_lib.DistributedVariable):
    if val._policy:  # pylint: disable=protected-access
      return not val._policy._is_mirrored()  # pylint: disable=protected-access
  return not isinstance(val, values_lib.Mirrored)


class CachingScopeLocal(threading.local):
  """Class for maintaining thread local state for caching scope."""

  def __init__(self):
    super(CachingScopeLocal, self).__init__()
    self.new_cache_scope_count = 0
    self.cache_scope_exited_count = 0

  def enter_scope(self):
    self.new_cache_scope_count += 1

  def exit_scope(self):
    self.cache_scope_exited_count += 1

  def in_caching_scope(self):
    return self.new_cache_scope_count > self.cache_scope_exited_count


caching_scope_local = CachingScopeLocal()


@contextlib.contextmanager
def cache_variable_reads():
  """Scope for caching variable reads for AggregatingVariable.

  The variable reads for AggregatingVariable inside this scope are cached. i.e.
  the first read of variable reads the value from possibly remote handle, but
  subsequent reads are returned using local cached value.

  For example:
  strategy = ParameterServerStrategy...
  with strategy.scope():
    # Variable v is of AggregatingVariable type with actual variable residing
    # on PS.
    v = tf.Variable(1.0)

  with distribute_utils.cache_variable_reads():
    v.read_value()  # Reads value 1.0
    v.assign(constant_op.constant(5.0))  # v changes to 5.0
    t1 = v.read_value()
    t2 = v.read_value()  # Both t1 & t2 return cached value 1.0 from local CPU.

  Notes about cache_variable_reads scope:
  1. Nesting of scope cache_variable_reads() is not supported
  2. And when caching scope is enabled, the thread enabling the cache and
    mirrored_run._MirroredReplicaThread threads spawned from it will have
    caching enabled.

  Yields:
    A context for caching variables.
  """

  try:
    if caching_scope_local.in_caching_scope():
      # There is nested cache scope, which is not supported.
      raise ValueError("cache_variable_reads scope cannot be nested")
    caching_scope_local.enter_scope()
    yield
  finally:
    caching_scope_local.exit_scope()


# The following mapping indicates the policy that you must use for a given
# variable `synchronization` and `aggregation` pair.
# OnWritePolicy is used for:
# (synchronization=Auto, aggregation=NONE,SUM,MEAN,ONLY_FIRST_REPLICA)
# (synchronization=ON_WRITE, aggregation=NONE,SUM,MEAN,ONLY_FIRST_REPLICA)
# OnReadPolicy is used for:
# (synchronization=ON_READ, aggregation=NONE,SUM,MEAN,ONLY_FIRST_REPLICA)
VARIABLE_POLICY_MAPPING = {
    vs.VariableSynchronization.ON_WRITE: values_lib.OnWritePolicy,
    vs.VariableSynchronization.ON_READ: values_lib.OnReadPolicy,
}

VARIABLE_CLASS_MAPPING = {
    "VariableClass": values_lib.DistributedVariable,
    vs.VariableSynchronization.ON_WRITE: values_lib.MirroredVariable,
    vs.VariableSynchronization.ON_READ: values_lib.SyncOnReadVariable,
}

TPU_VARIABLE_POLICY_MAPPING = {
    vs.VariableSynchronization.ON_WRITE: tpu_values_lib.TPUOnWritePolicy,
    vs.VariableSynchronization.ON_READ: tpu_values_lib.TPUOnReadPolicy,
}

TPU_VARIABLE_CLASS_MAPPING = {
    "VariableClass": tpu_values_lib.TPUDistributedVariable,
    vs.VariableSynchronization.ON_WRITE: tpu_values_lib.TPUMirroredVariable,
    vs.VariableSynchronization.ON_READ: tpu_values_lib.TPUSyncOnReadVariable,
}
