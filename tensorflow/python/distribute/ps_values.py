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
"""Various classes representing distributed values for PS."""

import contextlib
import copy
import threading
import weakref

import numpy as np

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util.lazy_loader import LazyLoader

load_context = LazyLoader(
    "load_context", globals(),
    "tensorflow.python.keras.saving.saved_model.load_context"
)


# Variable used in PSStrategy TF 1, TF2 and CentralStorageStrategy.
class AggregatingVariable(resource_variable_ops.BaseResourceVariable,
                          core.Tensor):
  """A wrapper around a variable that aggregates updates across replicas."""

  def __init__(self, strategy, v, aggregation):
    self._distribute_strategy = strategy
    self._v = v
    # NOTE: We don't use "_distributed_container" here because we don't want
    # to trigger that code path in regroup().
    v._aggregating_container = weakref.ref(self)  # pylint: disable=protected-access
    self._aggregation = aggregation

  def __deepcopy__(self, memo):
    """Perform a deepcopy of the `AggregatingVariable`.

    Unlike the deepcopy of a regular tf.Variable, this keeps the original
    strategy and devices of the `AggregatingVariable`.  To avoid confusion
    with the behavior of deepcopy on a regular `Variable` (which does
    copy into new devices), we only allow a deepcopy of a `AggregatingVariable`
    within its originating strategy scope.

    Args:
      memo: The memoization object for `deepcopy`.

    Returns:
      A deep copy of the current `AggregatingVariable`.

    Raises:
      RuntimeError: If trying to deepcopy into a different strategy.
    """
    with ds_context.enter_or_assert_strategy(self._distribute_strategy):
      v = copy.deepcopy(self._v, memo)

    copied_variable = type(self)(
        strategy=self._distribute_strategy,
        v=v,
        aggregation=self._aggregation)

    memo[id(self)] = copied_variable

    return copied_variable

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
              values_util.aggregation_error_msg.format(
                  variable_type="AggregatingVariable"))

        def merge_fn(strategy,
                     value,
                     use_locking=False,
                     name=None,
                     read_value=True):
          v = values_util.apply_aggregation(strategy, value, self._aggregation,
                                            self)
          if name and isinstance(name, values.PerReplica):
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

  def value(self):
    return self._v.value()

  def read_value(self):
    return self._v.read_value()

  def sparse_read(self, indices, name=None):
    return self._v.sparse_read(indices, name=name)

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
    if isinstance(self._v, CachingVariable):
      return self._v._gather_saveables_for_checkpoint()  # pylint:disable=protected-access
    return {trackable.VARIABLE_VALUE_KEY: self._v}

  def _map_resources(self, save_options):
    """For implementing `Trackable`."""
    # By delegating this method to the wrapped variable, SavedModel with
    # AggregatingVariable are identical to SavedModel with normal variables.
    obj_map, resource_map = self._v._map_resources(save_options)  # pylint:disable=protected-access
    obj_map[self] = obj_map[self._v]
    return obj_map, resource_map

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
    return self._v._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


class CachingVariable(resource_variable_ops.BaseResourceVariable, core.Tensor):
  """A wrapper around a variable that caches read value locally."""

  def __init__(self, v):
    self._v = v
    self._cache = None
    self._current_new_cache_scope_count = 0

  def get(self):
    return self._v

  def __getattr__(self, name):
    return getattr(self._v, name)

  def read_value(self):
    if distribute_utils.caching_scope_local.in_caching_scope():
      return self.cached_read_value()
    return self._v.read_value()

  def sparse_read(self, indices, name=None):
    return self._v.sparse_read(indices, name=name)

  def cached_read_value(self):
    if (distribute_utils.caching_scope_local.new_cache_scope_count >
        self._current_new_cache_scope_count):
      self._current_new_cache_scope_count += 1
      self._cache = None

    with ops.device("CPU:0"):
      if self._cache is not None:
        return self._cache
      else:
        self._cache = array_ops.identity(self._v)
        return self._cache

  def assign_sub(self, *args, **kwargs):
    return self._v.assign_sub(*args, **kwargs)

  def assign_add(self, *args, **kwargs):
    return self._v.assign_add(*args, **kwargs)

  def assign(self, *args, **kwargs):
    return self._v.assign(*args, **kwargs)

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

  def value(self):
    if distribute_utils.caching_scope_local.in_caching_scope():
      return self.cached_read_value()
    return self._v.value()

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

  @property
  def constraint(self):
    return self._v.constraint

  def __array__(self, dtype=None):
    return np.asarray(self.numpy(), dtype=dtype)

  def __complex__(self):
    return complex(self.value().numpy())

  def __int__(self):
    return int(self.value().numpy())

  def __float__(self):
    return float(self.value().numpy())

  def numpy(self):
    if context.executing_eagerly():
      return self.read_value().numpy()
    else:
      raise NotImplementedError(
          "numpy() is only available when eager execution is enabled.")

  def __str__(self):
    return str(self._v)

  def __repr__(self):
    return repr(self._v)

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    if distribute_utils.caching_scope_local.in_caching_scope():
      return self.cached_read_value()
    return self._v._dense_var_to_tensor(dtype=dtype, name=name, as_ref=False)  # pylint: disable=protected-access

  @classmethod
  def _overload_overloadable_operators(cls):
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      # Overloading __eq__ or __ne__ does not work as expected.
      if operator == "__eq__" or operator == "__ne__":
        continue
      cls._tensor_overload_operator(operator)

  @classmethod
  def _tensor_overload_operator(cls, operator):
    """Delegate an operator overload to `ops.Tensor`."""
    tensor_operator = getattr(ops.Tensor, operator)

    def _operator(v, *args, **kwargs):
      return tensor_operator(v.value(), *args, **kwargs)  # pylint: disable=protected-access
    setattr(cls, operator, _operator)

  def _gather_saveables_for_checkpoint(self):
    return {trackable.VARIABLE_VALUE_KEY: self._v}

  def _map_resources(self, save_options):
    """For implementing `Trackable`."""
    # By delegating this method to the wrapped variable, SavedModel with
    # AggregatingVariable are identical to SavedModel with normal variables.
    obj_map, resource_map = self._v._map_resources(save_options)  # pylint:disable=protected-access
    obj_map[self] = obj_map[self._v]
    return obj_map, resource_map


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_aggregate(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype, name, as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(AggregatingVariable,
                                        _tensor_conversion_aggregate)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_caching(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype, name, as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(CachingVariable,
                                        _tensor_conversion_caching)

CachingVariable._overload_overloadable_operators()  # pylint: disable=protected-access


class DistributedTable(lookup_ops.StaticHashTable):
  """A distributed StaticHashTable for ParameterServerStrategy.

  An instance of DistributedTable has copies of a StaticHashTable and its
  resource handle on the coordinator of each worker, created at the
  DistributedTable instance initialization time with initializers on each
  worker. Users can call methods on a DistributedTable as if it were a
  StaticHashTable, which leads to execution with the resource local to the
  consumer worker (or the coordinator, if calling from the coordinator). This
  implementation relies on the fact that the methods of StaticHashTable are
  queried with the resource handle (instead of the python object).

  Currently, at saving time, a DistributedTable is saved as a StaticHashTable on
  the coordinator, and restoring a DistributedTable from SavedModel is not
  supported.
  """

  def __init__(self, strategy, wrapped_creator):

    self._coordinator_instance = wrapped_creator()
    self._wrapped_creator = wrapped_creator
    self._coordinator = strategy._cluster_coordinator
    # self._distributed_table is a RemoteValue mapping worker_index to
    # RemoteValue that wraps a resource handle on the worker
    self._distributed_table = None
    self._distributed_table_creation_lock = threading.Lock()

    if not save_context.in_save_context():
      self._maybe_build_distributed_table()

  def __getattr__(self, attr):
    # This allows copy.copy(DistributedTable), e.g. at saving time.
    # (DistributedVariable uses the same fix.) When copying an object, copy.copy
    # doesn't invoke its __init__ method, instead it makes a new empty object,
    # then copies the attributes over. copy.copy looks for attributes like
    # "__setstate__" in case the object implements its custom unpickling. Since
    # DistributedTable doesn't have those attributes defined, __getattr__ will
    # be invoked, which tries to access the `_coordinator_instance` attribute.
    # But that doesn't exist either because this is an empty object, and again
    # __getattr__ is invoked, leading to an infinite recursion.
    if attr == "_coordinator_instance":
      raise AttributeError()

    if attr in self._coordinator_instance.__dict__:
      attr_value = self._coordinator_instance.__dict__[attr]
      if callable(attr_value):

        def wrapper(*args, **kwargs):
          return attr_value(self, *args, **kwargs)

        return wrapper
      elif isinstance(attr_value, property):
        return attr_value
      else:
        return getattr(self._coordinator_instance, attr)
    else:
      return getattr(self._coordinator_instance, attr)

  def resource_handle_call_time_value(self):
    """Returns a closure to run for a resource handle at call time and its spec.

    This function is called in self.resource_handle to create a placeholder
    which returns a resource handle on some worker or on the coordinator.
    """

    def closure():
      # function to be evaluated at function call time, returning a nest of
      # tensors compatible with `spec`.
      dispatch_context = coordinator_context.get_current_dispatch_context()
      if dispatch_context:
        remote_value = self._distributed_table._values[  # pylint: disable=protected-access
            dispatch_context.worker_index]
        dispatch_context.maybe_rebuild_remote_values(remote_value)
        ret = dispatch_context.maybe_get_remote_value(remote_value)
        return ret

      else:
        return self._coordinator_instance.resource_handle

    return closure, tensor_spec.TensorSpec([], dtype=dtypes.resource)

  def _maybe_build_distributed_table(self):
    """Create table objects and resources on each worker if hasn't been created."""
    with self._distributed_table_creation_lock:
      if not self._distributed_table:

        def create_copy():
          new_table = self._wrapped_creator()
          ret = new_table.resource_handle
          return ret

        self._distributed_table = (
            self._coordinator._create_per_worker_resources(create_copy))  # pylint: disable=protected-access

  @property
  def resource_handle(self):
    if context.executing_eagerly() or save_context.in_save_context():
      return self._coordinator_instance.resource_handle
    else:
      self._maybe_build_distributed_table()
      closure, spec = self.resource_handle_call_time_value()
      return ops.get_default_graph().capture_call_time_value(
          closure,
          spec,
          default_value=self._coordinator_instance.resource_handle)

  @property
  def is_distributed_table(self):
    return True

  def __tf_experimental_restore_capture__(
      self, concrete_function, internal_capture):
    closure, spec = self.resource_handle_call_time_value()
    concrete_function.graph.replace_capture_with_deferred_capture(
        self._coordinator_instance.resource_handle,
        closure,
        spec,
        default_value=self._coordinator_instance.resource_handle,
        placeholder=internal_capture)
    return concrete_function.graph.deferred_external_captures[-1]


_local_resource_restore_context = threading.local()


def get_current_local_resource_restore_context():
  try:
    return _local_resource_restore_context.current
  except AttributeError:
    return None


@contextlib.contextmanager
def with_local_resource_restore_context(instance):
  previous_context = getattr(_local_resource_restore_context, "current", None)
  _local_resource_restore_context.current = LocalResourceRestoreContext(
      instance)
  yield
  _local_resource_restore_context.current = previous_context


class LocalResourceRestoreContext(object):
  """Class holding information of a distributed instance, e.g. StaticHashTable.

  Pairing use with context manager `with_local_resource_restore_context` allows
  operations under this context manager to conveniently gets information of a
  component of the `RestoredDistributedTable` (and other restored distributed
  `CapturableResource` if we're supporting their distribution in the future),
  instead of looking it up from the mapping of the worker-to-resource handle.
  This is especially useful when we know which instance the operations should
  execute with and the mapping is not available yet.
  """

  def __init__(self, instance):
    self.instance = instance


class RestoredDistributedTable(DistributedTable):
  """A restored and distributed StaticHashTable for ParameterServerStrategy."""

  def resource_handle_call_time_value(self):
    """Returns a closure to run for a resource handle at call time and its spec.

    This function is called in self.resource_handle to create a placeholder
    which returns a resource handle on some worker or on the coordinator.
    """

    def closure():
      # function to be evaluated at function call time, returning a nest of
      # tensors compatible with `spec`.
      dispatch_context = coordinator_context.get_current_dispatch_context()
      if dispatch_context:
        local_resource_restore_context = (
            get_current_local_resource_restore_context())

        # A LocalResourceRestoreContext is entered in the process of remote
        # table creation and initialization if we're in the process of loading
        # from a SavedModel. A LocalResourceRestoreContext carries the
        # information regarding which table is being created and initialized. In
        # order to initialize a table, we need the restored `_initialize`
        # function, which captures this closure as table resource. And when this
        # closure is executed, we will read the table info from the
        # LocalResourceRestoreContext and return its handle, rather than
        # following the normal procedure of fetching from
        # `self._distributed_table`, because we're still in the middle of
        # building `self._distributed_table`.
        if local_resource_restore_context:
          remote_value = local_resource_restore_context.instance.resource_handle

        else:
          remote_value = self._distributed_table._values[  # pylint: disable=protected-access
              dispatch_context.worker_index]

        dispatch_context.maybe_rebuild_remote_values(remote_value)
        ret = dispatch_context.maybe_get_remote_value(remote_value)
        return ret

      else:

        return self._coordinator_instance.resource_handle

    return closure, tensor_spec.TensorSpec(shape=(), dtype=dtypes.resource)

  def __setattr__(self, name, value):
    if name in ["_create_resource", "_initialize", "_destroy_resource"]:
      # When a StaticHashTable is loaded with `tf.saved_model.load`, it becomes
      # a RestoredResource with dummy `_create_resource`, `_initialize`, and
      # `_destroy_resource" methods. Similarly, when loaded with
      # `tf.keras.models.load_model`, its initializer becomes a dummy one. In
      # both cases, these methods needs to be set to some RestoredFunctions
      # through `__setattr__`. Thus we need to store and set these methods for
      # the distributed tables (a.k.a. `self._distributed_table`) on the
      # workers too, besides setting for the coordinator instance. However, we
      # cannot set them at this point, since the distributed tables have not
      # been created. We store them in '_restored_function' and set them to the
      # distributed tables when they're created in
      # `self._maybe_build_distributed_table.create_copy`.
      if load_context.in_load_context() or (
          "RestoredStaticHashtable" in self._wrapped.__class__.__name__):

        if not hasattr(self, "_restored_function"):
          self._restored_function = {}
        self._restored_function[name] = value
      return self._coordinator_instance.__setattr__(name, value)
    else:
      return super(RestoredDistributedTable, self).__setattr__(name, value)

  def _create_resource(self):
    """A function that creates a resource handle for a table on coordinator."""
    return self._coordinator_instance._create_resource()  # pylint: disable=protected-access

  def _initialize(self):
    """A function that initializes the resource."""
    return self._coordinator_instance._initialize()  # pylint: disable=protected-access

  def _destroy_resource(self):
    """A function that destroys the resource."""
    return self._coordinator_instance._destroy_resource()  # pylint: disable=protected-access

  def _maybe_build_distributed_table(self):
    """Create table objects and resources on each worker if hasn't been created."""
    with self._distributed_table_creation_lock:
      if not self._distributed_table:

        def create_copy():
          new_table = self._wrapped_creator()

          if hasattr(self, "_restored_function"):
            with with_local_resource_restore_context(new_table):
              for name, tf_function in self._restored_function.items():
                setattr(new_table, name, tf_function)
              init_op = new_table._initialize()  # pylint: disable=protected-access
              if not context.executing_eagerly():
                ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)

          ret = new_table.resource_handle
          return ret

        self._distributed_table = (
            self._coordinator._create_per_worker_resources(create_copy))  # pylint: disable=protected-access
