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
import functools
import threading
import weakref

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import numpy_compat


TRACKABLE_RESOURCE_METHODS = [
    "_create_resource", "_initialize", "_destroy_resource"
]


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
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
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
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      f = kwargs.pop("f")
      if distribute_lib.in_cross_replica_context():
        if distribute_lib.get_update_replica_id() is not None:
          # We are calling an assign function in an update context.
          return f(self._v, *args, **kwargs)

        # We are calling an assign function in cross replica context, wrap it in
        # an update call.
        return self._distribute_strategy.extended.update(
            self, f, args=args, kwargs=kwargs)
      else:
        replica_context = distribute_lib.get_replica_context()
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
  def op(self) -> ops.Operation:
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

  def _export_to_saved_model_graph(self, object_map, tensor_map,
                                   options, **kwargs):
    """For implementing `Trackable`."""
    # By delegating this method to the wrapped variable, SavedModel with
    # AggregatingVariable are identical to SavedModel with normal variables.
    resource_list = self._v._export_to_saved_model_graph(object_map, tensor_map,  # pylint:disable=protected-access
                                                         options, **kwargs)
    object_map[self] = object_map[self._v]
    return resource_list

  def _copy_trackable_to_cpu(self, object_map):
    """For implementing `Trackable`."""
    # Create a copy of `self._v` to object_map, then create a new copy of self
    # that wraps the copy of `self._v`.
    # When updating value, only the lowest-level variable will actually do that,
    # the copy of `AggregatingVariable` is more like a shell.
    self._v._copy_trackable_to_cpu(object_map)  # pylint:disable=protected-access
    if self not in object_map:
      # If copy of `self` not populated yet, initialize one.
      object_map[self] = AggregatingVariable(self._distribute_strategy,
                                             object_map[self._v],
                                             self._aggregation)

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
  def op(self) -> ops.Operation:
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
    return numpy_compat.np_asarray(self.numpy(), dtype=dtype)

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
    for operator in tensor.Tensor.OVERLOADABLE_OPERATORS:
      # Overloading __eq__ or __ne__ does not work as expected.
      if operator == "__eq__" or operator == "__ne__":
        continue
      cls._tensor_overload_operator(operator)

  @classmethod
  def _tensor_overload_operator(cls, operator):
    """Delegate an operator overload to `tensor.Tensor`."""
    tensor_operator = getattr(tensor.Tensor, operator)

    def _operator(v, *args, **kwargs):
      return tensor_operator(v.value(), *args, **kwargs)  # pylint: disable=protected-access
    setattr(cls, operator, _operator)

  def _gather_saveables_for_checkpoint(self):
    return {trackable.VARIABLE_VALUE_KEY: self._v}

  def _export_to_saved_model_graph(self, object_map, tensor_map,
                                   options, **kwargs):
    """For implementing `Trackable`."""
    # By delegating this method to the wrapped variable, SavedModel with
    # AggregatingVariable are identical to SavedModel with normal variables.
    resource_list = self._v._export_to_saved_model_graph(object_map, tensor_map,  # pylint:disable=protected-access
                                                         options, **kwargs)
    object_map[self] = object_map[self._v]
    return resource_list

  def _copy_trackable_to_cpu(self, object_map):
    """For implementing `Trackable`."""
    # Create a copy of `self._v` to object_map, then create a new copy of self
    # that wraps the copy of `self._v`.
    # When updating value, only the lowest-level variable will actually do that,
    # the copy of `CachingVariable` is more like a shell.
    self._v._copy_trackable_to_cpu(object_map)  # pylint:disable=protected-access
    if self not in object_map:
      # If copy of `self` not populated yet, initialize one.
      object_map[self] = CachingVariable(object_map[self._v])


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_aggregate(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype, name, as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    AggregatingVariable, _tensor_conversion_aggregate)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_caching(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype, name, as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    CachingVariable, _tensor_conversion_caching)

CachingVariable._overload_overloadable_operators()  # pylint: disable=protected-access


class PerWorkerVariable(resource_variable_ops.BaseResourceVariable):
  """A wrapper around unsynced variables created on workers.

  `PerWorkerVariable`s are variables that are stored on workers and not
  synchronized. A `PerWorkerVariable` is really a wrapper around multiple
  independent `Variable`s stored on independent worker machines. 
  `PerWorkerVariable` is currently only tested and supported when used with
  `ParameterServerStrategy`. A `PerWorkerVariable` can be created by creating a
  `Variable` within strategy scope and using the `per_worker_variable` flag,
  e.g.:

  ```
  with strategy.scope():
    var = tf.Variable(initial_value=0.0, per_worker_variable=True)
  ```

  The implementation modifies the graph to ensure that a worker's local version
  of the variable is used for computation at call time, while needing only one
  function trace and requiring no code changes beyond the `per_worker_variable`
  flag. `PerWorkerVariable`s can thus be treated like a standard `Variable`, but
  support is experimental and not all ops have been tested.

  All per-worker values can be retrieved and read into a list via
  `PerWorkerVariable.read_all()`.

  Caveats:
    - `PerWorkerVariable`s should not be used as direct inputs to a
      `tf.function`. That is, they should not appear in a tf.function header as
      an input argument. However they can still be read and manipulated in a
      `tf.function`.
    - The `shape` argument must be fully-defined (no `None` entries) or left
      empty. Partially-defined shapes are not yet supported.
    - Automatic control dependencies do not work with `PerWorkerVariable`s, so
      returning a `PerWorkerVariable` is not supported, and `read_all()` should 
      be used to retrieve values. (TODO: b/286052052)
    - `PerWorkerVariable`s should not be created within a `tf.function`.
  """

  def __init__(self, strategy, next_creator, **kwargs):
    self._coordinator = strategy._cluster_coordinator
    self._per_worker_vars = None
    self._var_creator = functools.partial(next_creator, **kwargs)

    self._coordinator_instance = next_creator(**kwargs)

    # Set ResourceVariable attributes based on kwargs
    if kwargs.get("in_graph_mode") is None:
      with ops.init_scope():
        self._in_graph_mode = not context.executing_eagerly()
    else:
      self._in_graph_mode = kwargs["in_graph_mode"]

    self._cached_value = None
    self._shape = self._coordinator_instance.shape
    self._dtype = self._coordinator_instance.dtype
    self._trainable = False  # not supported
    self._unique_id = kwargs.get("unique_id")
    if kwargs.get("handle_name") is None:
      self._handle_name = "Variable:0"
    else:
      self._handle_name = kwargs["handle_name"] + ":0"
    self._validate_shape = kwargs.get("validate_shape", True)

  @classmethod
  def _variable_call(cls, *args, **kwargs):
    """Override to be a no-op to avoid metaclass creating ResourceVariables."""
    return None

  @property
  def handle(self):
    if context.executing_eagerly() or save_context.in_save_context():
      return self._coordinator_instance.handle
    else:
      self._maybe_create_per_worker_vars()
      closure, spec = self.handle_call_time_value()
      return ops.get_default_graph().capture_call_time_value(
          closure,
          spec)

  def handle_call_time_value(self):
    """Returns a closure to run for a handle at call time and its spec.

    This function is called in self.handle to create a placeholder
    which returns a handle on some worker or on the coordinator.
    """

    def closure():
      dispatch_context = coordinator_context.get_current_dispatch_context()
      if dispatch_context:
        remote_value = self._per_worker_vars._values[  # pylint: disable=protected-access
            dispatch_context.worker_index]
        ret = dispatch_context.maybe_get_remote_value(remote_value)
        return ret.handle
      else:
        # Only needed for tracing
        return self._coordinator_instance.handle
    return closure, PerWorkerVariableSpec(
        value=self._coordinator_instance.handle)

  def _maybe_create_per_worker_vars(self):
    """Create variable on each worker if it hasn't been created."""
    if not self._per_worker_vars:
      self._per_worker_vars = (
          self._coordinator._create_per_worker_variables(self._var_creator))  # pylint: disable=protected-access

  def read_all(self):
    """Synchronously read variables from all workers into a list of Tensors."""
    return [wv.get() for wv in self._per_worker_vars._values]  # pylint: disable=protected-access


class PerWorkerVariableSpec(tensor.TensorSpec):
  def __init__(self, value=None, name=None):
    super().__init__(value.shape, value.dtype, name=name)
    self._value = value

  def placeholder_value(self, placeholder_context):
    placeholder = super().placeholder_value(placeholder_context)
    handle_data_util.set_handle_data(placeholder, self._value._handle_data)  # pylint: disable=protected-access
    return placeholder


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
    distribute_lib.distribution_strategy_input_api_counter.get_cell(
        self.__class__.__name__, "PSSDistributedLookupTable").increase_by(1)
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
        ret = dispatch_context.maybe_get_remote_value(remote_value)
        return ret

      else:
        return self._coordinator_instance.resource_handle

    return closure, tensor.TensorSpec([], dtype=dtypes.resource)

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

  def __init__(self, strategy, wrapped_creator):
    # Wait for all resource functions to have been set before building the table
    self._has_resource_functions = threading.Condition()
    super().__init__(strategy, wrapped_creator)

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

        ret = dispatch_context.maybe_get_remote_value(remote_value)
        return ret

      else:

        return self._coordinator_instance.resource_handle

    return closure, tensor.TensorSpec(shape=(), dtype=dtypes.resource)

  def __setattr__(self, name, value):
    if name in TRACKABLE_RESOURCE_METHODS:
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
      if not hasattr(self, "_restored_function"):
        self._restored_function = {}
      self._restored_function[name] = value
      if all(method in self._restored_function
             for method in TRACKABLE_RESOURCE_METHODS):
        with self._has_resource_functions:
          self._has_resource_functions.notify_all()
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
          # Wait until all resource functions are available before setting them
          # on new_table.
          with self._has_resource_functions:
            while not hasattr(self, "_restored_function") or any(
                method not in self._restored_function
                for method in TRACKABLE_RESOURCE_METHODS):
              self._has_resource_functions.wait()

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
