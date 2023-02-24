# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Ops to use variables as resources."""

# pylint: disable=g-bad-name
import contextlib
import functools
import weakref

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_resource_variable_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

acd.register_read_only_resource_op("ReadVariableOp")
acd.register_read_only_resource_op("VariableShape")
acd.register_read_only_resource_op("ResourceGather")
acd.register_read_only_resource_op("ResourceGatherNd")
acd.register_read_only_resource_op("_ReadVariablesOp")

# TODO(allenl): Remove this alias and migrate callers.
get_resource_handle_data = handle_data_util.get_resource_handle_data


def get_eager_safe_handle_data(handle):
  """Get the data handle from the Tensor `handle`."""
  assert isinstance(handle, ops.Tensor)

  if isinstance(handle, ops.EagerTensor):
    return handle._handle_data  # pylint: disable=protected-access
  else:
    return get_resource_handle_data(handle)


def _set_handle_shapes_and_types(tensor, handle_data, graph_mode):
  """Sets the shape inference result HandleData on tensor.

  Args:
    tensor: A `Tensor` or `EagerTensor`.
    handle_data: A `CppShapeInferenceResult.HandleData`.
    graph_mode: A python bool.
  """
  tensor._handle_data = handle_data  # pylint: disable=protected-access
  if not graph_mode:
    return

  # Not an EagerTensor, so a graph tensor.
  shapes, types = zip(
      *[(pair.shape, pair.dtype) for pair in handle_data.shape_and_type])
  ranks = [len(s.dim) if not s.unknown_rank else -1 for s in shapes]
  shapes = [
      [d.size for d in s.dim]  # pylint: disable=g-complex-comprehension
      if not s.unknown_rank else None for s in shapes
  ]
  with tensor._op.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
    pywrap_tf_session.TF_GraphSetOutputHandleShapesAndTypes_wrapper(
        c_graph,
        tensor._as_tf_output(),  # pylint: disable=protected-access
        shapes,
        ranks,
        types)


def _combine_handle_data(handle, initial_value):
  """Concats HandleData from tensors `handle` and `initial_value`.

  Args:
    handle: A `Tensor` of dtype `resource`.
    initial_value: A `Tensor`.

  Returns:
    A `CppShapeInferenceResult.HandleData`.  If `initial_value` has dtype
    `variant`, the `HandleData` contains the concatenation of the shape_and_type
    from both `handle` and `initial_value`.

  Raises:
    RuntimeError: If handle, which was returned by VarHandleOp, either has
      no handle data, or its len(handle_data.shape_and_type) != 1.
  """
  assert handle.dtype == dtypes.resource

  variable_handle_data = get_eager_safe_handle_data(handle)

  if initial_value.dtype != dtypes.variant:
    return variable_handle_data

  extra_handle_data = get_eager_safe_handle_data(initial_value)
  if extra_handle_data is not None and extra_handle_data.is_set:
    if (variable_handle_data is None or not variable_handle_data.is_set or
        len(variable_handle_data.shape_and_type) != 1):
      raise RuntimeError(
          "Expected VarHandleOp to return a length==1 shape_and_type, "
          f"but saw: '{variable_handle_data}'")
    variable_handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)
  return variable_handle_data


def _variable_handle_from_shape_and_dtype(shape,
                                          dtype,
                                          shared_name,
                                          name,
                                          graph_mode,
                                          initial_value=None):
  """Create a variable handle, copying in handle data from `initial_value`."""
  container = ops.get_default_graph()._container  # pylint: disable=protected-access
  if container is None:
    container = ""
  shape = tensor_shape.as_shape(shape)
  dtype = dtypes.as_dtype(dtype)
  if not graph_mode:
    if shared_name is not None:
      raise errors.InternalError(
          node_def=None,
          op=None,
          message="Using an explicit shared_name is "
          "not allowed when executing eagerly.")
    shared_name = context.anonymous_name()

  handle = gen_resource_variable_ops.var_handle_op(
      shape=shape,
      dtype=dtype,
      shared_name=shared_name,
      name=name,
      container=container)
  if initial_value is None:
    initial_value = handle
  if graph_mode:
    full_handle_data = _combine_handle_data(handle, initial_value)
    _set_handle_shapes_and_types(handle, full_handle_data, graph_mode)
    return handle
  else:
    handle_data = handle_data_util.create_handle_data(shape, dtype)
    if initial_value is not None and initial_value.dtype == dtypes.variant:
      extra_handle_data = get_eager_safe_handle_data(initial_value)
      if extra_handle_data is not None and extra_handle_data.is_set:
        if (not handle_data.is_set or len(handle_data.shape_and_type) != 1):
          raise RuntimeError(
              "Expected VarHandleOp to return a length==1 shape_and_type, "
              f"but saw: '{handle_data}'")
        handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)

    _set_handle_shapes_and_types(handle, handle_data, graph_mode)
    return handle


def eager_safe_variable_handle(initial_value, shape, shared_name, name,
                               graph_mode):
  """Creates a variable handle with information to do shape inference.

  The dtype is read from `initial_value` and stored in the returned
  resource tensor's handle data.

  If `initial_value.dtype == tf.variant`, we additionally extract the handle
  data (if any) from `initial_value` and append it to the `handle_data`.
  In this case, the returned tensor's handle data is in the form

  ```
  is_set: true
  shape_and_type {
    shape {
      // initial_value.shape
    }
    dtype: DT_VARIANT
  }
  shape_and_type {
    // handle_data(initial_value).shape_and_type[0]
  }
  shape_and_type {
    // handle_data(initial_value).shape_and_type[1]
  }
  ...
  ```

  Ops that read from this tensor, such as `ReadVariableOp` and
  `AssignVariableOp`, know that `handle_data(handle).shape_and_type[1:]`
  correspond to the handle data of the variant(s) stored in the Variable.

  Args:
    initial_value: A `Tensor`.
    shape: The shape of the handle data. Can be `TensorShape(None)` (i.e.
      unknown shape).
    shared_name: A string.
    name: A string.
    graph_mode: A python bool.

  Returns:
    The handle, a `Tensor` of type `resource`.
  """
  dtype = initial_value.dtype.base_dtype
  return _variable_handle_from_shape_and_dtype(shape, dtype, shared_name, name,
                                               graph_mode, initial_value)


@contextlib.contextmanager
def _handle_graph(handle):
  # Note: might have an eager tensor but not be executing eagerly when building
  # functions.
  if (context.executing_eagerly() or isinstance(handle, ops.EagerTensor) or
      ops.has_default_graph()):
    yield
  else:
    with handle.graph.as_default():
      yield


class EagerResourceDeleter:
  """An object which cleans up a resource handle.

  An alternative to defining a __del__ method on an object. The intended use is
  that ResourceVariables or other objects with resource handles will maintain a
  single reference to this object. When the parent object is collected, this
  object will be too. Even if the parent object is part of a reference cycle,
  the cycle will be collectable.
  """

  __slots__ = ["_handle", "_handle_device", "_context"]

  def __init__(self, handle, handle_device):
    if not isinstance(handle, ops.Tensor):
      raise ValueError(
          (f"Passed handle={handle} to EagerResourceDeleter. Was expecting "
           f"the handle to be a `tf.Tensor`."))
    self._handle = handle
    self._handle_device = handle_device
    # This is held since the __del__ function runs an op, and if the context()
    # is collected before this object, there will be a segfault when running the
    # op.
    self._context = context.context()

  def __del__(self):
    # Resources follow object-identity when executing eagerly, so it is safe to
    # delete the resource we have a handle to.
    try:
      # A packed EagerTensor doesn't own any resource.
      if isinstance(self._handle, ops.EagerTensor) and self._handle.is_packed:
        return
      # This resource was created in eager mode. However, this destructor may be
      # running in graph mode (especially during unit tests). To clean up
      # successfully, we switch back into eager mode temporarily.
      with context.eager_mode():
        with ops.device(self._handle_device):
          gen_resource_variable_ops.destroy_resource_op(
              self._handle, ignore_lookup_error=True)
    except TypeError:
      # Suppress some exceptions, mainly for the case when we're running on
      # module deletion. Things that can go wrong include the context module
      # already being unloaded, self._handle._handle_data no longer being
      # valid, and so on. Printing warnings in these cases is silly
      # (exceptions raised from __del__ are printed as warnings to stderr).
      pass  # 'NoneType' object is not callable when the handle has been
      # partially unloaded.
    except AttributeError:
      pass  # 'NoneType' object has no attribute 'eager_mode' when context has
      # been unloaded. Will catch other module unloads as well.


def shape_safe_assign_variable_handle(handle, shape, value, name=None):
  """Helper that checks shape compatibility and assigns variable."""
  with _handle_graph(handle):
    value_tensor = ops.convert_to_tensor(value)
  shape.assert_is_compatible_with(value_tensor.shape)
  return gen_resource_variable_ops.assign_variable_op(
      handle, value_tensor, name=name)


def _maybe_set_handle_data(dtype, handle, tensor):
  if dtype == dtypes.variant:
    # For DT_VARIANT types, the handle's shape_and_type[1:] stores the
    # variant's handle data.  Extract it.
    handle_data = get_eager_safe_handle_data(handle)
    if handle_data.is_set and len(handle_data.shape_and_type) > 1:
      tensor._handle_data = (  # pylint: disable=protected-access
          cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData(
              is_set=True, shape_and_type=handle_data.shape_and_type[1:]))


def variable_accessed(variable):
  """Records that `variable` was accessed for the tape and FuncGraph."""
  if hasattr(ops.get_default_graph(), "watch_variable"):
    ops.get_default_graph().watch_variable(variable)
  if variable.trainable:
    tape.variable_accessed(variable)


class BaseResourceVariable(variables.VariableV1, core.Tensor):
  """A python variable from an existing handle."""

  # TODO(wangpeng): Deprecate `constraint` when callers no long pass it in.
  def __init__(  # pylint: disable=super-init-not-called
      self,
      trainable=None,
      shape=None,
      dtype=None,
      handle=None,
      constraint=None,
      synchronization=None,
      aggregation=None,
      distribute_strategy=None,
      name=None,
      unique_id=None,
      handle_name=None,
      graph_element=None,
      initial_value=None,
      initializer_op=None,
      is_initialized_op=None,
      cached_value=None,
      save_slice_info=None,
      caching_device=None,
      in_graph_mode=None,
      validate_shape=True,
      **unused_kwargs):
    """Creates a variable from a handle.

    Args:
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      shape: The variable's shape. This shape can be set to tf.TensorShape(None)
        in order to assign values of different shapes to this variable.
        Otherwise (i.e. if the shape is fully determined), it will trigger run
        time checks to ensure that each assignment is of the same shape.
      dtype: The variable's dtype.
      handle: The variable's handle
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: The distribution strategy this variable was created
        under.
      name: The name for this variable.
      unique_id: Internal. Unique ID for this variable's handle.
      handle_name: The name for the variable's handle.
      graph_element: Optional, required only in session.run-mode. Pre-created
        tensor which reads this variable's value.
      initial_value: Optional. Variable's initial value.
      initializer_op: Operation which assigns the variable's initial value.
      is_initialized_op: Pre-created operation to check whether this variable is
        initialized.
      cached_value: Pre-created operation to read this variable in a specific
        device.
      save_slice_info: Metadata for variable partitioning.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      in_graph_mode: whether we are executing in TF1 graph mode. If None, will
        detect within the function. This is to avoid repeated init_scope()
        conetxt entrances which can add up.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
    """
    if in_graph_mode is None:
      with ops.init_scope():
        self._in_graph_mode = not context.executing_eagerly()
    else:
      self._in_graph_mode = in_graph_mode
    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            synchronization, aggregation, trainable, name))
    self._trainable = trainable
    self._synchronization = synchronization
    self._aggregation = aggregation
    self._save_slice_info = save_slice_info
    self._initial_value = initial_value
    self._initializer_op = initializer_op
    self._is_initialized_op = is_initialized_op
    self._graph_element = graph_element
    self._caching_device = caching_device
    self._cached_value = cached_value
    self._distribute_strategy = distribute_strategy
    # Store the graph key so optimizers know how to only retrieve variables from
    # this graph. Guaranteed to be the same as the eager graph_key.
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    self._shape = tensor_shape.as_shape(shape)
    self._dtype = dtypes.as_dtype(dtype)
    self._handle = handle
    self._unique_id = unique_id
    if handle_name is None:
      self._handle_name = "Variable:0"
    else:
      self._handle_name = handle_name + ":0"
    self._constraint = constraint
    self._cached_shape_as_list = None
    self._validate_shape = validate_shape

  def __repr__(self):
    if context.executing_eagerly() and not self._in_graph_mode:
      # If we cannot read the value for any reason (e.g. variable uninitialized
      # during tf.function tracing), still produce a __repr__. Note that for
      # async eager, errors due to uninitialized variables will raise in
      # ops.value_text when the handle is resolved, so we need to keep that
      # under the try...except if we want to suppress them.
      try:
        with ops.device(self.device):
          value_text = ops.value_text(self.read_value(), is_repr=True)
      except:  # pylint: disable=bare-except
        value_text = "numpy=<unavailable>"

      return "<tf.Variable '%s' shape=%s dtype=%s, %s>" % (
          self.name, self.get_shape(), self.dtype.name, value_text)
    else:
      return "<tf.Variable '%s' shape=%s dtype=%s>" % (
          self.name, self.get_shape(), self.dtype.name)

  def __tf_tracing_type__(self, signature_context):
    alias_id = signature_context.alias_global_id(self._handle._id)  # pylint:disable=protected-access
    # TODO(xjun): Create variable placeholders directly from VariableSpec
    # without using original values.
    signature_context.add_placeholder(alias_id, self)
    return VariableSpec(shape=self.shape,
                        dtype=self.dtype,
                        trainable=self.trainable,
                        alias_id=alias_id)

  @contextlib.contextmanager
  def _assign_dependencies(self):
    """Makes assignments depend on the cached value, if any.

    This prevents undefined behavior with reads not ordered wrt writes.

    Yields:
      None.
    """
    if self._cached_value is not None:
      with ops.control_dependencies([self._cached_value]):
        yield
    else:
      yield

  def __array__(self, dtype=None):
    """Allows direct conversion to a numpy array.

    >>> np.array(tf.Variable([1.0]))
    array([1.], dtype=float32)

    Returns:
      The variable value as a numpy array.
    """
    # You can't return `self.numpy()` here because for scalars
    # that raises:
    #     ValueError: object __array__ method not producing an array
    # Even `self.read_value().__array__()` and `self.read_value()._numpy()` give
    # the same error. The `EagerTensor` class must be doing something behind the
    # scenes to make `np.array(tf.constant(1))` work.
    return np.asarray(self.numpy(), dtype=dtype)

  def __nonzero__(self):
    return self.__bool__()

  def __bool__(self):
    return bool(self.read_value())

  def __copy__(self):
    return self

  def __deepcopy__(self, memo):
    if not context.executing_eagerly():
      raise NotImplementedError(
          "__deepcopy__() is only available when eager execution is enabled.")
    copied_variable = ResourceVariable(
        initial_value=self.read_value(),
        trainable=self._trainable,
        constraint=self._constraint,
        dtype=self._dtype,
        name=self._shared_name,
        distribute_strategy=self._distribute_strategy,
        synchronization=self.synchronization,
        aggregation=self.aggregation)
    memo[self._unique_id] = copied_variable
    return copied_variable

  @property
  def dtype(self):
    """The dtype of this variable."""
    return self._dtype

  @property
  def device(self):
    """The device this variable is on."""
    return self.handle.device

  @property
  def graph(self):
    """The `Graph` of this variable."""
    return self.handle.graph

  @property
  def name(self):
    """The name of the handle for this variable."""
    return self._handle_name

  @property
  def shape(self):
    """The shape of this variable."""
    return self._shape

  def set_shape(self, shape):
    self._shape = self._shape.merge_with(shape)

  def _shape_as_list(self):
    if self.shape.ndims is None:
      return None
    return [dim.value for dim in self.shape.dims]

  def _shape_tuple(self):
    shape = self._shape_as_list()
    if shape is None:
      return None
    return tuple(shape)

  @property
  def create(self):
    """The op responsible for initializing this variable."""
    if not self._in_graph_mode:
      raise RuntimeError("This operation is not supported "
                         "when eager execution is enabled.")
    return self._initializer_op

  @property
  def handle(self):
    """The handle by which this variable can be accessed."""
    return self._handle

  def value(self):
    """A cached operation which reads the value of this variable."""
    if self._cached_value is not None:
      return self._cached_value
    with ops.colocate_with(None, ignore_existing=True):
      return self._read_variable_op()

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._graph_element

  @property
  def initializer(self):
    """The op responsible for initializing this variable."""
    return self._initializer_op

  @property
  def initial_value(self):
    """Returns the Tensor used as the initial value for the variable."""
    if context.executing_eagerly():
      raise RuntimeError("This property is not supported "
                         "when eager execution is enabled.")
    return self._initial_value

  @property
  def constraint(self):
    """Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
    return self._constraint

  @property
  def op(self):
    """The op for this variable."""
    return self.handle.op

  @property
  def trainable(self):
    return self._trainable

  @property
  def synchronization(self):
    return self._synchronization

  @property
  def aggregation(self):
    return self._aggregation

  def eval(self, session=None):
    """Evaluates and returns the value of this variable."""
    if context.executing_eagerly():
      raise RuntimeError("This operation is not supported "
                         "when eager execution is enabled.")
    return self._graph_element.eval(session=session)

  def numpy(self):
    if context.executing_eagerly():
      return self.read_value().numpy()
    raise NotImplementedError(
        "numpy() is only available when eager execution is enabled.")

  @deprecated(None, "Prefer Dataset.range instead.")
  def count_up_to(self, limit):
    """Increments this variable until it reaches `limit`.

    When that Op is run it tries to increment the variable by `1`. If
    incrementing the variable would bring it above `limit` then the Op raises
    the exception `OutOfRangeError`.

    If no error is raised, the Op outputs the value of the variable before
    the increment.

    This is essentially a shortcut for `count_up_to(self, limit)`.

    Args:
      limit: value at which incrementing the variable raises an error.

    Returns:
      A `Tensor` that will hold the variable value before the increment. If no
      other Op modifies this variable, the values produced will all be
      distinct.
    """
    return gen_state_ops.resource_count_up_to(
        self.handle, limit=limit, T=self.dtype)

  def _export_to_saved_model_graph(self, object_map=None, tensor_map=None,
                                   options=None, **kwargs):
    """For implementing `Trackable`."""
    new_variable = None
    if options.experimental_variable_policy._save_variable_devices():  # pylint:disable=protected-access
      with ops.device(self.device):
        new_variable = copy_to_graph_uninitialized(self)
    else:
      new_variable = copy_to_graph_uninitialized(self)
    object_map[self] = new_variable
    tensor_map[self.handle] = new_variable.handle
    return [self.handle]

  def _serialize_to_tensors(self):
    """Implements Trackable._serialize_to_tensors."""

    def _read_variable_closure():
      v = self
      with ops.device(v.device):
        if context.executing_eagerly() and not v.is_initialized():
          # A SaveSpec tensor value of `None` indicates that the variable is
          # uninitialized.
          return None
        # Read the variable without making a copy to limit memory usage.
        x = v.read_value_no_copy()
        # To allow variables placed on non-CPU devices to be checkpointed,
        # we copy them to CPU on the same machine first.
        with ops.device("/device:CPU:0"):
          return array_ops.identity(x)

    return {
        trackable.VARIABLE_VALUE_KEY:
            tensor_callable.Callable(
                _read_variable_closure, dtype=self.dtype, device=self.device)
    }

  def _restore_from_tensors(self, restored_tensors):
    """Implements Trackable._restore_from_tensors."""
    with ops.device(self.device):
      restored_tensor = array_ops.identity(
          restored_tensors[trackable.VARIABLE_VALUE_KEY])
      try:
        assigned_variable = shape_safe_assign_variable_handle(
            self.handle, self.shape, restored_tensor)
      except ValueError as e:
        raise ValueError(
            f"Received incompatible tensor with shape {restored_tensor.shape} "
            f"when attempting to restore variable with shape {self.shape} "
            f"and name {self.name}.") from e
      return assigned_variable

  def _read_variable_op(self, no_copy=False):
    """Reads the value of the variable.

    If the variable is in copy-on-read mode and `no_copy` is True, the variable
    is converted to copy-on-write mode before it is read.

    Args:
      no_copy: Whether to prevent a copy of the variable.

    Returns:
      The value of the variable.
    """
    variable_accessed(self)

    def read_and_set_handle(no_copy):
      if no_copy and forward_compat.forward_compatible(2022, 5, 3):
        gen_resource_variable_ops.disable_copy_on_read(self.handle)
      result = gen_resource_variable_ops.read_variable_op(
          self.handle, self._dtype)
      _maybe_set_handle_data(self._dtype, self.handle, result)
      return result

    if getattr(self, "_caching_device", None) is not None:
      with ops.colocate_with(None, ignore_existing=True):
        with ops.device(self._caching_device):
          result = read_and_set_handle(no_copy)
    else:
      result = read_and_set_handle(no_copy)

    if not context.executing_eagerly():
      # Note that if a control flow context is active the input of the read op
      # might not actually be the handle. This line bypasses it.
      tape.record_operation(
          "ReadVariableOp", [result], [self.handle],
          backward_function=lambda x: [x],
          forward_function=lambda x: [x])
    return result

  def read_value(self):
    """Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.

    Returns:
      The value of the variable.
    """
    with ops.name_scope("Read"):
      value = self._read_variable_op()
    # Return an identity so it can get placed on whatever device the context
    # specifies instead of the device where the variable is.
    return array_ops.identity(value)

  def read_value_no_copy(self):
    """Constructs an op which reads the value of this variable without copy.

    The variable is read without making a copy even when it has been sparsely
    accessed. Variables in copy-on-read mode will be converted to copy-on-write
    mode.

    Returns:
      The value of the variable.
    """
    with ops.name_scope("Read"):
      value = self._read_variable_op(no_copy=True)
    # Return an identity so it can get placed on whatever device the context
    # specifies instead of the device where the variable is.
    return array_ops.identity(value)

  def sparse_read(self, indices, name=None):
    """Reads the value of this variable sparsely, using `gather`."""
    with ops.name_scope("Gather" if name is None else name) as name:
      variable_accessed(self)
      value = gen_resource_variable_ops.resource_gather(
          self.handle, indices, dtype=self._dtype, name=name)

      if self._dtype == dtypes.variant:
        # For DT_VARIANT types, the handle's shape_and_type[1:] stores the
        # variant's handle data.  Extract it.
        handle_data = get_eager_safe_handle_data(self.handle)
        if handle_data.is_set and len(handle_data.shape_and_type) > 1:
          value._handle_data = (  # pylint: disable=protected-access
              cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData(
                  is_set=True, shape_and_type=handle_data.shape_and_type[1:]))

    return array_ops.identity(value)

  def gather_nd(self, indices, name=None):
    """Reads the value of this variable sparsely, using `gather_nd`."""
    with ops.name_scope("GatherNd" if name is None else name) as name:
      if self.trainable:
        variable_accessed(self)
      value = gen_resource_variable_ops.resource_gather_nd(
          self.handle, indices, dtype=self._dtype, name=name)

    return array_ops.identity(value)

  def to_proto(self, export_scope=None):
    """Converts a `ResourceVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    if context.executing_eagerly():
      raise RuntimeError("This operation is not supported "
                         "when eager execution is enabled.")
    if export_scope is None or self.handle.name.startswith(export_scope):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(self.handle.name,
                                                   export_scope)
      if self._initial_value is not None:
        # This is inside an if-statement for backwards compatibility, since
        # self._initial_value might be None for variables constructed from old
        # protos.
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      var_def.initializer_name = ops.strip_name_scope(self.initializer.name,
                                                      export_scope)
      if self._cached_value is not None:
        var_def.snapshot_name = ops.strip_name_scope(self._cached_value.name,
                                                     export_scope)
      else:
        # Store the graph_element here
        var_def.snapshot_name = ops.strip_name_scope(self._graph_element.name,
                                                     export_scope)
      var_def.is_resource = True
      var_def.trainable = self.trainable
      var_def.synchronization = self.synchronization.value
      var_def.aggregation = self.aggregation.value
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(
            self._save_slice_info.to_proto(export_scope=export_scope))
      return var_def
    else:
      return None

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    if context.executing_eagerly():
      raise RuntimeError("This operation is not supported "
                         "when eager execution is enabled.")
    return ResourceVariable(
        variable_def=variable_def, import_scope=import_scope)

  __array_priority__ = 100

  def is_initialized(self, name=None):
    """Checks whether a resource variable has been initialized.

    Outputs boolean scalar indicating whether the tensor has been initialized.

    Args:
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `bool`.
    """
    return gen_resource_variable_ops.var_is_initialized_op(self.handle, name)

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    """Subtracts a value from this variable.

    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name to use for the operation.
      read_value: A `bool`. Whether to read and return the new value of the
        variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
    # TODO(apassos): this here and below is not atomic. Consider making it
    # atomic if there's a way to do so without a performance cost for those who
    # don't need it.
    with _handle_graph(self.handle), self._assign_dependencies():
      assign_sub_op = gen_resource_variable_ops.assign_sub_variable_op(
          self.handle,
          ops.convert_to_tensor(delta, dtype=self.dtype),
          name=name)
    if read_value:
      return self._lazy_read(assign_sub_op)
    return assign_sub_op

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    """Adds a value to this variable.

    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name to use for the operation.
      read_value: A `bool`. Whether to read and return the new value of the
        variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
    with _handle_graph(self.handle), self._assign_dependencies():
      assign_add_op = gen_resource_variable_ops.assign_add_variable_op(
          self.handle,
          ops.convert_to_tensor(delta, dtype=self.dtype),
          name=name)
    if read_value:
      return self._lazy_read(assign_add_op)
    return assign_add_op

  def _lazy_read(self, op):
    variable_accessed(self)
    return _UnreadVariable(
        handle=self.handle,
        dtype=self.dtype,
        shape=self._shape,
        in_graph_mode=self._in_graph_mode,
        parent_op=op,
        unique_id=self._unique_id)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    """Assigns a new value to this variable.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name to use for the assignment.
      read_value: A `bool`. Whether to read and return the new value of the
        variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
    # Note: not depending on the cached value here since this can be used to
    # initialize the variable.
    with _handle_graph(self.handle):
      value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
      if not self._shape.is_compatible_with(value_tensor.shape):
        if self.name is None:
          tensor_name = ""
        else:
          tensor_name = " " + str(self.name)
        raise ValueError(
            (f"Cannot assign value to variable '{tensor_name}': Shape mismatch."
             f"The variable shape {self._shape}, and the "
             f"assigned value shape {value_tensor.shape} are incompatible."))
      kwargs = {}
      if forward_compat.forward_compatible(2022, 3, 23):
        # If the shape is fully defined, we do a runtime check with the shape of
        # value.
        validate_shape = self._validate_shape and self._shape.is_fully_defined()
        kwargs["validate_shape"] = validate_shape
      assign_op = gen_resource_variable_ops.assign_variable_op(
          self.handle, value_tensor, name=name, **kwargs)
      if read_value:
        return self._lazy_read(assign_op)
    return assign_op

  def __reduce__(self):
    # The implementation mirrors that of __deepcopy__.
    return functools.partial(
        ResourceVariable,
        initial_value=self.numpy(),
        trainable=self.trainable,
        name=self._shared_name,
        dtype=self.dtype,
        constraint=self.constraint,
        distribute_strategy=self._distribute_strategy), ()

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    """Subtracts `tf.IndexedSlices` from this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be subtracted from this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_sub(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    """Adds `tf.IndexedSlices` to this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be added to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_add(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    """Updates this variable with the max of `tf.IndexedSlices` and itself.

    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of max with this
        variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_max(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    """Updates this variable with the min of `tf.IndexedSlices` and itself.

    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of min with this
        variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_min(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    """Multiply this variable by `tf.IndexedSlices`.

    Args:
      sparse_delta: `tf.IndexedSlices` to multiply this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_mul(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    """Divide this variable by `tf.IndexedSlices`.

    Args:
      sparse_delta: `tf.IndexedSlices` to divide this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_div(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    """Assigns `tf.IndexedSlices` to this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        gen_resource_variable_ops.resource_scatter_update(
            self.handle,
            sparse_delta.indices,
            ops.convert_to_tensor(sparse_delta.values, self.dtype),
            name=name))

  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    """Assigns `tf.IndexedSlices` to this variable batch-wise.

    Analogous to `batch_gather`. This assumes that this variable and the
    sparse_delta IndexedSlices have a series of leading dimensions that are the
    same for all of them, and the updates are performed on the last dimension of
    indices. In other words, the dimensions should be the following:

    `num_prefix_dims = sparse_delta.indices.ndims - 1`
    `batch_dim = num_prefix_dims + 1`
    `sparse_delta.updates.shape = sparse_delta.indices.shape + var.shape[
         batch_dim:]`

    where

    `sparse_delta.updates.shape[:num_prefix_dims]`
    `== sparse_delta.indices.shape[:num_prefix_dims]`
    `== var.shape[:num_prefix_dims]`

    And the operation performed can be expressed as:

    `var[i_1, ..., i_n,
         sparse_delta.indices[i_1, ..., i_n, j]] = sparse_delta.updates[
            i_1, ..., i_n, j]`

    When sparse_delta.indices is a 1D tensor, this operation is equivalent to
    `scatter_update`.

    To avoid this operation one can looping over the first `ndims` of the
    variable and using `scatter_update` on the subtensors that result of slicing
    the first dimension. This is a valid option for `ndims = 1`, but less
    efficient than this implementation.

    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
      raise TypeError(f"Argument `sparse_delta` must be a "
                      f"`tf.IndexedSlices`. Received arg: {sparse_delta}")
    return self._lazy_read(
        state_ops.batch_scatter_update(
            self,
            sparse_delta.indices,
            sparse_delta.values,
            use_locking=use_locking,
            name=name))

  def scatter_nd_sub(self, indices, updates, name=None):
    """Applies sparse subtraction to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_sub(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to ref would look like this:

        [1, -9, 3, -6, -6, 6, 7, -4]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_sub(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))

  def scatter_nd_add(self, indices, updates, name=None):
    """Applies sparse addition to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        add = ref.scatter_nd_add(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(add)
    ```

    The resulting update to ref would look like this:

        [1, 13, 3, 14, 14, 6, 7, 20]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_add(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))

  def scatter_nd_update(self, indices, updates, name=None):
    """Applies sparse assignment to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_update(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to ref would look like this:

        [1, 11, 3, 10, 9, 6, 7, 12]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_update(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))

  def scatter_nd_max(self, indices, updates, name=None):
    """Updates this variable with the max of `tf.IndexedSlices` and itself.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_max(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))

  def scatter_nd_min(self, indices, updates, name=None):
    """Updates this variable with the min of `tf.IndexedSlices` and itself.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
    return self._lazy_read(
        gen_state_ops.resource_scatter_nd_min(
            self.handle,
            indices,
            ops.convert_to_tensor(updates, self.dtype),
            name=name))

  def _write_object_proto(self, proto, options):
    """Writes additional information of the variable into the SavedObject proto.

    Subclasses of ResourceVariables could choose to override this method to
    customize extra information to provide when saving a SavedModel.

    Ideally, this should contain the logic in
    write_object_proto_for_resource_variable but `DistributedValue` is an
    outlier at the momemnt. Once `DistributedValue` becomes a proper
    ResourceVariable, we should remove the helper method below.

    Args:
      proto: `SavedObject` proto to update.
      options: A `SaveOption` instance that configures save behavior.
    """
    write_object_proto_for_resource_variable(self, proto, options)

  def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask,
                            end_mask, ellipsis_mask, new_axis_mask,
                            shrink_axis_mask):
    with _handle_graph(self.handle), self._assign_dependencies():
      return self._lazy_read(
          gen_array_ops.resource_strided_slice_assign(
              ref=self.handle,
              begin=begin,
              end=end,
              strides=strides,
              value=ops.convert_to_tensor(value, dtype=self.dtype),
              name=name,
              begin_mask=begin_mask,
              end_mask=end_mask,
              ellipsis_mask=ellipsis_mask,
              new_axis_mask=new_axis_mask,
              shrink_axis_mask=shrink_axis_mask))

  def __complex__(self):
    return complex(self.value().numpy())

  def __int__(self):
    return int(self.value().numpy())

  def __long__(self):
    return long(self.value().numpy())

  def __float__(self):
    return float(self.value().numpy())

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    del name
    if dtype is not None and not dtype.is_compatible_with(self.dtype):
      raise ValueError(
          f"Incompatible type conversion requested to type {dtype.name} for "
          f"`tf.Variable of type {self.dtype.name}. (Variable: {self})")
    if as_ref:
      return self.read_value().op.inputs[0]
    else:
      return self.value()

  def __iadd__(self, unused_other):
    raise RuntimeError("`variable += value` with `tf.Variable`s is not "
                       "supported. Use `variable.assign_add(value)` to modify "
                       "the variable, or `out = variable + value` if you "
                       "need to get a new output Tensor.")

  def __isub__(self, unused_other):
    raise RuntimeError("`variable -= value` with `tf.Variable`s is not "
                       "supported. Use `variable.assign_sub(value)` to modify "
                       "the variable, or `out = variable * value` if you "
                       "need to get a new output Tensor.")

  def __imul__(self, unused_other):
    raise RuntimeError("`var *= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var * value)` to modify "
                       "the variable, or `out = var * value` if you "
                       "need to get a new output Tensor.")

  def __idiv__(self, unused_other):
    raise RuntimeError("`var /= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var / value)` to modify "
                       "the variable, or `out = var / value` if you "
                       "need to get a new output Tensor.")

  def __itruediv__(self, unused_other):
    raise RuntimeError("`var /= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var / value)` to modify "
                       "the variable, or `out = var / value` if you "
                       "need to get a new output Tensor.")

  def __irealdiv__(self, unused_other):
    raise RuntimeError("`var /= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var / value)` to modify "
                       "the variable, or `out = var / value` if you "
                       "need to get a new output Tensor.")

  def __ipow__(self, unused_other):
    raise RuntimeError("`var **= value` with `tf.Variable`s is not "
                       "supported. Use `var.assign(var ** value)` to modify "
                       "the variable, or `out = var ** value` if you "
                       "need to get a new output Tensor.")


class ResourceVariableGradient(
    composite_tensor_gradient.CompositeTensorGradient):
  """CompositeTensorGradient protocol for ResourceVariable."""

  # TODO(b/246997907): update this method to return value.handle.
  def get_gradient_components(self, value):
    """Returns the components of `value` that should be included in gradients.

    For a ResourceVariable, its gradient component is its handle tensor.
    For now, we return the ResourceVariable because the gradient infrastructure
    has special logics to handle ResourceVariables. We should remove those
    special logics and return the handle tensor.

    Args:
      value: A `ResourceVariable`.

    Returns:
      `value` itself.
    """
    return value

  def replace_gradient_components(self, value, component_grads):
    """Replaces the gradient components in `value` with `component_grads`.

    The gradient of a ResourceVariable is either None or a Tensor. So we don't
    need `value`'s TypeSpec or non-gradient components in this method.

    Args:
      value: A `ResourceVariable` with its gradient components compatible with
        `component_grads`.
      component_grads: A `Tensor` or None as the gradient result.

    Returns:
      The `component_grads`, which is either a `Tensor` or None.
    """
    return component_grads


class ResourceVariable(BaseResourceVariable, composite_tensor.CompositeTensor):
  """Variable based on resource handles.

  See the [Variables How To](https://tensorflow.org/guide/variables)
  for a high level overview.

  A `ResourceVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `ResourceVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  Just like any `Tensor`, variables created with
  `tf.Variable(use_resource=True)` can be used as inputs for other Ops in the
  graph. Additionally, all the operators overloaded for the `Tensor` class are
  carried over to variables, so you can also add nodes to the graph by just
  doing arithmetic on variables.

  Unlike ref-based variable, a ResourceVariable has well-defined semantics. Each
  usage of a ResourceVariable in a TensorFlow graph adds a read_value operation
  to the graph. The Tensors returned by a read_value operation are guaranteed to
  see all modifications to the value of the variable which happen in any
  operation on which the read_value depends on (either directly, indirectly, or
  via a control dependency) and guaranteed to not see any modification to the
  value of the variable from operations that depend on the read_value operation.
  Updates from operations that have no dependency relationship to the read_value
  operation might or might not be visible to read_value.

  For example, if there is more than one assignment to a ResourceVariable in
  a single session.run call there is a well-defined value for each operation
  which uses the variable's value if the assignments and the read are connected
  by edges in the graph. Consider the following example, in which two writes
  can cause tf.Variable and tf.ResourceVariable to behave differently:

  ```python
  a = tf.Variable(1.0, use_resource=True)
  a.initializer.run()

  assign = a.assign(2.0)
  with tf.control_dependencies([assign]):
    b = a.read_value()
  with tf.control_dependencies([b]):
    other_assign = a.assign(3.0)
  with tf.control_dependencies([other_assign]):
    # Will print 2.0 because the value was read before other_assign ran. If
    # `a` was a tf.Variable instead, 2.0 or 3.0 could be printed.
    tf.compat.v1.Print(b, [b]).eval()
  ```
  """

  def __init__(
      self,  # pylint: disable=super-init-not-called
      initial_value=None,
      trainable=None,
      collections=None,
      validate_shape=True,  # pylint: disable=unused-argument
      caching_device=None,
      name=None,
      dtype=None,
      variable_def=None,
      import_scope=None,
      constraint=None,
      distribute_strategy=None,
      synchronization=None,
      aggregation=None,
      shape=None,
      handle=None,
      experimental_enable_variable_lifting=None,
      ):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. Can also be a callable with
        no argument that returns the initial value when called. (Note that
        initializer functions from init_ops.py must first be bound to a shape
        before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `ResourceVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        ResourceVariable. Only used when `variable_def` is provided.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      distribute_strategy: The tf.distribute.Strategy this variable is being
        created inside of.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      handle: (optional) The handle of a `tf.Variable`. If provided, only
        `trainable`, `shape`, `dtype`, and `handle` will be used to construct
        this `tf.Variable`.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is `None`, which signifies that this `Variable` will not be added to any
    collections.
    @end_compatibility
    """
    if variable_def:
      if initial_value is not None:
        raise ValueError(f"The variable_def and initial_value args to "
                         f"`tf.Variable` are mutually exclusive, but got both: "
                         f"variable_def={variable_def},\n"
                         f"initial_value={initial_value}")
      if context.executing_eagerly():
        raise ValueError(f"Creating a `tf.Variable` with a `variable_def` arg "
                         f"is not supported when eager execution is enabled. "
                         f"Got: variable_def={variable_def}")
      self._init_from_proto(
          variable_def,
          import_scope=import_scope,
          validate_shape=validate_shape)
    elif handle is not None:
      self._init_from_handle(trainable=trainable,
                             shape=shape,
                             dtype=dtype,
                             handle=handle)
    else:
      self._init_from_args(
          initial_value=initial_value,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          name=name,
          dtype=dtype,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation,
          shape=shape,
          distribute_strategy=distribute_strategy,
          validate_shape=validate_shape,
          experimental_enable_variable_lifting=experimental_enable_variable_lifting,
          )

  # CompositeTensor method
  @property
  def _type_spec(self):
    return VariableSpec.from_value(self)

  # CompositeTensor method
  def _shape_invariant_to_type_spec(self, shape):
    return VariableSpec(shape, self.dtype, self.trainable)

  # CompositeTensorGradient protocol
  __composite_gradient__ = ResourceVariableGradient()

  def _init_from_args(
      self,
      initial_value=None,
      trainable=None,
      collections=None,
      caching_device=None,
      name=None,
      dtype=None,
      constraint=None,
      synchronization=None,
      aggregation=None,
      distribute_strategy=None,
      shape=None,
      validate_shape=True,
      experimental_enable_variable_lifting=None,
  ):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound to
        a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: DistributionStrategy under which this variable was
        created.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the `GLOBAL_VARIABLES` or
    `TRAINABLE_VARIABLES` collections, and the `collections` argument is
    ignored.
    @end_compatibility
    """
    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            synchronization, aggregation, trainable, name))
    if experimental_enable_variable_lifting is None:
      experimental_enable_variable_lifting = True
    if initial_value is None:
      raise ValueError("The `initial_value` arg to `tf.Variable` must "
                       "be specified except when you are not providing a "
                       "`variable_def`. You provided neither.")
    init_from_fn = callable(initial_value)

    if isinstance(initial_value, ops.Tensor) and hasattr(
        initial_value, "graph") and initial_value.graph.building_function:
      raise ValueError(f"Argument `initial_value` ({initial_value}) could not "
                       "be lifted out of a `tf.function`. "
                       f"(Tried to create variable with name='{name}'). "
                       "To avoid this error, when constructing `tf.Variable`s "
                       "inside of `tf.function` you can create the "
                       "`initial_value` tensor in a "
                       "`tf.init_scope` or pass a callable `initial_value` "
                       "(e.g., `tf.Variable(lambda : "
                       "tf.truncated_normal([10, 40]))`). "
                       "Please file a feature request if this "
                       "restriction inconveniences you.")

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          f"collections argument to Variable constructor must be a list, "
          f"tuple, or set. Got {collections} of type {type(collections)}")
    if constraint is not None and not callable(constraint):
      raise ValueError(f"Argument `constraint` must be None or a callable. "
                       f"a callable. Got a {type(constraint)}:  {constraint}")

    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
    if experimental_enable_variable_lifting:
      maybe_init_scope = ops.init_scope
    else:
      maybe_init_scope = contextlib.nullcontext
    with maybe_init_scope():
      with ops.name_scope(
          name,
          "Variable", [] if init_from_fn else [initial_value],
          skip_on_eager=False) as name:
        # pylint: disable=protected-access
        handle_name = ops.name_from_scope_name(name)
        if self._in_graph_mode:
          shared_name = handle_name
          unique_id = shared_name
        else:
          # When in eager mode, use a uid for the shared_name, to prevent
          # accidental sharing.
          unique_id = "%s_%d" % (handle_name, ops.uid())
          shared_name = None  # Never shared
        # Use attr_scope and device(None) to simulate the behavior of
        # colocate_with when the variable we want to colocate with doesn't
        # yet exist.
        device_context_manager = (
            ops.device if self._in_graph_mode else ops.NullContextmanager)
        attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                s=[compat.as_bytes("loc:@%s" % handle_name)]))
        with ops.get_default_graph()._attr_scope({"_class": attr}):
          with ops.name_scope("Initializer"), device_context_manager(None):
            if init_from_fn:
              initial_value = initial_value()
            if isinstance(initial_value, trackable.CheckpointInitialValue):
              self._maybe_initialize_trackable()
              self._update_uid = initial_value.checkpoint_position.restore_uid
              initial_value = initial_value.wrapped_value
            initial_value = ops.convert_to_tensor(
                initial_value, name="initial_value", dtype=dtype)
          if shape is not None:
            if not initial_value.shape.is_compatible_with(shape):
              raise ValueError(
                  f"In this `tf.Variable` creation, the initial value's shape "
                  f"({initial_value.shape}) is not compatible with "
                  f"the explicitly supplied `shape` argument ({shape}).")
          else:
            shape = initial_value.shape
          handle = eager_safe_variable_handle(
              initial_value=initial_value,
              shape=shape,
              shared_name=shared_name,
              name=name,
              graph_mode=self._in_graph_mode)
          handle._parent_trackable = weakref.ref(self)
          handle._name = handle_name + ":0"
          handle._unique_id = unique_id
        # pylint: disable=protected-access
        if (self._in_graph_mode and initial_value is not None and
            initial_value.op._get_control_flow_context() is not None):
          raise ValueError(
              f"The `initial_value` passed to `tf.Variable` {name} is from "
              f"inside a control-flow  construct, such as a loop or "
              f"conditional. When creating a "
              f"`tf.Variable` inside a loop or conditional, use a lambda as "
              f"the `initial_value`. Got: initial_value=({initial_value})")
        # pylint: enable=protected-access
        dtype = initial_value.dtype.base_dtype

        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            is_initialized_op = (
                gen_resource_variable_ops.var_is_initialized_op(handle))
          if initial_value is not None:
            # pylint: disable=g-backslash-continuation
            with ops.name_scope("Assign") as n, \
                 ops.colocate_with(None, ignore_existing=True), \
                 ops.device(handle.device):
              # pylint: disable=protected-access
              initializer_op = (
                  gen_resource_variable_ops.assign_variable_op(
                      handle,
                      variables._try_guard_against_uninitialized_dependencies(
                          name, initial_value),
                      name=n))
              # pylint: enable=protected-access
            # pylint: enable=g-backslash-continuation
          with ops.name_scope("Read"):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(handle.device):
              value = gen_resource_variable_ops.read_variable_op(handle, dtype)
              _maybe_set_handle_data(dtype, handle, value)
            graph_element = value
            if caching_device is not None:
              # Variables may be created in a tf.device() or ops.colocate_with()
              # context. At the same time, users would expect caching device to
              # be independent of this context, and/or would not expect the
              # current device context to be merged with the caching device
              # spec.  Therefore we reset the colocation stack before creating
              # the cached value. Note that resetting the colocation stack will
              # also reset the device stack.
              with ops.colocate_with(None, ignore_existing=True):
                with ops.device(caching_device):
                  cached_value = array_ops.identity(value)
            else:
              cached_value = None
        else:
          gen_resource_variable_ops.assign_variable_op(handle, initial_value)
          is_initialized_op = None
          initializer_op = None
          graph_element = None
          if caching_device:
            with ops.device(caching_device):
              cached_value = gen_resource_variable_ops.read_variable_op(
                  handle, dtype)
              _maybe_set_handle_data(dtype, handle, cached_value)
          else:
            cached_value = None

        if cached_value is not None:
          # Store the variable object so that the original variable can be
          # accessed to generate functions that are compatible with SavedModel.
          cached_value._cached_variable = weakref.ref(self)  # pylint: disable=protected-access

        if self._in_graph_mode:
          # Eager variables are only added to collections if they are part of an
          # eager variable store (otherwise in an interactive session they would
          # hog memory and cause OOM). This is done in ops/variable_scope.py.
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)
      initial_value = initial_value if self._in_graph_mode else None
      super(ResourceVariable, self).__init__(
          trainable=trainable,
          shape=shape,
          dtype=dtype,
          handle=handle,
          synchronization=synchronization,
          constraint=constraint,
          aggregation=aggregation,
          distribute_strategy=distribute_strategy,
          name=name,
          unique_id=unique_id,
          handle_name=handle_name,
          graph_element=graph_element,
          initial_value=initial_value,
          initializer_op=initializer_op,
          is_initialized_op=is_initialized_op,
          cached_value=cached_value,
          caching_device=caching_device,
          validate_shape=validate_shape,
      )

  def _init_from_proto(self,
                       variable_def,
                       import_scope=None,
                       validate_shape=True):
    """Initializes from `VariableDef` proto."""
    # Note that init_from_proto is currently not supported in Eager mode.
    assert not context.executing_eagerly()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError(f"The `variable_def` you passed to `tf.Variable` is "
                       f"Trying to restore a TF 1.x Reference Variable "
                       f"as a TF 2.x ResourceVariable. This is unsupported. "
                       f"Got variable_def={variable_def}")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.variable_name, import_scope=import_scope),
        allow_operation=False)
    self._shape = tensor_shape.TensorShape(self._handle.op.get_attr("shape"))
    self._handle_name = self._handle.name
    self._unique_id = self._handle_name
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.initializer_name, import_scope=import_scope))
    # Check whether initial_value_name exists for backwards compatibility.
    if (hasattr(variable_def, "initial_value_name") and
        variable_def.initial_value_name):
      self._initial_value = g.as_graph_element(
          ops.prepend_name_scope(
              variable_def.initial_value_name, import_scope=import_scope))
    else:
      self._initial_value = None
    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            variable_def.synchronization, variable_def.aggregation,
            variable_def.trainable, variable_def.variable_name))
    self._synchronization = synchronization
    self._aggregation = aggregation
    self._trainable = trainable
    if variable_def.snapshot_name:
      snapshot = g.as_graph_element(
          ops.prepend_name_scope(
              variable_def.snapshot_name, import_scope=import_scope))
      if snapshot.op.type != "ReadVariableOp":
        self._cached_value = snapshot
      else:
        self._cached_value = None
      while snapshot.op.type != "ReadVariableOp":
        snapshot = snapshot.op.inputs[0]
      self._graph_element = snapshot
    else:
      self._cached_value = None
      # Legacy case for protos without the snapshot name; assume it's the
      # following.
      self._graph_element = g.get_tensor_by_name(self._handle.op.name +
                                                 "/Read/ReadVariableOp:0")
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def,
          import_scope=import_scope)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._dtype = dtypes.as_dtype(self._handle.op.get_attr("dtype"))
    self._constraint = None
    self._validate_shape = validate_shape

  def _init_from_handle(self,
                        trainable=None,
                        shape=None,
                        dtype=None,
                        handle=None):
    handle_data = get_eager_safe_handle_data(handle)
    if not handle_data.is_set:
      # The handle may not have the handle shape and dtype if it was created
      # using tf.placeholder.
      handle_data = handle_data_util.create_handle_data(shape, dtype)
      handle_data_util.set_handle_data(handle, handle_data)
    # pylint: disable=protected-access
    if hasattr(handle, "_name") and isinstance(handle._name, str):
      handle_name = handle._name.rstrip(":0")
    else:
      handle_name = None
    # pylint: enable=protected-access
    unique_id = getattr(handle, "_unique_id", None)
    super().__init__(
        trainable=trainable, shape=shape, dtype=dtype, handle=handle,
        unique_id=unique_id, handle_name=handle_name)


class UninitializedVariable(BaseResourceVariable):
  """A variable with no initializer."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      trainable=None,
      caching_device=None,
      name=None,
      shape=None,
      dtype=None,
      constraint=None,
      synchronization=None,
      aggregation=None,
      extra_handle_data=None,
      distribute_strategy=None,
      **unused_kwargs):
    """Creates the variable handle.

    Args:
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      shape: The variable's shape.
      dtype: The variable's dtype.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      extra_handle_data: Optional, another resource handle or Tensor with handle
        data to merge with `shape` and `dtype`.
      distribute_strategy: The tf.distribute.Strategy this variable is being
        created inside of.
    """
    with ops.init_scope():
      # Here we are detecting eagerness within an init_scope, so this will only
      # be true when we are running in TF1 graph mode.
      self._in_graph_mode = not context.executing_eagerly()
      with ops.name_scope(name, "Variable", skip_on_eager=False) as name:
        handle_name = ops.name_from_scope_name(name)
        if self._in_graph_mode:
          shared_name = handle_name
          unique_id = shared_name
        else:
          unique_id = "%s_%d" % (handle_name, ops.uid())
          shared_name = None  # Never shared
        handle = _variable_handle_from_shape_and_dtype(
            shape=shape,
            dtype=dtype,
            shared_name=shared_name,
            name=name,
            graph_mode=self._in_graph_mode,
            initial_value=extra_handle_data)
        handle._parent_trackable = weakref.ref(self)
        handle._name = handle_name + ":0"
        handle._unique_id = unique_id

        if self._in_graph_mode:
          # We only need to add the read_variable_op in TF1.
          with ops.name_scope("Read"):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(handle.device):
              value = gen_resource_variable_ops.read_variable_op(handle, dtype)
              _maybe_set_handle_data(dtype, handle, value)
            graph_element = value
          ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, self)
          # Do *not* add to TRAINABLE_VARIABLES here, even if self._trainable,
          # because retraining or frozen use of imported SavedModels is
          # controlled at higher levels of model building.
        else:
          graph_element = None
    super(UninitializedVariable, self).__init__(
        distribute_strategy=distribute_strategy,
        shape=shape,
        dtype=dtype,
        unique_id=unique_id,
        handle_name=handle_name,
        constraint=constraint,
        handle=handle,
        graph_element=graph_element,
        trainable=trainable,
        synchronization=synchronization,
        aggregation=aggregation,
        in_graph_mode=self._in_graph_mode)


_pywrap_utils.RegisterType("ResourceVariable", ResourceVariable)
math_ops._resource_variable_type = ResourceVariable  # pylint: disable=protected-access


def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
tensor_conversion_registry.register_tensor_conversion_function(
    BaseResourceVariable, _dense_var_to_tensor)


class _UnreadVariable(BaseResourceVariable):
  """Represents a future for a read of a variable.

  Pretends to be the tensor if anyone looks.
  """

  def __init__(self, handle, dtype, shape, in_graph_mode, parent_op, unique_id):
    if isinstance(handle, ops.EagerTensor):
      handle_name = ""
    else:
      handle_name = handle.name
    # Only create a graph_element if we're in session.run-land as only
    # session.run requires a preexisting tensor to evaluate. Otherwise we can
    # avoid accidentally reading the variable.
    if context.executing_eagerly() or ops.inside_function():
      graph_element = None
    else:
      with ops.control_dependencies([parent_op]):
        graph_element = gen_resource_variable_ops.read_variable_op(
            handle, dtype)
        _maybe_set_handle_data(dtype, handle, graph_element)
    super(_UnreadVariable, self).__init__(
        handle=handle,
        shape=shape,
        handle_name=handle_name,
        unique_id=unique_id,
        dtype=dtype,
        graph_element=graph_element)
    self._parent_op = parent_op

  @property
  def name(self):
    if self._in_graph_mode:
      return self._parent_op.name
    else:
      return "UnreadVariable"

  def value(self):
    return self._read_variable_op()

  def read_value(self):
    return self._read_variable_op()

  def _read_variable_op(self):
    with ops.control_dependencies([self._parent_op]):
      result = gen_resource_variable_ops.read_variable_op(
          self._handle, self._dtype)
      _maybe_set_handle_data(self._dtype, self._handle, result)
      return result

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).assign_sub(delta, use_locking, name,
                                                     read_value)

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).assign_add(delta, use_locking, name,
                                                     read_value)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).assign(value, use_locking, name,
                                                 read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_sub(sparse_delta, use_locking,
                                                      name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_add(sparse_delta, use_locking,
                                                      name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_max(sparse_delta, use_locking,
                                                      name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_min(sparse_delta, use_locking,
                                                      name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_mul(sparse_delta, use_locking,
                                                      name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_div(sparse_delta, use_locking,
                                                      name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable,
                   self).scatter_update(sparse_delta, use_locking, name)

  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable,
                   self).batch_scatter_update(sparse_delta, use_locking, name)

  def scatter_nd_sub(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_nd_sub(indices, updates, name)

  def scatter_nd_add(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_nd_add(indices, updates, name)

  def scatter_nd_update(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable,
                   self).scatter_nd_update(indices, updates, name)

  def scatter_nd_max(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_nd_max(indices, updates, name)

  def scatter_nd_min(self, indices, updates, name=None):
    with ops.control_dependencies([self._parent_op]):
      return super(_UnreadVariable, self).scatter_nd_min(indices, updates, name)

  @property
  def op(self):
    """The op for this variable."""
    return self._parent_op


@ops.RegisterGradient("ReadVariableOp")
def _ReadGrad(_, grad):
  """Gradient for read op."""
  return grad


def variable_shape(handle, out_type=dtypes.int32):
  handle_data = get_eager_safe_handle_data(handle)
  if handle_data is None or not handle_data.is_set:
    return gen_resource_variable_ops.variable_shape(handle, out_type=out_type)
  shape_proto = handle_data.shape_and_type[0].shape
  if shape_proto.unknown_rank or any(x.size == -1 for x in shape_proto.dim):
    return gen_resource_variable_ops.variable_shape(handle, out_type=out_type)
  return constant_op.constant([x.size for x in shape_proto.dim], dtype=out_type)


@ops.RegisterGradient("ResourceGather")
def _GatherGrad(op, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = variable_shape(handle)
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return (indexed_slices.IndexedSlices(values, indices, params_shape), None)


def _to_proto_fn(v, export_scope=None):
  """Converts Variable and ResourceVariable to VariableDef for collections."""
  return v.to_proto(export_scope=export_scope)


def _from_proto_fn(v, import_scope=None):
  """Creates Variable or ResourceVariable from VariableDef as needed."""
  if v.is_resource:
    return ResourceVariable.from_proto(v, import_scope=import_scope)
  return variables.Variable.from_proto(v, import_scope=import_scope)


ops.register_proto_function(
    ops.GraphKeys.GLOBAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.TRAINABLE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.LOCAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.MODEL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.GLOBAL_STEP,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.METRIC_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)


@tf_export("__internal__.ops.is_resource_variable", v1=[])
def is_resource_variable(var):
  """"Returns True if `var` is to be considered a ResourceVariable."""
  return isinstance(var, BaseResourceVariable) or hasattr(
      var, "_should_act_as_resource_variable")


def copy_to_graph_uninitialized(var):
  """Copies an existing variable to a new graph, with no initializer."""
  # Like ResourceVariable.__deepcopy__, but does not set an initializer on the
  # new variable.
  # pylint: disable=protected-access
  new_variable = UninitializedVariable(
      trainable=var.trainable,
      constraint=var._constraint,
      shape=var.shape,
      dtype=var.dtype,
      name=var._shared_name,
      synchronization=var.synchronization,
      aggregation=var.aggregation,
      extra_handle_data=var.handle)
  new_variable._maybe_initialize_trackable()
  # pylint: enable=protected-access
  return new_variable


ops.NotDifferentiable("Assert")
ops.NotDifferentiable("VarIsInitializedOp")
ops.NotDifferentiable("VariableShape")


# TODO(b/246356867): This is the draft implementation. Currently VariableSpec is
# the only class using them. Move them to a separate file when necessary.
class StructurePattern:
  pass


class PLeaf(StructurePattern):
  """Represents a singleton leaf StructurePattern."""

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super().__new__(cls)
    return cls.instance


class PList(StructurePattern):
  """Represents a list of StructurePatterns."""

  def __init__(self, *components):
    self.components = list(components)

  def __eq__(self, other):
    return isinstance(other, PList) and self.components == other.components


class VariableSpec(tensor_spec.DenseSpec):
  """Describes a tf.Variable.

  A `VariableSpec` provides metadata describing the `tf.Variable` objects
  accepted or returned by TensorFlow 2.x APIs.
  """

  __slots__ = ["trainable", "alias_id"]

  value_type = property(lambda self: ResourceVariable)

  def __init__(self, shape, dtype=dtypes.float32, trainable=True,
               alias_id=None):
    super(VariableSpec, self).__init__(shape, dtype=dtype)
    self.trainable = trainable
    self.alias_id = alias_id

  def is_compatible_with(self, spec_or_value):
    """Returns True if `spec_or_value` is compatible with this `VariableSpec`.

    `spec_or_value` is considered to be compatible with this `VariableSpec` if

    * `spec_or_value` is a `Variable` or `VariableSpec`,
    * their shapes are compatible,
    * their dtypes are the same,
    * they are both trainable or not trainable.
    * they share the same alias_id if `spec_or_value` is a `VariableSpec`.

    Example:

    >>> v = tf.Variable([1., 2., 3.])
    >>> spec = VariableSpec([None])
    >>> spec.is_compatible_with(v)
    True
    >>> v = tf.Variable(1)
    >>> spec.is_compatible_with(v)
    False

    Args:
      spec_or_value: A VariableSpec or Variable to compare against.

    Returns:
      True if `spec_or_value` is compatible with this `VariableSpec`.
    """
    if not isinstance(spec_or_value, (type(self), self.value_type)):
      return False
    compatible = (self.shape.is_compatible_with(spec_or_value.shape) and
                  self.dtype == spec_or_value.dtype and
                  self.trainable == spec_or_value.trainable)
    if isinstance(spec_or_value, type(self)):
      # alias_id must be the same to be compatible.
      return compatible and self.alias_id == spec_or_value.alias_id
    return compatible

  @classmethod
  def from_value(cls, value):
    """Creates a `VariableSpec` from the given `Variable`.

    `value`'s shape, dtype, and trainable attributes will be used to create
    the new `VariableSpec`.

    Example:

    >>> v = tf.Variable([1., 2., 3.])
    >>> VariableSpec.from_value(v)
    VariableSpec(shape=(3,), dtype=tf.float32, trainable=True, alias_id=None)

    Args:
      value: A Variable.

    Returns:
      A `VariableSpec` created from `value`.
    """
    return cls(value.shape, dtype=value.dtype, trainable=value.trainable)

  def _to_components(self, value):
    return [value.handle]

  def _from_components(self, components):
    if not isinstance(components, (list, tuple)):
      raise TypeError(f"Components of a ResourceVariable must be a list or "
                      f"tuple, got f{components} instead.")
    if len(components) != 1:
      raise ValueError(f"Components of a ResourceVariable must only contain "
                       f"its resource handle, got f{components} instead.")
    handle = components[0]
    if not isinstance(handle, ops.Tensor) or handle.dtype != dtypes.resource:
      raise ValueError(f"The handle of a ResourceVariable must be a resource "
                       f"tensor, got {handle} instead.")
    return ResourceVariable(trainable=self.trainable,
                            shape=self.shape,
                            dtype=self.dtype,
                            handle=handle)

  @property
  def _component_specs(self):
    return [tensor_spec.TensorSpec([], dtypes.resource)]

  def _serialize(self):
    return self.shape, self.dtype, self.trainable, self.alias_id

  # TraceType method
  def is_subtype_of(self, other):
    if type(self) is not type(other):
      return False

    # Remove this once we add alias_id to all CompositeTensors with
    # ResourceVariable components.
    if self.alias_id is None and other.alias_id is None:
      return super().is_subtype_of(other)

    if self.alias_id is None or other.alias_id is None:
      raise NotImplementedError(f"VariableSpec.is_subtype_of doesn't support "
                                f"alias_id=None, got self: {self} and other: "
                                f"{other}.")

    return super().is_subtype_of(other)

  # TraceType method
  def most_specific_common_supertype(self, others):
    if any(type(self) is not type(other) for other in others):
      return None

    # It is a special case for tf.nest, which often takes CompositeTensors and
    # converts to TypeSpecs internally, such as tf.nest.assert_same_structure.
    if (self.alias_id is None and
        all(other.alias_id is None for other in others)):
      return super().most_specific_common_supertype(others)

    if self.alias_id is None or any(other.alias_id is None for other in others):
      raise NotImplementedError(f"VariableSpec.most_specific_common_supertype "
                                f"doesn't support alias_id=None, got self: "
                                f"{self} and others: {others}.")

    return super().most_specific_common_supertype(others)

  # TraceType method
  def placeholder_value(self, placeholder_context):
    if placeholder_context.unnest_only:
      return self

    name = self.name or placeholder_context.naming_scope
    context_graph = placeholder_context.context_graph
    if placeholder_context.has_placeholder(self.alias_id):
      # Get reference to the existing variable if alias_id already
      # exists in the PlaceholderContext
      variable = placeholder_context.get_placeholder(self.alias_id)
    else:
      spec = tensor_spec.TensorSpec([], dtypes.resource)
      spec_context = trace_type.InternalPlaceholderContext(
          context_graph.outer_graph)
      spec_context.update_naming_scope(name)
      placeholder = spec.placeholder_value(spec_context)
      variable = self._from_components([placeholder])
      # (b/262771247) ShardedVariable break without this and VariableSpecs
      # without alias_id are not TraceTypes.
      if self.alias_id is not None:
        placeholder_context.add_placeholder(self.alias_id, variable)
    # Capture the Variable's placeholder within the default graph of
    # the current thread.
    placeholder = context_graph.capture(variable.handle, name=name)
    placeholder.op._set_attr(  # pylint: disable=protected-access
        "_user_specified_name",
        attr_value_pb2.AttrValue(s=compat.as_bytes(name)))
    return variable

  def _to_tensors(self, value):
    assert isinstance(value, BaseResourceVariable)
    return [value.handle]

  def _get_structure(self):
    # shape, dtype, trainable, and alias_id are all leaves.
    return PList(PLeaf(), PLeaf(), PLeaf(), PLeaf())

  def __repr__(self):
    return (f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype!r}, "
            f"trainable={self.trainable!r}, alias_id={self.alias_id!r})")

  def __hash__(self):
    return hash((self.shape, self.dtype, self.trainable, self.alias_id))

  def __eq__(self, other):
    return (type(self) is type(other) and self.shape == other.shape and
            self.dtype == other.dtype and self.trainable == other.trainable and
            self.alias_id == other.alias_id)


nested_structure_coder.register_codec(
    nested_structure_coder.BuiltInTypeSpecCodec(
        VariableSpec, struct_pb2.TypeSpecProto.VARIABLE_SPEC
    )
)


_pywrap_utils.RegisterType("VariableSpec", VariableSpec)


def write_object_proto_for_resource_variable(resource_variable,
                                             proto,
                                             options,
                                             enforce_naming=True):
  """Writes additional information of the variable into the SavedObject proto.

  This allows users to define a `hook` to provide extra information of the
  variable to the SavedObject.

  For example, DistributedVariable class would fill in components in the
  distributed context.

  Args:
    resource_variable: A `ResourceVariable` or `DistributedValue` that has the
      information to be saved into the proto.
    proto: `SavedObject` proto to update.
    options: A `SaveOption` instance that configures save behavior.
    enforce_naming: A bool determining whether to check that names end in the
      expected string ':0'
  """
  proto.variable.SetInParent()
  if enforce_naming and not resource_variable.name.endswith(":0"):
    raise ValueError(f"Cowardly refusing to save variable "
                     f"{resource_variable.name} because of "
                     f"unexpected suffix in the name (expected ':0')"
                     f"which won't be restored.")
  proto.variable.name = meta_graph._op_name(resource_variable.name)  # pylint: disable=protected-access
  proto.variable.trainable = resource_variable.trainable
  proto.variable.dtype = resource_variable.dtype.as_datatype_enum
  proto.variable.synchronization = resource_variable.synchronization.value
  proto.variable.aggregation = resource_variable.aggregation.value
  proto.variable.shape.CopyFrom(resource_variable.shape.as_proto())
  if options.experimental_variable_policy._save_variable_devices(  # pylint: disable=protected-access
  ):
    if hasattr(resource_variable, "device"):
      proto.variable.device = resource_variable.device
