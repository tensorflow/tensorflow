# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Classes and functions used to construct graphs."""
# pylint: disable=g-bad-name
import collections
from collections.abc import Callable, Iterator, Sequence
import contextlib
import copy
import enum
import re
import sys
import threading
import types
from typing import cast, TypeVar, Any, AnyStr, NoReturn, Optional, Pattern, Union, ContextManager

from absl import app
import numpy as np
from numpy import typing as npt

from google.protobuf import message
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
# pywrap_tensorflow must be imported first to avoid protobuf issues.
# (b/143110113)
# pylint: disable=invalid-import-order,g-bad-import-order,unused-import
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
# pylint: enable=invalid-import-order,g-bad-import-order,unused-import
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export

_T = TypeVar("_T")
GraphType = TypeVar("GraphType", bound="Graph")
OpStatsType = TypeVar("OpStatsType", bound="OpStats")
OperationType = TypeVar("OperationType", bound="Operation")
EagerTensorType = TypeVar("EagerTensorType", bound="_EagerTensorBase")


# TODO(b/307794935): Remove after bug is fixed.
is_oss = True  # Updated by copybara

# Temporary global switches determining if we should enable the work-in-progress
# calls to the C API. These will be removed once all functionality is supported.
_USE_C_API: bool = True
_USE_C_SHAPES: bool = True


_api_usage_gauge = monitoring.BoolGauge(
    "/tensorflow/api/ops_eager_execution",
    "Whether ops.enable_eager_execution() is called.")

_control_flow_api_gauge = monitoring.BoolGauge(
    "/tensorflow/api/enable_control_flow_v2",
    "Whether enable_control_flow_v2() is called.")

_tf_function_api_gauge = monitoring.BoolGauge(
    "/tensorflow/api/tf_function",
    "Whether tf.function() is used.")

# pylint: disable=protected-access
_DTYPES_INTERN_TABLE: dict[types_pb2.DataType, dtypes.DType] = (
    dtypes._INTERN_TABLE)
# pylint: enable=protected-access


def tensor_id(tensor) -> Any:
  """Returns a unique identifier for this Tensor."""
  return tensor._id  # pylint: disable=protected-access


class _UserDeviceSpec(object):
  """Store user-specified device and provide computation of merged device."""

  def __init__(self, device_name_or_function) -> None:
    self._device_name_or_function = device_name_or_function
    self.display_name = str(self._device_name_or_function)
    self.function = device_name_or_function
    self.raw_string = None

    if isinstance(device_name_or_function, pydev.MergeDevice):
      self.is_null_merge = device_name_or_function.is_null_merge

    elif callable(device_name_or_function):
      self.is_null_merge = False
      dev_func = self._device_name_or_function
      func_name = function_utils.get_func_name(dev_func)
      func_code = function_utils.get_func_code(dev_func)
      if func_code:
        fname = func_code.co_filename
        lineno = func_code.co_firstlineno
      else:
        fname = "unknown"
        lineno = -1
      self.display_name = "%s<%s, %d>" % (func_name, fname, lineno)

    elif device_name_or_function is None:
      # NOTE(taylorrobie): This MUST be False. None signals a break in the
      #   device stack, so `is_null_merge` must be False for such a case to
      #   allow callers to safely skip over null merges without missing a None.
      self.is_null_merge = False

    else:
      self.raw_string = device_name_or_function
      self.function = pydev.merge_device(device_name_or_function)
      self.is_null_merge = self.function.is_null_merge

    # We perform this check in __init__ because it is of non-trivial cost,
    # and self.string_merge is typically called many times.
    self.fast_string_merge = isinstance(self.function, pydev.MergeDevice)

  def string_merge(self, node_def) -> str:
    if self.fast_string_merge:
      return self.function.shortcut_string_merge(node_def)

    return compat.as_str(_device_string(self.function(node_def)))


class NullContextmanager(contextlib.AbstractContextManager[None]):

  def __init__(self, *args, **kwargs) -> None:
    pass

  def __enter__(self) -> None:
    pass

  def __exit__(self, type_arg, value_arg, traceback_arg) -> bool:
    return False  # False values do not suppress exceptions


def _as_graph_element(obj):
  """Convert `obj` to a graph element if possible, otherwise return `None`.

  Args:
    obj: Object to convert.

  Returns:
    The result of `obj._as_graph_element()` if that method is available;
        otherwise `None`.
  """
  conv_fn = getattr(obj, "_as_graph_element", None)
  if conv_fn and callable(conv_fn):
    return conv_fn()
  return None


# Deprecated - legacy purposes only.
def is_dense_tensor_like(t) -> bool:
  return isinstance(t, core_tf_types.Tensor)


def uid() -> int:
  """A unique (within this program execution) integer."""
  return pywrap_tfe.TFE_Py_UID()


def numpy_text(tensor, is_repr=False) -> str:
  """Human readable representation of a tensor's numpy value."""
  if tensor.dtype.is_numpy_compatible:
    # pylint: disable=protected-access
    tensor_numpy = tensor._numpy()
    if is_repr:
      if np.isscalar(tensor_numpy) and not isinstance(tensor_numpy, bytes):
        # .item() converts the numpy scalars to python items.
        text = repr(tensor_numpy.item())
      else:
        text = repr(tensor_numpy)
    else:
      text = str(tensor_numpy)
    # pylint: enable=protected-access
  else:
    text = "<unprintable>"
  if "\n" in text:
    text = "\n" + text
  return text


def value_text(tensor, is_repr=False) -> AnyStr:
  """Either the NumPy value or a custom TensorFlow formatting of `tensor`.

  Custom formatting is used for custom device tensors, e.g. parallel tensors
  with multiple components on different devices.

  Args:
    tensor: The tensor to format.
    is_repr: Controls the style/verbosity of formatting.

  Returns:
    The formatted tensor.
  """
  # pylint: disable=protected-access  # friend access
  if tensor._prefer_custom_summarizer():
    text = tensor._summarize_value()
    # pylint: enable=protected-access
    if is_repr:
      text = "value=" + text
  else:
    text = numpy_text(tensor, is_repr=is_repr)
    if is_repr:
      text = "numpy=" + text
  return text


@tf_export("__internal__.SymbolicTensor")
class SymbolicTensor(pywrap_tf_session.PyTensor, tensor_lib.Tensor):
  """A symbolic tensor from a graph or tf.function."""

  def __new__(cls, op, value_index, dtype, unique_id=None) -> "SymbolicTensor":
    if unique_id is None:
      unique_id = uid()
    return pywrap_tf_session.PyTensor.__new__(
        SymbolicTensor, op, value_index, dtypes.as_dtype(dtype), unique_id
    )

  def __copy__(self) -> "SymbolicTensor":
    cls = self.__class__
    result = cls.__new__(cls, self.op, self.value_index, self.dtype, self._id)
    result.__dict__.update(self.__dict__)
    return result


def _create_graph_constant(
    value, dtype, shape, name, verify_shape, allow_broadcast
) -> tensor_lib.Tensor:
  """Create a graph constant and invoke constant callbacks."""
  g = get_default_graph()
  tensor_value = attr_value_pb2.AttrValue()
  tensor_value.tensor.CopyFrom(
      tensor_util.make_tensor_proto(
          value, dtype=dtype, shape=shape, verify_shape=verify_shape,
          allow_broadcast=allow_broadcast))
  dtype_value = attr_value_pb2.AttrValue(type=tensor_value.tensor.dtype)
  attrs = {"value": tensor_value, "dtype": dtype_value}
  const_tensor = g._create_op_internal(  # pylint: disable=protected-access
      "Const", [], [dtype_value.type], attrs=attrs, name=name).outputs[0]

  if op_callbacks.should_invoke_op_callbacks():
    # TODO(b/147670703): Once the special-op creation code paths
    # are unified. Remove this `if` block.
    callback_outputs = op_callbacks.invoke_op_callbacks(
        "Const", tuple(), attrs, (const_tensor,), op_name=name, graph=g)
    if callback_outputs is not None:
      [const_tensor] = callback_outputs
  return const_tensor


class _EagerTensorBase(
    tensor_lib.Tensor, internal.NativeObject, core_tf_types.Value):
  """Base class for EagerTensor."""

  # __complex__, __int__, __float__ and __index__ may copy the tensor to CPU and
  # only work for scalars; values are cast as per numpy.
  def __complex__(self) -> complex:
    return complex(self._numpy())

  def __int__(self) -> int:
    return int(self._numpy())

  def __float__(self) -> float:
    return float(self._numpy())

  def __index__(self) -> int:
    return cast(np.ndarray, self._numpy()).__index__()

  def __bool__(self) -> bool:
    x = self._numpy()
    if isinstance(x, np.ndarray):
      return bool(x.size > 0 and x)
    else:
      return bool(x)

  __nonzero__ = __bool__

  def __format__(self, format_spec) -> str:
    if self._prefer_custom_summarizer():
      return self._summarize_value().__format__(format_spec)
    elif self.dtype.is_numpy_compatible:
      # Not numpy_text here, otherwise the __format__ behaves differently.
      return self._numpy().__format__(format_spec)
    else:
      return "<unprintable>".__format__(format_spec)  # pytype: disable=attribute-error

  def __reduce__(self):
    return convert_to_tensor, (self._numpy(),)

  def __copy__(self: EagerTensorType) -> EagerTensorType:
    # Eager Tensors are immutable so it's safe to return themselves as a copy.
    return self

  def __deepcopy__(self: EagerTensorType, memo) -> EagerTensorType:
    # Eager Tensors are immutable so it's safe to return themselves as a copy.
    del memo
    return self

  def __str__(self) -> str:
    return "tf.Tensor(%s, shape=%s, dtype=%s)" % (
        value_text(self, is_repr=False), self.shape, self.dtype.name)

  def __repr__(self) -> str:
    return "<tf.Tensor: shape=%s, dtype=%s, %s>" % (
        self.shape, self.dtype.name, value_text(self, is_repr=True))

  def __len__(self) -> int:
    """Returns the length of the first dimension in the Tensor."""
    if not self.shape.ndims:
      raise TypeError("Scalar tensor has no `len()`")
    # pylint: disable=protected-access
    try:
      return self._shape_tuple()[0]
    except core._NotOkStatusException as e:
      raise core._status_to_exception(e) from None

  def __array__(self, dtype=None) -> np.ndarray:
    a = self._numpy()
    if not dtype:
      return cast(np.ndarray, a)

    return np.array(a, dtype=dtype)

  def __hash__(self) -> int:
    # EagerTensors are never hashable.
    raise TypeError("Tensor is unhashable. "
                    "Instead, use tensor.ref() as the key.")

  def _numpy_internal(self) -> npt.ArrayLike:
    raise NotImplementedError()

  def _numpy(self) -> npt.ArrayLike:
    try:
      return self._numpy_internal()
    except core._NotOkStatusException as e:  # pylint: disable=protected-access
      raise core._status_to_exception(e) from None  # pylint: disable=protected-access

  @property
  def dtype(self) -> dtypes.DType:
    # Note: using the intern table directly here as this is
    # performance-sensitive in some models.
    return dtypes._INTERN_TABLE[self._datatype_enum()]  # pylint: disable=protected-access

  def numpy(self) -> npt.ArrayLike:
    """Copy of the contents of this Tensor into a NumPy array or scalar.

    Unlike NumPy arrays, Tensors are immutable, so this method has to copy
    the contents to ensure safety. Use `memoryview` to get a readonly
    view of the contents without doing a copy:

    >>> t = tf.constant([42])
    >>> np.asarray(memoryview(t))
    array([42], dtype=int32)

    Note that `memoryview` is only zero-copy for Tensors on CPU. If a Tensor
    is on GPU, it will have to be transferred to CPU first in order for
    `memoryview` to work.

    Returns:
      A NumPy array of the same shape and dtype or a NumPy scalar, if this
      Tensor has rank 0.

    Raises:
      ValueError: If the dtype of this Tensor does not have a compatible
        NumPy dtype.
    """
    # TODO(slebedev): Consider avoiding a copy for non-CPU or remote tensors.
    maybe_arr = self._numpy()  # pylint: disable=protected-access
    return maybe_arr.copy() if isinstance(maybe_arr, np.ndarray) else maybe_arr

  @property
  def backing_device(self):
    """Returns the name of the device holding this tensor's memory.

    `.backing_device` is usually the same as `.device`, which returns
    the device on which the kernel of the operation that produced this tensor
    ran. However, some operations can produce tensors on a different device
    (e.g., an operation that executes on the GPU but produces output tensors
    in host memory).
    """
    raise NotImplementedError()

  def _datatype_enum(self) -> NoReturn:
    raise NotImplementedError()

  def _shape_tuple(self) -> NoReturn:
    """The shape of this Tensor, as a tuple.

    This is more performant than tuple(shape().as_list()) as it avoids
    two list and one object creation. Marked private for now as from an API
    perspective, it would be better to have a single performant way of
    getting a shape rather than exposing shape() and shape_tuple()
    (and heaven forbid, shape_list() etc. as well!). Punting on that for now,
    but ideally one would work things out and remove the need for this method.

    Returns:
      tuple with the shape.
    """
    raise NotImplementedError()

  def _rank(self) -> NoReturn:
    """Integer rank of this Tensor.

    Unlike regular Tensors, the rank is always known for EagerTensors.

    This is more performant than len(self._shape_tuple())

    Returns:
      Integer rank
    """
    raise NotImplementedError()

  def _num_elements(self) -> NoReturn:
    """Number of elements of this Tensor.

    Unlike regular Tensors, the number of elements is always known for
    EagerTensors.

    This is more performant than tensor.shape.num_elements

    Returns:
      Long - num elements in the tensor
    """
    raise NotImplementedError()

  def _copy_to_device(self, device_name) -> NoReturn:  # pylint: disable=redefined-outer-name
    raise NotImplementedError()

  @staticmethod
  def _override_operator(name, func) -> None:
    setattr(_EagerTensorBase, name, func)

  def _copy_nograd(
      self: EagerTensorType, ctx=None, device_name=None,
  ) -> EagerTensorType:
    """Copies tensor to dest device, but doesn't record the operation."""
    # Creates a new tensor on the dest device.
    if ctx is None:
      ctx = context.context()
    if device_name is None:
      device_name = ctx.device_name
    # pylint: disable=protected-access
    try:
      ctx.ensure_initialized()
      new_tensor = self._copy_to_device(device_name)
    except core._NotOkStatusException as e:
      raise core._status_to_exception(e) from None
    return new_tensor

  def _copy(
      self: EagerTensorType, ctx=None, device_name=None,
  ) -> EagerTensorType:
    """Copies tensor to dest device."""
    new_tensor = self._copy_nograd(ctx, device_name)
    # Record the copy on tape and define backprop copy as well.
    if context.executing_eagerly():
      self_device = self.device

      def grad_fun(dresult):
        return [
            dresult._copy(device_name=self_device)
            if hasattr(dresult, "_copy") else dresult
        ]

      record.record_operation("_copy", [new_tensor], [self], grad_fun)
    return new_tensor
    # pylint: enable=protected-access

  @property
  def shape(self) -> tensor_shape.TensorShape:
    if self._tensor_shape is None:  # pylint: disable=access-member-before-definition
      # pylint: disable=protected-access
      try:
        # `_tensor_shape` is declared and defined in the definition of
        # `EagerTensor`, in C.
        self._tensor_shape = tensor_shape.TensorShape(self._shape_tuple())
      except core._NotOkStatusException as e:
        raise core._status_to_exception(e) from None

    return self._tensor_shape

  def get_shape(self) -> tensor_shape.TensorShape:
    """Alias of Tensor.shape."""
    return self.shape

  def _shape_as_list(self) -> list[int]:
    """The shape of the tensor as a list."""
    return list(self._shape_tuple())

  @deprecation.deprecated(
      None, "Use tf.identity with explicit device placement instead.")
  def cpu(self: EagerTensorType) -> EagerTensorType:
    """A copy of this Tensor with contents backed by host memory."""
    return self._copy(context.context(), "CPU:0")

  @deprecation.deprecated(None, "Use tf.identity instead.")
  def gpu(self: EagerTensorType, gpu_index=0) -> EagerTensorType:
    """A copy of this Tensor with contents backed by memory on the GPU.

    Args:
      gpu_index: Identifies which GPU to place the contents on the returned
        Tensor in.

    Returns:
      A GPU-memory backed Tensor object initialized with the same contents
      as this Tensor.
    """
    return self._copy(context.context(), "GPU:" + str(gpu_index))

  def set_shape(self, shape) -> None:
    # pylint: disable=protected-access
    shape = tensor_shape.as_shape(shape)
    shape_dims = shape._dims
    if shape_dims is None:
      return
    self_dims = self.shape._dims
    if len(shape_dims) != len(self_dims):
      raise ValueError(f"Tensor's shape {self.shape} is not compatible "
                       f"with supplied shape {shape}.")
    for shape_dim, self_dim in zip(shape_dims, self_dims):
      if shape_dim is not None and self_dim != shape_dim:
        raise ValueError(f"Tensor's shape {self.shape} is not compatible "
                         f"with supplied shape {shape}.")
    # pylint: enable=protected-access

  # Methods not supported / implemented for Eager Tensors.
  @property
  def op(self) -> NoReturn:
    raise AttributeError(
        "Tensor.op is undefined when eager execution is enabled.")

  @property
  def graph(self) -> NoReturn:
    raise AttributeError(
        "Tensor.graph is undefined when eager execution is enabled.")

  @property
  def name(self) -> NoReturn:
    raise AttributeError(
        "Tensor.name is undefined when eager execution is enabled.")

  @property
  def value_index(self) -> NoReturn:
    raise AttributeError(
        "Tensor.value_index is undefined when eager execution is enabled.")

  def consumers(self) -> NoReturn:
    raise NotImplementedError(
        "Tensor.consumers is undefined when eager execution is enabled.")

  def _add_consumer(self, consumer) -> NoReturn:
    raise NotImplementedError(
        "_add_consumer not supported when eager execution is enabled.")

  def _as_node_def_input(self) -> NoReturn:
    raise NotImplementedError(
        "_as_node_def_input not supported when eager execution is enabled.")

  def _as_tf_output(self) -> NoReturn:
    raise NotImplementedError(
        "_as_tf_output not supported when eager execution is enabled.")

  def eval(self, feed_dict=None, session=None) -> NoReturn:
    raise NotImplementedError(
        "eval is not supported when eager execution is enabled, "
        "is .numpy() what you're looking for?")

  def __tf_tensor__(
      self, dtype: Optional[dtypes.DType] = None, name: Optional[str] = None
      ) -> tensor_lib.Tensor:
    if not context.executing_eagerly():
      graph = get_default_graph()
      if not graph.building_function:
        raise RuntimeError(
            _add_error_prefix(
                "Attempting to capture an EagerTensor without "
                "building a function.",
                name=name))
      return graph.capture(self, name=name)
    return super().__tf_tensor__(dtype, name)

  def _capture_as_const(self, name) -> Optional[tensor_lib.Tensor]:
    """Capture the EagerTensor to a graph constant tensor."""
    with control_dependencies(None):
      constant_value = tensor_util.constant_value(self)
      if constant_value is None:
        # Some eager tensors, e.g. parallel tensors, are not convertible to
        # a single constant. Return None in this case and the caller graph
        # would create a placeholder instead.
        return None

      const_tensor = _create_graph_constant(
          constant_value, dtype=self.dtype, shape=self.shape, name=name,
          verify_shape=False, allow_broadcast=True)
    return const_tensor


# This call creates an EagerTensor class, as a subclass of _EagerTensorBase, and
# registers it with the current module.
# It is exposed as an __internal__ api for now (b/171081052), though we
# expect it to be eventually covered by tf Tensor types and typing.
EagerTensor = tf_export("__internal__.EagerTensor", v1=[])(
    pywrap_tfe.TFE_Py_InitEagerTensor(_EagerTensorBase))


def _add_error_prefix(msg: str, *, name: Optional[str] = None) -> str:
  return msg if name is None else f"{name}: {msg}"


def pack_eager_tensors(tensors, ctx=None) -> EagerTensor:
  """Pack multiple `EagerTensor`s of the same dtype and shape.

  Args:
    tensors: a list of EagerTensors to pack.
    ctx: context.context().

  Returns:
    A packed EagerTensor.
  """
  if not isinstance(tensors, list):
    raise TypeError(f"tensors must be a list, but got a {type(tensors)}")

  if not tensors:
    raise ValueError("Cannot pack an empty list of tensors.")

  dtype = tensors[0].dtype
  shape = tensors[0].shape
  handle_data = tensors[0]._handle_data  # pylint: disable=protected-access
  is_resource = dtype == dtypes.resource
  for i in range(len(tensors)):
    t = tensors[i]
    if not isinstance(t, EagerTensor):
      raise TypeError(f"All tensors being packed must be EagerTensor. "
                      f"Found an item of type {type(t)}.")

    if t.dtype != dtype:
      raise ValueError(
          f"All tensors being packed should have the same dtype {dtype}, "
          f"but the {i}-th tensor is of dtype {t.dtype}")
    if t.shape != shape:
      raise ValueError(
          f"All tensors being packed should have the same shape {shape}, "
          f"but the {i}-th tensor is of shape {t.shape}")
    # pylint: disable=protected-access
    if is_resource and t._handle_data != handle_data:
      raise ValueError(
          f"All tensors being packed should have the same handle data "
          f"{handle_data}, "
          f"but the {i}-th tensor is of handle data {t._handle_data}")
    # pylint: enable=protected-access

  if ctx is None:
    ctx = context.context()

  # Propagate handle data for resource variables
  packed_tensor = ctx.pack_eager_tensors(tensors)
  if handle_data is not None:
    packed_tensor._handle_data = handle_data  # pylint: disable=protected-access

  def grad_fun(_):
    raise ValueError(
        "Computing gradients through pack_eager_tensors is not supported.")

  record.record_operation("pack_eager_tensors", [packed_tensor], tensors,
                          grad_fun)

  return packed_tensor


@profiler_trace.trace_wrapper("convert_to_tensor")
def convert_to_tensor(
    value,
    dtype=None,
    name=None,
    as_ref=False,
    preferred_dtype=None,
    dtype_hint=None,
    # TODO(b/268347915): Remove argument.
    ctx=None,  # pylint: disable=unused-argument
    accepted_result_types=(tensor_lib.Tensor,),
) -> Union[EagerTensor, SymbolicTensor]:
  """Implementation of the public convert_to_tensor."""
  # TODO(b/142518781): Fix all call-sites and remove redundant arg
  preferred_dtype = preferred_dtype or dtype_hint
  return tensor_conversion_registry.convert(
      value, dtype, name, as_ref, preferred_dtype, accepted_result_types
  )


internal_convert_to_tensor: Callable[
    ..., Union[EagerTensor, SymbolicTensor]] = convert_to_tensor


def internal_convert_n_to_tensor(
    values,
    dtype=None,
    name=None,
    as_ref=False,
    preferred_dtype=None,
    # TODO(b/268347915): Remove argument.
    ctx=None) -> list[Union[EagerTensor, SymbolicTensor]]:  # pylint: disable=unused-argument
  """Converts `values` to a list of `Tensor` objects.

  Args:
    values: A list of objects that can be consumed by `tf.convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` objects.
    name: (Optional.) A name prefix to used when a new `Tensor` is created, in
      which case element `i` will be given the name `name + '_' + i`.
    as_ref: True if the caller wants the results as ref tensors.
    preferred_dtype: Optional element type for the returned tensors, used when
      dtype is None. In some cases, a caller may not have a dtype in mind when
      converting to a tensor, so preferred_dtype can be used as a soft
      preference.  If the conversion to `preferred_dtype` is not possible, this
      argument has no effect.
    ctx: Unused. Present for API backwards compatibility.

  Returns:
    A list of `Tensor` and/or `IndexedSlices` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  if not isinstance(values, collections_abc.Sequence):
    raise TypeError("values must be a sequence.")
  ret = []
  for i, value in enumerate(values):
    n = None if name is None else "%s_%d" % (name, i)
    ret.append(
        convert_to_tensor(
            value,
            dtype=dtype,
            name=n,
            as_ref=as_ref,
            preferred_dtype=preferred_dtype))
  return ret


def convert_n_to_tensor(
    values, dtype=None, name=None, preferred_dtype=None
) ->  list[Union[EagerTensor, SymbolicTensor]]:
  """Converts `values` to a list of `Tensor` objects.

  Args:
    values: A list of objects that can be consumed by `tf.convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` objects.
    name: (Optional.) A name prefix to used when a new `Tensor` is created, in
      which case element `i` will be given the name `name + '_' + i`.
    preferred_dtype: Optional element type for the returned tensors, used when
      dtype is None. In some cases, a caller may not have a dtype in mind when
      converting to a tensor, so preferred_dtype can be used as a soft
      preference.  If the conversion to `preferred_dtype` is not possible, this
      argument has no effect.

  Returns:
    A list of `Tensor` and/or `IndexedSlices` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  return internal_convert_n_to_tensor(
      values=values,
      dtype=dtype,
      name=name,
      preferred_dtype=preferred_dtype,
      as_ref=False)


def convert_to_tensor_or_composite(
    value, dtype=None, name=None
) -> Union[EagerTensor, SymbolicTensor, composite_tensor.CompositeTensor]:
  """Converts the given object to a `Tensor` or `CompositeTensor`.

  If `value` is a `CompositeTensor` it is returned unmodified. Otherwise, it
  is converted to a `Tensor` using `convert_to_tensor()`.

  Args:
    value: A `CompositeTensor` or an object that can be consumed by
      `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `CompositeTensor`.
    name: (Optional.) A name to use if a new `Tensor` is created.

  Returns:
    A `Tensor` or `CompositeTensor`, based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  return internal_convert_to_tensor_or_composite(
      value=value, dtype=dtype, name=name, as_ref=False)


def internal_convert_to_tensor_or_composite(
    value, dtype=None,
    name=None,
    as_ref=False
) -> Union[EagerTensor, SymbolicTensor, composite_tensor.CompositeTensor]:
  """Converts the given object to a `Tensor` or `CompositeTensor`.

  If `value` is a `CompositeTensor` it is returned unmodified.  Otherwise, it
  is converted to a `Tensor` using `convert_to_tensor()`.

  Args:
    value: A `CompositeTensor`, or an object that can be consumed by
      `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `CompositeTensor`.
    name: (Optional.) A name to use if a new `Tensor` is created.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A `Tensor` or `CompositeTensor`, based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  if isinstance(value, composite_tensor.CompositeTensor):
    value_dtype = getattr(value, "dtype", None)
    if dtype and not dtypes.as_dtype(dtype).is_compatible_with(value_dtype):
      raise ValueError(f"Tensor conversion dtype mismatch. "
                       f"Requested dtype is {dtypes.as_dtype(dtype).name}, "
                       f"Tensor has dtype {value.dtype.name}: {value!r}")
    return value
  else:
    return convert_to_tensor(
        value,
        dtype=dtype,
        name=name,
        as_ref=as_ref,
        accepted_result_types=(
            tensor_lib.Tensor, composite_tensor.CompositeTensor))


def internal_convert_n_to_tensor_or_composite(
    values,
    dtype=None,
    name=None,
    as_ref=False
) -> list[Union[
    EagerTensor, SymbolicTensor, composite_tensor.CompositeTensor, type(None)]]:
  """Converts `values` to a list of `Tensor` or `CompositeTensor` objects.

  Any `CompositeTensor` objects in `values` are returned unmodified.

  Args:
    values: A list of `None`, `CompositeTensor`, or objects that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor`s or
      `CompositeTensor`s.
    name: (Optional.) A name prefix to used when a new `Tensor` is created, in
      which case element `i` will be given the name `name + '_' + i`.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A list of `Tensor`, `CompositeTensor`, and/or `None` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  if not isinstance(values, collections_abc.Sequence):
    raise TypeError("values must be a sequence.")
  ret = []
  for i, value in enumerate(values):
    if value is None:
      ret.append(value)
    else:
      n = None if name is None else "%s_%d" % (name, i)
      ret.append(
          internal_convert_to_tensor_or_composite(
              value, dtype=dtype, name=n, as_ref=as_ref))
  return ret


def convert_n_to_tensor_or_composite(
    values, dtype=None, name=None
) -> list[Union[
    EagerTensor, SymbolicTensor, composite_tensor.CompositeTensor, type(None)]]:
  """Converts `values` to a list of `Output` or `CompositeTensor` objects.

  Any `CompositeTensor` objects in `values` are returned unmodified.

  Args:
    values: A list of `None`, `CompositeTensor``, or objects that can be
      consumed by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor`s or
      `CompositeTensor`s.
    name: (Optional.) A name prefix to used when a new `Tensor` is created, in
      which case element `i` will be given the name `name + '_' + i`.

  Returns:
    A list of `Tensor` and/or `CompositeTensor` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  return internal_convert_n_to_tensor_or_composite(
      values=values, dtype=dtype, name=name, as_ref=False)


def _device_string(dev_spec) -> str:
  if pydev.is_device_spec(dev_spec):
    return dev_spec.to_string()
  else:
    return dev_spec


def _NodeDef(op_type, name, attrs=None) -> node_def_pb2.NodeDef:
  """Create a NodeDef proto.

  Args:
    op_type: Value for the "op" attribute of the NodeDef proto.
    name: Value for the "name" attribute of the NodeDef proto.
    attrs: Dictionary where the key is the attribute name (a string)
      and the value is the respective "attr" attribute of the NodeDef proto (an
      AttrValue).

  Returns:
    A node_def_pb2.NodeDef protocol buffer.
  """
  node_def = node_def_pb2.NodeDef(op=compat.as_bytes(op_type),
                                  name=compat.as_bytes(name))
  if attrs:
    for k, v in attrs.items():
      node_def.attr[k].CopyFrom(v)
  return node_def


# Copied from core/framework/node_def_util.cc
# TODO(mrry,josh11b): Consolidate this validation in C++ code.
_VALID_OP_NAME_REGEX: Pattern[str] = re.compile(
    r"^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$")
_VALID_SCOPE_NAME_REGEX: Pattern[str] = re.compile(
    r"^[A-Za-z0-9_.\\/>-]*$")


@tf_export("__internal__.create_c_op", v1=[])
@traceback_utils.filter_traceback
def _create_c_op(graph,
                 node_def,
                 inputs,
                 control_inputs,
                 op_def=None,
                 extract_traceback=True) -> pywrap_tf_session.TF_Operation:
  """Creates a TF_Operation.

  Args:
    graph: a `Graph`.
    node_def: `node_def_pb2.NodeDef` for the operation to create.
    inputs: A flattened list of `Tensor`s. This function handles grouping
      tensors into lists as per attributes in the `node_def`.
    control_inputs: A list of `Operation`s to set as control dependencies.
    op_def: Optional. `op_def_pb2.OpDef` for the operation to create. If not
      specified, is looked up from the `graph` using `node_def.op`.
    extract_traceback: if True, extract the current Python traceback to the
      TF_Operation.

  Returns:
    A wrapped TF_Operation*.
  """
  if op_def is None:
    op_def = graph.op_def_for_type(node_def.op)  # pylint: disable=protected-access
  # TODO(skyewm): op_def_library.apply_op() flattens the incoming inputs.
  # Refactor so we don't have to do this here.
  inputs = _reconstruct_sequence_inputs(op_def, inputs, node_def.attr)
  # pylint: disable=protected-access
  with graph._c_graph.get() as c_graph:
    op_desc = pywrap_tf_session.TF_NewOperation(c_graph,
                                                compat.as_str(node_def.op),
                                                compat.as_str(node_def.name))
  if node_def.device:
    pywrap_tf_session.TF_SetDevice(op_desc, compat.as_str(node_def.device))
  # Add inputs
  for op_input in inputs:
    if isinstance(op_input, (list, tuple)):
      pywrap_tf_session.TF_AddInputList(op_desc,
                                        [t._as_tf_output() for t in op_input])
    else:
      pywrap_tf_session.TF_AddInput(op_desc, op_input._as_tf_output())

  # Add control inputs
  for control_input in control_inputs:
    pywrap_tf_session.TF_AddControlInput(op_desc, control_input._c_op)
  # pylint: enable=protected-access

  # Add attrs
  for name, attr_value in node_def.attr.items():
    serialized = attr_value.SerializeToString()
    # TODO(skyewm): this creates and deletes a new TF_Status for every attr.
    # It might be worth creating a convenient way to re-use the same status.
    pywrap_tf_session.TF_SetAttrValueProto(op_desc, compat.as_str(name),
                                           serialized)

  try:
    c_op = pywrap_tf_session.TF_FinishOperation(op_desc)
  except errors.InvalidArgumentError as e:
    # Convert to ValueError for backwards compatibility.
    raise ValueError(e.message)

  # Record the current Python stack trace as the creating stacktrace of this
  # TF_Operation.
  if extract_traceback:
    pywrap_tf_session.TF_SetOpStackTrace(
        c_op, tf_stack.extract_stack(stacklevel=3)
    )

  return c_op


@tf_export("Operation")
class Operation(pywrap_tf_session.PyOperation):
  """Represents a graph node that performs computation on tensors.

  An `Operation` is a node in a `tf.Graph` that takes zero or more `Tensor`
  objects as input, and produces zero or more `Tensor` objects as output.
  Objects of type `Operation` are created by calling a Python op constructor
  (such as `tf.matmul`) within a `tf.function` or under a `tf.Graph.as_default`
  context manager.

  For example, within a `tf.function`, `c = tf.matmul(a, b)` creates an
  `Operation` of type "MatMul" that takes tensors `a` and `b` as input, and
  produces `c` as output.

  If a `tf.compat.v1.Session` is used, an `Operation` of a `tf.Graph` can be
  executed by passing it to `tf.Session.run`. `op.run()` is a shortcut for
  calling `tf.compat.v1.get_default_session().run(op)`.
  """

  @classmethod
  def from_node_def(
      cls: type[OperationType],
      node_def,
      g,
      inputs=None,
      output_types=None,
      control_inputs=None,
      input_types=None,
      original_op=None,
      op_def=None,
  ) -> OperationType:
    r"""Creates an `Operation`.

    NOTE: This constructor validates the name of the `Operation` (passed
    as `node_def.name`). Valid `Operation` names match the following
    regular expression:

        [A-Za-z0-9.][A-Za-z0-9_.\\-/]*

    Args:
      node_def: `node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`. Used for
        attributes of `node_def_pb2.NodeDef`, typically `name`, `op`, and
        `device`.  The `input` attribute is irrelevant here as it will be
        computed when generating the model.
      g: `Graph`. The parent graph.
      inputs: list of `Tensor` objects. The inputs to this `Operation`.
      output_types: list of `DType` objects.  List of the types of the `Tensors`
        computed by this operation.  The length of this list indicates the
        number of output endpoints of the `Operation`.
      control_inputs: list of operations or tensors from which to have a control
        dependency.
      input_types: List of `DType` objects representing the types of the tensors
        accepted by the `Operation`.  By default uses `[x.dtype.base_dtype for x
        in inputs]`.  Operations that expect reference-typed inputs must specify
        these explicitly.
      original_op: Optional. Used to associate the new `Operation` with an
        existing `Operation` (for example, a replica with the op that was
        replicated).
      op_def: Optional. The `op_def_pb2.OpDef` proto that describes the op type
        that this `Operation` represents.

    Raises:
      TypeError: if control inputs are not Operations or Tensors,
        or if `node_def` is not a `NodeDef`,
        or if `g` is not a `Graph`,
        or if `inputs` are not tensors,
        or if `inputs` and `input_types` are incompatible.
      ValueError: if the `node_def` name is not valid.

    Returns:
      Operation object.
    """
    if not isinstance(g, Graph):
      raise TypeError(f"Argument g must be a Graph. "
                      f"Received an instance of type {type(g)}")

    if not isinstance(node_def, node_def_pb2.NodeDef):
      raise TypeError(f"Argument node_def must be a NodeDef. "
                      f"Received an instance of type: {type(node_def)}.")
    if node_def.ByteSize() >= (1 << 31) or node_def.ByteSize() < 0:
      raise ValueError(
          f"Cannot create a tensor proto whose content is larger than 2GB. "
          f"Size of tensor is {node_def.ByteSize()} bytes.")

    # TODO(mdan): This does not belong here. Graph::AddNode should handle it.
    if not _VALID_OP_NAME_REGEX.match(node_def.name):
      raise ValueError(
          f"`{node_def.name}` is not a valid node name. "
          f"Accepted names conform to Regex /{_VALID_OP_NAME_REGEX}/")

    # FIXME(b/225400189): output_types is unused. Consider remove it from
    # the argument list.
    del output_types

    if inputs is None:
      inputs = []
    elif not isinstance(inputs, list):
      raise TypeError(f"Argument inputs shall be a list of Tensors. "
                      f"Received an instance of type {type(inputs)}")
    for a in inputs:
      if not isinstance(a, tensor_lib.Tensor):
        raise TypeError(f"Items of argument inputs shall be Tensor. "
                        f"Received an instance of type {type(a)}.")
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in inputs]
    else:
      if not all(
          x.is_compatible_with(i.dtype) for i, x in zip(inputs, input_types)):
        raise TypeError("In op '%s', input types (%s) are not compatible "
                        "with expected types (%s)" %
                        (node_def.name, [i.dtype for i in inputs], input_types))

    # Build the list of control inputs.
    control_input_ops = []
    if control_inputs:
      for c in control_inputs:
        control_op = None
        if isinstance(c, Operation):
          control_op = c
        elif isinstance(c, (tensor_lib.Tensor, internal.IndexedSlices)):
          control_op = c.op
        else:
          raise TypeError(f"Control input must be an Operation, "
                          f"a Tensor, or IndexedSlices. "
                          f"Received an instance of type {type(c)}.")
        control_input_ops.append(control_op)

    # Initialize c_op from node_def and other inputs
    c_op = _create_c_op(g, node_def, inputs, control_input_ops, op_def=op_def)
    self = Operation(c_op, SymbolicTensor)
    self._init(g)

    self._original_op = original_op

    # Post process for control flows.
    self._control_flow_post_processing(input_tensors=inputs)

    return self

  @classmethod
  def _from_c_op(cls: type[OperationType], c_op, g) -> OperationType:
    """Create an Operation from a TF_Operation.

    For internal use only: This is useful for creating Operation for ops
    indirectly created by C API methods, e.g. the ops created by
    TF_ImportGraphDef.

    Args:
      c_op: a TF_Operation.
      g: A Graph.

    Returns:
      an Operation object.
    """
    self = Operation(c_op, SymbolicTensor)
    self._init(g)
    return self

  def _init(self, graph: "Graph") -> None:
    """Initializes Operation from a TF_Operation."""
    self.graph = graph
    self._original_op = None

    # This will be set by self.inputs.
    self._inputs_val = None

    # List of _UserDevSpecs holding code location of device context manager
    # invocations and the users original argument to them.
    self._device_code_locations = None
    # Dict mapping op name to file and line information for op colocation
    # context managers.
    self._colocation_code_locations = None
    self._control_flow_context = self.graph._get_control_flow_context()  # pylint: disable=protected-access

    # Gradient function for this op. There are three ways to specify gradient
    # function, and first available gradient gets used, in the following order.
    # 1. self._gradient_function
    # 2. Gradient name registered by "_gradient_op_type" attribute.
    # 3. Gradient name registered by op.type.
    self._gradient_function = None

    self._init_outputs()
    self._id_value = self.graph._add_op(self)  # pylint: disable=protected-access

  def _control_flow_post_processing(self, input_tensors=None) -> None:
    """Add this op to its control flow context.

    This may add new ops and change this op's inputs. self.inputs must be
    available before calling this method.

    Args:
      input_tensors: (Optional.) A list of `Tensors` corresponding to the inputs
        of this op, which should be equivalent to `self.inputs`. Pass this
        argument to avoid evaluating `self.inputs` unnecessarily.
    """
    if input_tensors is None:
      input_tensors = self.inputs
    for input_tensor in input_tensors:
      control_flow_util.CheckInputFromValidContext(self, input_tensor.op)
    if self._control_flow_context is not None:
      self._control_flow_context.AddOp(self)

  def colocation_groups(self) -> list[bytes]:
    """Returns the list of colocation groups of the op."""
    default_colocation_group = [compat.as_bytes("loc:@%s" % self.name)]
    try:
      class_attr = self.get_attr("_class")
    except ValueError:
      # This op has no explicit colocation group, so it is itself its
      # own root of a colocation group.
      return default_colocation_group

    attr_groups = [
        class_name for class_name in class_attr
        if class_name.startswith(b"loc:@")
    ]

    # If there are no colocation groups in the explicit _class field,
    # return the default colocation group.
    return attr_groups if attr_groups else default_colocation_group

  def values(self) -> tuple[Any, ...]:
    """DEPRECATED: Use outputs."""
    return tuple(self.outputs)

  def _get_control_flow_context(self):
    """Returns the control flow context of this op.

    Returns:
      A context object.
    """
    return self._control_flow_context

  def _set_control_flow_context(self, ctx) -> None:
    """Sets the current control flow context of this op.

    Args:
      ctx: a context object.
    """
    self._control_flow_context = ctx

  @property
  def _id(self) -> int:
    """The unique integer id of this operation."""
    return self._id_value

  @property
  def device(self) -> str:
    """The name of the device to which this op has been assigned, if any.

    Returns:
      The string name of the device to which this op has been
      assigned, or an empty string if it has not been assigned to a
      device.
    """
    return pywrap_tf_session.TF_OperationDevice(self._c_op)

  @property
  def _device_assignments(self) -> list[traceable_stack.TraceableObject]:
    """Code locations for device context managers active at op creation.

    This property will return a list of traceable_stack.TraceableObject
    instances where .obj is a string representing the assigned device
    (or information about the function that would be applied to this op
    to compute the desired device) and the filename and lineno members
    record the location of the relevant device context manager.

    For example, suppose file_a contained these lines:

      file_a.py:
        15: with tf.device('/gpu:0'):
        16:   node_b = tf.constant(4, name='NODE_B')

    Then a TraceableObject t_obj representing the device context manager
    would have these member values:

      t_obj.obj -> '/gpu:0'
      t_obj.filename = 'file_a.py'
      t_obj.lineno = 15

    and node_b.op._device_assignments would return the list [t_obj].

    Returns:
      [str: traceable_stack.TraceableObject, ...] as per this method's
      description, above.
    """
    return self._device_code_locations or []

  @property
  def _colocation_dict(self) -> dict[str, traceable_stack.TraceableObject]:
    """Code locations for colocation context managers active at op creation.

    This property will return a dictionary for which the keys are nodes with
    which this Operation is colocated, and for which the values are
    traceable_stack.TraceableObject instances.  The TraceableObject instances
    record the location of the relevant colocation context manager but have the
    "obj" field set to None to prevent leaking private data.

    For example, suppose file_a contained these lines:

      file_a.py:
        14: node_a = tf.constant(3, name='NODE_A')
        15: with tf.compat.v1.colocate_with(node_a):
        16:   node_b = tf.constant(4, name='NODE_B')

    Then a TraceableObject t_obj representing the colocation context manager
    would have these member values:

      t_obj.obj -> None
      t_obj.filename = 'file_a.py'
      t_obj.lineno = 15

    and node_b.op._colocation_dict would return the dictionary

      { 'NODE_A': t_obj }

    Returns:
      {str: traceable_stack.TraceableObject} as per this method's description,
      above.
    """
    locations_dict = self._colocation_code_locations or {}
    return locations_dict.copy()

  @property
  def _output_types(self) -> list[int]:
    """List this operation's output types.

    Returns:
      List of the types of the Tensors computed by this operation.
      Each element in the list is an integer whose value is one of
      the TF_DataType enums defined in pywrap_tf_session.h
      The length of this list indicates the number of output endpoints
      of the operation.
    """
    num_outputs = pywrap_tf_session.TF_OperationNumOutputs(self._c_op)
    output_types = [
        int(pywrap_tf_session.TF_OperationOutputType(self._tf_output(i)))
        for i in range(num_outputs)
    ]

    return output_types

  def _set_device(self, device) -> None:  # pylint: disable=redefined-outer-name
    """Set the device of this operation.

    Args:
      device: string or device..  The device to set.
    """
    self._set_device_from_string(compat.as_str(_device_string(device)))

  def _update_input(self, index, tensor) -> None:
    """Update the input to this operation at the given index.

    NOTE: This is for TF internal use only. Please don't use it.

    Args:
      index: the index of the input to update.
      tensor: the Tensor to be used as the input at the given index.

    Raises:
      TypeError: if tensor is not a Tensor,
        or if input tensor type is not convertible to dtype.
      ValueError: if the Tensor is from a different graph.
    """
    if not isinstance(tensor, tensor_lib.Tensor):
      raise TypeError("tensor must be a Tensor: %s" % tensor)

    _assert_same_graph(self, tensor)

    # Reset cached inputs.
    self._inputs_val = None
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      pywrap_tf_session.UpdateEdge(
          c_graph,
          tensor._as_tf_output(),  # pylint: disable=protected-access
          self._tf_input(index))

  def _add_while_inputs(self, tensors) -> None:
    """See AddWhileInputHack in python_api.h.

    NOTE: This is for TF internal use only. Please don't use it.

    Args:
      tensors: list of Tensors

    Raises:
      TypeError: if tensor is not a Tensor,
        or if input tensor type is not convertible to dtype.
      ValueError: if the Tensor is from a different graph.
    """
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      for tensor in tensors:
        if not isinstance(tensor, tensor_lib.Tensor):
          raise TypeError("tensor must be a Tensor: %s" % tensor)
        _assert_same_graph(self, tensor)

        # Reset cached inputs.
        self._inputs_val = None
        pywrap_tf_session.AddWhileInputHack(
            c_graph,  # pylint: disable=protected-access
            tensor._as_tf_output(),  # pylint: disable=protected-access
            self._c_op)

  def __str__(self) -> str:
    return str(self.node_def)

  def __repr__(self) -> str:
    return "<tf.Operation '%s' type=%s>" % (self.name, self.type)

  def __tf_tensor__(self, dtype=None, name=None) -> NoReturn:
    """Raises a helpful error."""
    raise TypeError("can't convert Operation '{}' to Tensor".format(self.name))

  @property
  def inputs(self) -> Sequence[tensor_lib.Tensor]:
    """The sequence of `Tensor` objects representing the data inputs of this op."""
    if self._inputs_val is None:
      # pylint: disable=protected-access
      self._inputs_val = tuple(
          self.graph._get_tensor_by_tf_output(i)
          for i in pywrap_tf_session.GetOperationInputs(self._c_op))
      # pylint: enable=protected-access
    return self._inputs_val

  @property
  def _input_types(self) -> list[dtypes.DType]:
    num_inputs = pywrap_tf_session.TF_OperationNumInputs(self._c_op)
    input_types = [
        dtypes.as_dtype(
            pywrap_tf_session.TF_OperationInputType(self._tf_input(i)))
        for i in range(num_inputs)
    ]
    return input_types

  @property
  def traceback(self):
    """Returns the call stack from when this operation was constructed."""
    # FIXME(b/225423591): This object contains a dangling reference if _c_op
    # goes out of scope.
    return pywrap_tf_session.TF_OperationGetStackTrace(self._c_op)

  @property
  def node_def(self) -> node_def_pb2.NodeDef:
    return node_def_pb2.NodeDef.FromString(self._node_def)

  @property
  def op_def(self) -> op_def_pb2.OpDef:
    return op_def_pb2.OpDef.FromString(self._op_def)

  def _set_attr(self, attr_name, attr_value) -> None:
    """Private method used to set an attribute in the node_def."""
    buf = pywrap_tf_session.TF_NewBufferFromString(
        compat.as_bytes(attr_value.SerializeToString()))
    try:
      self._set_attr_with_buf(attr_name, buf)
    finally:
      pywrap_tf_session.TF_DeleteBuffer(buf)

  def _set_attr_with_buf(self, attr_name, attr_buf) -> None:
    """Set an attr in the node_def with a pre-allocated buffer."""
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      # pylint: disable=protected-access
      pywrap_tf_session.SetAttr(c_graph, self._c_op, attr_name, attr_buf)
      # pylint: enable=protected-access

  def _set_func_attr(self, attr_name, func_name) -> None:
    """Private method used to set a function attribute in the node_def."""
    func = attr_value_pb2.NameAttrList(name=func_name)
    self._set_attr(attr_name, attr_value_pb2.AttrValue(func=func))

  def _set_func_list_attr(self, attr_name, func_names) -> None:
    """Private method used to set a list(function) attribute in the node_def."""
    funcs = [attr_value_pb2.NameAttrList(name=func_name)
             for func_name in func_names]
    funcs_list = attr_value_pb2.AttrValue.ListValue(func=funcs)
    self._set_attr(attr_name, attr_value_pb2.AttrValue(list=funcs_list))

  def _set_type_list_attr(self, attr_name, data_types) -> None:
    """Private method used to set a list(type) attribute in the node_def."""
    if not data_types:
      return
    if isinstance(data_types[0], dtypes.DType):
      data_types = [dt.as_datatype_enum for dt in data_types]
    types_list = attr_value_pb2.AttrValue.ListValue(type=data_types)
    self._set_attr(attr_name, attr_value_pb2.AttrValue(list=types_list))

  def _set_shape_list_attr(self, attr_name, shapes) -> None:
    """Private method used to set a list(shape) attribute in the node_def."""
    shapes = [s.as_proto() for s in shapes]
    shapes_list = attr_value_pb2.AttrValue.ListValue(shape=shapes)
    self._set_attr(attr_name, attr_value_pb2.AttrValue(list=shapes_list))

  def _clear_attr(self, attr_name) -> None:
    """Private method used to clear an attribute in the node_def."""
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      # pylint: disable=protected-access
      pywrap_tf_session.ClearAttr(c_graph, self._c_op, attr_name)
      # pylint: enable=protected-access

  def get_attr(self, name):
    """Returns the value of the attr of this op with the given `name`.

    Args:
      name: The name of the attr to fetch.

    Returns:
      The value of the attr, as a Python object.

    Raises:
      ValueError: If this op does not have an attr with the given `name`.
    """
    fields = ("s", "i", "f", "b", "type", "shape", "tensor", "func")
    try:
      with c_api_util.tf_buffer() as buf:   # pytype: disable=wrong-arg-count
        pywrap_tf_session.TF_OperationGetAttrValueProto(self._c_op, name, buf)
        data = pywrap_tf_session.TF_GetBuffer(buf)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(e.message)
    x = attr_value_pb2.AttrValue()
    x.ParseFromString(data)

    oneof_value = x.WhichOneof("value")
    if oneof_value is None:
      return []
    if oneof_value == "list":
      for f in fields:
        if getattr(x.list, f):
          if f == "type":
            return [dtypes.as_dtype(t) for t in x.list.type]
          else:
            return list(getattr(x.list, f))
      return []
    if oneof_value == "type":
      return dtypes.as_dtype(x.type)
    assert oneof_value in fields, "Unsupported field type in " + str(x)
    return getattr(x, oneof_value)

  def _get_attr_type(self, name) -> dtypes.DType:
    """Returns the `DType` value of the attr of this op with the given `name`."""
    try:
      dtype_enum = pywrap_tf_session.TF_OperationGetAttrType(self._c_op, name)
      return _DTYPES_INTERN_TABLE[dtype_enum]
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(e.message)

  def _get_attr_bool(self, name) -> bool:
    """Returns the `bool` value of the attr of this op with the given `name`."""
    try:
      return pywrap_tf_session.TF_OperationGetAttrBool(self._c_op, name)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(e.message)

  def _get_attr_int(self, name) -> int:
    """Returns the `int` value of the attr of this op with the given `name`."""
    try:
      return pywrap_tf_session.TF_OperationGetAttrInt(self._c_op, name)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(e.message)

  def experimental_set_type(self, type_proto) -> None:
    """Sets the corresponding node's `experimental_type` field.

    See the description of `NodeDef.experimental_type` for more info.

    Args:
      type_proto: A FullTypeDef proto message. The root type_if of this object
        must be `TFT_PRODUCT`, even for ops which only have a singlre return
        value.
    """
    with self.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
      if (type_proto.type_id
          not in (full_type_pb2.TFT_UNSET, full_type_pb2.TFT_PRODUCT)):
        raise ValueError("error setting the type of ", self.name,
                         ": expected TFT_UNSET or TFT_PRODUCT, got ",
                         type_proto.type_id)
      with c_api_util.tf_buffer(type_proto.SerializeToString()) as serialized:
        pywrap_tf_session.SetFullType(c_graph, self._c_op, serialized)  # pylint:disable=protected-access

  def run(self, feed_dict=None, session=None) -> None:
    """Runs this operation in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for this operation.

    *N.B.* Before invoking `Operation.run()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to run to this operation. If
        none, the default session will be used.
    """
    _run_using_default_session(self, feed_dict, self.graph, session)

gradient_registry: registry.Registry
_gradient_registry: registry.Registry
# TODO(b/185395742): Clean up usages of _gradient_registry
gradient_registry = _gradient_registry = registry.Registry("gradient")


@tf_export("RegisterGradient")
class RegisterGradient(object):
  """A decorator for registering the gradient function for an op type.

  This decorator is only used when defining a new op type. For an op
  with `m` inputs and `n` outputs, the gradient function is a function
  that takes the original `Operation` and `n` `Tensor` objects
  (representing the gradients with respect to each output of the op),
  and returns `m` `Tensor` objects (representing the partial gradients
  with respect to each input of the op).

  For example, assuming that operations of type `"Sub"` take two
  inputs `x` and `y`, and return a single output `x - y`, the
  following gradient function would be registered:

  ```python
  @tf.RegisterGradient("Sub")
  def _sub_grad(unused_op, grad):
    return grad, tf.negative(grad)
  ```

  The decorator argument `op_type` is the string type of an
  operation. This corresponds to the `OpDef.name` field for the proto
  that defines the operation.
  """

  __slots__ = ["_op_type"]

  def __init__(self, op_type):
    """Creates a new decorator with `op_type` as the Operation type.

    Args:
      op_type: The string type of an operation. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.

    Raises:
      TypeError: If `op_type` is not string.
    """
    if not isinstance(op_type, str):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f: _T) -> _T:
    """Registers the function `f` as gradient function for `op_type`."""
    gradient_registry.register(f, self._op_type)
    return f


@deprecation.deprecated_endpoints("NotDifferentiable", "NoGradient")
@tf_export("no_gradient", v1=["no_gradient", "NotDifferentiable", "NoGradient"])
def no_gradient(op_type: str) -> None:
  """Specifies that ops of type `op_type` is not differentiable.

  This function should *not* be used for operations that have a
  well-defined gradient that is not yet implemented.

  This function is only used when defining a new op type. It may be
  used for ops such as `tf.size()` that are not differentiable.  For
  example:

  ```python
  tf.no_gradient("Size")
  ```

  The gradient computed for 'op_type' will then propagate zeros.

  For ops that have a well-defined gradient but are not yet implemented,
  no declaration should be made, and an error *must* be thrown if
  an attempt to request its gradient is made.

  Args:
    op_type: The string type of an operation. This corresponds to the
      `OpDef.name` field for the proto that defines the operation.

  Raises:
    TypeError: If `op_type` is not a string.

  """
  if not isinstance(op_type, str):
    raise TypeError("op_type must be a string")
  gradient_registry.register(None, op_type)


# Aliases for the old names, will be eventually removed.
NoGradient: Callable[[str], None] = no_gradient
NotDifferentiable: Callable[[str], None] = no_gradient


def get_gradient_function(op):
  """Returns the function that computes gradients for "op"."""
  if not op.inputs:
    return None

  gradient_function = op._gradient_function  # pylint: disable=protected-access
  if gradient_function:
    return gradient_function

  try:
    op_type = op.get_attr("_gradient_op_type")
  except ValueError:
    op_type = op.type
  return gradient_registry.lookup(op_type)


def set_shape_and_handle_data_for_outputs(_) -> None:
  """No op. TODO(b/74620627): Remove this."""
  pass


class OpStats(object):
  """A holder for statistics about an operator.

  This class holds information about the resource requirements for an op,
  including the size of its weight parameters on-disk and how many FLOPS it
  requires to execute forward inference.

  If you define a new operation, you can create a function that will return a
  set of information about its usage of the CPU and disk space when serialized.
  The function itself takes a Graph object that's been set up so you can call
  methods like get_tensor_by_name to help calculate the results, and a NodeDef
  argument.

  """

  __slots__ = ["_statistic_type", "_value"]

  def __init__(self, statistic_type, value=None) -> None:
    """Sets up the initial placeholders for the statistics."""
    self.statistic_type = statistic_type
    self.value = value

  @property
  def statistic_type(self):
    return self._statistic_type

  @statistic_type.setter
  def statistic_type(self, statistic_type):
    self._statistic_type = statistic_type

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value

  def __iadd__(self: OpStatsType, other: OpStatsType) -> OpStatsType:
    if other.statistic_type != self.statistic_type:
      raise ValueError("Can't add an OpStat of type %s to one of %s." %
                       (self.statistic_type, other.statistic_type))
    if self.value is None:
      self.value = other.value
    elif other.value is not None:
      self._value += other.value  # pytype: disable=attribute-error
    return self


_stats_registry: registry.Registry = registry.Registry("statistical functions")


class RegisterStatistics(object):
  """A decorator for registering the statistics function for an op type.

  This decorator can be defined for an op type so that it gives a
  report on the resources used by an instance of an operator, in the
  form of an OpStats object.

  Well-known types of statistics include these so far:

  - flops: When running a graph, the bulk of the computation happens doing
    numerical calculations like matrix multiplications. This type allows a node
    to return how many floating-point operations it takes to complete. The
    total number of FLOPs for a graph is a good guide to its expected latency.

  You can add your own statistics just by picking a new type string, registering
  functions for the ops you care about, and then calling get_stats_for_node_def.

  If a statistic for an op is registered multiple times, a KeyError will be
  raised.

  Since the statistics is counted on a per-op basis. It is not suitable for
  model parameters (capacity), which is expected to be counted only once, even
  if it is shared by multiple ops. (e.g. RNN)

  For example, you can define a new metric called doohickey for a Foo operation
  by placing this in your code:

  ```python
  @ops.RegisterStatistics("Foo", "doohickey")
  def _calc_foo_bojangles(unused_graph, unused_node_def):
    return ops.OpStats("doohickey", 20)
  ```

  Then in client code you can retrieve the value by making this call:

  ```python
  doohickey = ops.get_stats_for_node_def(graph, node_def, "doohickey")
  ```

  If the NodeDef is for an op with a registered doohickey function, you'll get
  back the calculated amount in doohickey.value, or None if it's not defined.

  """

  __slots__ = ["_op_type", "_statistic_type"]

  def __init__(self, op_type, statistic_type) -> None:
    """Saves the `op_type` as the `Operation` type."""
    if not isinstance(op_type, str):
      raise TypeError("op_type must be a string.")
    if "," in op_type:
      raise TypeError("op_type must not contain a comma.")
    self._op_type = op_type
    if not isinstance(statistic_type, str):
      raise TypeError("statistic_type must be a string.")
    if "," in statistic_type:
      raise TypeError("statistic_type must not contain a comma.")
    self._statistic_type = statistic_type

  def __call__(self, f: _T) -> _T:
    """Registers "f" as the statistics function for "op_type"."""
    _stats_registry.register(f, self._op_type + "," + self._statistic_type)
    return f


def get_stats_for_node_def(graph, node, statistic_type) -> Any:
  """Looks up the node's statistics function in the registry and calls it.

  This function takes a Graph object and a NodeDef from a GraphDef, and if
  there's an associated statistics method, calls it and returns a result. If no
  function has been registered for the particular node type, it returns an empty
  statistics object.

  Args:
    graph: A Graph object that's been set up with the node's graph.
    node: A NodeDef describing the operator.
    statistic_type: A string identifying the statistic we're interested in.

  Returns:
    An OpStats object containing information about resource usage.
  """

  try:
    stats_func = _stats_registry.lookup(node.op + "," + statistic_type)
    result = stats_func(graph, node)
  except LookupError:
    result = OpStats(statistic_type)
  return result


def name_from_scope_name(name) -> str:
  """Returns the name of an op given the name of its scope.

  Args:
    name: the name of the scope.

  Returns:
    the name of the op (equal to scope name minus any trailing slash).
  """
  return name[:-1] if (name and name[-1] == "/") else name


_MUTATION_LOCK_GROUP: int = 0
_SESSION_RUN_LOCK_GROUP: int = 1


@tf_contextlib.contextmanager
def resource_creator_scope(resource_type, resource_creator) -> Iterator[None]:
  with get_default_graph()._resource_creator_scope(resource_type,  # pylint: disable=protected-access
                                                   resource_creator):
    yield


@tf_export("Graph")
class Graph(pywrap_tf_session.PyGraph):
  """A TensorFlow computation, represented as a dataflow graph.

  Graphs are used by `tf.function`s to represent the function's computations.
  Each graph contains a set of `tf.Operation` objects, which represent units of
  computation; and `tf.Tensor` objects, which represent the units of data that
  flow between operations.

  ### Using graphs directly (deprecated)

  A `tf.Graph` can be constructed and used directly without a `tf.function`, as
  was required in TensorFlow 1, but this is deprecated and it is recommended to
  use a `tf.function` instead. If a graph is directly used, other deprecated
  TensorFlow 1 classes are also required to execute the graph, such as a
  `tf.compat.v1.Session`.

  A default graph can be registered with the `tf.Graph.as_default` context
  manager. Then, operations will be added to the graph instead of being executed
  eagerly. For example:

  ```python
  g = tf.Graph()
  with g.as_default():
    # Define operations and tensors in `g`.
    c = tf.constant(30.0)
    assert c.graph is g
  ```

  `tf.compat.v1.get_default_graph()` can be used to obtain the default graph.

  Important note: This class *is not* thread-safe for graph construction. All
  operations should be created from a single thread, or external
  synchronization must be provided. Unless otherwise specified, all methods
  are not thread-safe.

  A `Graph` instance supports an arbitrary number of "collections"
  that are identified by name. For convenience when building a large
  graph, collections can store groups of related objects: for
  example, the `tf.Variable` uses a collection (named
  `tf.GraphKeys.GLOBAL_VARIABLES`) for
  all variables that are created during the construction of a graph. The caller
  may define additional collections by specifying a new name.
  """

  def __init__(self) -> None:
    """Creates a new, empty Graph."""
    super().__init__()
    # Protects core state that can be returned via public accessors.
    # Thread-safety is provided on a best-effort basis to support buggy
    # programs, and is not guaranteed by the public `tf.Graph` API.
    #
    # NOTE(mrry): This does not protect the various stacks. A warning will
    # be reported if these are used from multiple threads
    self._lock = threading.RLock()
    # The group lock synchronizes Session.run calls with methods that create
    # and mutate ops (e.g. Graph.create_op()). This synchronization is
    # necessary because it's illegal to modify an operation after it's been run.
    # The group lock allows any number of threads to mutate ops at the same time
    # but if any modification is going on, all Session.run calls have to wait.
    # Similarly, if one or more Session.run calls are going on, all mutate ops
    # have to wait until all Session.run calls have finished.
    self._group_lock = lock_util.GroupLock(num_groups=2)
    # Maps a name used in the graph to the next id to use for that name.
    self._names_in_use = {}
    self._stack_state_is_thread_local = False
    self._thread_local = threading.local()
    # Functions that will be applied to choose a device if none is specified.
    # In TF2.x or after switch_to_thread_local(),
    # self._thread_local._device_function_stack is used instead.
    self._graph_device_function_stack = traceable_stack.TraceableStack()
    # Default original_op applied to new ops.
    self._default_original_op = None
    # Current control flow context. It could be either CondContext or
    # WhileContext defined in ops/control_flow_ops.py
    self._control_flow_context = None
    # A new node will depend of the union of all of the nodes in the stack.
    # In TF2.x or after switch_to_thread_local(),
    # self._thread_local._control_dependencies_stack is used instead.
    self._graph_control_dependencies_stack = []
    # Arbitrary collections of objects.
    self._collections = {}
    # The graph-level random seed
    self._seed = None
    # A dictionary of attributes that should be applied to all ops.
    self._attr_scope_map = {}
    # A map from op type to the kernel label that should be used.
    self._op_to_kernel_label_map = {}
    # A map from op type to an alternative op type that should be used when
    # computing gradients.
    self._gradient_override_map = {}
    # A map from op type to a gradient function that should be used instead.
    self._gradient_function_map = {}
    # True if the graph is considered "finalized".  In that case no
    # new operations can be added.
    self._finalized = False
    # Functions defined in the graph
    self._functions = collections.OrderedDict()
    # Default GraphDef versions
    self._graph_def_versions = versions_pb2.VersionDef(
        producer=versions.GRAPH_DEF_VERSION,
        min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER)
    self._building_function = False
    # Stack of colocate_with ops. In TF2.x or after switch_to_thread_local(),
    # self._thread_local._colocation_stack is used instead.
    self._graph_colocation_stack = traceable_stack.TraceableStack()
    # Set of tensors that are dangerous to feed!
    self._unfeedable_tensors = object_identity.ObjectIdentitySet()
    # Set of operations that are dangerous to fetch!
    self._unfetchable_ops = set()
    # A map of tensor handle placeholder to tensor dtype.
    self._handle_feeders = {}
    # A map from tensor handle to its read op.
    self._handle_readers = {}
    # A map from tensor handle to its move op.
    self._handle_movers = {}
    # A map from tensor handle to its delete op.
    self._handle_deleters = {}
    # Allow optimizers and other objects to pseudo-uniquely key graphs (this key
    # will be shared when defining function graphs, for example, so optimizers
    # being called inside function definitions behave as if they were seeing the
    # actual outside graph).
    self._graph_key = "graph-key-%d/" % (uid(),)
    # A string with the last reduction method passed to
    # losses.compute_weighted_loss(), or None.
    # Backward compatibility with optimizer V1 use cases.
    self._last_loss_reduction = None
    # Required only for backward compatibility with optimizer V1 use cases.
    self._is_loss_scaled_by_optimizer = False
    self._container = ""

    # The current AutomaticControlDependencies context manager.
    self.experimental_acd_manager = None
    # Set to True if this graph is being built in an
    # AutomaticControlDependencies context.
    # Deprecated: use acd_manager instead.
    self._add_control_dependencies = False

    # Cache for OpDef protobufs retrieved via the C API.
    self._op_def_cache = {}
    # Cache for constant results of `reduced_shape()`. The keys are pairs of
    # tuples: (input_shape_tuple, reduction_indices_tuple), and the values
    # are pairs of tuples: (output_shape_kept_dims, tile_scaling).
    self._reduced_shape_cache = {}

    if tf2.enabled():
      self.switch_to_thread_local()

  # `Graph` now _is_ the C graph, but we have many places that manually attempt
  # to manipulate the _c_graph object. Leave these accessors here until these
  # are cleaned up.
  @property
  def _c_graph(self):
    return self

  def __enter__(self: GraphType) -> GraphType:
    return self

  def __exit__(self, *args) -> None:
    return

  def get(self: GraphType) -> GraphType:
    return self

  # Note: this method is private because the API of tf.Graph() is public and
  # frozen, and this functionality is still not ready for public visibility.
  @tf_contextlib.contextmanager
  def _variable_creator_scope(self, creator, priority=100) -> Iterator[None]:
    """Scope which defines a variable creation function.

    Args:
      creator: A callable taking `next_creator` and `kwargs`. See the
        `tf.variable_creator_scope` docstring.
      priority: Creators with a higher `priority` are called first. Within the
        same priority, creators are called inner-to-outer.

    Yields:
      `_variable_creator_scope` is a context manager with a side effect, but
      doesn't return a value.

    Raises:
      RuntimeError: If variable creator scopes are not properly nested.
    """
    # This step keeps a reference to the existing stack, and it also initializes
    # self._thread_local._variable_creator_stack if it doesn't exist yet.
    old = self._variable_creator_stack
    new = list(old)
    new.append((priority, creator))
    # Sorting is stable, so we'll put higher-priority creators later in the list
    # but otherwise maintain registration order.
    new.sort(key=lambda item: item[0])
    self._thread_local._variable_creator_stack = new  # pylint: disable=protected-access
    try:
      yield
    finally:
      if self._thread_local._variable_creator_stack is not new:  # pylint: disable=protected-access
        raise RuntimeError(
            "Exiting variable_creator_scope without proper nesting.")
      self._thread_local._variable_creator_stack = old  # pylint: disable=protected-access

  # TODO(b/192405401): unify resource_creator_scope with variable_creator_scope.
  # pylint: disable=protected-access
  @tf_contextlib.contextmanager
  def _resource_creator_scope(self, resource_type, creator) -> Iterator[None]:
    """Scope which defines a resource creation function used by some resource.

    The resource should be a subclass of CapturableResource with a class method
    `cls._resource_type`, the output of which is what the `resource_type`
    argument should be. By default, `cls._resource_type` returns the class name,
    `cls.__name__`. Given a scope, creators being added with the same
    `resource_type` argument will be composed together to apply to all classes
    with this `_resource_type`.


    `creator` is expected to be a function with the following signature:

    ```
      def resource_creator(next_creator, *a, **kwargs)
    ```

    The creator is supposed to eventually call the next_creator to create an
    instance if it does want to create an instance and not call
    the class initialization method directly. This helps make creators
    composable. A creator may choose to create multiple instances, return
    already existing instances, or simply register that an instance was created
    and defer to the next creator in line. Creators can also modify keyword
    arguments seen by the next creators.

    Valid keyword arguments in `kwargs` depends on the specific resource
    class. For StaticHashTable, this may be:
    * initializer: The table initializer to use.
    * default_value: The value to use if a key is missing in the table.
    * name: Optional name for the table, default to None.


    Args:
      resource_type: the output of the resource class's `_resource_type` method.
      creator: the passed creator for the resource.

    Yields:
      A scope in which the creator is active

    Raises:
      RuntimeError: If resource_creator_scope is existed without proper nesting.
    """
    # This step keeps a reference to the existing stack, and it also initializes
    # self._thread_local._variable_creator_stack if it doesn't exist yet.
    old = self._resource_creator_stack
    new = copy.deepcopy(old)
    if isinstance(resource_type, (list, tuple)):
      for r in resource_type:
        new[r].append(creator)
    else:
      new[resource_type].append(creator)
    self._thread_local._resource_creator_stack = new
    try:
      yield
    finally:
      if self._thread_local._resource_creator_stack is not new:
        raise RuntimeError(
            "Exiting resource_creator_scope without proper nesting.")
      self._thread_local._resource_creator_stack = old

  @property
  def _resource_creator_stack(self) -> dict[str, list[Callable[..., Any]]]:
    if not hasattr(self._thread_local, "_resource_creator_stack"):
      self._thread_local._resource_creator_stack = collections.defaultdict(list)
    return self._thread_local._resource_creator_stack

  @_resource_creator_stack.setter
  def _resource_creator_stack(
      self,
      resource_creator_stack: dict[str, list[Callable[..., Any]]],
  ) -> None:
    self._thread_local._resource_creator_stack = resource_creator_stack
  # pylint: enable=protected-access

  # Note: this method is private because the API of tf.Graph() is public and
  # frozen, and this functionality is still not ready for public visibility.
  @property
  def _variable_creator_stack(self) -> list[tuple[int, Callable[..., Any]]]:
    if not hasattr(self._thread_local, "_variable_creator_stack"):
      self._thread_local._variable_creator_stack = []  # pylint: disable=protected-access

    # This previously returned a copy of the stack instead of the stack itself,
    # to guard against accidental mutation. Consider, however, code that wants
    # to save and restore the variable creator stack:
    #     def f():
    #       original_stack = graph._variable_creator_stack
    #       graph._variable_creator_stack = new_stack
    #       ...  # Some code
    #       graph._variable_creator_stack = original_stack
    #
    # And lets say you have some code that calls this function with some
    # variable_creator:
    #     def g():
    #       with variable_scope.variable_creator_scope(creator):
    #         f()
    # When exiting the variable creator scope, it would see a different stack
    # object than it expected leading to a "Exiting variable_creator_scope
    # without proper nesting" error.
    return self._thread_local._variable_creator_stack  # pylint: disable=protected-access

  @_variable_creator_stack.setter
  def _variable_creator_stack(
      self,
      variable_creator_stack: list[tuple[int, Callable[..., Any]]],
  ) -> None:
    self._thread_local._variable_creator_stack = variable_creator_stack  # pylint: disable=protected-access

  def _check_not_finalized(self) -> None:
    """Check if the graph is finalized.

    Raises:
      RuntimeError: If the graph finalized.
    """
    if self._finalized:
      raise RuntimeError("Graph is finalized and cannot be modified.")

  @property
  def graph_def_versions(self) -> versions_pb2.VersionDef:
    # pylint: disable=line-too-long
    """The GraphDef version information of this graph.

    For details on the meaning of each version, see
    [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto).

    Returns:
      A `VersionDef`.
    """
    return versions_pb2.VersionDef.FromString(self._version_def)

  @property
  def seed(self) -> Optional[int]:
    """The graph-level random seed of this graph."""
    return self._seed

  @seed.setter
  def seed(self, seed: int) -> None:
    self._seed = seed

  @property
  def finalized(self) -> bool:
    """True if this graph has been finalized."""
    return self._finalized

  def finalize(self) -> None:
    """Finalizes this graph, making it read-only.

    After calling `g.finalize()`, no new operations can be added to
    `g`.  This method is used to ensure that no operations are added
    to a graph when it is shared between multiple threads, for example
    when using a `tf.compat.v1.train.QueueRunner`.
    """
    self._finalized = True

  def _unsafe_unfinalize(self) -> None:
    """Opposite of `finalize`.

    Internal interface.

    NOTE: Unfinalizing a graph could have negative impact on performance,
    especially in a multi-threaded environment.  Unfinalizing a graph
    when it is in use by a Session may lead to undefined behavior. Ensure
    that all sessions using a graph are closed before calling this method.
    """
    self._finalized = False

  def _get_control_flow_context(self):
    """Returns the current control flow context.

    Returns:
      A context object.
    """
    return self._control_flow_context

  def _set_control_flow_context(self, ctx) -> None:
    """Sets the current control flow context.

    Args:
      ctx: a context object.
    """
    self._control_flow_context = ctx

  def _copy_functions_to_graph_def(self, graph_def, starting_bytesize) -> None:
    """If this graph contains functions, copy them to `graph_def`."""
    bytesize = starting_bytesize
    for f in self._functions.values():
      bytesize += f.cached_definition.ByteSize()
      if bytesize >= (1 << 31) or bytesize < 0:
        raise ValueError("GraphDef cannot be larger than 2GB.")
      graph_def.library.function.extend([f.cached_definition])
      if getattr(f, "grad_func_name", None):
        grad_def = function_pb2.GradientDef()
        grad_def.function_name = f.name
        grad_def.gradient_func = f.grad_func_name
        graph_def.library.gradient.extend([grad_def])

  def _as_graph_def(
      self, from_version=None, add_shapes=False, use_pybind11_proto=False,
  ) -> tuple[graph_pb2.GraphDef, int]:
    # pylint: disable=line-too-long
    """Returns a serialized `GraphDef` representation of this graph.

    The serialized `GraphDef` can be imported into another `Graph`
    (using `tf.import_graph_def`) or used with the
    [C++ Session API](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/api_docs/cc/index.md).

    This method is thread-safe.

    Args:
      from_version: Optional.  If this is set, returns a `GraphDef` containing
        only the nodes that were added to this graph since its `version`
        property had the given value.
      add_shapes: If true, adds an "_output_shapes" list attr to each node with
        the inferred shapes of each of its outputs.
      use_pybind11_proto: If true, uses the c++ pybind11_proto api to get the
        GraphDef proto directly from c++, instead of through a TF buffer.

    Returns:
      A tuple containing a
      [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer, and the version of the graph to which that
      `GraphDef` corresponds.

    Raises:
      ValueError: If the `graph_def` would be too large.

    """
    # pylint: enable=line-too-long
    with self._lock:
      if use_pybind11_proto:
        with self._c_graph.get() as c_graph:
          graph = graph_pb2.GraphDef()
          graph.CopyFrom(pywrap_tf_session.TF_GraphToGraphDefPybind(c_graph))
      else:
        with c_api_util.tf_buffer() as buf:   # pytype: disable=wrong-arg-count
          with self._c_graph.get() as c_graph:
            pywrap_tf_session.TF_GraphToGraphDef(c_graph, buf)
            data = pywrap_tf_session.TF_GetBuffer(buf)
        graph = graph_pb2.GraphDef()
        graph.ParseFromString(compat.as_bytes(data))
      # Strip the experimental library field iff it's empty.
      if not graph.library.function:
        graph.ClearField("library")

      if add_shapes:
        for node in graph.node:
          op = self._get_operation_by_name(node.name)
          if op.outputs:
            node.attr["_output_shapes"].list.shape.extend(
                [output.get_shape().as_proto() for output in op.outputs])
        for function_def in graph.library.function:
          defined_function = self._functions[function_def.signature.name]
          try:
            func_graph = defined_function.graph
          except AttributeError:
            # _DefinedFunction doesn't have a graph, _EagerDefinedFunction
            # does. Both rely on ops.py, so we can't really isinstance check
            # them.
            continue
          input_shapes = function_def.attr["_input_shapes"]
          try:
            func_graph_inputs = func_graph.inputs
          except AttributeError:
            continue
          # TODO(b/141471245): Fix the inconsistency when inputs of func graph
          # are appended during gradient computation of while/cond.
          assert len(input_shapes.list.shape) in [0, len(func_graph_inputs)]
          # If the function_def has inputs already filled out, skip this step.
          if not input_shapes.list.shape:
            for input_tensor, arg_def in zip(func_graph_inputs,
                                             function_def.signature.input_arg):
              input_shapes.list.shape.add().CopyFrom(
                  input_tensor.get_shape().as_proto())
              if input_tensor.dtype == dtypes.resource:
                _copy_handle_data_to_arg_def(input_tensor, arg_def)

          for output_tensor, arg_def in zip(func_graph.outputs,
                                            function_def.signature.output_arg):
            if output_tensor.dtype == dtypes.resource:
              _copy_handle_data_to_arg_def(output_tensor, arg_def)

          for node in function_def.node_def:
            try:
              op = func_graph.get_operation_by_name(node.name)
            except KeyError:
              continue
            outputs = op.outputs

            if op.type == "StatefulPartitionedCall":
              # Filter out any extra outputs (possibly added by function
              # backpropagation rewriting).
              num_outputs = len(node.attr["Tout"].list.type)
              outputs = outputs[:num_outputs]

            node.attr["_output_shapes"].list.shape.extend(
                [output.get_shape().as_proto() for output in outputs])

    return graph, self.version

  def as_graph_def(
      self, from_version=None, add_shapes=False, use_pybind11_proto=False
  ) -> graph_pb2.GraphDef:
    # pylint: disable=line-too-long
    """Returns a serialized `GraphDef` representation of this graph.

    The serialized `GraphDef` can be imported into another `Graph`
    (using `tf.import_graph_def`) or used with the
    [C++ Session API](../../api_docs/cc/index.md).

    This method is thread-safe.

    Args:
      from_version: Optional.  If this is set, returns a `GraphDef` containing
        only the nodes that were added to this graph since its `version`
        property had the given value.
      add_shapes: If true, adds an "_output_shapes" list attr to each node with
        the inferred shapes of each of its outputs.
      use_pybind11_proto: If true, If true, uses the c++ pybind11_proto api to
        get the GraphDef proto directly from c++, instead of through a TF
        buffer. See https://github.com/pybind/pybind11_protobuf for reference.

    Returns:
      A
      [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer.

    Raises:
      ValueError: If the `graph_def` would be too large.
    """
    # pylint: enable=line-too-long
    if is_oss:
      use_pybind11_proto = False
    result, _ = self._as_graph_def(
        from_version, add_shapes, use_pybind11_proto=use_pybind11_proto
    )
    return result

  def _is_function(self, name) -> bool:
    """Tests whether 'name' is registered in this graph's function library.

    Args:
      name: string op name.

    Returns:
      bool indicating whether or not 'name' is registered in function library.
    """
    return compat.as_str(name) in self._functions

  def _get_function(self, name):
    """Returns the function definition for 'name'.

    Args:
      name: string function name.

    Returns:
      The function def proto.
    """
    return self._functions.get(compat.as_str(name), None)

  def _add_function_recursive(self, function, overwrite=False) -> None:
    """Adds function to the graph including other functions in its graph."""

    if self._is_function(function.name):
      if overwrite:
        self._remove_function(function.name)
        self._add_function(function)
    else:
      self._add_function(function)

    if hasattr(function, "children"):
      for f in function.children:  # pylint: disable=protected-access
        if self._is_function(f.name):
          if overwrite:
            self._remove_function(f.name)
            self._add_function(f)
        else:
          self._add_function(f)

  def _add_function(self, function) -> None:
    """Adds a function to the graph.

    After the function has been added, you can call to the function by
    passing the function name in place of an op name to
    `Graph.create_op()`.

    Args:
      function: A `_DefinedFunction` object.

    Raises:
      ValueError: if another function is defined with the same name.
    """
    self._check_not_finalized()

    name = function.name
    # Sanity checks on gradient definition for deprecated _DefinedFunction.
    if getattr(function, "grad_func_name", None) and getattr(
        function, "python_grad_func", None
    ):
      raise ValueError("Gradient defined twice for function %s" % name)

    # Add function to graph
    # pylint: disable=protected-access
    with self._c_graph.get() as c_graph:
      with function._c_func.get() as func:
        if getattr(function, "_grad_func", None):
          # For deprecated _DefinedFunction.
          with function._grad_func._c_func.get() as gradient:
            pywrap_tf_session.TF_GraphCopyFunction(c_graph, func, gradient)
        else:
          pywrap_tf_session.TF_GraphCopyFunction(c_graph, func, None)
    # pylint: enable=protected-access

    self._functions[compat.as_str(name)] = function

    # Need a new-enough consumer to support the functions we add to the graph.
    if self._graph_def_versions.min_consumer < 12:
      self._graph_def_versions.min_consumer = 12

  def _remove_function(self, name) -> None:
    self._check_not_finalized()
    if not self._is_function(name):
      raise ValueError(f"Function {name!r} is not found in {self!r}.")

    with self._c_graph.get() as c_graph:
      pywrap_tf_session.TF_GraphRemoveFunction(c_graph, compat.as_bytes(name))
      del self._functions[compat.as_str(name)]

  @property
  def building_function(self) -> bool:
    """Returns True iff this graph represents a function."""
    return self._building_function

  # Helper functions to create operations.
  @deprecated_args(None,
                   "Shapes are always computed; don't use the compute_shapes "
                   "as it has no effect.", "compute_shapes")
  @traceback_utils.filter_traceback
  def create_op(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_shapes=True,
      compute_device=True) -> "Operation":
    """Creates an `Operation` in this graph.

    This is a low-level interface for creating an `Operation`. Most
    programs will not call this method directly, and instead use the
    Python op constructors, such as `tf.constant()`, which add ops to
    the default graph.

    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: (Optional) A list of `DType` objects that will be the types of the
        tensors that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of the
        tensors that the operation consumes. By default, uses the base `DType`
        of each input in `inputs`. Operations that expect reference-typed inputs
        must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_shapes: (Optional.) Deprecated. Has no effect (shapes are always
        computed).
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.

    Raises:
      TypeError: if any of the inputs is not a `Tensor`.
      ValueError: if colocation conflicts with existing device assignment.

    Returns:
      An `Operation` object.
    """
    del compute_shapes
    for idx, a in enumerate(inputs):
      if not isinstance(a, tensor_lib.Tensor):
        raise TypeError("Input #%d is not a tensor: %s" % (idx, a))
    return self._create_op_internal(op_type, inputs, dtypes, input_types, name,
                                    attrs, op_def, compute_device)

  def _create_op_internal(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True) -> "Operation":
    """Creates an `Operation` in this graph.

    Implements `Graph.create_op()` without the overhead of the deprecation
    wrapper.

    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: (Optional) A list of `DType` objects that will be the types of the
        tensors that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of the
        tensors that the operation consumes. By default, uses the base `DType`
        of each input in `inputs`. Operations that expect reference-typed inputs
        must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.

    Raises:
      ValueError: if colocation conflicts with existing device assignment.

    Returns:
      An `Operation` object.
    """
    self._check_not_finalized()
    if name is None:
      name = op_type
    # If a names ends with a '/' it is a "name scope" and we use it as-is,
    # after removing the trailing '/'.
    if name and name[-1] == "/":
      name = name_from_scope_name(name)
    else:
      name = self.unique_name(name)

    node_def = _NodeDef(op_type, name, attrs)

    input_ops = set(t.op for t in inputs)
    control_inputs = self._control_dependencies_for_inputs(input_ops)
    # _create_op_helper mutates the new Operation. `_mutation_lock` ensures a
    # Session.run call cannot occur between creating and mutating the op.
    with self._mutation_lock():
      ret = Operation.from_node_def(
          node_def,
          self,
          inputs=inputs,
          output_types=dtypes,
          control_inputs=control_inputs,
          input_types=input_types,
          original_op=self._default_original_op,
          op_def=op_def,
      )
      self._create_op_helper(ret, compute_device=compute_device)
    return ret

  def _create_op_from_tf_operation(
      self, c_op, compute_device=True,
  ) -> "Operation":
    """Creates an `Operation` in this graph from the supplied TF_Operation.

    This method is like create_op() except the new Operation is constructed
    using `c_op`. The returned Operation will have `c_op` as its _c_op
    field. This is used to create Operation objects around TF_Operations created
    indirectly by the C API (e.g. by TF_ImportGraphDef, TF_FinishWhile).

    This function does not call Operation._control_flow_post_processing or
    Graph._control_dependencies_for_inputs (since the inputs may not be
    available yet). The caller is responsible for calling these methods.

    Args:
      c_op: a wrapped TF_Operation
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.

    Returns:
      An `Operation` object.
    """
    self._check_not_finalized()
    ret = Operation._from_c_op(c_op=c_op, g=self)  # pylint: disable=protected-access
    # If a name_scope was created with ret.name but no nodes were created in it,
    # the name will still appear in _names_in_use even though the name hasn't
    # been used. This is ok, just leave _names_in_use as-is in this case.
    # TODO(skyewm): make the C API guarantee no name conflicts.
    name_key = ret.name.lower()
    if name_key not in self._names_in_use:
      self._names_in_use[name_key] = 1
    self._create_op_helper(ret, compute_device=compute_device)
    return ret

  def _create_op_helper(self, op, compute_device=True) -> None:
    """Common logic for creating an op in this graph."""
    # Apply any additional attributes requested. Do not overwrite any existing
    # attributes.
    for key, value in self._attr_scope_map.items():
      try:
        op.get_attr(key)
      except ValueError:
        if callable(value):
          value = value(op.node_def)
          if not isinstance(value, (type(None), attr_value_pb2.AttrValue)):
            raise TypeError(
                "Callable for scope map key '%s' must return either None or "
                "an AttrValue protocol buffer; but it returned: %s" %
                (key, value))
        if value:
          op._set_attr(key, value)  # pylint: disable=protected-access

    # Apply a kernel label if one has been specified for this op type.
    try:
      kernel_label = self._op_to_kernel_label_map[op.type]
      op._set_attr("_kernel",  # pylint: disable=protected-access
                   attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:
      pass

    op._gradient_function = self._gradient_function_map.get(op.type)  # pylint: disable=protected-access

    # Apply the overriding op type for gradients if one has been specified for
    # this op type.
    try:
      mapped_op_type = self._gradient_override_map[op.type]
      op._set_attr("_gradient_op_type",  # pylint: disable=protected-access
                   attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
      pass

    self._record_op_seen_by_control_dependencies(op)

    if compute_device:
      self._apply_device_functions(op)

    # Snapshot the colocation stack metadata before we might generate error
    # messages using it.  Note that this snapshot depends on the actual stack
    # and is independent of the op's _class attribute.
    # pylint: disable=protected-access
    op._colocation_code_locations = self._snapshot_colocation_stack_metadata()
    # pylint: enable=protected-access

    if self._colocation_stack:
      all_colocation_groups = []
      is_device_set = False
      for colocation_op in self._colocation_stack.peek_objs():
        try:
          all_colocation_groups.extend(colocation_op.colocation_groups())
        except AttributeError:
          pass
        if colocation_op.device and not is_device_set:
          # pylint: disable=protected-access
          op._set_device(colocation_op.device)
          # pylint: enable=protected-access
          is_device_set = True

      all_colocation_groups = sorted(set(all_colocation_groups))
      # pylint: disable=protected-access
      op._set_attr(
          "_class",
          attr_value_pb2.AttrValue(
              list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))
      # pylint: enable=protected-access

    # Sets "container" attribute if
    # (1) self._container is not None
    # (2) "is_stateful" is set in OpDef
    # (3) "container" attribute is in OpDef
    # (4) "container" attribute is None
    if self._container and op._is_stateful:  # pylint: disable=protected-access
      try:
        container_attr = op.get_attr("container")
      except ValueError:
        # "container" attribute is not in OpDef
        pass
      else:
        if not container_attr:
          op._set_attr("container", attr_value_pb2.AttrValue(  # pylint: disable=protected-access
              s=compat.as_bytes(self._container)))

  def _add_new_tf_operations(self, compute_devices=True) -> list["Operation"]:
    """Creates `Operations` in this graph for any new TF_Operations.

    This is useful for when TF_Operations are indirectly created by the C API
    outside of the Operation constructor (e.g. by TF_ImportGraphDef,
    TF_FinishWhile). This ensures there are corresponding Operations for all
    TF_Operations in the underlying TF_Graph.

    Args:
      compute_devices: (Optional.) If True, device functions will be executed to
        compute the device properties of each new Operation.

    Returns:
      A list of the new `Operation` objects.
    """
    self._check_not_finalized()

    # Create all Operation objects before accessing their inputs since an op may
    # be created before its inputs.
    new_ops = [
        self._create_op_from_tf_operation(c_op, compute_device=compute_devices)
        for c_op in self.new_operations()
    ]

    # pylint: disable=protected-access
    for op in new_ops:
      new_control_inputs = self._control_dependencies_for_inputs(op.inputs)
      op._add_control_inputs(new_control_inputs)
      op._control_flow_post_processing()
    # pylint: enable=protected-access

    return new_ops

  def as_graph_element(
      self, obj, allow_tensor=True, allow_operation=True,
  ) -> Union[tensor_lib.Tensor, "Operation"]:
    """Returns the object referred to by `obj`, as an `Operation` or `Tensor`.

    This function validates that `obj` represents an element of this
    graph, and gives an informative error message if it is not.

    This function is the canonical way to get/validate an object of
    one of the allowed types from an external argument reference in the
    Session API.

    This method may be called concurrently from multiple threads.

    Args:
      obj: A `Tensor`, an `Operation`, or the name of a tensor or operation. Can
        also be any object with an `_as_graph_element()` method that returns a
        value of one of these types. Note: `_as_graph_element` will be called
        inside the graph's lock and so may not modify the graph.
      allow_tensor: If true, `obj` may refer to a `Tensor`.
      allow_operation: If true, `obj` may refer to an `Operation`.

    Returns:
      The `Tensor` or `Operation` in the Graph corresponding to `obj`.

    Raises:
      TypeError: If `obj` is not a type we support attempting to convert
        to types.
      ValueError: If `obj` is of an appropriate type but invalid. For
        example, an invalid string.
      KeyError: If `obj` is not an object in the graph.
    """
    if self._finalized:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

    with self._lock:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

  def _as_graph_element_locked(
      self, obj, allow_tensor, allow_operation,
  ) -> Union[tensor_lib.Tensor, "Operation"]:
    """See `Graph.as_graph_element()` for details."""
    # The vast majority of this function is figuring
    # out what an API user might be doing wrong, so
    # that we can give helpful error messages.
    #
    # Ideally, it would be nice to split it up, but we
    # need context to generate nice error messages.

    if allow_tensor and allow_operation:
      types_str = "Tensor or Operation"
    elif allow_tensor:
      types_str = "Tensor"
    elif allow_operation:
      types_str = "Operation"
    else:
      raise ValueError("allow_tensor and allow_operation can't both be False.")

    temp_obj = _as_graph_element(obj)
    if temp_obj is not None:
      obj = temp_obj

    # If obj appears to be a name...
    if isinstance(obj, compat.bytes_or_text_types):
      name = compat.as_str(obj)

      if ":" in name and allow_tensor:
        # Looks like a Tensor name and can be a Tensor.
        try:
          op_name, out_n = name.split(":")
          out_n = int(out_n)
        except:
          raise ValueError("The name %s looks a like a Tensor name, but is "
                           "not a valid one. Tensor names must be of the "
                           "form \"<op_name>:<output_index>\"." % repr(name))
        try:
          op = self._get_operation_by_name(op_name)
        except KeyError as exc:
          raise KeyError(
              "The name %s refers to a Tensor which does not "
              "exist. The operation, %s, does not exist in the "
              "graph." % (repr(name), repr(op_name))
          ) from exc

        try:
          return op.outputs[out_n]
        except:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, exists but only has "
                         "%s outputs." %
                         (repr(name), repr(op_name), len(op.outputs)))

      elif ":" in name and not allow_tensor:
        # Looks like a Tensor name but can't be a Tensor.
        raise ValueError("Name %s appears to refer to a Tensor, not a %s." %
                         (repr(name), types_str))

      elif ":" not in name and allow_operation:
        # Looks like an Operation name and can be an Operation.
        try:
          op = self._get_operation_by_name(name)
        except KeyError as exc:
          raise KeyError(
              "The name %s refers to an Operation not in the graph."
              % repr(name)
          ) from exc
        return op

      elif ":" not in name and not allow_operation:
        # Looks like an Operation name but can't be an Operation.
        try:
          op = self._get_operation_by_name(name)
          # Yep, it's an Operation name
          err_msg = ("The name %s refers to an Operation, not a %s." %
                     (repr(name), types_str))
        except KeyError:
          err_msg = ("The name %s looks like an (invalid) Operation name, "
                     "not a %s." % (repr(name), types_str))
        err_msg += (" Tensor names must be of the form "
                    "\"<op_name>:<output_index>\".")
        raise ValueError(err_msg)

    elif isinstance(obj, tensor_lib.Tensor) and allow_tensor:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Tensor %s is not an element of this graph." % obj)
      return obj
    elif isinstance(obj, Operation) and allow_operation:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Operation %s is not an element of this graph." % obj)
      return obj
    else:
      # We give up!
      raise TypeError("Can not convert a %s into a %s." %
                      (type(obj).__name__, types_str))

  def get_operation_by_name(self, name) -> "Operation":
    """Returns the `Operation` with the given `name`.

    This method may be called concurrently from multiple threads.

    Args:
      name: The name of the `Operation` to return.

    Returns:
      The `Operation` with the given `name`.

    Raises:
      TypeError: If `name` is not a string.
      KeyError: If `name` does not correspond to an operation in this graph.
    """

    if not isinstance(name, str):
      raise TypeError("Operation names are strings (or similar), not %s." %
                      type(name).__name__)
    op = cast(
        Operation,
        self.as_graph_element(name, allow_tensor=False, allow_operation=True)
    )
    return op

  def _get_operation_by_tf_operation(self, tf_oper) -> "Operation":
    op_name = pywrap_tf_session.TF_OperationName(tf_oper)
    return self._get_operation_by_name(op_name)

  def get_tensor_by_name(self, name) -> tensor_lib.Tensor:
    """Returns the `Tensor` with the given `name`.

    This method may be called concurrently from multiple threads.

    Args:
      name: The name of the `Tensor` to return.

    Returns:
      The `Tensor` with the given `name`.

    Raises:
      TypeError: If `name` is not a string.
      KeyError: If `name` does not correspond to a tensor in this graph.
    """
    # Names should be strings.
    if not isinstance(name, str):
      raise TypeError("Tensor names are strings (or similar), not %s." %
                      type(name).__name__)
    tensor = cast(
        tensor_lib.Tensor,
        self.as_graph_element(name, allow_tensor=True, allow_operation=False)
    )
    return tensor

  def _get_tensor_by_tf_output(self, tf_output) -> tensor_lib.Tensor:
    """Returns the `Tensor` representing `tf_output`.

    Note that there is only one such `Tensor`, i.e. multiple calls to this
    function with the same TF_Output value will always return the same `Tensor`
    object.

    Args:
      tf_output: A wrapped `TF_Output` (the C API equivalent of `Tensor`).

    Returns:
      The `Tensor` that represents `tf_output`.
    """
    op = self._get_operation_by_tf_operation(tf_output.oper)
    return op.outputs[tf_output.index]

  def op_def_for_type(self, type) -> op_def_pb2.OpDef:  # pylint: disable=redefined-builtin
    """Returns the `OpDef` proto for `type`. `type` is a string."""
    # NOTE: No locking is required because the lookup and insertion operations
    # on Python dictionaries are atomic.
    try:
      return self._op_def_cache[type]
    except KeyError:
      self._op_def_cache[type] = op_def_pb2.OpDef.FromString(
          self._op_def_for_type(type)
      )
      return self._op_def_cache[type]

  def as_default(self) -> ContextManager["Graph"]:
    """Returns a context manager that makes this `Graph` the default graph.

    This method should be used if you want to create multiple graphs
    in the same process. For convenience, a global default graph is
    provided, and all ops will be added to this graph if you do not
    create a new graph explicitly.

    Use this method with the `with` keyword to specify that ops created within
    the scope of a block should be added to this graph. In this case, once
    the scope of the `with` is exited, the previous default graph is set again
    as default. There is a stack, so it's ok to have multiple nested levels
    of `as_default` calls.

    The default graph is a property of the current thread. If you
    create a new thread, and wish to use the default graph in that
    thread, you must explicitly add a `with g.as_default():` in that
    thread's function.

    The following code examples are equivalent:

    ```python
    # 1. Using Graph.as_default():
    g = tf.Graph()
    with g.as_default():
      c = tf.constant(5.0)
      assert c.graph is g

    # 2. Constructing and making default:
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      assert c.graph is g
    ```

    If eager execution is enabled ops created under this context manager will be
    added to the graph instead of executed eagerly.

    Returns:
      A context manager for using this graph as the default graph.
    """
    return _default_graph_stack.get_controller(self)

  @property
  def collections(self) -> list[str]:
    """Returns the names of the collections known to this graph."""
    return list(self._collections)

  def add_to_collection(self, name, value) -> None:
    """Stores `value` in the collection with the given `name`.

    Note that collections are not sets, so it is possible to add a value to
    a collection several times.

    Args:
      name: The key for the collection. The `GraphKeys` class contains many
        standard names for collections.
      value: The value to add to the collection.
    """  # pylint: disable=g-doc-exception
    self._check_not_finalized()
    with self._lock:
      if name not in self._collections:
        self._collections[name] = [value]
      else:
        self._collections[name].append(value)

  def add_to_collections(self, names, value) -> None:
    """Stores `value` in the collections given by `names`.

    Note that collections are not sets, so it is possible to add a value to
    a collection several times. This function makes sure that duplicates in
    `names` are ignored, but it will not check for pre-existing membership of
    `value` in any of the collections in `names`.

    `names` can be any iterable, but if `names` is a string, it is treated as a
    single collection name.

    Args:
      names: The keys for the collections to add to. The `GraphKeys` class
        contains many standard names for collections.
      value: The value to add to the collections.
    """
    # Make sure names are unique, but treat strings as a single collection name
    names = (names,) if isinstance(names, str) else set(names)
    for name in names:
      self.add_to_collection(name, value)

  def get_collection_ref(self, name) -> list[Any]:
    """Returns a list of values in the collection with the given `name`.

    If the collection exists, this returns the list itself, which can
    be modified in place to change the collection.  If the collection does
    not exist, it is created as an empty list and the list is returned.

    This is different from `get_collection()` which always returns a copy of
    the collection list if it exists and never creates an empty collection.

    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.

    Returns:
      The list of values in the collection with the given `name`, or an empty
      list if no value has been added to that collection.
    """  # pylint: disable=g-doc-exception
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        coll_list = []
        self._collections[name] = coll_list
      return coll_list

  def get_collection(self, name, scope=None) -> list[Any]:
    """Returns a list of values in the collection with the given `name`.

    This is different from `get_collection_ref()` which always returns the
    actual collection list if it exists in that it returns a new list each time
    it is called.

    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.
      scope: (Optional.) A string. If supplied, the resulting list is filtered
        to include only items whose `name` attribute matches `scope` using
        `re.match`. Items without a `name` attribute are never returned if a
        scope is supplied. The choice of `re.match` means that a `scope` without
        special tokens filters by prefix.

    Returns:
      The list of values in the collection with the given `name`, or
      an empty list if no value has been added to that collection. The
      list contains the values in the order under which they were
      collected.
    """  # pylint: disable=g-doc-exception
    with self._lock:
      collection = self._collections.get(name, None)
      if collection is None:
        return []
      if scope is None:
        return list(collection)
      else:
        c = []
        regex = re.compile(scope)
        for item in collection:
          try:
            if regex.match(item.name):
              c.append(item)
          except AttributeError:
            # Collection items with no name are ignored.
            pass
        return c

  def get_all_collection_keys(self) -> list[str]:
    """Returns a list of collections used in this graph."""
    with self._lock:
      return [x for x in self._collections if isinstance(x, str)]

  def clear_collection(self, name) -> None:
    """Clears all values in a collection.

    Args:
      name: The key for the collection. The `GraphKeys` class contains many
        standard names for collections.
    """
    self._check_not_finalized()
    with self._lock:
      if name in self._collections:
        del self._collections[name]

  @tf_contextlib.contextmanager
  def _original_op(self, op) -> Iterator[None]:
    """Python 'with' handler to help annotate ops with their originator.

    An op may have an 'original_op' property that indicates the op on which
    it was based. For example a replica op is based on the op that was
    replicated and a gradient op is based on the op that was differentiated.

    All ops created in the scope of this 'with' handler will have
    the given 'op' as their original op.

    Args:
      op: The Operation that all ops created in this scope will have as their
        original op.

    Yields:
      Nothing.
    """
    old_original_op = self._default_original_op
    self._default_original_op = op
    try:
      yield
    finally:
      self._default_original_op = old_original_op

  @property
  def _name_stack(self) -> str:
    # This may be called from a thread where name_stack doesn't yet exist.
    if not hasattr(self._thread_local, "_name_stack"):
      self._thread_local._name_stack = ""
    return self._thread_local._name_stack

  @_name_stack.setter
  def _name_stack(self, name_stack: str) -> None:
    self._thread_local._name_stack = name_stack

  # pylint: disable=g-doc-return-or-yield,line-too-long
  @tf_contextlib.contextmanager
  def name_scope(self, name) -> Iterator[str]:
    """Returns a context manager that creates hierarchical names for operations.

    A graph maintains a stack of name scopes. A `with name_scope(...):`
    statement pushes a new name onto the stack for the lifetime of the context.

    The `name` argument will be interpreted as follows:

    * A string (not ending with '/') will create a new name scope, in which
      `name` is appended to the prefix of all operations created in the
      context. If `name` has been used before, it will be made unique by
      calling `self.unique_name(name)`.
    * A scope previously captured from a `with g.name_scope(...) as
      scope:` statement will be treated as an "absolute" name scope, which
      makes it possible to re-enter existing scopes.
    * A value of `None` or the empty string will reset the current name scope
      to the top-level (empty) name scope.

    For example:

    ```python
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0, name="c")
      assert c.op.name == "c"
      c_1 = tf.constant(6.0, name="c")
      assert c_1.op.name == "c_1"

      # Creates a scope called "nested"
      with g.name_scope("nested") as scope:
        nested_c = tf.constant(10.0, name="c")
        assert nested_c.op.name == "nested/c"

        # Creates a nested scope called "inner".
        with g.name_scope("inner"):
          nested_inner_c = tf.constant(20.0, name="c")
          assert nested_inner_c.op.name == "nested/inner/c"

        # Create a nested scope called "inner_1".
        with g.name_scope("inner"):
          nested_inner_1_c = tf.constant(30.0, name="c")
          assert nested_inner_1_c.op.name == "nested/inner_1/c"

          # Treats `scope` as an absolute name scope, and
          # switches to the "nested/" scope.
          with g.name_scope(scope):
            nested_d = tf.constant(40.0, name="d")
            assert nested_d.op.name == "nested/d"

            with g.name_scope(""):
              e = tf.constant(50.0, name="e")
              assert e.op.name == "e"
    ```

    The name of the scope itself can be captured by `with
    g.name_scope(...) as scope:`, which stores the name of the scope
    in the variable `scope`. This value can be used to name an
    operation that represents the overall result of executing the ops
    in a scope. For example:

    ```python
    inputs = tf.constant(...)
    with g.name_scope('my_layer') as scope:
      weights = tf.Variable(..., name="weights")
      biases = tf.Variable(..., name="biases")
      affine = tf.matmul(inputs, weights) + biases
      output = tf.nn.relu(affine, name=scope)
    ```

    NOTE: This constructor validates the given `name`. Valid scope
    names match one of the following regular expressions:

        [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (for scopes at the root)
        [A-Za-z0-9_.\\-/]* (for other scopes)

    Args:
      name: A name for the scope.

    Returns:
      A context manager that installs `name` as a new name scope.

    Raises:
      ValueError: If `name` is not a valid scope name, according to the rules
        above.
    """
    if name:
      if isinstance(name, compat.bytes_or_text_types):
        name = compat.as_str(name)

      if self._name_stack:
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        if not _VALID_SCOPE_NAME_REGEX.match(name):
          raise ValueError(
              f"'{name}' is not a valid scope name. A scope name has to match "
              f"the following pattern: {_VALID_SCOPE_NAME_REGEX.pattern}")
      else:
        # Scopes created in the root must match the more restrictive
        # op name regex, which constrains the initial character.
        if not _VALID_OP_NAME_REGEX.match(name):
          raise ValueError(
              f"'{name}' is not a valid root scope name. A root scope name has "
              f"to match the following pattern: {_VALID_OP_NAME_REGEX.pattern}")
    old_stack = self._name_stack
    if not name:  # Both for name=None and name="" we re-set to empty scope.
      new_stack = ""
      returned_scope = ""
    elif name[-1] == "/":
      new_stack = name_from_scope_name(name)
      returned_scope = name
    else:
      new_stack = self.unique_name(name)
      returned_scope = new_stack + "/"
    self._name_stack = new_stack
    try:
      yield returned_scope
    finally:
      self._name_stack = old_stack

  # pylint: enable=g-doc-return-or-yield,line-too-long

  def unique_name(self, name, mark_as_used=True) -> str:
    """Return a unique operation name for `name`.

    Note: You rarely need to call `unique_name()` directly.  Most of
    the time you just need to create `with g.name_scope()` blocks to
    generate structured names.

    `unique_name` is used to generate structured names, separated by
    `"/"`, to help identify operations when debugging a graph.
    Operation names are displayed in error messages reported by the
    TensorFlow runtime, and in various visualization tools such as
    TensorBoard.

    If `mark_as_used` is set to `True`, which is the default, a new
    unique name is created and marked as in use. If it's set to `False`,
    the unique name is returned without actually being marked as used.
    This is useful when the caller simply wants to know what the name
    to be created will be.

    Args:
      name: The name for an operation.
      mark_as_used: Whether to mark this name as being used.

    Returns:
      A string to be passed to `create_op()` that will be used
      to name the operation being created.
    """
    if self._name_stack:
      name = self._name_stack + "/" + name

    # For the sake of checking for names in use, we treat names as case
    # insensitive (e.g. foo = Foo).
    name_key = name.lower()
    i = self._names_in_use.get(name_key, 0)
    # Increment the number for "name_key".
    if mark_as_used:
      self._names_in_use[name_key] = i + 1
    if i > 0:
      base_name_key = name_key
      # Make sure the composed name key is not already used.
      while name_key in self._names_in_use:
        name_key = "%s_%d" % (base_name_key, i)
        i += 1
      # Mark the composed name_key as used in case someone wants
      # to call unique_name("name_1").
      if mark_as_used:
        self._names_in_use[name_key] = 1

      # Return the new name with the original capitalization of the given name.
      name = "%s_%d" % (name, i - 1)
    return name

  def get_name_scope(self) -> str:
    """Returns the current name scope.

    For example:

    ```python
    with tf.name_scope('scope1'):
      with tf.name_scope('scope2'):
        print(tf.compat.v1.get_default_graph().get_name_scope())
    ```
    would print the string `scope1/scope2`.

    Returns:
      A string representing the current name scope.
    """
    return self._name_stack

  @tf_contextlib.contextmanager
  def _colocate_with_for_gradient(self, op, gradient_uid,
                                  ignore_existing=False) -> Iterator[None]:
    with self.colocate_with(op, ignore_existing):
      if gradient_uid is not None:
        ctx = _get_enclosing_context(self)
        if ctx is not None:
          ctx.EnterGradientColocation(op, gradient_uid)
          try:
            yield
          finally:
            ctx.ExitGradientColocation(op, gradient_uid)
        else:
          yield
      else:
        yield

  @tf_contextlib.contextmanager
  def colocate_with(self, op, ignore_existing=False) -> Iterator[None]:
    """Returns a context manager that specifies an op to colocate with.

    Note: this function is not for public use, only for internal libraries.

    For example:

    ```python
    a = tf.Variable([1.0])
    with g.colocate_with(a):
      b = tf.constant(1.0)
      c = tf.add(a, b)
    ```

    `b` and `c` will always be colocated with `a`, no matter where `a`
    is eventually placed.

    **NOTE** Using a colocation scope resets any existing device constraints.

    If `op` is `None` then `ignore_existing` must be `True` and the new
    scope resets all colocation and device constraints.

    Args:
      op: The op to colocate all created ops with, or `None`.
      ignore_existing: If true, only applies colocation of this op within the
        context, rather than applying all colocation properties on the stack.
        If `op` is `None`, this value must be `True`.

    Raises:
      ValueError: if op is None but ignore_existing is False.

    Yields:
      A context manager that specifies the op with which to colocate
      newly created ops.
    """
    if op is None and not ignore_existing:
      raise ValueError("Trying to reset colocation (op is None) but "
                       "ignore_existing is not True")
    op, device_only_candidate = _op_to_colocate_with(op, self)

    # By default, colocate_with resets the device function stack,
    # since colocate_with is typically used in specific internal
    # library functions where colocation is intended to be "stronger"
    # than device functions.
    #
    # In the future, a caller may specify that device_functions win
    # over colocation, in which case we can add support.
    device_fn_tmp = self._device_function_stack
    self._device_function_stack = traceable_stack.TraceableStack()

    if ignore_existing:
      current_stack = self._colocation_stack
      self._colocation_stack = traceable_stack.TraceableStack()

    if op is not None:
      # offset refers to the stack frame used for storing code location.
      # We use 4, the sum of 1 to use our caller's stack frame and 3
      # to jump over layers of context managers above us.
      self._colocation_stack.push_obj(op, offset=4)
      if device_only_candidate is not None:
        self._colocation_stack.push_obj(device_only_candidate, offset=4)
    elif not ignore_existing:
      raise ValueError("Trying to reset colocation (op is None) but "
                       "ignore_existing is not True")
    try:
      yield
    finally:
      # Restore device function stack
      self._device_function_stack = device_fn_tmp
      if op is not None:
        self._colocation_stack.pop_obj()
        if device_only_candidate is not None:
          self._colocation_stack.pop_obj()

      # Reset the colocation stack if requested.
      if ignore_existing:
        self._colocation_stack = current_stack

  def _add_device_to_stack(
      self, device_name_or_function, offset=0,
  ) -> _UserDeviceSpec:
    """Add device to stack manually, separate from a context manager."""
    total_offset = 1 + offset
    spec = _UserDeviceSpec(device_name_or_function)
    self._device_function_stack.push_obj(spec, offset=total_offset)
    return spec

  @tf_contextlib.contextmanager
  def device(self, device_name_or_function) -> Iterator[None]:
    # pylint: disable=line-too-long
    """Returns a context manager that specifies the default device to use.

    The `device_name_or_function` argument may either be a device name
    string, a device function, or None:

    * If it is a device name string, all operations constructed in
      this context will be assigned to the device with that name, unless
      overridden by a nested `device()` context.
    * If it is a function, it will be treated as a function from
      Operation objects to device name strings, and invoked each time
      a new Operation is created. The Operation will be assigned to
      the device with the returned name.
    * If it is None, all `device()` invocations from the enclosing context
      will be ignored.

    For information about the valid syntax of device name strings, see
    the documentation in
    [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).

    For example:

    ```python
    with g.device('/device:GPU:0'):
      # All operations constructed in this context will be placed
      # on GPU 0.
      with g.device(None):
        # All operations constructed in this context will have no
        # assigned device.

    # Defines a function from `Operation` to device string.
    def matmul_on_gpu(n):
      if n.type == "MatMul":
        return "/device:GPU:0"
      else:
        return "/cpu:0"

    with g.device(matmul_on_gpu):
      # All operations of type "MatMul" constructed in this context
      # will be placed on GPU 0; all other operations will be placed
      # on CPU 0.
    ```

    **N.B.** The device scope may be overridden by op wrappers or
    other library code. For example, a variable assignment op
    `v.assign()` must be colocated with the `tf.Variable` `v`, and
    incompatible device scopes will be ignored.

    Args:
      device_name_or_function: The device name or function to use in the
        context.

    Yields:
      A context manager that specifies the default device to use for newly
      created ops.

    Raises:
      RuntimeError: If device scopes are not properly nested.
    """
    self._add_device_to_stack(device_name_or_function, offset=2)
    old_top_of_stack = self._device_function_stack.peek_top_obj()
    try:
      yield
    finally:
      new_top_of_stack = self._device_function_stack.peek_top_obj()
      if old_top_of_stack is not new_top_of_stack:
        raise RuntimeError("Exiting device scope without proper scope nesting.")
      self._device_function_stack.pop_obj()

  def _apply_device_functions(self, op) -> None:
    """Applies the current device function stack to the given operation."""
    # Apply any device functions in LIFO order, so that the most recently
    # pushed function has the first chance to apply a device to the op.
    # We apply here because the result can depend on the Operation's
    # signature, which is computed in the Operation constructor.
    # pylint: disable=protected-access
    prior_device_string = None
    for device_spec in self._device_function_stack.peek_objs():
      if device_spec.is_null_merge:
        continue

      if device_spec.function is None:
        break

      device_string = device_spec.string_merge(op)

      # Take advantage of the fact that None is a singleton and Python interns
      # strings, since identity checks are faster than equality checks.
      if device_string is not prior_device_string:
        op._set_device_from_string(device_string)
        prior_device_string = device_string
    op._device_code_locations = self._snapshot_device_function_stack_metadata()
    # pylint: enable=protected-access

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def container(self, container_name) -> Iterator[str]:
    """Returns a context manager that specifies the resource container to use.

    Stateful operations, such as variables and queues, can maintain their
    states on devices so that they can be shared by multiple processes.
    A resource container is a string name under which these stateful
    operations are tracked. These resources can be released or cleared
    with `tf.Session.reset()`.

    For example:

    ```python
    with g.container('experiment0'):
      # All stateful Operations constructed in this context will be placed
      # in resource container "experiment0".
      v1 = tf.Variable([1.0])
      v2 = tf.Variable([2.0])
      with g.container("experiment1"):
        # All stateful Operations constructed in this context will be
        # placed in resource container "experiment1".
        v3 = tf.Variable([3.0])
        q1 = tf.queue.FIFOQueue(10, tf.float32)
      # All stateful Operations constructed in this context will be
      # be created in the "experiment0".
      v4 = tf.Variable([4.0])
      q1 = tf.queue.FIFOQueue(20, tf.float32)
      with g.container(""):
        # All stateful Operations constructed in this context will be
        # be placed in the default resource container.
        v5 = tf.Variable([5.0])
        q3 = tf.queue.FIFOQueue(30, tf.float32)

    # Resets container "experiment0", after which the state of v1, v2, v4, q1
    # will become undefined (such as uninitialized).
    tf.Session.reset(target, ["experiment0"])
    ```

    Args:
      container_name: container name string.

    Returns:
      A context manager for defining resource containers for stateful ops,
        yields the container name.
    """
    original_container = self._container
    self._container = container_name
    try:
      yield self._container
    finally:
      self._container = original_container

  # pylint: enable=g-doc-return-or-yield

  class _ControlDependenciesController(object):
    """Context manager for `control_dependencies()`."""

    def __init__(self, graph, control_inputs) -> None:
      """Create a new `_ControlDependenciesController`.

      A `_ControlDependenciesController` is the context manager for
      `with tf.control_dependencies()` blocks.  These normally nest,
      as described in the documentation for `control_dependencies()`.

      The `control_inputs` argument list control dependencies that must be
      added to the current set of control dependencies.  Because of
      uniquification the set can be empty even if the caller passed a list of
      ops.  The special value `None` indicates that we want to start a new
      empty set of control dependencies instead of extending the current set.

      In that case we also clear the current control flow context, which is an
      additional mechanism to add control dependencies.

      Args:
        graph: The graph that this controller is managing.
        control_inputs: List of ops to use as control inputs in addition to the
          current control dependencies.  None to indicate that the dependencies
          should be cleared.
      """
      self._graph = graph
      if control_inputs is None:
        self._control_inputs_val = []
        self._new_stack = True
      else:
        self._control_inputs_val = control_inputs
        self._new_stack = False
      self._seen_nodes = set()
      self._old_stack = None
      self._old_control_flow_context = None

    # pylint: disable=protected-access

    def __enter__(self) -> None:
      if self._new_stack:
        # Clear the control_dependencies graph.
        self._old_stack = self._graph._control_dependencies_stack
        self._graph._control_dependencies_stack = []
        # Clear the control_flow_context too.
        self._old_control_flow_context = self._graph._get_control_flow_context()
        self._graph._set_control_flow_context(None)
      self._graph._push_control_dependencies_controller(self)

    def __exit__(self, unused_type, unused_value, unused_traceback) -> None:
      self._graph._pop_control_dependencies_controller(self)
      if self._new_stack:
        self._graph._control_dependencies_stack = self._old_stack
        self._graph._set_control_flow_context(self._old_control_flow_context)

    # pylint: enable=protected-access

    @property
    def control_inputs(self):
      return self._control_inputs_val

    def add_op(self, op) -> None:
      if isinstance(op, tensor_lib.Tensor):
        op = op.ref()
      self._seen_nodes.add(op)

    def op_in_group(self, op) -> bool:
      if isinstance(op, tensor_lib.Tensor):
        op = op.ref()
      return op in self._seen_nodes

  def _push_control_dependencies_controller(self, controller) -> None:
    self._control_dependencies_stack.append(controller)

  def _pop_control_dependencies_controller(self, controller) -> None:
    assert self._control_dependencies_stack[-1] is controller
    self._control_dependencies_stack.pop()

  def _current_control_dependencies(self) -> set[Operation]:
    ret = set()
    for controller in self._control_dependencies_stack:
      for op in controller.control_inputs:
        ret.add(op)
    return ret

  def _control_dependencies_for_inputs(self, input_ops) -> list[Operation]:
    """For an op that takes `input_ops` as inputs, compute control inputs.

    The returned control dependencies should yield an execution that
    is equivalent to adding all control inputs in
    self._control_dependencies_stack to a newly created op. However,
    this function attempts to prune the returned control dependencies
    by observing that nodes created within the same `with
    control_dependencies(...):` block may have data dependencies that make
    the explicit approach redundant.

    Args:
      input_ops: The data input ops for an op to be created.

    Returns:
      A list of control inputs for the op to be created.
    """
    ret = []
    for controller in self._control_dependencies_stack:
      # If any of the input_ops already depends on the inputs from controller,
      # we say that the new op is dominated (by that input), and we therefore
      # do not need to add control dependencies for this controller's inputs.
      dominated = False
      for op in input_ops:
        if controller.op_in_group(op):
          dominated = True
          break
      if not dominated:
        # Don't add a control input if we already have a data dependency on i.
        # NOTE(mrry): We do not currently track transitive data dependencies,
        #   so we may add redundant control inputs.
        ret.extend(c for c in controller.control_inputs if c not in input_ops)
    return ret

  def _record_op_seen_by_control_dependencies(self, op) -> None:
    """Record that the given op depends on all registered control dependencies.

    Args:
      op: An Operation.
    """
    for controller in self._control_dependencies_stack:
      controller.add_op(op)

  def control_dependencies(
      self, control_inputs,
  ) -> _ControlDependenciesController:
    """Returns a context manager that specifies control dependencies.

    Use with the `with` keyword to specify that all operations constructed
    within the context should have control dependencies on
    `control_inputs`. For example:

    ```python
    with g.control_dependencies([a, b, c]):
      # `d` and `e` will only run after `a`, `b`, and `c` have executed.
      d = ...
      e = ...
    ```

    Multiple calls to `control_dependencies()` can be nested, and in
    that case a new `Operation` will have control dependencies on the union
    of `control_inputs` from all active contexts.

    ```python
    with g.control_dependencies([a, b]):
      # Ops constructed here run after `a` and `b`.
      with g.control_dependencies([c, d]):
        # Ops constructed here run after `a`, `b`, `c`, and `d`.
    ```

    You can pass None to clear the control dependencies:

    ```python
    with g.control_dependencies([a, b]):
      # Ops constructed here run after `a` and `b`.
      with g.control_dependencies(None):
        # Ops constructed here run normally, not waiting for either `a` or `b`.
        with g.control_dependencies([c, d]):
          # Ops constructed here run after `c` and `d`, also not waiting
          # for either `a` or `b`.
    ```

    *N.B.* The control dependencies context applies *only* to ops that
    are constructed within the context. Merely using an op or tensor
    in the context does not add a control dependency. The following
    example illustrates this point:

    ```python
    # WRONG
    def my_func(pred, tensor):
      t = tf.matmul(tensor, tensor)
      with tf.control_dependencies([pred]):
        # The matmul op is created outside the context, so no control
        # dependency will be added.
        return t

    # RIGHT
    def my_func(pred, tensor):
      with tf.control_dependencies([pred]):
        # The matmul op is created in the context, so a control dependency
        # will be added.
        return tf.matmul(tensor, tensor)
    ```

    Also note that though execution of ops created under this scope will trigger
    execution of the dependencies, the ops created under this scope might still
    be pruned from a normal tensorflow graph. For example, in the following
    snippet of code the dependencies are never executed:

    ```python
      loss = model.loss()
      with tf.control_dependencies(dependencies):
        loss = loss + tf.constant(1)  # note: dependencies ignored in the
                                      # backward pass
      return tf.gradients(loss, model.variables)
    ```

    This is because evaluating the gradient graph does not require evaluating
    the constant(1) op created in the forward pass.

    Args:
      control_inputs: A list of `Operation` or `Tensor` objects which must be
        executed or computed before running the operations defined in the
        context.  Can also be `None` to clear the control dependencies.

    Returns:
     A context manager that specifies control dependencies for all
     operations constructed within the context.

    Raises:
      TypeError: If `control_inputs` is not a list of `Operation` or
        `Tensor` objects.
    """
    if control_inputs is None:
      return self._ControlDependenciesController(self, None)
    # First convert the inputs to ops, and deduplicate them.
    # NOTE(mrry): Other than deduplication, we do not currently track direct
    #   or indirect dependencies between control_inputs, which may result in
    #   redundant control inputs.
    control_ops = []
    current = self._current_control_dependencies()
    for c in control_inputs:
      # The hasattr(handle) is designed to match ResourceVariables. This is so
      # control dependencies on a variable or on an unread variable don't
      # trigger reads.
      if (isinstance(c, internal.IndexedSlices) or
          (hasattr(c, "_handle") and hasattr(c, "op"))):
        c = c.op
      c = self.as_graph_element(c)
      if isinstance(c, tensor_lib.Tensor):
        c = c.op  # pytype: disable=attribute-error
      elif not isinstance(c, Operation):
        raise TypeError("Control input must be Operation or Tensor: %s" % c)
      if c not in current:
        control_ops.append(c)
        current.add(c)
        # Mark this op with an attribute indicating that it is used as a manual
        # control dep in order to allow tracking how common utilization of
        # manual control deps in graphs run through the MLIR Bridge are. See
        # go/manual-control-dependencies-bridge for details.
        # pylint: disable=protected-access
        c._set_attr("_has_manual_control_dependencies",  # pytype: disable=attribute-error
                    attr_value_pb2.AttrValue(b=True))
        # pylint: enable=protected-access
    return self._ControlDependenciesController(self, control_ops)

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _attr_scope(self, attr_map) -> Iterator[None]:
    """EXPERIMENTAL: A context manager for setting attributes on operators.

    This context manager can be used to add additional
    attributes to operators within the scope of the context.

    For example:

       with ops.Graph().as_default() as g:
         f_1 = Foo()  # No extra attributes
         with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=False)}):
           f_2 = Foo()  # Additional attribute _a=False
           with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=True)}):
             f_3 = Foo()  # Additional attribute _a=False
             with g._attr_scope({"_a": None}):
               f_4 = Foo()  # No additional attributes.

    Args:
      attr_map: A dictionary mapping attr name strings to AttrValue protocol
        buffers or None.

    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.

    Raises:
      TypeError: If attr_map is not a dictionary mapping
        strings to AttrValue protobufs.
    """
    if not isinstance(attr_map, dict):
      raise TypeError("attr_map must be a dictionary mapping "
                      "strings to AttrValue protocol buffers")
    # The saved_attrs dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_attrs = {}
    # Install the given attribute
    for name, attr in attr_map.items():
      if not (isinstance(name, str) and
              (isinstance(attr, (type(None), attr_value_pb2.AttrValue)) or
               callable(attr))):
        raise TypeError("attr_map must be a dictionary mapping "
                        "strings to AttrValue protocol buffers or "
                        "callables that emit AttrValue protocol buffers")
      try:
        saved_attrs[name] = self._attr_scope_map[name]
      except KeyError:
        pass
      if attr is None:
        del self._attr_scope_map[name]
      else:
        self._attr_scope_map[name] = attr
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the attributes set for this context, and restore any saved
      # attributes.
      for name, attr in attr_map.items():
        try:
          self._attr_scope_map[name] = saved_attrs[name]
        except KeyError:
          del self._attr_scope_map[name]

  # pylint: enable=g-doc-return-or-yield

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _kernel_label_map(self, op_to_kernel_label_map) -> Iterator[None]:
    """EXPERIMENTAL: A context manager for setting kernel labels.

    This context manager can be used to select particular
    implementations of kernels within the scope of the context.

    For example:

        with ops.Graph().as_default() as g:
          f_1 = Foo()  # Uses the default registered kernel for the Foo op.
          with g.kernel_label_map({"Foo": "v_2"}):
            f_2 = Foo()  # Uses the registered kernel with label "v_2"
                         # for the Foo op.
            with g.kernel_label_map({"Foo": "v_3"}):
              f_3 = Foo()  # Uses the registered kernel with label "v_3"
                           # for the Foo op.
              with g.kernel_label_map({"Foo": ""}):
                f_4 = Foo()  # Uses the default registered kernel
                             # for the Foo op.

    Args:
      op_to_kernel_label_map: A dictionary mapping op type strings to kernel
        label strings.

    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.

    Raises:
      TypeError: If op_to_kernel_label_map is not a dictionary mapping
        strings to strings.
    """
    if not isinstance(op_to_kernel_label_map, dict):
      raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_labels dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_labels = {}
    # Install the given label
    for op_type, label in op_to_kernel_label_map.items():
      if not (isinstance(op_type, str) and
              isinstance(label, str)):
        raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_labels[op_type] = self._op_to_kernel_label_map[op_type]
      except KeyError:
        pass
      self._op_to_kernel_label_map[op_type] = label
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, label in op_to_kernel_label_map.items():
        try:
          self._op_to_kernel_label_map[op_type] = saved_labels[op_type]
        except KeyError:
          del self._op_to_kernel_label_map[op_type]

  # pylint: enable=g-doc-return-or-yield

  @tf_contextlib.contextmanager
  def _override_gradient_function(
      self, gradient_function_map,
  ) -> Iterator[None]:
    """Specify gradient function for the given op type."""

    # This is an internal API and we don't need nested context for this.
    # TODO(mdan): make it a proper context manager.
    assert not self._gradient_function_map
    self._gradient_function_map = gradient_function_map
    try:
      yield
    finally:
      self._gradient_function_map = {}

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def gradient_override_map(self, op_type_map) -> Iterator[None]:
    """EXPERIMENTAL: A context manager for overriding gradient functions.

    This context manager can be used to override the gradient function
    that will be used for ops within the scope of the context.

    For example:

    ```python
    @tf.RegisterGradient("CustomSquare")
    def _custom_square_grad(op, grad):
      # ...

    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      s_1 = tf.square(c)  # Uses the default gradient for tf.square.
      with g.gradient_override_map({"Square": "CustomSquare"}):
        s_2 = tf.square(s_2)  # Uses _custom_square_grad to compute the
                              # gradient of s_2.
    ```

    Args:
      op_type_map: A dictionary mapping op type strings to alternative op type
        strings.

    Returns:
      A context manager that sets the alternative op type to be used for one
      or more ops created in that context.

    Raises:
      TypeError: If `op_type_map` is not a dictionary mapping strings to
        strings.
    """
    if not isinstance(op_type_map, dict):
      raise TypeError("op_type_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_mappings dictionary stores any currently-set mappings that
    # will be overridden by this context manager.
    saved_mappings = {}
    # Install the given label
    for op_type, mapped_op_type in op_type_map.items():
      if not (isinstance(op_type, str) and
              isinstance(mapped_op_type, str)):
        raise TypeError("op_type_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_mappings[op_type] = self._gradient_override_map[op_type]
      except KeyError:
        pass
      self._gradient_override_map[op_type] = mapped_op_type
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, mapped_op_type in op_type_map.items():
        try:
          self._gradient_override_map[op_type] = saved_mappings[op_type]
        except KeyError:
          del self._gradient_override_map[op_type]

  # pylint: enable=g-doc-return-or-yield

  def prevent_feeding(self, tensor) -> None:
    """Marks the given `tensor` as unfeedable in this graph."""
    self._unfeedable_tensors.add(tensor)

  def is_feedable(self, tensor) -> bool:
    """Returns `True` if and only if `tensor` is feedable."""
    return tensor not in self._unfeedable_tensors

  def prevent_fetching(self, op) -> None:
    """Marks the given `op` as unfetchable in this graph."""
    self._unfetchable_ops.add(op)

  def is_fetchable(self, tensor_or_op) -> bool:
    """Returns `True` if and only if `tensor_or_op` is fetchable."""
    if isinstance(tensor_or_op, tensor_lib.Tensor):
      return tensor_or_op.op not in self._unfetchable_ops
    else:
      return tensor_or_op not in self._unfetchable_ops

  def switch_to_thread_local(self) -> None:
    """Make device, colocation and dependencies stacks thread-local.

    Device, colocation and dependencies stacks are not thread-local be default.
    If multiple threads access them, then the state is shared.  This means that
    one thread may affect the behavior of another thread.

    After this method is called, the stacks become thread-local.  If multiple
    threads access them, then the state is not shared.  Each thread uses its own
    value; a thread doesn't affect other threads by mutating such a stack.

    The initial value for every thread's stack is set to the current value
    of the stack when `switch_to_thread_local()` was first called.
    """
    if not self._stack_state_is_thread_local:
      self._stack_state_is_thread_local = True

  @property
  def _device_function_stack(self) -> traceable_stack.TraceableStack:
    if self._stack_state_is_thread_local:
      # This may be called from a thread where device_function_stack doesn't yet
      # exist.
      # pylint: disable=protected-access
      if not hasattr(self._thread_local, "_device_function_stack"):
        stack_copy_for_this_thread = self._graph_device_function_stack.copy()
        self._thread_local._device_function_stack = stack_copy_for_this_thread
      return self._thread_local._device_function_stack
      # pylint: enable=protected-access
    else:
      return self._graph_device_function_stack

  @property
  def _device_functions_outer_to_inner(self):
    user_device_specs = self._device_function_stack.peek_objs()
    device_functions = [spec.function for spec in user_device_specs]
    device_functions_outer_to_inner = list(reversed(device_functions))
    return device_functions_outer_to_inner

  def _snapshot_device_function_stack_metadata(
      self,
  ) -> list[traceable_stack.TraceableObject]:
    """Return device function stack as a list of TraceableObjects.

    Returns:
      [traceable_stack.TraceableObject, ...] where each TraceableObject's .obj
      member is a displayable name for the user's argument to Graph.device, and
      the filename and lineno members point to the code location where
      Graph.device was called directly or indirectly by the user.
    """
    snapshot = []
    for obj in self._device_function_stack.peek_traceable_objs():
      obj_copy = obj.copy_metadata()
      obj_copy.obj = obj.obj.display_name
      snapshot.append(obj_copy)
    return snapshot

  @_device_function_stack.setter
  def _device_function_stack(
      self, device_function_stack: traceable_stack.TraceableStack,
  ) -> None:
    if self._stack_state_is_thread_local:
      # pylint: disable=protected-access
      self._thread_local._device_function_stack = device_function_stack
      # pylint: enable=protected-access
    else:
      self._graph_device_function_stack = device_function_stack

  @property
  def _colocation_stack(self) -> traceable_stack.TraceableStack:
    """Return thread-local copy of colocation stack."""
    if self._stack_state_is_thread_local:
      # This may be called from a thread where colocation_stack doesn't yet
      # exist.
      # pylint: disable=protected-access
      if not hasattr(self._thread_local, "_colocation_stack"):
        stack_copy_for_this_thread = self._graph_colocation_stack.copy()
        self._thread_local._colocation_stack = stack_copy_for_this_thread
      return self._thread_local._colocation_stack
      # pylint: enable=protected-access
    else:
      return self._graph_colocation_stack

  def _snapshot_colocation_stack_metadata(
      self,
  ) -> dict[str, traceable_stack.TraceableObject]:
    """Return colocation stack metadata as a dictionary."""
    return {
        traceable_obj.obj.name: traceable_obj.copy_metadata()
        for traceable_obj in self._colocation_stack.peek_traceable_objs()
    }

  @_colocation_stack.setter
  def _colocation_stack(
      self, colocation_stack: traceable_stack.TraceableStack,
  ) -> None:
    if self._stack_state_is_thread_local:
      # pylint: disable=protected-access
      self._thread_local._colocation_stack = colocation_stack
      # pylint: enable=protected-access
    else:
      self._graph_colocation_stack = colocation_stack

  @property
  def _control_dependencies_stack(self) -> list[_ControlDependenciesController]:
    if self._stack_state_is_thread_local:
      # This may be called from a thread where control_dependencies_stack
      # doesn't yet exist.
      if not hasattr(self._thread_local, "_control_dependencies_stack"):
        self._thread_local._control_dependencies_stack = (
            self._graph_control_dependencies_stack[:])
      return self._thread_local._control_dependencies_stack
    else:
      return self._graph_control_dependencies_stack

  @_control_dependencies_stack.setter
  def _control_dependencies_stack(
      self,
      control_deps_controllers: list[_ControlDependenciesController],
  ) -> None:
    if self._stack_state_is_thread_local:
      self._thread_local._control_dependencies_stack = control_deps_controllers
    else:
      self._graph_control_dependencies_stack = control_deps_controllers

  @property
  def _distribution_strategy_stack(self) -> list[Any]:
    """A stack to maintain distribution strategy context for each thread."""
    if not hasattr(self._thread_local, "_distribution_strategy_stack"):
      self._thread_local._distribution_strategy_stack = []  # pylint: disable=protected-access
    return self._thread_local._distribution_strategy_stack  # pylint: disable=protected-access

  @_distribution_strategy_stack.setter
  def _distribution_strategy_stack(
      self, _distribution_strategy_stack: list[Any],
  ) -> None:
    self._thread_local._distribution_strategy_stack = (  # pylint: disable=protected-access
        _distribution_strategy_stack)

  @property
  def _global_distribute_strategy_scope(self):
    """For implementing `tf.distribute.set_strategy()`."""
    if not hasattr(self._thread_local, "distribute_strategy_scope"):
      self._thread_local.distribute_strategy_scope = None
    return self._thread_local.distribute_strategy_scope

  @_global_distribute_strategy_scope.setter
  def _global_distribute_strategy_scope(self, distribute_strategy_scope):
    self._thread_local.distribute_strategy_scope = (distribute_strategy_scope)

  def _mutation_lock(self) -> lock_util.GroupLock._Context:
    """Returns a lock to guard code that creates & mutates ops.

    See the comment for self._group_lock for more info.
    """
    return self._group_lock.group(_MUTATION_LOCK_GROUP)

  def _session_run_lock(self) -> lock_util.GroupLock._Context:
    """Returns a lock to guard code for Session.run.

    See the comment for self._group_lock for more info.
    """
    return self._group_lock.group(_SESSION_RUN_LOCK_GROUP)


# TODO(agarwal): currently device directives in an outer eager scope will not
# apply to inner graph mode code. Fix that.


@tf_export(v1=["device"])
def device(device_name_or_function) -> ContextManager[None]:
  """Wrapper for `Graph.device()` using the default graph.

  See `tf.Graph.device` for more details.

  Args:
    device_name_or_function: The device name or function to use in the context.

  Returns:
    A context manager that specifies the default device to use for newly
    created ops.

  Raises:
    RuntimeError: If eager execution is enabled and a function is passed in.
  """
  if context.executing_eagerly():
    if callable(device_name_or_function):
      raise RuntimeError(
          "tf.device does not support functions when eager execution "
          "is enabled.")
    return context.device(device_name_or_function)
  elif executing_eagerly_outside_functions():
    @tf_contextlib.contextmanager
    def combined(device_name_or_function):
      with get_default_graph().device(device_name_or_function):
        if not callable(device_name_or_function):
          with context.device(device_name_or_function):
            yield
        else:
          yield
    return combined(device_name_or_function)
  else:
    return get_default_graph().device(device_name_or_function)


@tf_export("device", v1=[])
def device_v2(device_name) -> ContextManager[None]:
  """Specifies the device for ops created/executed in this context.

  This function specifies the device to be used for ops created/executed in a
  particular context. Nested contexts will inherit and also create/execute
  their ops on the specified device. If a specific device is not required,
  consider not using this function so that a device can be automatically
  assigned.  In general the use of this function is optional. `device_name` can
  be fully specified, as in "/job:worker/task:1/device:cpu:0", or partially
  specified, containing only a subset of the "/"-separated fields. Any fields
  which are specified will override device annotations from outer scopes.

  For example:

  ```python
  with tf.device('/job:foo'):
    # ops created here have devices with /job:foo
    with tf.device('/job:bar/task:0/device:gpu:2'):
      # ops created here have the fully specified device above
    with tf.device('/device:gpu:1'):
      # ops created here have the device '/job:foo/device:gpu:1'
  ```

  Args:
    device_name: The device name to use in the context.

  Returns:
    A context manager that specifies the default device to use for newly
    created ops.

  Raises:
    RuntimeError: If a function is passed in.
  """
  if callable(device_name):
    raise RuntimeError("tf.device does not support functions.")
  return device(device_name)


@tf_export(v1=["container"])
def container(container_name) -> ContextManager[str]:
  """Wrapper for `Graph.container()` using the default graph.

  Args:
    container_name: The container string to use in the context.

  Returns:
    A context manager that specifies the default container to use for newly
    created stateful ops.
  """
  return get_default_graph().container(container_name)


def _colocate_with_for_gradient(
    op, gradient_uid, ignore_existing=False,
) -> ContextManager[None]:
  """Returns a context manager for colocating op gradients with an op.

  Internal API. In eager mode, returns a context manager that sets the default
  device for new ops to the same device as the given op. Does the same if a
  function is currently being built (i.e. the current mode is graph, but the
  overall mode is eager).

  In all other cases, returns a `Graph.colocate_with()` context manager,
  optionally accounting for gradients (if a gradient UID is specified).

  Args:
    op: Operation or Tensor with which to colocate.
    gradient_uid: Optional gradient UID to enable colocation of gradients during
      compilation.
    ignore_existing: See `Graph.colocate_with()`.

  Returns:
    A context manager used to colocate ops and gradients with the specified
    operation.
  """
  if context.executing_eagerly():
    if op is not None:
      if not hasattr(op, "device"):
        op = convert_to_tensor(op)
      return device(op.device)
    else:
      return NullContextmanager()
  else:
    default_graph = get_default_graph()
    if isinstance(op, EagerTensor):
      if default_graph.building_function:
        return default_graph.device(op.device)
      else:
        raise ValueError("Encountered an Eager-defined Tensor during graph "
                         "construction, but a function was not being built.")
    return default_graph._colocate_with_for_gradient(
        op, gradient_uid=gradient_uid, ignore_existing=ignore_existing)


# Internal interface to colocate_with. colocate_with has been deprecated from
# public API. There are still a few internal uses of colocate_with. Add internal
# only API for those uses to avoid deprecation warning.
def colocate_with(op, ignore_existing=False) -> ContextManager[None]:
  return _colocate_with_for_gradient(op, None, ignore_existing=ignore_existing)


@deprecation.deprecated(
    date=None, instructions="Colocations handled automatically by placer.")
@tf_export(v1=["colocate_with"])
def _colocate_with(op, ignore_existing=False) -> ContextManager[None]:
  return colocate_with(op, ignore_existing)


@tf_export("control_dependencies")
def control_dependencies(
    control_inputs,
) -> Graph._ControlDependenciesController:
  """Wrapper for `Graph.control_dependencies()` using the default graph.

  See `tf.Graph.control_dependencies` for more details.

  In TensorFlow 2 with eager and/or Autograph, you should not need this method
  most of the times, as ops execute in the expected order thanks to automatic
  control dependencies. Only use it to manually control ordering, for example as
  a workaround to known issues such as `tf.function` with `tf.debugging.assert*`
  and `tf.py_function`.
  For example:

  >>> @tf.function(
  ...   input_signature=[tf.TensorSpec([None, None], tf.float32),
  ...                    tf.TensorSpec([None, None], tf.float32)])
  ... def my_assert_func_1(x, bias):
  ...   # `tf.function` attempts to execute `tf.math.add` in parallel to
  ...   # `assert_equal`. As a result an error can get raised from `tf.math.add`
  ...   # without triggering the assertion error.
  ...   tf.assert_equal(tf.shape(x)[1],
  ...                   tf.shape(bias)[1],
  ...                   message='bad shape')
  ...   return x + bias

  >>> # Error raised in either `add` or `assert`
  >>> my_assert_func_1(tf.ones((2, 5)), tf.ones((2, 7)))
  Traceback (most recent call last):
     ...
  InvalidArgumentError: ...


  >>> @tf.function(
  ...   input_signature=[tf.TensorSpec([None, None], tf.float32),
  ...                    tf.TensorSpec([None, None], tf.float32)])
  ... def my_assert_func_2(x, bias):
  ...   with tf.control_dependencies(
  ...       [tf.assert_equal(tf.shape(x)[1],
  ...                       tf.shape(bias)[1],
  ...                       message='bad shape')]):
  ...     return x + bias

  >>> # Error raised in `assert`
  >>> my_assert_func_2(tf.ones((2, 5)), tf.ones((2, 7)))
  Traceback (most recent call last):
     ...
  InvalidArgumentError: ...

  When eager execution is enabled, any callable object in the `control_inputs`
  list will be called.

  Args:
    control_inputs: A list of `Operation` or `Tensor` objects which must be
      executed or computed before running the operations defined in the context.
      Can also be `None` to clear the control dependencies. If eager execution
      is enabled, any callable object in the `control_inputs` list will be
      called.

  Returns:
   A context manager that specifies control dependencies for all
   operations constructed within the context.
  """
  if context.executing_eagerly():
    if control_inputs:
      # Execute any pending callables.
      for control in control_inputs:
        if callable(control):
          control()
    return NullContextmanager()
  else:
    return get_default_graph().control_dependencies(control_inputs)

# TODO(b/271463878): Remove in favor of direct references to `stack`.
get_default_session = stack.get_default_session


def _run_using_default_session(
    operation, feed_dict, graph, session=None) -> None:
  """Uses the default session to run "operation".

  Args:
    operation: The Operation to be run.
    feed_dict: A dictionary that maps Tensor objects (or tensor names) to lists,
      numpy ndarrays, TensorProtos, or strings.
    graph: The graph in which "operation" is defined.
    session: (Optional) A different session to use to run "operation".

  Raises:
    ValueError: If no default session is available; the default session
      does not have "graph" as its graph; or if "session" is specified,
      and it does not have "graph" as its graph.
  """
  if session is None:
    session = stack.get_default_session()
    if session is None:
      raise ValueError("Cannot execute operation using `run()`: No default "
                       "session is registered. Use `with "
                       "sess.as_default():` or pass an explicit session to "
                       "`run(session=sess)`")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to execute operation: "
                       "the operation's graph is different from the "
                       "session's graph. Pass an explicit session to "
                       "run(session=sess).")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to execute operation: "
                       "the operation's graph is different from the session's "
                       "graph.")
  session.run(operation, feed_dict)


class _DefaultGraphStack(stack.DefaultStack[Graph]):  # pylint: disable=protected-access
  """A thread-local stack of objects for providing an implicit default graph."""

  def __init__(self) -> None:
    super(_DefaultGraphStack, self).__init__()
    self._global_default_graph = None

  def get_default(self) -> Graph:
    """Override that returns a global default if the stack is empty."""
    if self.stack:
      return self.stack[-1]
    elif self._global_default_graph:
      return self._global_default_graph
    else:
      self._global_default_graph = Graph()
      return self._global_default_graph

  def _GetGlobalDefaultGraph(self) -> Graph:
    if self._global_default_graph is None:
      # TODO(mrry): Perhaps log that the default graph is being used, or set
      #   provide some other feedback to prevent confusion when a mixture of
      #   the global default graph and an explicit graph are combined in the
      #   same process.
      self._global_default_graph = Graph()
    return self._global_default_graph

  def reset(self) -> None:
    super(_DefaultGraphStack, self).reset()
    self._global_default_graph = None

  @tf_contextlib.contextmanager
  def get_controller(self, default) -> Iterator[Graph]:
    context.context().context_switches.push(default.building_function,
                                            default.as_default,
                                            default._device_function_stack)
    try:
      with super(_DefaultGraphStack,
                 self).get_controller(default) as g, context.graph_mode():  # pytype: disable=wrong-arg-count
        yield g
    finally:
      # If an exception is raised here it may be hiding a related exception in
      # the try-block (just above).
      context.context().context_switches.pop()


_default_graph_stack: _DefaultGraphStack = _DefaultGraphStack()


# Shared helper used in init_scope and executing_eagerly_outside_functions
# to obtain the outermost context that is not building a function, and the
# innermost non empty device stack.
def _get_outer_context_and_inner_device_stack(
) -> tuple[Callable[[], ContextManager[Graph]], traceable_stack.TraceableStack]:
  """Get the outermost context not building a function."""
  default_graph = get_default_graph()
  outer_context = None
  innermost_nonempty_device_stack = default_graph._device_function_stack  # pylint: disable=protected-access

  if not _default_graph_stack.stack:
    # If the default graph stack is empty, then we cannot be building a
    # function. Install the global graph (which, in this case, is also the
    # default graph) as the outer context.
    if default_graph.building_function:
      raise RuntimeError("The global graph is building a function.")
    outer_context = default_graph.as_default
  else:
    # Find a context that is not building a function.
    for stack_entry in reversed(context.context().context_switches.stack):
      if not innermost_nonempty_device_stack:
        innermost_nonempty_device_stack = stack_entry.device_stack
      if not stack_entry.is_building_function:
        outer_context = stack_entry.enter_context_fn
        break

    if outer_context is None:
      # As a last resort, obtain the global default graph; this graph doesn't
      # necessarily live on the graph stack (and hence it doesn't necessarily
      # live on the context stack), but it is stored in the graph stack's
      # encapsulating object.
      outer_context = _default_graph_stack._GetGlobalDefaultGraph().as_default  # pylint: disable=protected-access

  if outer_context is None:
    # Sanity check; this shouldn't be triggered.
    raise RuntimeError("All graphs are building functions, and no "
                       "eager context was previously active.")

  return outer_context, innermost_nonempty_device_stack


# pylint: disable=g-doc-return-or-yield,line-too-long
@tf_export("init_scope")
@tf_contextlib.contextmanager
def init_scope() -> Iterator[None]:
  """A context manager that lifts ops out of control-flow scopes and function-building graphs.

  There is often a need to lift variable initialization ops out of control-flow
  scopes, function-building graphs, and gradient tapes. Entering an
  `init_scope` is a mechanism for satisfying these desiderata. In particular,
  entering an `init_scope` has three effects:

    (1) All control dependencies are cleared the moment the scope is entered;
        this is equivalent to entering the context manager returned from
        `control_dependencies(None)`, which has the side-effect of exiting
        control-flow scopes like `tf.cond` and `tf.while_loop`.

    (2) All operations that are created while the scope is active are lifted
        into the lowest context on the `context_stack` that is not building a
        graph function. Here, a context is defined as either a graph or an eager
        context. Every context switch, i.e., every installation of a graph as
        the default graph and every switch into eager mode, is logged in a
        thread-local stack called `context_switches`; the log entry for a
        context switch is popped from the stack when the context is exited.
        Entering an `init_scope` is equivalent to crawling up
        `context_switches`, finding the first context that is not building a
        graph function, and entering it. A caveat is that if graph mode is
        enabled but the default graph stack is empty, then entering an
        `init_scope` will simply install a fresh graph as the default one.

    (3) The gradient tape is paused while the scope is active.

  When eager execution is enabled, code inside an init_scope block runs with
  eager execution enabled even when tracing a `tf.function`. For example:

  ```python
  tf.compat.v1.enable_eager_execution()

  @tf.function
  def func():
    # A function constructs TensorFlow graphs,
    # it does not execute eagerly.
    assert not tf.executing_eagerly()
    with tf.init_scope():
      # Initialization runs with eager execution enabled
      assert tf.executing_eagerly()
  ```

  Raises:
    RuntimeError: if graph state is incompatible with this initialization.
  """
  # pylint: enable=g-doc-return-or-yield,line-too-long

  if context.executing_eagerly():
    # Fastpath.
    with record.stop_recording():
      yield
  else:
    # Retrieve the active name scope: entering an `init_scope` preserves
    # the name scope of the current context.
    scope = get_default_graph().get_name_scope()
    if scope and scope[-1] != "/":
      # Names that end with trailing slashes are treated by `name_scope` as
      # absolute.
      scope = scope + "/"

    outer_context, innermost_nonempty_device_stack = (
        _get_outer_context_and_inner_device_stack())

    outer_graph = None
    outer_device_stack = None
    try:
      with outer_context(), name_scope(
          scope, skip_on_eager=False), control_dependencies(
              None), record.stop_recording():
        context_manager = NullContextmanager
        context_manager_input = None
        if not context.executing_eagerly():
          # The device stack is preserved when lifting into a graph. Eager
          # execution doesn't implement device stacks and in particular it
          # doesn't support device functions, so in general it's not possible
          # to do the same when lifting into the eager context.
          outer_graph = get_default_graph()
          outer_device_stack = outer_graph._device_function_stack  # pylint: disable=protected-access
          outer_graph._device_function_stack = innermost_nonempty_device_stack  # pylint: disable=protected-access
        elif innermost_nonempty_device_stack is not None:
          for device_spec in innermost_nonempty_device_stack.peek_objs():
            if device_spec.function is None:
              break
            if device_spec.raw_string:
              context_manager = context.device
              context_manager_input = device_spec.raw_string
              break
            # It is currently not possible to have a device function in V2,
            # but in V1 we are unable to apply device functions in eager mode.
            # This means that we will silently skip some of the entries on the
            # device stack in V1 + eager mode.

        with context_manager(context_manager_input):
          yield
    finally:
      # If an exception is raised here it may be hiding a related exception in
      # try-block (just above).
      if outer_graph is not None:
        outer_graph._device_function_stack = outer_device_stack  # pylint: disable=protected-access


@tf_export(v1=["executing_eagerly_outside_functions"])
def executing_eagerly_outside_functions() -> bool:
  """Returns True if executing eagerly, even if inside a graph function.

  This function will check the outermost context for the program and see if
  it is in eager mode. It is useful comparing to `tf.executing_eagerly()`,
  which checks the current context and will return `False` within a
  `tf.function` body. It can be used to build library that behave differently
  in eager runtime and v1 session runtime (deprecated).

  Example:

  >>> tf.compat.v1.enable_eager_execution()
  >>> @tf.function
  ... def func():
  ...   # A function constructs TensorFlow graphs, it does not execute eagerly,
  ...   # but the outer most context is still eager.
  ...   assert not tf.executing_eagerly()
  ...   return tf.compat.v1.executing_eagerly_outside_functions()
  >>> func()
  <tf.Tensor: shape=(), dtype=bool, numpy=True>

  Returns:
    boolean, whether the outermost context is in eager mode.
  """
  if context.executing_eagerly():
    return True
  else:
    outer_context, _ = _get_outer_context_and_inner_device_stack()
    with outer_context():
      return context.executing_eagerly()


@tf_export("inside_function", v1=[])
def inside_function() -> bool:
  """Indicates whether the caller code is executing inside a `tf.function`.

  Returns:
    Boolean, True if the caller code is executing inside a `tf.function`
    rather than eagerly.

  Example:

  >>> tf.inside_function()
  False
  >>> @tf.function
  ... def f():
  ...   print(tf.inside_function())
  >>> f()
  True
  """
  return get_default_graph().building_function


@tf_export(v1=["enable_eager_execution"])
def enable_eager_execution(config=None, device_policy=None,
                           execution_mode=None) -> None:
  """Enables eager execution for the lifetime of this program.

  Eager execution provides an imperative interface to TensorFlow. With eager
  execution enabled, TensorFlow functions execute operations immediately (as
  opposed to adding to a graph to be executed later in a `tf.compat.v1.Session`)
  and
  return concrete values (as opposed to symbolic references to a node in a
  computational graph).

  For example:

  ```python
  tf.compat.v1.enable_eager_execution()

  # After eager execution is enabled, operations are executed as they are
  # defined and Tensor objects hold concrete values, which can be accessed as
  # numpy.ndarray`s through the numpy() method.
  assert tf.multiply(6, 7).numpy() == 42
  ```

  Eager execution cannot be enabled after TensorFlow APIs have been used to
  create or execute graphs. It is typically recommended to invoke this function
  at program startup and not in a library (as most libraries should be usable
  both with and without eager execution).

  @compatibility(TF2)
  This function is not necessary if you are using TF2. Eager execution is
  enabled by default.
  @end_compatibility

  Args:
    config: (Optional.) A `tf.compat.v1.ConfigProto` to use to configure the
      environment in which operations are executed. Note that
      `tf.compat.v1.ConfigProto` is also used to configure graph execution (via
      `tf.compat.v1.Session`) and many options within `tf.compat.v1.ConfigProto`
      are not implemented (or are irrelevant) when eager execution is enabled.
    device_policy: (Optional.) Policy controlling how operations requiring
      inputs on a specific device (e.g., a GPU 0) handle inputs on a different
      device  (e.g. GPU 1 or CPU). When set to None, an appropriate value will
      be picked automatically. The value picked may change between TensorFlow
      releases.
      Valid values:
      - DEVICE_PLACEMENT_EXPLICIT: raises an error if the
        placement is not correct.
      - DEVICE_PLACEMENT_WARN: copies the tensors which are not
        on the right device but logs a warning.
      - DEVICE_PLACEMENT_SILENT: silently copies the tensors.
        Note that this may hide performance problems as there is no notification
        provided when operations are blocked on the tensor being copied between
        devices.
      - DEVICE_PLACEMENT_SILENT_FOR_INT32: silently copies
        int32 tensors, raising errors on the other ones.
    execution_mode: (Optional.) Policy controlling how operations dispatched are
      actually executed. When set to None, an appropriate value will be picked
      automatically. The value picked may change between TensorFlow releases.
      Valid values:
      - SYNC: executes each operation synchronously.
      - ASYNC: executes each operation asynchronously. These
        operations may return "non-ready" handles.

  Raises:
    ValueError: If eager execution is enabled after creating/executing a
     TensorFlow graph, or if options provided conflict with a previous call
     to this function.
  """
  _api_usage_gauge.get_cell().set(True)
  logging.vlog(1, "Enabling eager execution")
  if context.default_execution_mode != context.EAGER_MODE:
    return enable_eager_execution_internal(
        config=config,
        device_policy=device_policy,
        execution_mode=execution_mode,
        server_def=None)


@tf_export(v1=["disable_eager_execution"])
def disable_eager_execution() -> None:
  """Disables eager execution.

  This function can only be called before any Graphs, Ops, or Tensors have been
  created.

  @compatibility(TF2)
  This function is not necessary if you are using TF2. Eager execution is
  enabled by default. If you want to use Graph mode please consider
  [tf.function](https://www.tensorflow.org/api_docs/python/tf/function).
  @end_compatibility
  """
  _api_usage_gauge.get_cell().set(False)
  logging.vlog(1, "Disabling eager execution")
  context.default_execution_mode = context.GRAPH_MODE
  c = context.context_safe()
  if c is not None:
    c._thread_local_data.is_eager = False  # pylint: disable=protected-access


def enable_eager_execution_internal(config=None,
                                    device_policy=None,
                                    execution_mode=None,
                                    server_def=None) -> None:
  """Enables eager execution for the lifetime of this program.

  Most of the doc string for enable_eager_execution is relevant here as well.

  Args:
    config: See enable_eager_execution doc string
    device_policy: See enable_eager_execution doc string
    execution_mode: See enable_eager_execution doc string
    server_def: (Optional.) A tensorflow::ServerDef proto. Enables execution on
      remote devices. GrpcServers need to be started by creating an identical
      server_def to this, and setting the appropriate task_indexes, so that the
      servers can communicate. It will then be possible to execute operations on
      remote devices.

  Raises:
    ValueError

  """
  if config is not None and not isinstance(config, config_pb2.ConfigProto):
    raise TypeError("config must be a tf.ConfigProto, but got %s" %
                    type(config))
  if device_policy not in (None, context.DEVICE_PLACEMENT_EXPLICIT,
                           context.DEVICE_PLACEMENT_WARN,
                           context.DEVICE_PLACEMENT_SILENT,
                           context.DEVICE_PLACEMENT_SILENT_FOR_INT32):
    raise ValueError("device_policy must be one of None, DEVICE_PLACEMENT_*")
  if execution_mode not in (None, context.SYNC, context.ASYNC):
    raise ValueError("execution_mode must be one of None, SYNC, " "ASYNC")
  if context.default_execution_mode == context.GRAPH_MODE:
    graph_mode_has_been_used = (
        _default_graph_stack._global_default_graph is not None)  # pylint: disable=protected-access
    if graph_mode_has_been_used:
      raise ValueError(
          "tf.enable_eager_execution must be called at program startup.")
  context.default_execution_mode = context.EAGER_MODE
  # pylint: disable=protected-access
  with context._context_lock:
    if context._context is None:
      context._set_context_locked(context.Context(
          config=config,
          device_policy=device_policy,
          execution_mode=execution_mode,
          server_def=server_def))
    elif ((config is not None and config is not context._context._config) or
          (device_policy is not None and
           device_policy is not context._context._device_policy) or
          (execution_mode is not None and
           execution_mode is not context._context._execution_mode)):
      raise ValueError(
          "Trying to change the options of an active eager"
          " execution. Context config: %s, specified config:"
          " %s. Context device policy: %s, specified device"
          " policy: %s. Context execution mode: %s, "
          " specified execution mode %s." %
          (context._context._config, config, context._context._device_policy,
           device_policy, context._context._execution_mode, execution_mode))
    else:
      # We already created everything, so update the thread local data.
      context._context._thread_local_data.is_eager = True

  # Monkey patch to get rid of an unnecessary conditional since the context is
  # now initialized.
  context.context = context.context_safe


def eager_run(main=None, argv=None) -> NoReturn:
  """Runs the program with an optional main function and argv list.

  The program will run with eager execution enabled.

  Example:
  ```python
  import tensorflow as tf
  # Import subject to future changes:

  def main(_):
    u = tf.constant(6.0)
    v = tf.constant(7.0)
    print(u * v)

  if __name__ == "__main__":
    tfe.run()
  ```

  Args:
    main: the main function to run.
    argv: the arguments to pass to it.
  """
  enable_eager_execution()
  app.run(main, argv)


@tf_export(v1=["reset_default_graph"])
def reset_default_graph() -> None:
  """Clears the default graph stack and resets the global default graph.

  NOTE: The default graph is a property of the current thread. This
  function applies only to the current thread.  Calling this function while
  a `tf.compat.v1.Session` or `tf.compat.v1.InteractiveSession` is active will
  result in undefined
  behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
  after calling this function will result in undefined behavior.

  @compatibility(TF2)
  `reset_default_graph` does not work with either eager execution or
  `tf.function`, and you should not invoke it directly. To migrate code that
  uses Graph-related functions to TF2, rewrite the code without them. See the
  [migration guide](https://www.tensorflow.org/guide/migrate) for more
  description about the behavior and semantic changes between Tensorflow 1 and
  Tensorflow 2.
  @end_compatibility

  Raises:
    AssertionError: If this function is called within a nested graph.
  """
  if not _default_graph_stack.is_cleared():
    raise AssertionError("Do not use tf.reset_default_graph() to clear "
                         "nested graphs. If you need a cleared graph, "
                         "exit the nesting and create a new graph.")
  _default_graph_stack.reset()


@tf_export(v1=["get_default_graph"])
def get_default_graph() -> Graph:
  """Returns the default graph for the current thread.

  The returned graph will be the innermost graph on which a
  `Graph.as_default()` context has been entered, or a global default
  graph if none has been explicitly created.

  NOTE: The default graph is a property of the current thread. If you
  create a new thread, and wish to use the default graph in that
  thread, you must explicitly add a `with g.as_default():` in that
  thread's function.

  @compatibility(TF2)
  `get_default_graph` does not work with either eager execution or
  `tf.function`, and you should not invoke it directly. To migrate code that
  uses Graph-related functions to TF2, rewrite the code without them. See the
  [migration guide](https://www.tensorflow.org/guide/migrate) for more
  description about the behavior and semantic changes between Tensorflow 1 and
  Tensorflow 2.
  @end_compatibility

  Returns:
    The default `Graph` being used in the current thread.
  """
  return _default_graph_stack.get_default()


def has_default_graph() -> bool:
  """Returns True if there is a default graph."""
  return len(_default_graph_stack.stack) >= 1


# Exported due to b/171079555
@tf_export("__internal__.get_name_scope", v1=[])
def get_name_scope() -> str:
  """Returns the current name scope in the default_graph.

  For example:

  ```python
  with tf.name_scope('scope1'):
    with tf.name_scope('scope2'):
      print(tf.get_name_scope())
  ```
  would print the string `scope1/scope2`.

  Returns:
    A string representing the current name scope.
  """
  if context.executing_eagerly():
    return context.context().scope_name.rstrip("/")
  return get_default_graph().get_name_scope()


def _assert_same_graph(original_item, item) -> None:
  """Fail if the 2 items are from different graphs.

  Args:
    original_item: Original item to check against.
    item: Item to check.

  Raises:
    ValueError: if graphs do not match.
  """
  original_graph = getattr(original_item, "graph", None)
  graph = getattr(item, "graph", None)
  if original_graph and graph and original_graph is not graph:
    raise ValueError(
        "%s must be from the same graph as %s (graphs are %s and %s)." %
        (item, original_item, graph, original_graph))


def _get_graph_from_inputs(op_input_list, graph=None) -> Graph:
  """Returns the appropriate graph to use for the given inputs.

  This library method provides a consistent algorithm for choosing the graph
  in which an Operation should be constructed:

  1. If the default graph is being used to construct a function, we
     use the default graph.
  2. If the "graph" is specified explicitly, we validate that all of the inputs
     in "op_input_list" are compatible with that graph.
  3. Otherwise, we attempt to select a graph from the first Operation-
     or Tensor-valued input in "op_input_list", and validate that all other
     such inputs are in the same graph.
  4. If the graph was not specified and it could not be inferred from
     "op_input_list", we attempt to use the default graph.

  Args:
    op_input_list: A list of inputs to an operation, which may include `Tensor`,
      `Operation`, and other objects that may be converted to a graph element.
    graph: (Optional) The explicit graph to use.

  Raises:
    TypeError: If op_input_list is not a list or tuple, or if graph is not a
      Graph.
    ValueError: If a graph is explicitly passed and not all inputs are from it,
      or if the inputs are from multiple graphs, or we could not find a graph
      and there was no default graph.

  Returns:
    The appropriate graph to use for the given inputs.

  """
  current_default_graph = get_default_graph()
  if current_default_graph.building_function:
    return current_default_graph

  op_input_list = tuple(op_input_list)  # Handle generators correctly
  if graph and not isinstance(graph, Graph):
    raise TypeError("Input graph needs to be a Graph: %s" % (graph,))

  # 1. We validate that all of the inputs are from the same graph. This is
  #    either the supplied graph parameter, or the first one selected from one
  #    the graph-element-valued inputs. In the latter case, we hold onto
  #    that input in original_graph_element so we can provide a more
  #    informative error if a mismatch is found.
  original_graph_element = None
  for op_input in op_input_list:
    graph_element = None
    if isinstance(op_input, (Operation, SymbolicTensor)):
      graph_element = op_input
    else:
      graph_element = _as_graph_element(op_input)

    if graph_element is not None:
      if not graph:
        original_graph_element = graph_element
        graph = getattr(graph_element, "graph", None)
      elif original_graph_element is not None:
        _assert_same_graph(original_graph_element, graph_element)
      elif graph_element.graph is not graph:
        raise ValueError("%s is not from the passed-in graph." % graph_element)

  # 2. If all else fails, we use the default graph, which is always there.
  return graph or current_default_graph


@tf_export(v1=["GraphKeys"])
class GraphKeys(object):
  """Standard names to use for graph collections.

  The standard library uses various well-known names to collect and
  retrieve values associated with a graph. For example, the
  `tf.Optimizer` subclasses default to optimizing the variables
  collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
  specified, but it is also possible to pass an explicit list of
  variables.

  The following standard keys are defined:

  * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
    across distributed environment (model variables are subset of these). See
    `tf.compat.v1.global_variables`
    for more details.
    Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
    and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
  * `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
    machine. Usually used for temporarily variables, like counters.
  * `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
    model for inference (feed forward).
  * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
    be trained by an optimizer. See
    `tf.compat.v1.trainable_variables`
    for more details.
  * `SUMMARIES`: the summary `Tensor` objects that have been created in the
    graph. See
    `tf.compat.v1.summary.merge_all`
    for more details.
  * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
    produce input for a computation. See
    `tf.compat.v1.train.start_queue_runners`
    for more details.
  * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
    keep moving averages.  See
    `tf.compat.v1.moving_average_variables`
    for more details.
  * `REGULARIZATION_LOSSES`: regularization losses collected during graph
    construction.

  The following standard keys are _defined_, but their collections are **not**
  automatically populated as many of the others are:

  * `WEIGHTS`
  * `BIASES`
  * `ACTIVATIONS`
  """

  # Key to collect Variable objects that are global (shared across machines).
  # Default collection for all variables, except local ones.
  GLOBAL_VARIABLES = "variables"
  # Key to collect local variables that are local to the machine and are not
  # saved/restored.
  LOCAL_VARIABLES = "local_variables"
  # Key to collect local variables which are used to accumulate internal state
  # to be used in tf.metrics.*.
  METRIC_VARIABLES = "metric_variables"
  # Key to collect model variables defined by layers.
  MODEL_VARIABLES = "model_variables"
  # Key to collect Variable objects that will be trained by the
  # optimizers.
  TRAINABLE_VARIABLES = "trainable_variables"
  # Key to collect summaries.
  SUMMARIES = "summaries"
  # Key to collect QueueRunners.
  QUEUE_RUNNERS = "queue_runners"
  # Key to collect table initializers.
  TABLE_INITIALIZERS = "table_initializer"
  # Key to collect asset filepaths. An asset represents an external resource
  # like a vocabulary file.
  ASSET_FILEPATHS = "asset_filepaths"
  # Key to collect Variable objects that keep moving averages.
  MOVING_AVERAGE_VARIABLES = "moving_average_variables"
  # Key to collect regularization losses at graph construction.
  REGULARIZATION_LOSSES = "regularization_losses"
  # Key to collect concatenated sharded variables.
  CONCATENATED_VARIABLES = "concatenated_variables"
  # Key to collect savers.
  SAVERS = "savers"
  # Key to collect weights
  WEIGHTS = "weights"
  # Key to collect biases
  BIASES = "biases"
  # Key to collect activations
  ACTIVATIONS = "activations"
  # Key to collect update_ops
  UPDATE_OPS = "update_ops"
  # Key to collect losses
  LOSSES = "losses"
  # Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
  SAVEABLE_OBJECTS = "saveable_objects"
  # Key to collect all shared resources used by the graph which need to be
  # initialized once per cluster.
  RESOURCES = "resources"
  # Key to collect all shared resources used in this graph which need to be
  # initialized once per session.
  LOCAL_RESOURCES = "local_resources"
  # Trainable resource-style variables.
  TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"

  # Key to indicate various ops.
  INIT_OP = "init_op"
  LOCAL_INIT_OP = "local_init_op"
  READY_OP = "ready_op"
  READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
  SUMMARY_OP = "summary_op"
  GLOBAL_STEP = "global_step"

  # Used to count the number of evaluations performed during a single evaluation
  # run.
  EVAL_STEP = "eval_step"
  TRAIN_OP = "train_op"

  # Key for control flow context.
  COND_CONTEXT = "cond_context"
  WHILE_CONTEXT = "while_context"

  # Used to store v2 summary names.
  _SUMMARY_COLLECTION = "_SUMMARY_V2"

  # List of all collections that keep track of variables.
  _VARIABLE_COLLECTIONS = [
      GLOBAL_VARIABLES,
      LOCAL_VARIABLES,
      METRIC_VARIABLES,
      MODEL_VARIABLES,
      TRAINABLE_VARIABLES,
      MOVING_AVERAGE_VARIABLES,
      CONCATENATED_VARIABLES,
      TRAINABLE_RESOURCE_VARIABLES,
  ]

  # Key for streaming model ports.
  # NOTE(yuanbyu): internal and experimental.
  _STREAMING_MODEL_PORTS = "streaming_model_ports"

  @decorator_utils.classproperty
  @deprecation.deprecated(None, "Use `tf.GraphKeys.GLOBAL_VARIABLES` instead.")
  def VARIABLES(cls):  # pylint: disable=no-self-argument
    return cls.GLOBAL_VARIABLES


def dismantle_graph(graph) -> None:
  """Cleans up reference cycles from a `Graph`.

  Helpful for making sure the garbage collector doesn't need to run after a
  temporary `Graph` is no longer needed.

  Args:
    graph: A `Graph` object to destroy. Neither it nor any of its ops are usable
      after this function runs.
  """
  graph._functions.clear()  # pylint: disable=protected-access
  graph.Dismantle()


@tf_export(v1=["add_to_collection"])
def add_to_collection(name, value) -> None:
  """Wrapper for `Graph.add_to_collection()` using the default graph.

  See `tf.Graph.add_to_collection`
  for more details.

  Args:
    name: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collection.

  @compatibility(eager)
  Collections are only supported in eager when variables are created inside
  an EagerVariableStore (e.g. as part of a layer or template).
  @end_compatibility
  """
  get_default_graph().add_to_collection(name, value)


@tf_export(v1=["add_to_collections"])
def add_to_collections(names, value) -> None:
  """Wrapper for `Graph.add_to_collections()` using the default graph.

  See `tf.Graph.add_to_collections`
  for more details.

  Args:
    names: The key for the collections. The `GraphKeys` class contains many
      standard names for collections.
    value: The value to add to the collections.

  @compatibility(eager)
  Collections are only supported in eager when variables are created inside
  an EagerVariableStore (e.g. as part of a layer or template).
  @end_compatibility
  """
  get_default_graph().add_to_collections(names, value)


@tf_export(v1=["get_collection_ref"])
def get_collection_ref(key) -> list[Any]:
  """Wrapper for `Graph.get_collection_ref()` using the default graph.

  See `tf.Graph.get_collection_ref`
  for more details.

  Args:
    key: The key for the collection. For example, the `GraphKeys` class contains
      many standard names for collections.

  Returns:
    The list of values in the collection with the given `name`, or an empty
    list if no value has been added to that collection.  Note that this returns
    the collection list itself, which can be modified in place to change the
    collection.

  @compatibility(eager)
  Collections are not supported when eager execution is enabled.
  @end_compatibility
  """
  return get_default_graph().get_collection_ref(key)


@tf_export(v1=["get_collection"])
def get_collection(key, scope=None) -> list[Any]:
  """Wrapper for `Graph.get_collection()` using the default graph.

  See `tf.Graph.get_collection`
  for more details.

  Args:
    key: The key for the collection. For example, the `GraphKeys` class contains
      many standard names for collections.
    scope: (Optional.) If supplied, the resulting list is filtered to include
      only items whose `name` attribute matches using `re.match`. Items without
      a `name` attribute are never returned if a scope is supplied and the
      choice or `re.match` means that a `scope` without special tokens filters
      by prefix.

  Returns:
    The list of values in the collection with the given `name`, or
    an empty list if no value has been added to that collection. The
    list contains the values in the order under which they were
    collected.

  @compatibility(eager)
  Collections are not supported when eager execution is enabled.
  @end_compatibility
  """
  return get_default_graph().get_collection(key, scope)


def get_all_collection_keys() -> list[str]:
  """Returns a list of collections used in the default graph."""
  return get_default_graph().get_all_collection_keys()


def name_scope(
    name, default_name=None, values=None, skip_on_eager=True,
) -> ContextManager[Optional[str]]:
  """Internal-only entry point for `name_scope*`.

  Internal ops do not use the public API and instead rely on
  `ops.name_scope` regardless of the execution mode. This function
  dispatches to the correct `name_scope*` implementation based on
  the arguments provided and the current mode. Specifically,

  * if `values` contains a graph tensor `Graph.name_scope` is used;
  * `name_scope_v1` is used in graph mode;
  * `name_scope_v2` -- in eager mode.

  Args:
    name: The name argument that is passed to the op function.
    default_name: The default name to use if the `name` argument is `None`.
    values: The list of `Tensor` arguments that are passed to the op function.
    skip_on_eager: Indicates to return NullContextmanager if executing eagerly.
      By default this is True since naming tensors and operations in eager mode
      have little use and cause unnecessary performance overhead. However, it is
      important to preserve variable names since they are often useful for
      debugging and saved models.

  Returns:
    `name_scope*` context manager.
  """
  if not context.executing_eagerly():
    return internal_name_scope_v1(name, default_name, values)

  if skip_on_eager:
    return NullContextmanager()

  name = default_name if name is None else name
  if values:
    # The presence of a graph tensor in `values` overrides the context.
    # TODO(slebedev): this is Keras-specific and should be removed.
    graph_value = next(
        (value for value in values if is_symbolic_tensor(value)), None
    )
    # pylint: enable=unidiomatic-typecheck
    if graph_value is not None:
      return graph_value.graph.name_scope(name)

  return name_scope_v2(name or "")


class internal_name_scope_v1(contextlib.AbstractContextManager[str]):  # pylint: disable=invalid-name
  """Graph-only version of `name_scope_v1`."""

  @property
  def name(self):
    return self._name

  def __init__(self, name, default_name=None, values=None) -> None:
    """Initialize the context manager.

    Args:
      name: The name argument that is passed to the op function.
      default_name: The default name to use if the `name` argument is `None`.
      values: The list of `Tensor` arguments that are passed to the op function.

    Raises:
      TypeError: if `default_name` is passed in but not a string.
    """
    if not (default_name is None or isinstance(default_name, str)):
      raise TypeError(
          "`default_name` type (%s) is not a string type. You likely meant to "
          "pass this into the `values` kwarg." % type(default_name))
    self._name = default_name if name is None else name
    self._default_name = default_name
    self._values = values

  def __enter__(self) -> str:
    """Start the scope block.

    Returns:
      The scope name.

    Raises:
      ValueError: if neither `name` nor `default_name` is provided
        but `values` are.
    """
    if self._name is None and self._values is not None:
      # We only raise an error if values is not None (provided) because
      # currently tf.name_scope(None) (values=None then) is sometimes used as
      # an idiom to reset to top scope.
      raise ValueError(
          "At least one of name (%s) and default_name (%s) must be provided."
          % (self._name, self._default_name))

    g = get_default_graph()
    if self._values and not g.building_function:
      # Specialize based on the knowledge that `_get_graph_from_inputs()`
      # ignores `inputs` when building a function.
      g_from_inputs = _get_graph_from_inputs(self._values)
      if g_from_inputs is not g:
        g = g_from_inputs
        self._g_manager = g.as_default()
        self._g_manager.__enter__()
      else:
        self._g_manager = None
    else:
      self._g_manager = None

    try:
      self._name_scope = g.name_scope(self._name)
      return self._name_scope.__enter__()
    except:
      if self._g_manager is not None:
        self._g_manager.__exit__(*sys.exc_info())
      raise

  def __exit__(self, *exc_info) -> None:
    self._name_scope.__exit__(*exc_info)
    if self._g_manager is not None:
      self._g_manager.__exit__(*exc_info)


# Named like a function for backwards compatibility with the
# @tf_contextlib.contextmanager version, which was switched to a class to avoid
# some object creation overhead.
@tf_export(v1=["name_scope"])
class name_scope_v1(contextlib.AbstractContextManager[Optional[str]]):  # pylint: disable=invalid-name
  """A context manager for use when defining a Python op.

  This context manager validates that the given `values` are from the
  same graph, makes that graph the default graph, and pushes a
  name scope in that graph (see
  `tf.Graph.name_scope`
  for more details on that).

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope(name, "MyOp", [a, b, c]) as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.convert_to_tensor(c, name="c")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```
  """

  __slots__ = ["_name", "_name_scope"]

  @property
  def name(self):
    return self._name

  def __init__(self, name, default_name=None, values=None) -> None:
    """Initialize the context manager.

    Args:
      name: The name argument that is passed to the op function.
      default_name: The default name to use if the `name` argument is `None`.
      values: The list of `Tensor` arguments that are passed to the op function.

    Raises:
      TypeError: if `default_name` is passed in but not a string.
    """
    self._name_scope = name_scope(
        name, default_name, values, skip_on_eager=False)
    self._name = default_name if name is None else name

  def __enter__(self) -> Optional[str]:
    return self._name_scope.__enter__()

  def __exit__(self, *exc_info) -> Optional[bool]:
    return self._name_scope.__exit__(*exc_info)


@tf_export("get_current_name_scope", v1=[])
def get_current_name_scope() -> str:
  """Returns current full name scope specified by `tf.name_scope(...)`s.

  For example,
  ```python
  with tf.name_scope("outer"):
    tf.get_current_name_scope()  # "outer"

    with tf.name_scope("inner"):
      tf.get_current_name_scope()  # "outer/inner"
  ```

  In other words, `tf.get_current_name_scope()` returns the op name prefix that
  will be prepended to, if an op is created at that place.

  Note that `@tf.function` resets the name scope stack as shown below.

  ```
  with tf.name_scope("outer"):

    @tf.function
    def foo(x):
      with tf.name_scope("inner"):
        return tf.add(x * x)  # Op name is "inner/Add", not "outer/inner/Add"
  ```
  """

  ctx = context.context()
  if ctx.executing_eagerly():
    return ctx.scope_name.rstrip("/")
  else:
    return get_default_graph().get_name_scope()


@tf_export("name_scope", v1=[])
class name_scope_v2(contextlib.AbstractContextManager[str]):
  """A context manager for use when defining a Python op.

  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope("MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.convert_to_tensor(c, name="c")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  When executed, the Tensors `a`, `b`, `c`, will have names `MyOp/a`, `MyOp/b`,
  and `MyOp/c`.

  Inside a `tf.function`, if the scope name already exists, the name will be
  made unique by appending `_n`. For example, calling `my_op` the second time
  will generate `MyOp_1/a`, etc.
  """

  __slots__ = ["_name", "_exit_fns"]

  def __init__(self, name) -> None:
    """Initialize the context manager.

    Args:
      name: The prefix to use on all names created within the name scope.

    Raises:
      ValueError: If name is not a string.
    """
    if not isinstance(name, str):
      raise ValueError("name for name_scope must be a string.")
    self._name = name
    self._exit_fns = []

  @property
  def name(self):
    return self._name

  def __enter__(self) -> str:
    """Start the scope block.

    Returns:
      The scope name.
    """
    ctx = context.context()
    if ctx.executing_eagerly():
      # Names are not auto-incremented in eager mode.
      # A trailing slash breaks out of nested name scopes, indicating a
      # fully specified scope name, for compatibility with Graph.name_scope.
      # This also prevents auto-incrementing.
      old_name = ctx.scope_name
      name = self._name
      if not name:
        scope_name = ""
      elif name[-1] == "/":
        scope_name = name
      elif old_name:
        scope_name = old_name + name + "/"
      else:
        scope_name = name + "/"
      ctx.scope_name = scope_name

      def _restore_name_scope(*_):
        ctx.scope_name = old_name

      self._exit_fns.append(_restore_name_scope)
    else:
      scope = get_default_graph().name_scope(self._name)
      scope_name = scope.__enter__()
      self._exit_fns.append(scope.__exit__)
    return scope_name

  def __exit__(
      self, type_arg: None, value_arg: None, traceback_arg: None,
  ) -> bool:
    self._exit_fns.pop()(type_arg, value_arg, traceback_arg)
    return False  # False values do not suppress exceptions

  def __getstate__(self) -> tuple[str, list[Callable[..., Any]]]:
    return self._name, self._exit_fns

  def __setstate__(self, state) -> None:
    self._name = state[0]
    self._exit_fns = state[1]


def strip_name_scope(name: str, export_scope) -> str:
  """Removes name scope from a name.

  Args:
    name: A `string` name.
    export_scope: Optional `string`. Name scope to remove.

  Returns:
    Name with name scope removed, or the original name if export_scope
    is None.
  """
  if export_scope:
    if export_scope[-1] == "/":
      export_scope = export_scope[:-1]

    try:
      # Strips export_scope/, export_scope///,
      # ^export_scope/, loc:@export_scope/.
      str_to_replace = r"([\^]|loc:@|^)" + export_scope + r"[\/]+(.*)"
      return re.sub(str_to_replace, r"\1\2", compat.as_str(name), count=1)
    except TypeError as e:
      # If the name is not of a type we can process, simply return it.
      logging.warning(e)
      return name
  else:
    return name


def prepend_name_scope(name: str, import_scope) -> str:
  """Prepends name scope to a name.

  Args:
    name: A `string` name.
    import_scope: Optional `string`. Name scope to add.

  Returns:
    Name with name scope added, or the original name if import_scope
    is None.
  """
  if import_scope:
    if import_scope[-1] == "/":
      import_scope = import_scope[:-1]

    try:
      str_to_replace = r"([\^]|loc:@|^)(.*)"
      return re.sub(str_to_replace, r"\1" + import_scope + r"/\2",
                    compat.as_str(name))
    except TypeError as e:
      # If the name is not of a type we can process, simply return it.
      logging.warning(e)
      return name
  else:
    return name


# pylint: disable=g-doc-return-or-yield
# pylint: disable=not-context-manager
@tf_export(v1=["op_scope"])
@tf_contextlib.contextmanager
def op_scope(values, name, default_name=None) -> Iterator[Optional[str]]:
  """DEPRECATED. Same as name_scope above, just different argument order."""
  logging.warn("tf.op_scope(values, name, default_name) is deprecated,"
               " use tf.name_scope(name, default_name, values)")
  with name_scope(name, default_name=default_name, values=values) as scope:
    yield scope


_proto_function_registry = registry.Registry("proto functions")


def register_proto_function(collection_name,
                            proto_type=None,
                            to_proto=None,
                            from_proto=None) -> None:
  """Registers `to_proto` and `from_proto` functions for collection_name.

  `to_proto` function converts a Python object to the corresponding protocol
  buffer, and returns the protocol buffer.

  `from_proto` function converts protocol buffer into a Python object, and
  returns the object..

  Args:
    collection_name: Name of the collection.
    proto_type: Protobuf type, such as `saver_pb2.SaverDef`,
      `variable_pb2.VariableDef`, `queue_runner_pb2.QueueRunnerDef`..
    to_proto: Function that implements Python object to protobuf conversion.
    from_proto: Function that implements protobuf to Python object conversion.
  """
  if to_proto and not callable(to_proto):
    raise TypeError("to_proto must be callable.")
  if from_proto and not callable(from_proto):
    raise TypeError("from_proto must be callable.")

  _proto_function_registry.register((proto_type, to_proto, from_proto),
                                    collection_name)


def get_collection_proto_type(
    collection_name,
) -> Optional[type[message.Message]]:
  """Returns the proto_type for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[0]
  except LookupError:
    return None


def get_to_proto_function(
    collection_name,
) -> Optional[Callable[[Any], message.Message]]:
  """Returns the to_proto function for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[1]
  except LookupError:
    return None


def get_from_proto_function(
    collection_name,
) -> Optional[Callable[[message.Message], Any]]:
  """Returns the from_proto function for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[2]
  except LookupError:
    return None


def _op_to_colocate_with(
    v, graph,
) -> tuple[Optional[Operation], Optional[Callable[[], None]]]:
  """Operation object corresponding to v to use for colocation constraints."""
  if v is None:
    return None, None
  if isinstance(v, Operation):
    return v, None

  # We always want to colocate with the reference op.
  # When 'v' is a ResourceVariable, the reference op is the handle creating op.
  #
  # What this should be is:
  # if isinstance(v, ResourceVariable):
  #   return v.handle.op, v
  # However, that would require a circular import dependency.
  # As of October 2018, there were attempts underway to remove
  # colocation constraints altogether. Assuming that will
  # happen soon, perhaps this hack to work around the circular
  # import dependency is acceptable.
  if hasattr(v, "handle") and isinstance(v.handle, tensor_lib.Tensor):
    device_only_candidate = lambda: None
    device_only_candidate.device = v.device
    device_only_candidate.name = v.name
    if graph.building_function:
      return graph.capture(v.handle).op, device_only_candidate
    else:
      return v.handle.op, device_only_candidate
  if isinstance(v, EagerTensor) and not context.executing_eagerly():
    return convert_to_tensor(v, as_ref=True).op, None
  elif isinstance(v, internal.NativeObject):
    return v.op, None
  else:
    return convert_to_tensor(v, as_ref=True).op, None


# Helper functions for op wrapper modules generated by `python_op_gen`.


def to_raw_op(f: types.FunctionType) -> Callable[..., Any]:
  """Make a given op wrapper function `f` raw.

  Raw op wrappers can only be called with keyword arguments.

  Args:
    f: An op wrapper function to make raw.

  Returns:
    Raw `f`.
  """
  # Copy `f` to get a new `__dict__`, otherwise `tf_export` will fail
  # due to double-registration.
  f = types.FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__,
                         f.__closure__)
  return kwarg_only(f)


def raise_from_not_ok_status(e, name) -> NoReturn:
  e.message += (" name: " + str(name if name is not None else ""))
  raise core._status_to_exception(e) from None  # pylint: disable=protected-access


def add_exit_callback_to_default_func_graph(fn) -> None:
  """Add a callback to run when the default function graph goes out of scope.

  Usage:

  ```python
  @tf.function
  def fn(x, v):
    expensive = expensive_object(v)
    add_exit_callback_to_default_func_graph(lambda: expensive.release())
    return g(x, expensive)

  fn(x=tf.constant(...), v=...)
  # `expensive` has been released.
  ```

  Args:
    fn: A callable that takes no arguments and whose output is ignored.
      To be executed when exiting func graph scope.

  Raises:
    RuntimeError: If executed when the current default graph is not a FuncGraph,
      or not currently executing in function creation mode (e.g., if inside
      an init_scope).
  """
  default_graph = get_default_graph()
  if not default_graph._building_function:  # pylint: disable=protected-access
    raise RuntimeError(
        "Cannot add scope exit callbacks when not building a function.  "
        "Default graph: {}".format(default_graph))
  default_graph._add_scope_exit_callback(fn)  # pylint: disable=protected-access


def _reconstruct_sequence_inputs(
    op_def, inputs, attrs,
) -> list[Union[tensor_lib.Tensor, list[tensor_lib.Tensor]]]:
  """Regroups a flat list of input tensors into scalar and sequence inputs.

  Args:
    op_def: The `op_def_pb2.OpDef` (for knowing the input types)
    inputs: a list of input `Tensor`s to the op.
    attrs: mapping from attr name to `attr_value_pb2.AttrValue` (these define
      how long each sequence is)

  Returns:
    A list of `Tensor`s (corresponding to scalar inputs) and lists of
    `Tensor`s (corresponding to sequence inputs).
  """
  grouped_inputs = []
  i = 0
  for input_arg in op_def.input_arg:
    if input_arg.number_attr:
      input_len = attrs[input_arg.number_attr].i
      is_sequence = True
    elif input_arg.type_list_attr:
      input_len = len(attrs[input_arg.type_list_attr].list.type)
      is_sequence = True
    else:
      input_len = 1
      is_sequence = False

    if is_sequence:
      grouped_inputs.append(inputs[i:i + input_len])
    else:
      grouped_inputs.append(inputs[i])
    i += input_len

  assert i == len(inputs)
  return grouped_inputs


# OFF mode is the current TF dtype promotion semantics - no dtype conversion
# allowed. LEGACY mode maintains the old Tf-NumPy promotion semantics, similar
# to NumPy's dtype promotion semantics. ALL mode allows all conversions while
# SAFE mode disallows “risky” promotions that can result in dtype widening or
# potential precision loss.
class PromoMode(enum.Enum):
  OFF: int = 0
  LEGACY: int = 1
  SAFE: int = 2
  ALL: int = 3


_dtype_conversion_mode: PromoMode = PromoMode.OFF


def get_dtype_conversion_mode() -> PromoMode:
  return _dtype_conversion_mode


# TODO(b/289395872): Make sure all WeakTensor construction is guarded with this
# check.
def is_auto_dtype_conversion_enabled() -> bool:
  return (
      _dtype_conversion_mode == PromoMode.ALL
      or _dtype_conversion_mode == PromoMode.SAFE
  )


def is_numpy_style_type_promotion() -> bool:
  return _dtype_conversion_mode == PromoMode.LEGACY


def set_dtype_conversion_mode(dtype_conversion_mode) -> None:
  """Enables the specified dtype conversion mode.

  Args:
    dtype_conversion_mode: a string that specifies dtype conversion mode. This
      string corresponds to a PromoMode Enum and can be 'off', 'legacy', 'safe'
      or 'all'.
  """
  global _dtype_conversion_mode
  _dtype_conversion_mode = _get_promo_mode_enum(dtype_conversion_mode)


def _get_promo_mode_enum(dtype_conversion_mode) -> PromoMode:
  """Returns the corresponding PromoMode enum value from string."""
  if dtype_conversion_mode == "off":
    return PromoMode.OFF
  if dtype_conversion_mode == "legacy":
    return PromoMode.LEGACY
  elif dtype_conversion_mode == "safe":
    return PromoMode.SAFE
  elif dtype_conversion_mode == "all":
    return PromoMode.ALL
  else:
    raise ValueError(
        f"The provided promotion mode {dtype_conversion_mode} does not exist."
        " Make sure the provided dtype conversion mode is one of the"
        " followings: 'off', 'legacy', 'safe' or 'all'."
    )


def promo_mode_enum_to_string(promo_safety_mode_enum) -> str:
  """Returns the corresponding PromoMode string value from PromoMode enum."""
  if promo_safety_mode_enum == PromoMode.OFF:
    return "off"
  if promo_safety_mode_enum == PromoMode.LEGACY:
    return "legacy"
  elif promo_safety_mode_enum == PromoMode.SAFE:
    return "safe"
  elif promo_safety_mode_enum == PromoMode.ALL:
    return "all"
  else:
    raise ValueError(
        f"The provided promotion mode {promo_safety_mode_enum} does not exist."
    )


_numpy_style_slicing: bool = False


def enable_numpy_style_slicing() -> None:
  """If called, follows NumPy's rules for slicing Tensors.

  Used for enabling NumPy behavior on slicing for TF NumPy.
  """
  global _numpy_style_slicing
  _numpy_style_slicing = True


def set_int_list_attr(op, attr_name, ints) -> None:
  """TF internal method used to set a list(int) attribute in the node_def."""
  ints_list = attr_value_pb2.AttrValue.ListValue(i=ints)
  op._set_attr(attr_name, attr_value_pb2.AttrValue(list=ints_list))  # pylint:disable=protected-access


def _get_enclosing_context(graph) -> Any:
  # pylint: disable=protected-access
  if graph is None:
    return None

  if graph._control_flow_context is not None:
    return graph._control_flow_context

  if graph.building_function and hasattr(graph, "outer_graph"):
    return _get_enclosing_context(graph.outer_graph)


# TODO(b/271463878): Remove in favor of direct references to `handle_data_util`.
get_resource_handle_data = handle_data_util.get_resource_handle_data


def _copy_handle_data_to_arg_def(tensor, arg_def) -> None:
  handle_data = handle_data_util.get_resource_handle_data(tensor)
  if handle_data.shape_and_type:
    shape_and_type = handle_data.shape_and_type[0]
    proto = arg_def.handle_data.add()
    proto.dtype = shape_and_type.dtype
    proto.shape.CopyFrom(handle_data.shape_and_type[0].shape)


@tf_export("is_symbolic_tensor", v1=["is_symbolic_tensor"])
def is_symbolic_tensor(tensor) -> bool:
  """Test if `tensor` is a symbolic Tensor.

  Args:
    tensor: a tensor-like object

  Returns:
    True if `tensor` is a symbolic tensor (not an eager tensor).
  """
  return isinstance(tensor, SymbolicTensor)
