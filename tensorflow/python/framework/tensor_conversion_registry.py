# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Registry for tensor conversion functions."""
# pylint: disable=g-bad-name
import collections
import threading

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export


_tensor_conversion_func_registry = collections.defaultdict(list)
_tensor_conversion_func_cache = {}
_tensor_conversion_func_lock = threading.Lock()

# Instances of these types should only be converted by internally-registered
# conversion functions.
_CONSTANT_OP_CONVERTIBLES = (
    int,
    float,
    np.generic,
    np.ndarray,
)


# TODO(josh11b): Add ctx argument to conversion_func() signature.
def register_tensor_conversion_function_internal(base_type,
                                                 conversion_func,
                                                 priority=100):
  """Internal version of register_tensor_conversion_function.

  See docstring of `register_tensor_conversion_function` for details.

  The internal version of the function allows registering conversions
  for types in the _UNCONVERTIBLE_TYPES tuple.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values run
      earlier than conversion functions with larger priority values. Defaults to
      100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
  """
  base_types = base_type if isinstance(base_type, tuple) else (base_type,)
  if any(not isinstance(x, type) for x in base_types):
    raise TypeError("Argument `base_type` must be a type or a tuple of types. "
                    f"Obtained: {base_type}")
  del base_types  # Only needed for validation.
  if not callable(conversion_func):
    raise TypeError("Argument `conversion_func` must be callable. Received "
                    f"{conversion_func}.")

  with _tensor_conversion_func_lock:
    _tensor_conversion_func_registry[priority].append(
        (base_type, conversion_func))
    _tensor_conversion_func_cache.clear()


@tf_export("register_tensor_conversion_function")
def register_tensor_conversion_function(base_type,
                                        conversion_func,
                                        priority=100):
  """Registers a function for converting objects of `base_type` to `Tensor`.

  The conversion function must have the following signature:

  ```python
      def conversion_func(value, dtype=None, name=None, as_ref=False):
        # ...
  ```

  It must return a `Tensor` with the given `dtype` if specified. If the
  conversion function creates a new `Tensor`, it should use the given
  `name` if specified. All exceptions will be propagated to the caller.

  The conversion function may return `NotImplemented` for some
  inputs. In this case, the conversion process will continue to try
  subsequent conversion functions.

  If `as_ref` is true, the function must return a `Tensor` reference,
  such as a `Variable`.

  NOTE: The conversion functions will execute in order of priority,
  followed by order of registration. To ensure that a conversion function
  `F` runs before another conversion function `G`, ensure that `F` is
  registered with a smaller priority than `G`.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values run
      earlier than conversion functions with larger priority values. Defaults to
      100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
  """
  base_types = base_type if isinstance(base_type, tuple) else (base_type,)
  if any(not isinstance(x, type) for x in base_types):
    raise TypeError("Argument `base_type` must be a type or a tuple of types. "
                    f"Obtained: {base_type}")
  if any(issubclass(x, _CONSTANT_OP_CONVERTIBLES) for x in base_types):
    raise TypeError("Cannot register conversions for Python numeric types and "
                    "NumPy scalars and arrays.")
  del base_types  # Only needed for validation.
  register_tensor_conversion_function_internal(
      base_type, conversion_func, priority)


def get(query):
  """Get conversion function for objects of `cls`.

  Args:
    query: The type to query for.

  Returns:
    A list of conversion functions in increasing order of priority.
  """
  conversion_funcs = _tensor_conversion_func_cache.get(query)
  if conversion_funcs is None:
    with _tensor_conversion_func_lock:
      # Has another thread populated the cache in the meantime?
      conversion_funcs = _tensor_conversion_func_cache.get(query)
      if conversion_funcs is None:
        conversion_funcs = []
        for _, funcs_at_priority in sorted(
            _tensor_conversion_func_registry.items()):
          conversion_funcs.extend(
              (base_type, conversion_func)
              for base_type, conversion_func in funcs_at_priority
              if issubclass(query, base_type))
        _tensor_conversion_func_cache[query] = conversion_funcs
  return conversion_funcs


def _add_error_prefix(msg, *, name=None):
  return msg if name is None else f"{name}: {msg}"


def convert(value,
            dtype=None,
            name=None,
            as_ref=False,
            preferred_dtype=None,
            accepted_result_types=(core.Symbol,)):
  """Converts `value` to a `Tensor` using registered conversion functions.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    as_ref: Optional boolean specifying if the returned value should be a
      reference-type `Tensor` (e.g. Variable). Pass-through to the registered
      conversion function. Defaults to `False`.
    preferred_dtype: Optional element type for the returned tensor.
      Used when dtype is None. In some cases, a caller may not have a dtype
      in mind when converting to a tensor, so `preferred_dtype` can be used
      as a soft preference. If the conversion to `preferred_dtype` is not
      possible, this argument has no effect.
    accepted_result_types: Optional collection of types as an allow-list
      for the returned value. If a conversion function returns an object
      which is not an instance of some type in this collection, that value
      will not be returned.

  Returns:
    A `Tensor` converted from `value`.

  Raises:
    ValueError: If `value` is a `Tensor` and conversion is requested
      to a `Tensor` with an incompatible `dtype`.
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """

  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  if preferred_dtype is not None:
    preferred_dtype = dtypes.as_dtype(preferred_dtype)

  if isinstance(value, core.TensorProtocol):
    return value.__tf_tensor__(dtype, name)

  for base_type, conversion_func in get(type(value)):
    # If dtype is None but preferred_dtype is not None, we try to
    # cast to preferred_dtype first.
    ret = None
    if dtype is None and preferred_dtype is not None:
      try:
        ret = conversion_func(
            value, dtype=preferred_dtype, name=name, as_ref=as_ref)
      except (TypeError, ValueError):
        # Could not coerce the conversion to use the preferred dtype.
        pass
      else:
        if (ret is not NotImplemented and
            ret.dtype.base_dtype != preferred_dtype.base_dtype):
          raise RuntimeError(
              _add_error_prefix(
                  f"Conversion function {conversion_func!r} for type "
                  f"{base_type} returned incompatible dtype: requested = "
                  f"{preferred_dtype.base_dtype.name}, "
                  f"actual = {ret.dtype.base_dtype.name}",
                  name=name))

    if ret is None:
      ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)

    if ret is NotImplemented:
      continue

    if not isinstance(ret, accepted_result_types):
      raise RuntimeError(
          _add_error_prefix(
              f"Conversion function {conversion_func!r} for type "
              f"{base_type} returned non-Tensor: {ret!r}",
              name=name))
    if dtype and not dtype.is_compatible_with(ret.dtype):
      raise RuntimeError(
          _add_error_prefix(
              f"Conversion function {conversion_func} for type {base_type} "
              f"returned incompatible dtype: requested = {dtype.name}, "
              f"actual = {ret.dtype.name}",
              name=name))
    return ret
  raise TypeError(
      _add_error_prefix(
          f"Cannot convert {value!r} with type {type(value)} to Tensor: "
          f"no conversion function registered.",
          name=name))
