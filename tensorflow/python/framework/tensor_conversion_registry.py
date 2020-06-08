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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import threading

import numpy as np
import six

from tensorflow.python.util import lazy_loader
from tensorflow.python.util.tf_export import tf_export

# Loaded lazily due to a circular dependency
# ops->tensor_conversion_registry->constant_op->ops.
constant_op = lazy_loader.LazyLoader(
    "constant_op", globals(),
    "tensorflow.python.framework.constant_op")


_tensor_conversion_func_registry = collections.defaultdict(list)
_tensor_conversion_func_cache = {}
_tensor_conversion_func_lock = threading.Lock()

# Instances of these types are always converted using
# `_default_conversion_function`.
_UNCONVERTIBLE_TYPES = six.integer_types + (
    float,
    np.generic,
    np.ndarray,
)


def _default_conversion_function(value, dtype, name, as_ref):
  del as_ref  # Unused.
  return constant_op.constant(value, dtype, name=name)


# TODO(josh11b): Add ctx argument to conversion_func() signature.
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
    raise TypeError("base_type must be a type or a tuple of types.")
  if any(issubclass(x, _UNCONVERTIBLE_TYPES) for x in base_types):
    raise TypeError("Cannot register conversions for Python numeric types and "
                    "NumPy scalars and arrays.")
  del base_types  # Only needed for validation.
  if not callable(conversion_func):
    raise TypeError("conversion_func must be callable.")

  with _tensor_conversion_func_lock:
    _tensor_conversion_func_registry[priority].append(
        (base_type, conversion_func))
    _tensor_conversion_func_cache.clear()


def get(query):
  """Get conversion function for objects of `cls`.

  Args:
    query: The type to query for.

  Returns:
    A list of conversion functions in increasing order of priority.
  """
  if issubclass(query, _UNCONVERTIBLE_TYPES):
    return [(query, _default_conversion_function)]

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
