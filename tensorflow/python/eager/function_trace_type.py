# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utitiles for Cache Key generation based on Function Trace Type."""

from typing import Optional, Sequence, Dict
import weakref

import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.types import trace


class GenericType(trace.TraceType):
  """Represents an arbitrary Python object."""

  def __init__(self, obj):
    self._object = obj
    self._object_hash = self._make_hash(obj)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    return self == other

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional[trace.TraceType]:
    return None

  def __eq__(self, other) -> bool:
    return isinstance(other, GenericType) and self._object == other._object

  def __hash__(self) -> int:
    return self._object_hash

  # TODO(b/195985838): Cleanup once Tensor protocol is implemented.
  def _make_hash(self, elem):
    """Deals with special cases while hashing arbitrary Python objects."""
    try:
      return hash(elem)
    except TypeError:
      # TODO(slebedev): consider using nest.
      if isinstance(elem, tuple):
        return hash(tuple(map(self._make_hash, elem)))

      # TFE_Py_EncodeArg weakrefs arguments it does not recognize, and we expect
      # all recognized types to be hashable.
      assert isinstance(elem, weakref.ReferenceType)
      v = elem()

      if resource_variable_ops.is_resource_variable(v):
        # We special case variables here to use unique_id as the cache key. This
        # ensures we have to retrace whenever a different variable is passed in.
        # This is needed to support cases where the user may use the id of a
        # variable in the function perhaps as a lookup in a dictionary.
        #
        # This choice leads to more retracing when we could have possibly used
        # the shape and dtype instead. However, we expect the number of
        # variables in a program to be bounded, and correspondingly the number
        # of retraces.
        #
        # Note we also include the class name to avoid collisions with strings.
        return hash((v.__class__, v._unique_id))  # pylint: disable=protected-access

      if self._is_ndarray(v):
        # Numpy arrays are not hashable, but when calling functions we treat
        # them in the same way as tf.Tensors.
        if not hasattr(v, "shape") or not hasattr(v, "dtype"):
          # TODO(tomhennigan) De-dup with _as_ndarray in _convert_numpy_inputs.
          v = self._as_ndarray(v)
        return hash(tensor_spec.TensorSpec(v.shape, v.dtype))

      raise ValueError(
          "Arguments to a tf.function must be a nested structure of "
          "Tensors, Variables, NumPy arrays, or hashable Python "
          f"objects, got {type(v)}.")

  def _as_ndarray(self, value):
    """Converts value to an ndarray, assumes _is_ndarray(value)."""
    # TODO(tomhennigan) Support __array_interface__ too (including for
    # _convert_numpy_inputs).
    return value.__array__()

  def _is_ndarray(self, value):
    """Tests whether the given value is an ndarray (and not a TF tensor/var)."""
    # TODO(tomhennigan) Support __array_interface__ too.
    return hasattr(value, "__array__") and not (
        isinstance(value, ops.Tensor) or
        isinstance(value, resource_variable_ops.BaseResourceVariable) or
        hasattr(value, "_should_act_as_resource_variable")

        # For legacy reasons we do not automatically promote Numpy strings.
        or isinstance(value, np.str_)
        # NumPy dtypes have __array__ as unbound methods.
        or isinstance(value, type)
        # CompositeTensors should be flattened instead.
        or isinstance(value, composite_tensor.CompositeTensor))


class CollectionType(trace.TraceType):
  """Represents a collection of TraceType objects.

  Attributes:
    components: The group of TraceTypes objects that this class represents.
  """

  def __init__(self, *components: trace.TraceType):
    self.components = components

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    if not isinstance(other, type(self)):
      return False

    if len(self.components) != len(other.components):
      return False

    if not all([
        component.is_subtype_of(other.components[i])
        for i, component in enumerate(self.components)
    ]):
      return False

    return True

  def most_specific_common_supertype(self, others: Sequence[trace.TraceType]):
    if not all([
        isinstance(other, type(self)) and
        len(self.components) == len(other.components) for other in others
    ]):
      return None

    new_components = []
    for i, component in enumerate(self.components):
      common = component.most_specific_common_supertype(
          *[other.components[i] for other in others])
      if common is None:
        return None
      else:
        new_components.append(common)

    return new_components

  def __eq__(self, other) -> bool:
    if not isinstance(other, type(self)):
      return False

    if len(self.components) != len(other.components):
      return False

    if not all([
        component == other.components[i]
        for i, component in enumerate(self.components)
    ]):
      return False

    return True

  def __hash__(self) -> int:
    return hash((type(self), self.components))


class TupleType(CollectionType):
  """Represents a tuple of TraceType objects."""
  pass


class ListType(CollectionType):
  """Represents a list of TraceType objects."""
  pass


class DictType(CollectionType):
  """Represents a dictionary of TraceType objects."""

  def __init__(self, mapping: Dict[trace.TraceType, trace.TraceType]):
    sorted_keys = sorted(mapping.keys(), key=hash)
    components = []
    for k in sorted_keys:
      components.append(TupleType(k, mapping[k]))

    super().__init__(*components)


def get_arg_spec(inputs, include_tensor_ranks_only,
                 encode_variables_by_resource_id, enable_full_trace_type):
  """Returns the trace type specification of a function's arguments.

  Args:
    inputs: Tuple/List/Dict structure containing the function arguments
    include_tensor_ranks_only: If Tensors should be considered by rank
    encode_variables_by_resource_id: If Variables should be considered by
      resource id
    enable_full_trace_type: If full usage of trace type protocol should be
      enabled. Otherwise, only a GenericType wrapper is added over the final
      results.

  Returns:
    A hashable object representing the function arguments.
  """
  if enable_full_trace_type:

    def parametrized_get_arg_spec(arg):
      return get_arg_spec(arg, include_tensor_ranks_only,
                          encode_variables_by_resource_id, True)

    if isinstance(inputs, tuple):
      return TupleType(*map(parametrized_get_arg_spec, inputs))

    if isinstance(inputs, list):
      return ListType(*map(parametrized_get_arg_spec, inputs))

    if isinstance(inputs, dict):
      traced = {
          parametrized_get_arg_spec(k): parametrized_get_arg_spec(v)
          for k, v in inputs.items()
      }
      return DictType(traced)

  return GenericType(
      pywrap_tfe.TFE_Py_EncodeArg(inputs, include_tensor_ranks_only,
                                  encode_variables_by_resource_id))
