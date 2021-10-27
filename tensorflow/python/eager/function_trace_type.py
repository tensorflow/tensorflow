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

from typing import Dict, Optional, Sequence, Type, Tuple
import weakref

import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.types import trace
from tensorflow.python.util import _pywrap_utils


class SignatureContext(trace.TracingContext):
  """Container for variables and flags shared across signature tracing."""

  def __init__(self, include_tensor_ranks_only=False):
    self._include_tensor_ranks_only = include_tensor_ranks_only
    self._global_to_local_id = {}

  # TODO(b/202772221): Consider dropping after alias pattern matching is
  # supported.
  def get_local_id(self, local_id):

    if local_id not in self._global_to_local_id:
      self._global_to_local_id[local_id] = len(self._global_to_local_id)

    return self._global_to_local_id[local_id]

  # TODO(b/202430155): Remove this flag after TraceType shape relaxation.
  @property
  def include_tensor_ranks_only(self):
    return self._include_tensor_ranks_only


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
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, GenericType) and self._object == other._object

  def __hash__(self) -> int:
    return self._object_hash

  def __repr__(self):
    return f"{self.__class__.__name__}(obj={self._object!r})"

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


_pywrap_utils.RegisterType("GenericType", GenericType)


class OrderedCollectionType(trace.TraceType):
  """Represents an ordered collection of TraceType objects.

  Attributes:
    components: The sequence of TraceType objects that this class represents.
  """

  def __init__(self, *components: trace.TraceType):
    self.components = components

  def _has_same_structure(self, other):
    if not isinstance(other, type(self)):
      return False

    if len(self.components) != len(other.components):
      return False

    return True

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """See base class."""
    if not self._has_same_structure(other):
      return False

    if not all([
        component.is_subtype_of(other.components[i])
        for i, component in enumerate(self.components)
    ]):
      return False

    return True

  def most_specific_common_supertype(self, others: Sequence[trace.TraceType]):
    """See base class."""
    if not all(self._has_same_structure(other) for other in others):
      return None

    new_components = []
    for i, component in enumerate(self.components):
      common = component.most_specific_common_supertype(
          [other.components[i] for other in others])
      if common is None:
        return None
      else:
        new_components.append(common)

    return type(self)(*new_components)

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not self._has_same_structure(other):
      return False

    return self.components == other.components

  def __hash__(self) -> int:
    return hash(self.components)

  def __repr__(self):
    return "{}(components={})".format(
        type(self).__name__, repr(self.components))


class ListType(OrderedCollectionType):
  pass


class TupleType(OrderedCollectionType):
  pass


class AttrsType(OrderedCollectionType):
  """Represents a class annotated by attr.s.

  Each attr.s class has a fixed, ordered set of attributes. Therefore, we only
  need to consider the class type and the underlying attributes. Extra
  metadata including attribute names can be ignored.
  """

  def __init__(self, classtype: Type[object],
               attributes: Tuple[trace.TraceType]):
    super().__init__(GenericType(classtype) + attributes)


_pywrap_utils.RegisterType("ListType", ListType)
_pywrap_utils.RegisterType("TupleType", TupleType)
_pywrap_utils.RegisterType("AttrsType", TupleType)


class DictType(trace.TraceType):
  """Represents a dictionary of TraceType objects.

  Attributes:
    mapping: A mapping from TraceType objects to TraceType objects.
  """

  def __init__(self, mapping: Dict[trace.TraceType, trace.TraceType]):
    self.mapping = mapping

  # TODO(b/202429845): Figure out how to subtype DictType.
  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """See base class."""
    return self == other

  def most_specific_common_supertype(self, others: Sequence[trace.TraceType]):
    """See base class."""
    return None

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, DictType):
      return False

    return self.mapping == other.mapping

  def __hash__(self) -> int:
    return hash(frozenset(self.mapping.keys()))

  def __repr__(self):
    return "{}(mapping={})".format(type(self).__name__, repr(self.mapping))


_pywrap_utils.RegisterType("DictType", DictType)


def get_arg_spec(inputs, include_tensor_ranks_only,
                 encode_variables_by_resource_id, use_full_trace_type):
  """Returns the trace type specification of a function's arguments.

  Args:
    inputs: Tuple/List/Dict structure containing the function arguments
    include_tensor_ranks_only: If Tensors should be considered by rank
    encode_variables_by_resource_id: If Variables should be considered by
      resource id
    use_full_trace_type: Uses the TraceType protocol wherever possible.

  Returns:
    A TraceType object representing the function arguments.
  """

  signature_context = SignatureContext(include_tensor_ranks_only)
  try:
    encoding = pywrap_tfe.TFE_Py_EncodeArg(inputs, signature_context,
                                           include_tensor_ranks_only,
                                           encode_variables_by_resource_id,
                                           use_full_trace_type)
    if use_full_trace_type:
      return encoding
    else:
      # TODO(b/201533914): Drop when use_full_trace_type flag is removed.
      return GenericType(encoding)

  except core._NotOkStatusException as e:  # pylint: disable=protected-access
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
