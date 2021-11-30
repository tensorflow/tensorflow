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

from typing import Dict, Hashable, Optional, Sequence, Tuple, Type, Callable

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import errors
from tensorflow.python.types import trace
from tensorflow.python.util import _pywrap_utils


class WeakrefDeletionObserver:
  """An observer for the event of deleting a weakref.

  This allows users of FunctionTraceType to be notified when an instance which
  depends on a weakref becomes invalid by the deletion of the weakref. In
  particular, tf.function caches can use this mechanism to clear the cache of
  keys that are no longer valid.

  We use the observer pattern and not just basic callbacks because the keys
  are typically created before they are used by the cache.
  """

  def __init__(self):
    self._triggered = False
    self._callables = []

  def add_listener(self, on_delete: Callable[[], None]):
    if self._triggered:
      on_delete()
    else:
      self._callables.append(on_delete)

  def weakref_deleted(self):
    self._triggered = True
    for c in self._callables:
      c()

  def __call__(self, _):
    """Call handler for convenience of use with weakref."""
    self.weakref_deleted()


class SignatureContext(trace.TracingContext):
  """Container for variables and flags shared across signature tracing."""

  def __init__(self, include_tensor_ranks_only=False):
    self._deletion_observer = WeakrefDeletionObserver()
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

  @property
  def deletion_observer(self):
    """Returns a functor which invalidates the current key when called."""
    return self._deletion_observer


class GenericType(trace.TraceType):
  """Represents an arbitrary Python object."""

  def __init__(self, obj):
    self._object = obj
    self._object_hash = hash(obj)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    return self == other

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional[trace.TraceType]:
    if not others:
      raise errors.InvalidArgumentError(
          "Argument `others` to function `most_specific_common_supertype` must be a non-empty Sequence."
      )

    return None

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, GenericType) and self._object == other._object

  def __hash__(self) -> int:
    return self._object_hash

  def __repr__(self):
    return f"{self.__class__.__name__}(obj={self._object!r})"


class WeakrefType(GenericType):
  """Represents weakref of an arbitrary Python object.

  When a function argument is a custom class, instead of making a copy of it
  just for the sake of function cache, a weakref is instead kept to save memory.
  """

  def __eq__(self, other):
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, WeakrefType):
      return False

    if self._object() is None or other._object() is None:
      return False

    if self._object() is other._object():
      return True

    return self._object == other._object

  def __hash__(self):
    return self._object_hash


_pywrap_utils.RegisterType("GenericType", GenericType)
_pywrap_utils.RegisterType("WeakrefType", WeakrefType)


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
    if not others:
      raise errors.InvalidArgumentError(
          "Argument `others` to function `most_specific_common_supertype` must be a non-empty Sequence."
      )

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
    super().__init__(GenericType(classtype), *attributes)


_pywrap_utils.RegisterType("ListType", ListType)
_pywrap_utils.RegisterType("TupleType", TupleType)
_pywrap_utils.RegisterType("AttrsType", AttrsType)


class DictType(trace.TraceType):
  """Represents a dictionary of TraceType objects.

  Attributes:
    mapping: A mapping from TraceType objects to TraceType objects.
  """

  def __init__(self, mapping: Dict[Hashable, trace.TraceType]):
    self.mapping = mapping

  def _has_same_structure(self, other):
    if not isinstance(other, DictType):
      return False

    return self.mapping.keys() == other.mapping.keys()

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """See base class."""
    if not self._has_same_structure(other):
      return False

    # We need all keys to be present because there can be logic relying on
    # their existence or lack thereof and hence can not guarantee subtype based
    # on a subset or superset of keys.
    # Only the tracing code can explicitly check for key dependencies and inform
    # that decision.
    return all(self.mapping[key].is_subtype_of(other.mapping[key])
               for key in self.mapping)

  def most_specific_common_supertype(self, others: Sequence[trace.TraceType]):
    """See base class."""

    if not others:
      raise errors.InvalidArgumentError(
          "Argument `others` to function `most_specific_common_supertype` must be a non-empty Sequence."
      )

    if not all(self._has_same_structure(other) for other in others):
      return None

    new_mapping = {}
    for key in self.mapping.keys():
      common = self.mapping[key].most_specific_common_supertype(
          [other.mapping[key] for other in others])
      if common is None:
        return None
      else:
        new_mapping[key] = common

    return DictType(new_mapping)

  def __eq__(self, other) -> bool:
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not self._has_same_structure(other):
      return False

    return self.mapping == other.mapping

  def __hash__(self) -> int:
    return hash(frozenset(self.mapping.keys()))

  def __repr__(self):
    return "{}(mapping={})".format(type(self).__name__, repr(self.mapping))


_pywrap_utils.RegisterType("DictType", DictType)


def make_function_signature(
    function_args,
    signature_context: SignatureContext,
    encode_variables_by_resource_id,
    use_full_trace_type) -> trace.TraceType:
  """Returns the trace type specification of a function's arguments.

  Args:
    function_args: Tuple/List/Dict structure containing the function arguments
    signature_context: The SignatureContext to be shared during protocol calls.
    encode_variables_by_resource_id: If Variables should be considered by
      resource id
    use_full_trace_type: Uses the TraceType protocol wherever possible.

  Returns:
    A TraceType object representing all the given inputs.
  """

  try:
    encoding = pywrap_tfe.TFE_Py_EncodeArg(
        function_args, signature_context,
        signature_context.include_tensor_ranks_only,
        encode_variables_by_resource_id, use_full_trace_type)
    if use_full_trace_type:
      return encoding
    else:
      # TODO(b/201533914): Drop when use_full_trace_type flag is removed.
      return GenericType(encoding)

  except core._NotOkStatusException as e:  # pylint: disable=protected-access
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
