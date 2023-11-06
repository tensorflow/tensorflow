# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""FuncGraph and related functionality."""

import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional

from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity


_EAGER_CONST_THRESHOLD = 128


class MutationAwareDict(py_collections.OrderedDict):
  """A dict with a mutation flag."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._mutated = True

  def pop(self, key, default=None):
    self._mutated = True
    return super().pop(key, default)

  def __setitem__(self, key, value):
    self._mutated = True
    return super().__setitem__(key, value)

  def __delitem__(self, key):
    self._mutated = True
    return super().__delitem__(key)

  def clear(self):
    self._mutated = True
    return super().clear()

  @property
  def mutated(self):
    return self._mutated

  @mutated.setter
  def mutated(self, value):
    self._mutated = value


class FunctionCaptures(object):
  """A container for all capture usages within FuncGraph."""

  def __init__(self):
    self._by_ref_internal = py_collections.OrderedDict()
    self._by_ref_external = py_collections.OrderedDict()
    self._by_ref_tracetype = py_collections.OrderedDict()
    self._by_val_internal = MutationAwareDict()
    self._by_val_external = MutationAwareDict()
    self._by_val_tracetype = py_collections.OrderedDict()

    # Set of external ops on which the graph has a control dependency
    self.control = object_identity.ObjectIdentitySet()

    # Cached properties derived from the above.
    self._cached_by_val_capture_tuples = []
    self._cached_capture_types = py_collections.OrderedDict()

  def clear(self):
    self._by_ref_internal.clear()
    self._by_ref_external.clear()
    self._by_ref_tracetype.clear()
    self._by_val_internal.clear()
    self._by_val_external.clear()

  def capture_by_value(
      self, graph: Any, tensor: core.Tensor, name: Optional[str] = None
  ) -> core.Tensor:
    """Captures `tensor` if it's external to this graph.

    If `tensor` is from a different graph, returns a placeholder for it.
    `tensor` and the placeholder will appear in self.captures, and the
    placeholder will appear in self.inputs.  Multiple calls to this method with
    the same `tensor` argument will return the same placeholder. If `tensor` is
    from this graph, returns `tensor`.

    Args:
      graph: The FuncGraph that captures this tensor.
      tensor: Tensor. May be from this FuncGraph or a different graph.
      name: Optional name if a placeholder is created.

    Returns:
      Tensor from this FuncGraph.

    Raises:
      InaccessibleTensorError: if any tensors are accessed in a manner that
      bypasses the mechanisms required for the data dependencies to be correctly
      wired.
    """
    if isinstance(tensor, core.Value):
      if name is None:
        # A unique (within the program execution) integer.
        name = str(pywrap_tfe.TFE_Py_UID())

      # Small EagerTensors are captured with Const ops
      if (
          tensor.dtype in dtypes.TF_VALUE_DTYPES
          and functools.reduce(lambda a, b: a * b, tensor.shape, 1)
          <= _EAGER_CONST_THRESHOLD
      ):
        graph_const = self.by_val_internal.get(id(tensor))
        if graph_const is None:
          graph_const = tensor._capture_as_const(name)  # pylint: disable=protected-access
          if graph_const is None:
            # Some eager tensors, e.g. parallel tensors, are not convertible to
            # a single constant. We'll use a placeholder for this case.
            graph_const = self._create_placeholder_helper(graph, tensor, name)
          self.add_or_replace(
              key=id(tensor),
              external=tensor,
              internal=graph_const,
              is_by_ref=False,
          )
          graph.inputs.append(graph_const)
        graph_const._record_tape(tensor)  # pylint: disable=protected-access
        return graph_const

      # Large EagerTensors and resources are captured with Placeholder ops
      return self._create_placeholder_helper(graph, tensor, name)

    if tensor.graph is not graph:
      graph._validate_in_scope(tensor)  # pylint: disable=protected-access
      if name is None:
        assert tensor.op is not None, (
            tensor.__class__,
            dir(tensor),
            tensor.__class__.__name__,
        )
        name = tensor.op.name
      # cond/while graphs override _capture_helper() so cannot call
      # self.create_placeholder_helper() here directly.
      return graph._capture_helper(tensor, name)  # pylint: disable=protected-access
    return tensor

  def add_or_replace(
      self,
      key: Hashable,
      external: Any,
      internal: core.Tensor,
      tracetype: Any = None,
      is_by_ref: bool = False,
  ) -> None:
    """Replace a already exsiting capture, otherwise add it."""
    if is_by_ref:
      self._by_ref_external[key] = external
      self._by_ref_internal[key] = internal
      self._by_ref_tracetype[key] = tracetype
    else:
      self._by_val_internal[key] = internal
      self._by_val_external[key] = external
      if tracetype is not None:
        self._by_val_tracetype[key] = tracetype
      else:
        self._by_val_tracetype[key] = trace_type.from_value(external)

  def pop(self, key: Hashable, is_by_ref: bool = False) -> Any:
    if is_by_ref:
      return (
          self._by_ref_external.pop(key, None),
          self._by_ref_internal.pop(key, None),
          self._by_ref_tracetype.pop(key, None),
      )
    else:
      return (
          self._by_val_external.pop(key, None),
          self._by_val_internal.pop(key, None),
          self._by_val_tracetype.pop(key, None),
      )

  def reset_captures(self, tensors, placeholders):
    """Set the captures with the provided list of captures & placeholder."""
    self._by_val_external = MutationAwareDict()
    self._by_val_internal = MutationAwareDict()
    self._by_val_tracetype = MutationAwareDict()
    for external, internal in zip(tensors, placeholders):
      key = id(external)
      self._by_val_external[key] = external
      self._by_val_internal[key] = internal
      self._by_val_tracetype[key] = trace_type.from_value(external)

  # TODO(panzf): make the method public after supporting lam() returns
  # non-tensor values. Currently, this method is only used by
  # FuncGraph._experimental_capture_side_input_by_ref(), which contains the
  # logics for converting non-tensor values to tensor.
  def _capture_by_ref(
      self, graph: Any, lam: Callable[[], Any], key: Hashable = None
  ) -> Any:
    """Used during tracing process to create/retrive by-ref captures.

    Args:
      graph: The FuncGraph that captures this tensor.
      lam: A callable that takes no arguments and returns tensor captures.
      key: A hashable identifier.

    Returns:
      Tensor from this FuncGraph.
    """
    # Check if the capture exists in self._by_ref
    if key is not None and key in self._by_ref_internal:
      return self._by_ref_internal[key]
    if key is None:
      key = len(self._by_ref_internal)
      while key in self._by_ref_internal:
        key += 1

    value_nested = lam()
    capture_trace_type = trace_type.from_value(value_nested)
    ctx = trace_type.InternalPlaceholderContext(graph)
    internal = capture_trace_type.placeholder_value(ctx)

    def lam_fn():
      # pytype: disable=attribute-error
      value = lam()
      return capture_trace_type.to_tensors(value)
      # pytype: enable=attribute-error

    self._by_ref_external[key] = lam_fn
    self._by_ref_internal[key] = internal
    self._by_ref_tracetype[key] = capture_trace_type
    return self._by_ref_internal[key]

  def merge_by_ref_with(self, other: "FunctionCaptures") -> None:
    """Add by-ref captures from `other` to `self` if not exist."""
    assert isinstance(other, FunctionCaptures)
    for key in other.by_ref_external:
      if key not in self._by_ref_external:
        self._by_ref_external[key] = other.by_ref_external[key]
        self._by_ref_tracetype[key] = other.by_ref_tracetype[key]

  # TODO(panzf): Return structured values instead of flat tensors.
  def get_by_ref_snapshot(self) -> Mapping[Hashable, Any]:
    """Get a snapshot of current values of by-ref captures."""
    snapshot = {}
    for key in self._by_ref_external:
      func = self._by_ref_external[key]
      try:
        value = func()
      except (AttributeError, RuntimeError):
        # b/269680071 In case of by-ref captures are unavailable at dispatch
        # time, use the predefined trace_type instead.
        value = self._by_ref_tracetype[key]
      snapshot[key] = value
    return snapshot

  def _create_placeholder_helper(
      self, graph: Any, tensor: core.Tensor, name: str
  ):
    """A helper function to create capture placeholder."""
    placeholder = self._by_val_internal.get(id(tensor))
    if placeholder is None:
      tracing_ctx = trace_type.InternalTracingContext()
      spec = trace_type.from_value(tensor, tracing_ctx)
      spec._name = name  # pylint: disable=protected-access
      if isinstance(tensor, core.Value) and tensor.is_packed:
        composite_device_name = tensor.device
      else:
        composite_device_name = None
      placeholder_ctx = trace_type.InternalPlaceholderContext(
          graph,
          with_none_control_dependencies=True,
          composite_device_name=composite_device_name,
      )
      placeholder = spec.placeholder_value(placeholder_ctx)
      self.add_or_replace(
          key=id(tensor), external=tensor, internal=placeholder, is_by_ref=False
      )
      graph.inputs.append(placeholder)
    placeholder._record_tape(tensor)  # pylint: disable=protected-access
    return placeholder

  def _recompute_cached_properties(self):
    """Regenerates cached properties if there have been mutations."""
    self._by_val_internal.mutated = False
    self._by_val_external.mutated = False
    assert len(self._by_val_internal) == len(self._by_val_external)
    self._cached_by_val_capture_tuples = []
    for key in self._by_val_internal:
      assert key in self._by_val_external
      internal = self._by_val_internal[key]
      external = self._by_val_external[key]
      self._cached_by_val_capture_tuples.append((external, internal))

    self._cached_capture_types = py_collections.OrderedDict(
        list(self._by_val_tracetype.items())
        + list(self._by_ref_tracetype.items())
    )

  @property
  def capture_types(self):
    if self._by_val_internal.mutated or self._by_val_external.mutated:
      self._recompute_cached_properties()
    return self._cached_capture_types

  @property
  def by_val_capture_tuples(self):
    if self._by_val_internal.mutated or self._by_val_external.mutated:
      self._recompute_cached_properties()
    return self._cached_by_val_capture_tuples

  @property
  def by_ref_internal(self):
    return self._by_ref_internal

  @property
  def by_ref_external(self):
    return self._by_ref_external

  @property
  def by_ref_tracetype(self):
    return self._by_ref_tracetype

  @property
  def by_val_internal(self):
    return self._by_val_internal

  @property
  def by_val_external(self):
    return self._by_val_external

  @property
  def by_val_tracetype(self):
    return self._by_val_tracetype
