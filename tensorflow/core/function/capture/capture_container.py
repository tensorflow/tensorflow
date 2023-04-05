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
import dataclasses
import functools
from typing import Any, Callable, Hashable, Mapping, Union, Optional

from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity


_EAGER_CONST_THRESHOLD = 128


# TODO(panzf): Remove idf and is_by_ref when splitting the container.
@dataclasses.dataclass(frozen=True)
class CaptureContainer():
  """A container for both by-reference and by-value captures.

  external: Used to record the tensor external to the func_graph.
     For by-value captures, it would be the original tensor.
     For by-reference captures, it would be the lambda function, which will be
     called later to get the capture's runtime value.
  internal: An internal placeholder for the capture, or a constant tensor.
    The external value of the capture will be fed to this internal placeholder
    when executing the func_graph as a side input.
  idf: A Hashable identifier for the capture.
  is_by_ref: A bool indicates if the capture is call by reference or value.
    This flag will determine how `CaptureContainer.internal` is used.
  """
  external: Any
  internal: core.Tensor
  idf: Hashable
  is_by_ref: bool = False


class CachedCaptureDict(py_collections.OrderedDict):
  """A dict like container for captures with cached tuples."""

  def __init__(self, *args, **kwargs):
    self._tuple_cache = []
    super().__init__(*args, **kwargs)

  def _recompute_tuple_cache(self):
    self._tuple_cache = [(
        c.external, c.internal) for c in self.values()]

  def pop(self, key, default=None):
    if key in self.keys():
      ret = super().pop(key, default)
      self._recompute_tuple_cache()
      return ret
    else:
      return default

  def __setitem__(self, key, value):
    assert isinstance(value, CaptureContainer)
    if key in self.keys():
      super().__setitem__(key, value)
      self._recompute_tuple_cache()
    else:
      super().__setitem__(key, value)
      self._tuple_cache.append((value.external, value.internal))

  def __delitem__(self, key):
    super().__delitem__(key)
    self._recompute_tuple_cache()

  def clear(self):
    self._tuple_cache = []
    super().clear()

  @property
  def tuple_cache(self):
    return self._tuple_cache


# TODO(panzf): Move lambda_fn to polymorphic function level
@dataclasses.dataclass(frozen=False)
class ByRefCaptureContainer():
  """A container for by-value captures.

  tracetype: TraceType of the capture
  internal: Nested structure that contains both placeholder and Python
    primitives.
  lambda_fn: lambda function that returns the nested structure of the captures.
  """
  tracetype: Any
  internal: Any
  lambda_fn: Callable[[], Any]


class FunctionCaptures(object):
  """A container for all capture usages within FuncGraph."""

  def __init__(self):
    # Dict that maps capture identifier -> CaptureContainer
    self._by_ref = py_collections.OrderedDict()
    self._by_val = CachedCaptureDict()
    # Set of external ops on which the graph has a control dependency
    self.control = object_identity.ObjectIdentitySet()

  def capture_by_value(
      self,
      graph: Any,
      tensor: core.Tensor,
      name: Optional[str] = None
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
      if (tensor.dtype in dtypes.TF_VALUE_DTYPES and
          functools.reduce(lambda a, b: a*b, tensor.shape, 1) <=
          _EAGER_CONST_THRESHOLD):
        capture = self.by_val_captures.get(id(tensor))
        if capture is None:
          graph_const = tensor._capture_as_const(name)  # pylint: disable=protected-access
          if graph_const is None:
            # Some eager tensors, e.g. parallel tensors, are not convertible to
            # a single constant. We'll use a placeholder for this case.
            graph_const = self._create_placeholder_helper(graph, tensor, name)
          self.add_or_replace(tensor, graph_const, id(tensor), False)
          graph.inputs.append(graph_const)
        else:
          graph_const = capture.internal
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
      value: Any,
      placeholder: core.Tensor,
      idf: Hashable,
      is_by_ref: bool = False):
    """Replace a already exsiting capture, otherwise add it."""
    capture = CaptureContainer(value, placeholder, idf, is_by_ref)
    if is_by_ref:
      self._by_ref[idf] = capture
    else:
      self._by_val[idf] = capture
    return capture

  def pop(self,
          idf: Hashable,
          is_by_ref: bool = False) -> Union[core.Tensor, None]:
    if is_by_ref:
      return self._by_ref.pop(idf, None)
    else:
      return self._by_val.pop(idf, None)

  def reset_captures(self, tensors, placeholders):
    """Set the captures with the provided list of captures & placeholder."""
    self._by_val = CachedCaptureDict()
    for external, internal in zip(tensors, placeholders):
      idf = id(external)
      c = CaptureContainer(external, internal, idf)
      self._by_val[idf] = c

  # TODO(panzf): make the method public after supporting lam() returns
  # non-tensor values. Currently, this method is only used by
  # FuncGraph._experimental_capture_side_input_by_ref(), which contains the
  # logics for converting non-tensor values to tensor.
  def _capture_by_ref(self,
                      graph: Any,
                      lam: Callable[[], Any],
                      idf: Hashable = None) -> Any:
    """Used during tracing process to create/retrive by-ref captures.

    Args:
      graph: The FuncGraph that captures this tensor.
      lam: A callable that takes no arguments and returns tensor captures.
      idf: A hashable identifier.

    Returns:
      Tensor from this FuncGraph.
    """
    # Check if the capture exists in self._by_ref
    if idf is not None and idf in self._by_ref:
      capture = self._by_ref[idf]
      return capture.internal
    if idf is None:
      idf = len(self._by_ref)
      while idf in self._by_ref:
        idf += 1

    value_nested = lam()
    capture_trace_type = trace_type.from_value(value_nested)
    ctx = trace_type.InternalPlaceholderContext(graph)
    internal = capture_trace_type.placeholder_value(ctx)

    def lam_fn():
      # pytype: disable=attribute-error
      value = lam()
      return capture_trace_type._to_tensors(value)  # pylint: disable=protected-access
      # pytype: enable=attribute-error

    capture = ByRefCaptureContainer(capture_trace_type, internal, lam_fn)
    self._by_ref[idf] = capture
    return self._by_ref[idf].internal

  def merge_by_ref_with(self, other: "FunctionCaptures") -> None:
    """Add by-ref captures from `other` to `self` if not exist."""
    assert isinstance(other, FunctionCaptures)
    for key, capture in other.by_ref_captures.items():
      if key not in self._by_ref:
        self._by_ref[key] = capture

  def get_by_ref_snapshot(self) -> Mapping[Hashable, Any]:
    """Get a snapshot of current values of by-ref captures."""
    snapshot = {}
    for key, capture in self._by_ref.items():
      func = capture.lambda_fn  # pytype: disable=attribute-error
      try:
        value = func()
      except (AttributeError, RuntimeError):
        # b/269680071 In case of by-ref captures are unavailable at dispatch
        # time, use the predefined trace_type instead.
        value = capture.tracetype  # pytype: disable=attribute-error
      snapshot[key] = value
    return snapshot

  def _create_placeholder_helper(
      self,
      graph: Any,
      tensor: core.Tensor,
      name: str):
    """A helper function to create capture placeholder."""
    capture = self._by_val.get(id(tensor))
    if capture is None:
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
          composite_device_name=composite_device_name)
      placeholder_ctx._spec_id_to_handledata = (  # pylint: disable=protected-access
          tracing_ctx.get_handledata_mapping()
      )
      placeholder = spec.placeholder_value(placeholder_ctx)
      self.add_or_replace(tensor, placeholder, id(tensor), False)
      graph.inputs.append(placeholder)
    else:
      placeholder = capture.internal
    placeholder._record_tape(tensor)  # pylint: disable=protected-access
    return placeholder

  @property
  def by_ref_captures(self):
    return self._by_ref

  @property
  def by_val_captures(self):
    return self._by_val

  @property
  def by_val_capture_tuples(self):
    return self._by_val.tuple_cache
