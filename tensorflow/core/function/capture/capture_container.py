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
import inspect
from typing import Any, Callable, Hashable, Mapping, Union

from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


_EAGER_CONST_THRESHOLD = 128


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
      graph: "FuncGraph",
      tensor: core.Tensor,
      name: str = None
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

  def capture_by_ref(self,
                     lam: Callable[[], Any],
                     idf: Hashable = None):
    """Create a by-referece capture if not exists."""
    # check if the capture exist in self._by_ref
    if idf is not None and idf in self._by_ref:
      capture = self._by_ref[idf]
      return capture.internal
    if idf is None:
      idf = len(self._by_ref)

    if context.executing_eagerly():
      return lam()
    placeholder = self._create_capture_placeholder(lam)
    capture = CaptureContainer(lam, placeholder, idf, is_by_ref=True)
    self._by_ref[idf] = capture
    return capture.internal

  def merge_by_ref_with(self, other: "FunctionCaptures"):
    """Add by-ref captures from `other` to `self` if not exist."""
    assert isinstance(other, FunctionCaptures)
    for key, capture in other.by_ref_captures.items():
      if key not in self._by_ref:
        self._by_ref[key] = capture

  def get_by_ref_snapshot(self) -> Mapping[Hashable, Any]:
    """Get a snapshot of current values of by-ref captures."""
    snapshot = {}
    for key, capture in self._by_ref.items():
      func = capture.external
      snapshot[key] = func()
    return snapshot

  def _create_placeholder_helper(
      self,
      graph: "FuncGraph",
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

  # TODO(panzf): Use FunctionType/TraceType to create placeholder here.
  def _create_capture_placeholder(self, func: Callable[[], Any]) -> ...:
    """Create placeholder if the input is tensor."""
    values_nest = func()

    values_flat = nest.flatten(values_nest)
    # Return values in flat format. It consists of placeholders and non-tensor
    # values.
    return_flat = []
    tensor_spec_flat = []
    # Create return_flat and replace tensors with None. Later, each None is
    # replaced again by corresponding placeholders
    for value in values_flat:
      if isinstance(value, core.Tensor):
        return_flat.append(None)
        tensor_spec_flat.append(type_spec.type_spec_from_value(value))
      elif isinstance(value, set) or isinstance(value, frozenset):
        raise NotImplementedError(
            (f"Side input returned by '{inspect.getsource(func).strip()}' "
             f"has element of {type(value)} type, which is currently not "
             "supported by tf.function."))
      else:
        return_flat.append(value)
    if tensor_spec_flat:

      def tensor_func():
        values = nest.flatten(func())
        return [value for value in values if isinstance(value, core.Tensor)]
      # TODO(panzf): remove get_default_graph after moving
      # capture_call_time_value to this class.
      graph = ops.get_default_graph()
      placeholder_flat = graph.capture_call_time_value(
          tensor_func, tensor_spec_flat)
      # replace None that represents tensors with placehoders
      flat_ptr = 0
      for idx, item in enumerate(return_flat):
        if item is None:
          return_flat[idx] = placeholder_flat[flat_ptr]
          flat_ptr += 1
    return_nest = nest.pack_sequence_as(values_nest, return_flat)
    return return_nest

  @property
  def by_ref_captures(self):
    return self._by_ref

  @property
  def by_val_captures(self):
    return self._by_val

  @property
  def by_val_capture_tuples(self):
    return self._by_val.tuple_cache
