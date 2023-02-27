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
from typing import Any, Callable, Hashable, Mapping, Union

from tensorflow.core.function import trace_type
from tensorflow.python.types import core
from tensorflow.python.util import object_identity


# TODO(panzf): Rename CaptureContainer to ByValCaptureContainer
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


# TODO(panzf): Move lambda_fn to polymorphic function level
@dataclasses.dataclass(frozen=False)
class ByRefCaptureContainer():
  """A container for by-value captures.

  tracetype: TraceType of the capture
  internal: Nested structure that contains both placeholder and Python
    primitives.
  idf: A Hashable identifier for the capture.
  lambda_fn: lambda function that returns the nested structure of the captures.
  """
  tracetype: Any
  internal: Any
  lambda_fn: Callable[[], Any]
  idf: Hashable


class FunctionCaptures(object):
  """A container for all capture usages within FuncGraph."""

  def __init__(self):
    self._by_val = py_collections.OrderedDict()
    self._by_ref = py_collections.OrderedDict()
    # Set of external ops on which the graph has a control dependency
    self.control = object_identity.ObjectIdentitySet()

  def capture_by_val(
      self,
      value: Any,
      placeholder: core.Tensor = None,
      idf: Hashable = None
  ) -> core.Tensor:
    assert idf == id(value), "By value captures must use id(tensor) as idf."
    capture = self.add_or_replace(value, placeholder, idf, is_by_ref=False)
    return capture.internal

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
    self._by_val = py_collections.OrderedDict()
    for external, internal in zip(tensors, placeholders):
      idf = id(external)
      c = CaptureContainer(external, internal, idf)
      self._by_val[idf] = c

  def capture_by_ref(self,
                     graph: Any,
                     lam: Callable[[], Any],
                     idf: Hashable = None) -> Any:
    """Used during tracing process to create/retrive by-ref captures."""
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

    capture = ByRefCaptureContainer(
        capture_trace_type, internal, lam, idf)
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
      except AttributeError:
        # b/269680071 In case of by-ref captures are unavailable at dispatch
        # time, use the predefined trace_type instead.
        value = capture.tracetype  # pytype: disable=attribute-error
      snapshot[key] = value
    return snapshot

  def capture_call_time_value(self,
                              placeholder_ctx,
                              closure,
                              spec,
                              key=None,
                              placeholder=None):
    """Returns a placeholder which at call time has the value closure().

    The `tf.function` supports the notion of captures, that is, it allows Python
    functions to have closure variables, which bind over some value outside the
    function. However, this name binding is "early binding" performed before the
    program is run, i.e.,
    ```
    @tf.function
    def f():
      return x

    x = tf.constant(1)
    f()  # returns 1

    x = tf.constant(2)
    f()  # still returns 1!
    ```
    while in Python, name binding is performed as the program is running.
    ```
    def f():
      return x

    x = 1
    f()  # returns 1

    x = 2
    f()  # returns 2
    ```
    `capture_call_time_value` allows tf.function to mimic late binding as a
    Python function does, by passing in a `closure` callable argument to be
    executed when the tf.function is invoked eagerly.  E.g.
    ```
    @tf.function
    def f():
      return ops.get_default_graph.capture_call_time_value(lambda: x)

    x = tf.constant(1)
    f()  # returns 1

    x = tf.constant(2)
    f()  # returns 2
    ```
    Note that a `capture_call_time_value` function itself does not work well in
    the saving process (since the tf.function in which it's called is not
    invoked eagerly) unless passed a `default_value` argument. At saving time,
    the `default_value` argument is returned instead.

    Args:
      placeholder_ctx: a trace.PlaceholderContext object.
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      key: optional. If not None, multiple calls to lazy_capture with the same
        key in the same graph will return the same placeholder, and the first
        closure will be used at function call time.
      placeholder: optional. If not None, the graph will take the passed-in
        `placeholder` as the internal capture instead of creating a new one.
        This is useful when loading from a SavedModel.

    Returns:
      Nest of placeholders which, at function call time, will be fed with the
      result of calling closure().

    Raises:
      ValueError: at function call time, if the return value of closure() is
       not compatible with `spec`.
    """
    if key is None:
      key = object()
    if key not in self._by_ref:
      if placeholder is None:
        placeholder = spec.placeholder_value(placeholder_ctx)
      capture = ByRefCaptureContainer(
          spec, placeholder, closure, key)
      self._by_ref[key] = capture
    return self._by_ref[key].internal

  @property
  def by_ref_captures(self):
    return self._by_ref

  @property
  def by_val_captures(self):
    return self._by_val

  @property
  def flat(self):
    """Return a list of callables that are used as the inputs to the graph."""
    flat = [c.external for c in self._by_val.values()]
    # Build fn that returns tensors only for by-ref captures
    for capture in self._by_ref.values():

      def make_fn(cap):
        def lam_fn():
          # pytype: disable=attribute-error
          value = cap.lambda_fn()
          return cap.tracetype._to_tensors(value)  # pylint: disable=protected-access
          # pytype: enable=attribute-error
        return lam_fn

      flat.append(make_fn(capture))
    return flat
