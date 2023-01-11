# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=unidiomatic-typecheck
"""ExportedConcreteFunction class and its associated functions.

Part of saved model utils, a shim layer for working with
functions exported/restored from saved models.
This functionality should ultimately be moved into a first-class core API.
"""

import gc
from tensorflow.python.trackable import base as trackable


# TODO(kathywu): Delete this class when ConcreteFunctions can be copied with new
# captures.
class ExportedConcreteFunction(trackable.Trackable):
  """A callable class that uses captures from the exported SavedModel graph."""
  __slots__ = ("function", "tensor_map")

  def __init__(self, function, tensor_map):
    self.function = function
    self.tensor_map = tensor_map

  def __call__(self, *args, **kwargs):
    _, _, filtered_flat_args = (
        self.function._function_spec.canonicalize_function_inputs(args, kwargs))
    export_captures = _map_captures_to_created_tensors(
        self.function.graph.captures, self.tensor_map, self.function)
    return self.function._call_flat(filtered_flat_args, export_captures)


def _map_captures_to_created_tensors(original_captures, tensor_map, function):
  """Maps eager tensors captured by a function to Graph resources for export.

  Args:
    original_captures: A dictionary mapping from tensors captured by the
      function to interior placeholders for those tensors (inside the function
      body).
    tensor_map: A dictionary mapping from resource tensors owned by the eager
      context to resource tensors in the exported graph.
    function: Function with the original captures. Only used when raising the
      AssertionError.

  Returns:
    A list of stand-in tensors which belong to the exported graph, corresponding
    to the function's captures.

  Raises:
    AssertionError: If the function references a resource which is not part of
      `tensor_map`.
  """
  export_captures = []
  for exterior, interior in original_captures:
    mapped_resource = tensor_map.get(exterior, None)
    if mapped_resource is None:
      _raise_untracked_capture_error(function.name, exterior, interior)
    export_captures.append(mapped_resource)
  return export_captures


def _raise_untracked_capture_error(function_name, capture,
                                   internal_capture=None,
                                   node_path=None):
  """Raises AssertionError due to being unable to export a function."""
  msg = ("Tried to export a function which references an 'untracked' resource. "
         "TensorFlow objects (e.g. tf.Variable) captured by functions must be "
         "'tracked' by assigning them to an attribute of a tracked object or "
         "assigned to an attribute of the main object directly. See the "
         "information below:"
         f"\n\tFunction name = {function_name}")

  if node_path is not None:
    msg += f"\n\tPath to Function = {node_path}"

  msg += f"\n\tCaptured Tensor = {capture}"
  msg += f"\n\t{_get_trackable_parent_error_string(capture)}"

  if internal_capture is not None:
    msg += f"\n\tInternal Tensor = {internal_capture}"
  raise AssertionError(msg)


def _get_trackable_parent_error_string(capture):
  """Gets error string with the capture's parent object."""
  parent = getattr(capture, "_parent_trackable", None)
  if parent is not None:
    return f"Trackable referencing this tensor = {parent()}"

  # Try to figure out where the resource came from by iterating over objects
  # which reference it. This is slow and doesn't help us figure out how to
  # match it to other objects when loading the SavedModel as a checkpoint,
  # so we can't continue saving. But we can at least tell the user what
  # needs attaching.
  trackable_referrers = []
  for primary_referrer in gc.get_referrers(capture):
    if isinstance(primary_referrer, trackable.Trackable):
      trackable_referrers.append(primary_referrer)
    for secondary_referrer in gc.get_referrers(primary_referrer):
      if isinstance(secondary_referrer, trackable.Trackable):
        trackable_referrers.append(secondary_referrer)
  return ("Trackable Python objects referring to this tensor "
          "(from gc.get_referrers, limited to two hops) = [\n\t\t{}]"
          .format("\n\t\t".join([repr(obj) for obj in trackable_referrers])))
