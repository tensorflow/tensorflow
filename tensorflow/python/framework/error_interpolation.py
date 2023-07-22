# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Function for interpolating formatted errors from the TensorFlow runtime.

Exposes the function `interpolate` to interpolate messages with tags of the form
{{type name}}.
"""

import collections
import os
import re
import site
import traceback

from tensorflow.core.framework import graph_debug_info_pb2

_NAME_REGEX = r"[A-Za-z0-9_.][A-Za-z0-9_.\-/]*?"
_TAG_REGEX = fr"{{{{(?P<type>{_NAME_REGEX}) (?P<name>{_NAME_REGEX})}}}}"
_INTERPOLATION_REGEX = fr"(?P<sep>.*?)(?P<tag>{_TAG_REGEX})"
_INTERPOLATION_PATTERN = re.compile(_INTERPOLATION_REGEX, re.DOTALL)

_ParseTag = collections.namedtuple("_ParseTag", ["type", "name"])


# Remove the last three path components from this module's file (i.e.
# python/framework/error_interpolation.py) so that we have an absolute path
# prefix to the root of the installation.
_FRAMEWORK_COMMON_PREFIX = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__)))

# Sub-directories under the common prefix that are considered part of the
# framework.
# Note that keras code lives outside of tensorflow directory, we need to walk
# up the directory tree and find it.
_FRAMEWORK_PATH_PREFIXES = [
    os.path.join(_FRAMEWORK_COMMON_PREFIX, "python") + os.sep,
    os.path.join(_FRAMEWORK_COMMON_PREFIX, "contrib") + os.sep,
    os.path.join(os.path.dirname(_FRAMEWORK_COMMON_PREFIX),
                 "py", "keras") + os.sep,
]

# Patterns of filename patterns that should be considered internal to
# the TensorFlow framework.
_FRAMEWORK_FILENAME_PATTERNS = [
    re.compile(r"<embedded"),
]

# This is for OSS keras, since the package is load from local python env,
# but we don't know exactly where it is installed. Matching to keyword
# "keras".
try:
  _FRAMEWORK_PATH_PREFIXES.extend([
      os.path.join(package_path, "keras") + os.sep
      for package_path in site.getsitepackages() + [site.getusersitepackages()]
  ])
except AttributeError:
  # if site.getsitepackages is not available somehow, we just use the "keras" as
  # the keyword to do the match.
  _FRAMEWORK_FILENAME_PATTERNS.append(re.compile(r"keras"))

# Patterns of filename patterns that should be considered external to
# TensorFlow regardless of framework prefix match.
_EXTERNAL_FILENAME_PATTERNS = [
    # Explicitly treat test frames as not part of the framework.
    re.compile(r"_test\.py$"),
]


def parse_message(message):
  """Extract function tags and node tags from a message.

  Tags are named tuples representing the string {{type name}}. For example,
  in "123{{node Foo}}456{{function_node Bar}}789", there are two tags: a node
  tag and a function tag.

  Args:
    message: An error message, possibly from an OpError.

  Returns:
    A tuple containing the original message with function nodes stripped,
    function tags, and node tags.

    For example, if message is "123{{node Foo}}456{{function_node Bar}}789"
    then this function returns ("123{{node Foo}}456789",
    [_ParseTag("function_node", "Bar")], [_ParseTag("node", "Foo")]).
  """
  error_message = []
  func_tags = []
  node_tags = []
  pos = 0
  for match in re.finditer(_INTERPOLATION_PATTERN, message):
    parsed_tag = _ParseTag(match.group("type"), match.group("name"))
    if parsed_tag.type == "function_node":
      error_message.append(match.group("sep"))
      func_tags.append(parsed_tag)
    else:
      error_message.append(match.group())
      node_tags.append(parsed_tag)
    pos = match.end()
  error_message.append(message[pos:])
  return "".join(error_message), func_tags, node_tags


def _compute_device_summary_from_list(name, device_assignment_list, prefix=""):
  """Return a summary of an op's device function stack.

  Args:
    name: The name of the op.
    device_assignment_list: The op._device_assignments list.
    prefix:  An optional string prefix used before each line of the multi-
        line string returned by this function.

  Returns:
    A multi-line string similar to:
        Device assignments active during op 'foo' creation:
          with tf.device(/cpu:0): <test_1.py:27>
          with tf.device(some_func<foo.py, 123>): <test_2.py:38>
    The first line will have no padding to its left by default.  Subsequent
    lines will have two spaces of left-padding.  Use the prefix argument
    to increase indentation.
  """
  if not device_assignment_list:
    message = "No device assignments were active during op '%s' creation."
    message %= name
    return prefix + message

  str_list = []
  str_list.append(
      "%sDevice assignments active during op '%s' creation:" % (prefix, name))

  for traceable_obj in device_assignment_list:
    location_summary = "<{file}:{line}>".format(
        file=traceable_obj.filename, line=traceable_obj.lineno)
    subs = {
        "prefix": prefix,
        "indent": "  ",
        "dev_name": traceable_obj.obj,
        "loc": location_summary,
    }
    str_list.append(
        "{prefix}{indent}with tf.device({dev_name}): {loc}".format(**subs))

  return "\n".join(str_list)


def _compute_device_assignment_summary_from_op(op, prefix=""):
  # pylint: disable=protected-access
  return _compute_device_summary_from_list(op.name, op._device_assignments,
                                           prefix)
  # pylint: enable=protected-access


def _compute_colocation_summary_from_dict(name, colocation_dict, prefix=""):
  """Return a summary of an op's colocation stack.

  Args:
    name: The op name.
    colocation_dict: The op._colocation_dict.
    prefix:  An optional string prefix used before each line of the multi-
        line string returned by this function.

  Returns:
    A multi-line string similar to:
        Node-device colocations active during op creation:
          with tf.compat.v1.colocate_with(test_node_1): <test_1.py:27>
          with tf.compat.v1.colocate_with(test_node_2): <test_2.py:38>
    The first line will have no padding to its left by default.  Subsequent
    lines will have two spaces of left-padding.  Use the prefix argument
    to increase indentation.
  """
  if not colocation_dict:
    message = "No node-device colocations were active during op '%s' creation."
    message %= name
    return prefix + message

  str_list = []
  str_list.append("%sNode-device colocations active during op '%s' creation:" %
                  (prefix, name))

  for coloc_name, location in colocation_dict.items():
    location_summary = "<{file}:{line}>".format(
        file=location.filename, line=location.lineno)
    subs = {
        "prefix": prefix,
        "indent": "  ",
        "name": coloc_name,
        "loc": location_summary,
    }
    str_list.append(
        "{prefix}{indent}with tf.colocate_with({name}): {loc}".format(**subs))

  return "\n".join(str_list)


def _compute_colocation_summary_from_op(op, prefix=""):
  """Fetch colocation file, line, and nesting and return a summary string."""
  # pylint: disable=protected-access
  return _compute_colocation_summary_from_dict(op.name, op._colocation_dict,
                                               prefix)
  # pylint: enable=protected-access


def _is_framework_filename(filename):
  """Returns whether a filename should be considered a part of the framework.

  A file is part of the framework if it does not match a pattern in
  _EXTERNAL_FILENAME_PATTERNS and it either matches a pattern in
  _FRAMEWORK_FILENAME_PATTERNS or starts with a _FRAMEWORK_PATH_PREFIXES prefix.

  Args:
    filename: A filename string.

  Returns:
    Whether the filename should be considered to be internal to the
    TensorFlow framework for the purposes of reporting errors.
  """
  for pattern in _EXTERNAL_FILENAME_PATTERNS:
    if pattern.search(filename):
      return False
  for pattern in _FRAMEWORK_FILENAME_PATTERNS:
    if pattern.search(filename):
      return True
  for prefix in _FRAMEWORK_PATH_PREFIXES:
    if filename.startswith(prefix):
      return True
  return False


def _find_index_of_defining_frame(tb):
  """Return index in op.traceback with first 'useful' frame.

  This method reads through the stack stored in op.traceback looking for the
  innermost frame which (hopefully) belongs to the caller.  It accomplishes this
  by rejecting frames deemed to be part of the TensorFlow framework (by
  pattern matching the filename).

  Args:
    tb: A list of traceback frames (as from Operation.traceback).

  Returns:
    Integer index into op.traceback where the first non-TF file was found
    (innermost to outermost), or 0 (for the outermost stack frame) if all files
    came from TensorFlow.
  """
  # Index 0 of traceback is the outermost frame.
  size = len(tb)
  filenames = [frame.filename for frame in tb]
  # We process the filenames from the innermost frame to outermost.
  for idx, filename in enumerate(reversed(filenames)):
    is_framework = _is_framework_filename(filename)
    if not is_framework:
      # Consider this to be the defining frame.
      return size - idx - 1
  return 0


# TODO(feyu): follow up with users of this function (saved model)
# to see what 'useful' means and whether we can obliviate this.
def _compute_useful_frames(tb, num):
  """Return a list of frames, which form a 'useful' stack.

  Starting from the defining frame to the outermost one, this method computes
  the contiguous portion of the 'useful' stack trace and returns the selected
  frames.

  Args:
    tb: A list of traceback frames (as from Operation.traceback).
    num: total number of frames to return.

  Returns:
    A list of frames.
  """
  defining_frame_index = _find_index_of_defining_frame(tb)
  # The stack trace is collected from two lines before the defining frame in the
  # model file to the outermost with `num` frames at most. These two extra lines
  # are included from the TensorFlow library to give the context which node is
  # defined.
  innermost_excluded = min(defining_frame_index + 2 + 1, len(tb))
  outermost_included = max(innermost_excluded - num, 0)
  return tb[outermost_included:innermost_excluded]


def create_graph_debug_info_def(func_named_operations):
  """Construct and returns a `GraphDebugInfo` protocol buffer.

  Args:
    func_named_operations: An iterable of (func_name, op.Operation) tuples
      where the Operation instances have a _traceback members. The func_name
      should be the empty string for operations in the top-level Graph.

  Returns:
    GraphDebugInfo protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
  # Creates an empty GraphDebugInfoDef proto.
  graph_debug_info_def = graph_debug_info_pb2.GraphDebugInfo()

  # Gets the file names and line numbers for the exported node names. Also
  # collects the unique file names.
  all_file_names = set()
  node_to_trace = {}
  for func_name, op in func_named_operations:
    if op.traceback is None:
      continue
    # Gets the stack trace of the operation and then the file location.
    node_name = op.name + "@" + func_name
    node_to_trace[node_name] = _compute_useful_frames(op.traceback, 10)
    for frame in node_to_trace[node_name]:
      all_file_names.add(frame.filename)

  # Sets the `files` field in the GraphDebugInfo proto
  graph_debug_info_def.files.extend(all_file_names)

  # Builds a mapping between file names and index of the `files` field, so we
  # only store the indexes for the nodes in the GraphDebugInfo.
  file_to_index = dict(
      [(y, x) for x, y in enumerate(graph_debug_info_def.files)])

  # Creates the FileLineCol proto for each node and sets the value in the
  # GraphDebugInfo proto. We only store the file name index for each node to
  # save the storage space.
  for node_name, frames in node_to_trace.items():
    trace_def = graph_debug_info_def.traces[node_name]
    for frame in reversed(frames):
      trace_def.file_line_cols.add(
          file_index=file_to_index[frame.filename],
          line=frame.lineno)

  return graph_debug_info_def


def merge_graph_debug_info_def(per_fn_info):
  """Construct and returns a `GraphDebugInfo` protocol buffer.

  Args:
    per_fn_info: An iterable of (func_name, GraphDebugInfo) tuples.

  Returns:
    GraphDebugInfo protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
  graph_debug_info_def = graph_debug_info_pb2.GraphDebugInfo()

  all_file_names = set()
  for _, fn_info in per_fn_info:
    all_file_names.update(fn_info.files)
  # Ensure determinism.
  all_file_names = sorted(all_file_names)

  graph_debug_info_def.files.extend(all_file_names)
  file_to_index = dict(
      [(y, x) for x, y in enumerate(graph_debug_info_def.files)])

  for fn_name, fn_info in per_fn_info:
    for fn_node_name, fn_trace in fn_info.traces.items():
      trace_def = graph_debug_info_def.traces[fn_node_name + "@" + fn_name]
      for fn_frame in fn_trace.file_line_cols:
        trace_def.file_line_cols.add(
            file_index=file_to_index[fn_info.files[fn_frame.file_index]],
            line=fn_frame.line,
            col=fn_frame.col,
            func=fn_frame.func,
            code=fn_frame.code,
        )

  return graph_debug_info_def


def _compute_field_dict(op):
  r"""Return a dictionary mapping interpolation tokens to values.

  Args:
    op: op.Operation object.

  Returns:
    A dictionary mapping string tokens to string values.  The keys are shown
    below along with example values.
    {
      "file": "tool_utils.py",
      "lineno": "124",
      "line": "  source code line",
      "defined_at": " (defined at tool_utils.py:124)",
      "colocations":
          '''Node-device colocations active during op creation:
               with tf.compat.v1.colocate_with(test_node_1): <test_1.py:27>
               with tf.compat.v1.colocate_with(test_node_2): <test_2.py:38>'''
      "devices":
          '''Device assignments active during op 'foo' creation:
               with tf.device(/cpu:0): <test_1.py:27>
               with tf.device(some_func<foo.py, 123>): <test_2.py:38>'''
      "devs_and_colocs": A concatenation of colocations and devices, e.g.
          '''Node-device colocations active during op creation:
               with tf.compat.v1.colocate_with(test_node_1): <test_1.py:27>
               with tf.compat.v1.colocate_with(test_node_2): <test_2.py:38>'''
             Device assignments active during op 'foo' creation:
               with tf.device(/cpu:0): <test_1.py:27>
               with tf.device(some_func<foo.py, 123>): <test_2.py:38>'''
    }
  """
  # TODO(xjun): colocation and device info are not displayed. Consider
  # removing them or using vlog.
  colocation_summary = _compute_colocation_summary_from_op(op)
  device_summary = _compute_device_assignment_summary_from_op(op)
  combined_summary = "\n".join([colocation_summary, device_summary])

  if op.traceback is None:
    # Some ops synthesized on as part of function or control flow definition
    # do not have tracebacks.
    filename = "<unknown>"
    definition_traceback = ""
    lineno = 0
    line = ""
    defined_at = "<unknown>"
  else:
    frame = op.traceback.last_user_frame()
    filename = frame.filename
    definition_traceback = traceback.format_list(op.traceback.get_user_frames())
    lineno = frame.lineno
    line = frame.line
    defined_at = f"{filename}:{lineno:d}"

  field_dict = {
      "colocations": colocation_summary,
      "devices": device_summary,
      "devs_and_colocs": combined_summary,
      "defined_at": defined_at,
      "file": filename,
      "lineno": lineno,
      "line": line,
      "definition_traceback": definition_traceback,
  }
  return field_dict


def _build_node_error_message(op):
  """Returns the formatted error message for the given op.

  Args:
    op: The node.

  Returns:
    The formatted error message for the given op with traceback.
  """
  node_error_message = [
      f"Detected at node {op.name!r} defined at (most recent call last):"
  ]
  field_dict = _compute_field_dict(op)

  # Add node traceback.
  for frame in field_dict["definition_traceback"]:
    if "<embedded" not in frame:
      node_error_message.extend(
          [f"  {line}" for line in frame.split("\n") if line.strip()])

  # Add node name.
  node_error_message.append(f"Node: {op.name!r}")

  return "\n".join(node_error_message)


def interpolate_graph(message, graph):
  """Interpolates an error message.

  The error message can contain tags of form `{{node_type node_name}}`
  which will be parsed to identify the tf.Graph and op. If the op contains
  traceback, the traceback will be attached to the error message.

  Args:
    message: A string to interpolate.
    graph: ops.Graph object containing all nodes referenced in the error
        message.

  Returns:
    The error message string with node definition traceback.
  """
  parsed_messaged, _, node_tags = parse_message(message)
  error_message = ["Graph execution error:", ""]
  for tag in node_tags:
    try:
      op = graph.get_operation_by_name(tag.name)
    except KeyError:
      continue
    else:
      error_message.append(_build_node_error_message(op))

  error_message.append(parsed_messaged.strip())
  return "\n".join(error_message)
