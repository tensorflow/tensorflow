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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os
import re

import six

from tensorflow.python.util import tf_stack

_NAME_REGEX = r"[A-Za-z0-9.][A-Za-z0-9_.\-/]*?"
_TAG_REGEX = r"{{{{({name}) ({name})}}}}".format(name=_NAME_REGEX)
_INTERPOLATION_REGEX = r"^(.*?)({tag})".format(tag=_TAG_REGEX)
_INTERPOLATION_PATTERN = re.compile(_INTERPOLATION_REGEX, re.DOTALL)

_ParseTag = collections.namedtuple("_ParseTag", ["type", "name"])

_BAD_FILE_SUBSTRINGS = [
    os.path.join("tensorflow", "python"),
    os.path.join("tensorflow", "contrib"),
    "<embedded",
]


def _parse_message(message):
  """Parses the message.

  Splits the message into separators and tags. Tags are named tuples
  representing the string {{type name}} and they are separated by
  separators. For example, in "123{{node Foo}}456{{node Bar}}789", there are
  two tags and three separators. The separators are the numeric characters.

  Args:
    message: String to parse

  Returns:
    (list of separator strings, list of _ParseTags).

    For example, if message is "123{{node Foo}}456" then this function
    returns (["123", "456"], [_ParseTag("node", "Foo")])
  """
  seps = []
  tags = []
  pos = 0
  while pos < len(message):
    match = re.match(_INTERPOLATION_PATTERN, message[pos:])
    if match:
      seps.append(match.group(1))
      tags.append(_ParseTag(match.group(3), match.group(4)))
      pos += match.end()
    else:
      break
  seps.append(message[pos:])
  return seps, tags


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
          with tf.colocate_with(test_node_1): <test_1.py:27>
          with tf.colocate_with(test_node_2): <test_2.py:38>
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


def _find_index_of_defining_frame_for_op(op):
  """Return index in op.traceback with first 'useful' frame.

  This method reads through the stack stored in op.traceback looking for the
  innermost frame which (hopefully) belongs to the caller.  It accomplishes this
  by rejecting frames whose filename appears to come from TensorFlow (see
  error_interpolation._BAD_FILE_SUBSTRINGS for the list of rejected substrings).

  Args:
    op: the Operation object for which we would like to find the defining
        location.

  Returns:
    Integer index into op.traceback where the first non-TF file was found
    (innermost to outermost), or 0 (for the outermost stack frame) if all files
    came from TensorFlow.
  """
  # Index 0 of tf_traceback is the outermost frame.
  tf_traceback = op.traceback
  size = len(tf_traceback)
  filenames = [frame[tf_stack.TB_FILENAME] for frame in tf_traceback]
  # We process the filenames from the innermost frame to outermost.
  for idx, filename in enumerate(reversed(filenames)):
    contains_bad_substrings = [ss in filename for ss in _BAD_FILE_SUBSTRINGS]
    if not any(contains_bad_substrings):
      return size - idx - 1
  return 0


def _get_defining_frame_from_op(op):
  """Find and return stack frame where op was defined."""
  frame_index = _find_index_of_defining_frame_for_op(op)
  return op.traceback[frame_index]


def compute_field_dict(op, strip_file_prefix=""):
  """Return a dictionary mapping interpolation tokens to values.

  Args:
    op: op.Operation object having a _traceback member.
    strip_file_prefix: The common path in the stacktrace. We remove the prefix
    from the file names.

  Returns:
    A dictionary mapping string tokens to string values.  The keys are shown
    below along with example values.
    {
      "file": "tool_utils.py",
      "line": "124",
      "defined_at": " (defined at tool_utils.py:124)",
      "colocations":
          '''Node-device colocations active during op creation:
               with tf.colocate_with(test_node_1): <test_1.py:27>
               with tf.colocate_with(test_node_2): <test_2.py:38>'''
      "devices":
          '''Device assignments active during op 'foo' creation:
               with tf.device(/cpu:0): <test_1.py:27>
               with tf.device(some_func<foo.py, 123>): <test_2.py:38>'''
      "devs_and_colocs": A concatenation of colocations and devices, e.g.
          '''Node-device colocations active during op creation:
               with tf.colocate_with(test_node_1): <test_1.py:27>
               with tf.colocate_with(test_node_2): <test_2.py:38>'''
             Device assignments active during op 'foo' creation:
               with tf.device(/cpu:0): <test_1.py:27>
               with tf.device(some_func<foo.py, 123>): <test_2.py:38>'''
    }
  """
  frame = _get_defining_frame_from_op(op)
  filename = frame[tf_stack.TB_FILENAME]
  if filename.startswith(strip_file_prefix):
    filename = filename[len(strip_file_prefix):]
  lineno = frame[tf_stack.TB_LINENO]
  defined_at = " (defined at %s:%d)" % (filename, lineno)
  colocation_summary = _compute_colocation_summary_from_op(op)
  device_summary = _compute_device_assignment_summary_from_op(op)
  combined_summary = "\n".join([colocation_summary, device_summary])

  field_dict = {
      "file": filename,
      "line": lineno,
      "defined_at": defined_at,
      "colocations": colocation_summary,
      "devices": device_summary,
      "devs_and_colocs": combined_summary,
  }
  return field_dict


def traceback_files_common_prefix(all_ops):
  """Determines the common prefix from the paths of the stacktrace of 'all_ops'.

  For example, if the paths are '/foo/bar/baz/' and '/foo/car', this would
  return '/foo'.

  Args:
    all_ops: All the input nodes in the form of a list of lists of ops.

  Returns:
    The common prefix.
  """
  files = set()
  for ops in all_ops:
    if ops is None:
      continue
    for op in ops:
      for frame in op.traceback:
        filename = frame[tf_stack.TB_FILENAME]
        if "<embedded" not in filename:
          files.add(filename)
  return os.path.split(os.path.commonprefix(list(files)))[0]


def _sources_for_node(name, graph):
  """Gets the top-level root input nodes for 'name' node.

  We recursively traverse the graph from 'name' node to its inputs and collect
  all the nodes which don't have any inputs.

  Args:
    name: The name of the node.
    graph: The graph containing the node.

  Returns:
    The unique top-level root input nodes.
  """
  def _helper(name, graph, seen_names, inputs):
    """Recursive helper. 'seen_names' and 'inputs' are mutated."""
    if name.startswith("^"):
      name = name[1:]
    try:
      tensor = graph.get_tensor_by_name(name)
      op = tensor.op
    except (KeyError, ValueError):
      try:
        op = graph.get_operation_by_name(name)
      except KeyError:
        return
    name = op.name
    if name in seen_names:
      return
    seen_names.add(name)
    if not op.node_def.input:
      inputs.add(op)
      return
    for n in op.node_def.input:
      _helper(n, graph, seen_names, inputs)

  names = set()
  inputs = set()
  _helper(name, graph, names, inputs)
  return list(inputs)


def _build_error_message(op, input_ops, common_prefix):
  """Returns the formatted error message for the given op.

  Args:
    op: The node.
    input_ops: The input nodes to the 'op' node
    common_prefix: The prefix path common to the stacktrace of inputs.

  Returns:
    The formatted error message for the given op. The error message also
    includes the information about the input sources for the given op.
  """
  field_dict = compute_field_dict(op, common_prefix)
  msg = "node %s%s " % (op.name, field_dict["defined_at"])
  input_debug_info = []
  # This stores the line numbers that we have already printed.
  done = set()
  done.add(field_dict["defined_at"])
  for op_inp in input_ops:
    field_dict_inp = compute_field_dict(op_inp, common_prefix)
    if field_dict_inp["defined_at"] not in done:
      input_debug_info.append(
          " %s%s" % (op_inp.name, field_dict_inp["defined_at"]))
      done.add(field_dict_inp["defined_at"])
  if input_debug_info:
    end_msg = ("\nInput Source operations connected to node %s:\n") % (op.name)
    end_msg += "\t\n".join(input_debug_info)
  else:
    end_msg = ""
  return msg, end_msg


def interpolate(error_message, graph):
  """Interpolates an error message.

  The error message can contain tags of the form `{{type name}}` which will be
  replaced. For example: "{{node <name>}}" would get expanded to:
  "node <name>(defined at <path>)".

  Args:
    error_message: A string to interpolate.
    graph: ops.Graph object containing all nodes referenced in the error
        message.

  Returns:
    The string with tags of the form {{type name}} interpolated.
  """
  seps, tags = _parse_message(error_message)
  subs = []
  end_msg = collections.defaultdict(list)
  tagged_ops = []

  for t in tags:
    try:
      op = graph.get_operation_by_name(t.name)
    except KeyError:
      op = None
    if op is None:
      tagged_ops.append(None)
    else:
      tagged_ops.append([op] + _sources_for_node(op.name, graph))

  common_prefix = traceback_files_common_prefix(tagged_ops)
  for tag, ops in zip(tags, tagged_ops):
    msg = "{{%s %s}}" % (tag.type, tag.name)
    if ops is not None:
      if tag.type == "node":
        msg, source_msg = _build_error_message(ops[0], ops[1:], common_prefix)
        if source_msg:
          end_msg["source_nodes"].append(source_msg)
      elif tag.type == "colocation_node":
        field_dict = compute_field_dict(ops[0], common_prefix)
        msg = "node %s%s placed on device %s " % (
            ops[0].name, field_dict["defined_at"], field_dict["devices"])
        end_msg["colocations"].append(field_dict["devs_and_colocs"])
    subs.append(msg)

  if "source_nodes" in end_msg:
    subs.append("\n\nErrors may have originated from an input operation.")
    subs.append("\n".join(end_msg["source_nodes"]))
    end_msg.pop("source_nodes", None)
  for k, messages in end_msg.items():
    subs.append("Additional information about %s:" % k)
    subs.append("\n".join(messages))

  return "".join(
      itertools.chain(*six.moves.zip_longest(seps, subs, fillvalue="")))
