# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Shared functions and classes for tfdbg command-line interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables


def _get_fetch_names(fetches):
  """Get a flattened list of the names in run() call fetches.

  Args:
    fetches: Fetches of the `Session.run()` call. It maybe a Tensor, an
      Operation or a Variable. It may also be nested lists, tuples or
      dicts. See doc of `Session.run()` for more details.

  Returns:
    (list of str) A flattened list of fetch names from `fetches`.
  """

  lines = []
  if isinstance(fetches, (list, tuple)):
    for fetch in fetches:
      lines.extend(_get_fetch_names(fetch))
  elif isinstance(fetches, dict):
    for key in fetches:
      lines.extend(_get_fetch_names(fetches[key]))
  else:
    # This ought to be a Tensor, an Operation or a Variable, for which the name
    # attribute should be available. (Bottom-out condition of the recursion.)
    lines.append(fetches.name)

  return lines


def _recommend_command(command, description, indent=2):
  """Generate a RichTextLines object that describes a recommended command.

  Args:
    command: (str) The command to recommend.
    description: (str) A description of what the the command does.
    indent: (int) How many spaces to indent in the beginning.

  Returns:
    (RichTextLines) Formatted text (with font attributes) for recommending the
      command.
  """

  indent_str = " " * indent
  lines = [indent_str + command + ":", indent_str + "  " + description]
  font_attr_segs = {0: [(indent, indent + len(command), "bold")]}

  return debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)


def get_run_start_intro(run_call_count, fetches, feed_dict, tensor_filters):
  """Generate formatted intro for run-start UI.

  Args:
    run_call_count: (int) Run call counter.
    fetches: Fetches of the `Session.run()` call. See doc of `Session.run()`
      for more details.
    feed_dict: Feeds to the `Session.run()` call. See doc of `Session.run()`
      for more details.
    tensor_filters: (dict) A dict from tensor-filter name to tensor-filter
      callable.

  Returns:
    (RichTextLines) Formatted intro message about the `Session.run()` call.
  """

  fetch_lines = _get_fetch_names(fetches)

  if not feed_dict:
    feed_dict_lines = ["(Empty)"]
  else:
    feed_dict_lines = []
    for feed_key in feed_dict:
      if isinstance(feed_key, six.string_types):
        feed_dict_lines.append(feed_key)
      else:
        feed_dict_lines.append(feed_key.name)

  intro_lines = [
      "======================================",
      "About to enter Session run() call #%d:" % run_call_count, "",
      "Fetch(es):"
  ]
  intro_lines.extend(["  " + line for line in fetch_lines])
  intro_lines.extend(["", "Feed dict(s):"])
  intro_lines.extend(["  " + line for line in feed_dict_lines])
  intro_lines.extend([
      "======================================", "",
      "Select one of the following commands to proceed ---->"
  ])

  out = debugger_cli_common.RichTextLines(intro_lines)

  out.extend(
      _recommend_command("run",
                         "Execute the run() call with debug tensor-watching"))
  out.extend(
      _recommend_command(
          "run -n", "Execute the run() call without debug tensor-watching"))
  out.extend(
      _recommend_command(
          "run -f <filter_name>",
          "Keep executing run() calls until a dumped tensor passes a given, "
          "registered filter (conditional breakpoint mode)."))

  more_font_attr_segs = {}
  more_lines = ["    Registered filter(s):"]

  if tensor_filters:
    filter_names = []
    for filter_name in tensor_filters:
      filter_names.append(filter_name)
      more_lines.append("        * " + filter_name)
      more_font_attr_segs[len(more_lines) - 1] = [(10, len(more_lines[-1]),
                                                   "green")]
  else:
    more_lines.append("        (None)")

  more_lines.extend([
      "",
      "For more details, see help below:"
      "",
  ])

  out.extend(
      debugger_cli_common.RichTextLines(
          more_lines, font_attr_segs=more_font_attr_segs))

  return out


def get_run_short_description(run_call_count, fetches, feed_dict):
  """Get a short description of the run() call.

  Args:
    run_call_count: (int) Run call counter.
    fetches: Fetches of the `Session.run()` call. See doc of `Session.run()`
      for more details.
    feed_dict: Feeds to the `Session.run()` call. See doc of `Session.run()`
      for more details.

  Returns:
    (str) A short description of the run() call, including information about
      the fetche(s) and feed(s).
  """

  description = "run #%d: " % run_call_count

  if isinstance(fetches, (ops.Tensor, ops.Operation, variables.Variable)):
    description += "1 fetch (%s); " % fetches.name
  else:
    # Could be (nested) list, tuple, dict or namedtuple.
    num_fetches = len(_get_fetch_names(fetches))
    if num_fetches > 1:
      description += "%d fetches; " % num_fetches
    else:
      description += "%d fetch; " % num_fetches

  if not feed_dict:
    description += "0 feeds"
  else:
    if len(feed_dict) == 1:
      for key in feed_dict:
        description += "1 feed (%s)" % key.name
    else:
      description += "%d feeds" % len(feed_dict)

  return description


def get_error_intro(tf_error):
  """Generate formatted intro for TensorFlow run-time error.

  Args:
    tf_error: (errors.OpError) TensorFlow run-time error object.

  Returns:
    (RichTextLines) Formatted intro message about the run-time OpError, with
      sample commands for debugging.
  """

  op_name = tf_error.op.name

  intro_lines = [
      "--------------------------------------",
      "!!! An error occurred during the run !!!",
      "",
      "You may use the following commands to debug:",
  ]
  intro_font_attr_segs = {1: [(0, len(intro_lines[1]), "blink")]}

  out = debugger_cli_common.RichTextLines(
      intro_lines, font_attr_segs=intro_font_attr_segs)

  out.extend(
      _recommend_command("ni %s" % op_name,
                         "Inspect information about the failing op."))
  out.extend(
      _recommend_command("li -r %s" % op_name,
                         "List inputs to the failing op, recursively."))
  out.extend(
      _recommend_command(
          "lt", "List all tensors dumped during the failing run() call."))

  more_lines = [
      "",
      "Op name:    " + op_name,
      "Error type: " + str(type(tf_error)),
      "",
      "Details:",
      str(tf_error),
      "",
      "WARNING: Using client GraphDef due to the error, instead of "
      "executor GraphDefs.",
      "--------------------------------------",
      "",
  ]

  out.extend(debugger_cli_common.RichTextLines(more_lines))

  return out
