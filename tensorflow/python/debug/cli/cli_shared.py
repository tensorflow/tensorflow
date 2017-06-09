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

import math

import numpy as np
import six

from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables

RL = debugger_cli_common.RichLine

# Default threshold number of elements above which ellipses will be used
# when printing the value of the tensor.
DEFAULT_NDARRAY_DISPLAY_THRESHOLD = 2000

COLOR_BLACK = "black"
COLOR_BLUE = "blue"
COLOR_CYAN = "cyan"
COLOR_GRAY = "gray"
COLOR_GREEN = "green"
COLOR_MAGENTA = "magenta"
COLOR_RED = "red"
COLOR_WHITE = "white"
COLOR_YELLOW = "yellow"

TIME_UNIT_US = "us"
TIME_UNIT_MS = "ms"
TIME_UNIT_S = "s"
TIME_UNITS = [TIME_UNIT_US, TIME_UNIT_MS, TIME_UNIT_S]


def bytes_to_readable_str(num_bytes, include_b=False):
  """Generate a human-readable string representing number of bytes.

  The units B, kB, MB and GB are used.

  Args:
    num_bytes: (`int` or None) Number of bytes.
    include_b: (`bool`) Include the letter B at the end of the unit.

  Returns:
    (`str`) A string representing the number of bytes in a human-readable way,
      including a unit at the end.
  """

  if num_bytes is None:
    return str(num_bytes)
  if num_bytes < 1024:
    result = "%d" % num_bytes
  elif num_bytes < 1048576:
    result = "%.2fk" % (num_bytes / 1024.0)
  elif num_bytes < 1073741824:
    result = "%.2fM" % (num_bytes / 1048576.0)
  else:
    result = "%.2fG" % (num_bytes / 1073741824.0)

  if include_b:
    result += "B"
  return result


def time_to_readable_str(value_us, force_time_unit=None):
  """Convert time value to human-readable string.

  Args:
    value_us: time value in microseconds.
    force_time_unit: force the output to use the specified time unit. Must be
      in TIME_UNITS.

  Returns:
    Human-readable string representation of the time value.

  Raises:
    ValueError: if force_time_unit value is not in TIME_UNITS.
  """
  if not value_us:
    return "0"
  if force_time_unit:
    if force_time_unit not in TIME_UNITS:
      raise ValueError("Invalid time unit: %s" % force_time_unit)
    order = TIME_UNITS.index(force_time_unit)
    time_unit = force_time_unit
    return "{:.10g}{}".format(value_us / math.pow(10.0, 3*order), time_unit)
  else:
    order = min(len(TIME_UNITS) - 1, int(math.log(value_us, 10) / 3))
    time_unit = TIME_UNITS[order]
    return "{:.3g}{}".format(value_us / math.pow(10.0, 3*order), time_unit)


def parse_ranges_highlight(ranges_string):
  """Process ranges highlight string.

  Args:
    ranges_string: (str) A string representing a numerical range of a list of
      numerical ranges. See the help info of the -r flag of the print_tensor
      command for more details.

  Returns:
    An instance of tensor_format.HighlightOptions, if range_string is a valid
      representation of a range or a list of ranges.
  """

  ranges = None

  def ranges_filter(x):
    r = np.zeros(x.shape, dtype=bool)
    for range_start, range_end in ranges:
      r = np.logical_or(r, np.logical_and(x >= range_start, x <= range_end))

    return r

  if ranges_string:
    ranges = command_parser.parse_ranges(ranges_string)
    return tensor_format.HighlightOptions(
        ranges_filter, description=ranges_string)
  else:
    return None


def format_tensor(tensor,
                  tensor_name,
                  np_printoptions,
                  print_all=False,
                  tensor_slicing=None,
                  highlight_options=None):
  """Generate formatted str to represent a tensor or its slices.

  Args:
    tensor: (numpy ndarray) The tensor value.
    tensor_name: (str) Name of the tensor, e.g., the tensor's debug watch key.
    np_printoptions: (dict) Numpy tensor formatting options.
    print_all: (bool) Whether the tensor is to be displayed in its entirety,
      instead of printing ellipses, even if its number of elements exceeds
      the default numpy display threshold.
      (Note: Even if this is set to true, the screen output can still be cut
       off by the UI frontend if it consist of more lines than the frontend
       can handle.)
    tensor_slicing: (str or None) Slicing of the tensor, e.g., "[:, 1]". If
      None, no slicing will be performed on the tensor.
    highlight_options: (tensor_format.HighlightOptions) options to highlight
      elements of the tensor. See the doc of tensor_format.format_tensor()
      for more details.

  Returns:
    (str) Formatted str representing the (potentially sliced) tensor.
  """

  if tensor_slicing:
    # Validate the indexing.
    value = command_parser.evaluate_tensor_slice(tensor, tensor_slicing)
    sliced_name = tensor_name + tensor_slicing
  else:
    value = tensor
    sliced_name = tensor_name

  if print_all:
    np_printoptions["threshold"] = value.size
  else:
    np_printoptions["threshold"] = DEFAULT_NDARRAY_DISPLAY_THRESHOLD

  return tensor_format.format_tensor(
      value,
      sliced_name,
      include_metadata=True,
      np_printoptions=np_printoptions,
      highlight_options=highlight_options)


def error(msg):
  """Generate a RichTextLines output for error.

  Args:
    msg: (str) The error message.

  Returns:
    (debugger_cli_common.RichTextLines) A representation of the error message
      for screen output.
  """

  return debugger_cli_common.rich_text_lines_from_rich_line_list([
      RL("ERROR: " + msg, COLOR_RED)])


def _get_fetch_name(fetch):
  """Obtain the name or string representation of a fetch.

  Args:
    fetch: The fetch in question.

  Returns:
    If the attribute 'name' is available, return the name. Otherwise, return
    str(fetch).
  """

  return fetch.name if hasattr(fetch, "name") else str(fetch)


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
    lines.append(_get_fetch_name(fetches))

  return lines


def _recommend_command(command, description, indent=2, create_link=False):
  """Generate a RichTextLines object that describes a recommended command.

  Args:
    command: (str) The command to recommend.
    description: (str) A description of what the command does.
    indent: (int) How many spaces to indent in the beginning.
    create_link: (bool) Whether a command link is to be applied to the command
      string.

  Returns:
    (RichTextLines) Formatted text (with font attributes) for recommending the
      command.
  """

  indent_str = " " * indent

  if create_link:
    font_attr = [debugger_cli_common.MenuItem("", command), "bold"]
  else:
    font_attr = "bold"

  lines = [RL(indent_str) + RL(command, font_attr) + ":",
           indent_str + "  " + description]

  return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)


def get_tfdbg_logo():
  """Make an ASCII representation of the tfdbg logo."""

  lines = [
      "",
      "TTTTTT FFFF DDD  BBBB   GGG ",
      "  TT   F    D  D B   B G    ",
      "  TT   FFF  D  D BBBB  G  GG",
      "  TT   F    D  D B   B G   G",
      "  TT   F    DDD  BBBB   GGG ",
      "",
  ]
  return debugger_cli_common.RichTextLines(lines)


def get_run_start_intro(run_call_count,
                        fetches,
                        feed_dict,
                        tensor_filters):
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
      "Session.run() call #%d:" % run_call_count,
      "", "Fetch(es):"
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
      _recommend_command(
          "run",
          "Execute the run() call with debug tensor-watching",
          create_link=True))
  out.extend(
      _recommend_command(
          "run -n",
          "Execute the run() call without debug tensor-watching",
          create_link=True))
  out.extend(
      _recommend_command(
          "run -t <T>",
          "Execute run() calls (T - 1) times without debugging, then "
          "execute run() once more with debugging and drop back to the CLI"))
  out.extend(
      _recommend_command(
          "run -f <filter_name>",
          "Keep executing run() calls until a dumped tensor passes a given, "
          "registered filter (conditional breakpoint mode)"))

  more_lines = ["    Registered filter(s):"]
  if tensor_filters:
    filter_names = []
    for filter_name in tensor_filters:
      filter_names.append(filter_name)
      command_menu_node = debugger_cli_common.MenuItem(
          "", "run -f %s" % filter_name)
      more_lines.append(RL("        * ") + RL(filter_name, command_menu_node))
  else:
    more_lines.append("        (None)")

  out.extend(
      debugger_cli_common.rich_text_lines_from_rich_line_list(more_lines))

  out.extend(
      _recommend_command(
          "invoke_stepper",
          "Use the node-stepper interface, which allows you to interactively "
          "step through nodes involved in the graph run() call and "
          "inspect/modify their values", create_link=True))

  out.append("")

  out.append_rich_line(RL("For more details, see ") +
                       RL("help.", debugger_cli_common.MenuItem("", "help")) +
                       ".")
  out.append("")

  # Make main menu for the run-start intro.
  menu = debugger_cli_common.Menu()
  menu.append(debugger_cli_common.MenuItem("run", "run"))
  menu.append(debugger_cli_common.MenuItem(
      "invoke_stepper", "invoke_stepper"))
  menu.append(debugger_cli_common.MenuItem("exit", "exit"))
  out.annotations[debugger_cli_common.MAIN_MENU_KEY] = menu

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
    description += "1 fetch (%s); " % _get_fetch_name(fetches)
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
        description += "1 feed (%s)" % (
            key if isinstance(key, six.string_types) else key.name)
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
      RL("!!! An error occurred during the run !!!", "blink"),
      "",
      "You may use the following commands to debug:",
  ]

  out = debugger_cli_common.rich_text_lines_from_rich_line_list(intro_lines)

  out.extend(
      _recommend_command("ni -a -d -t %s" % op_name,
                         "Inspect information about the failing op.",
                         create_link=True))
  out.extend(
      _recommend_command("li -r %s" % op_name,
                         "List inputs to the failing op, recursively.",
                         create_link=True))

  out.extend(
      _recommend_command(
          "lt",
          "List all tensors dumped during the failing run() call.",
          create_link=True))

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
