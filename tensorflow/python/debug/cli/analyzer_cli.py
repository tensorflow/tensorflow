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
"""CLI Backend for the Analyzer Part of the Debugger.

The analyzer performs post hoc analysis of dumped intermediate tensors and
graph structure information from debugged Session.run() calls.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import re

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils

RL = debugger_cli_common.RichLine

# String constants for the depth-dependent hanging indent at the beginning
# of each line.
HANG_UNFINISHED = "|  "  # Used for unfinished recursion depths.
HANG_FINISHED = "   "
HANG_SUFFIX = "|- "

# String constant for displaying depth and op type.
DEPTH_TEMPLATE = "(%d) "
OP_TYPE_TEMPLATE = "[%s] "

# String constants for control inputs/outputs, etc.
CTRL_LABEL = "(Ctrl) "
ELLIPSIS = "..."

SORT_TENSORS_BY_TIMESTAMP = "timestamp"
SORT_TENSORS_BY_DUMP_SIZE = "dump_size"
SORT_TENSORS_BY_OP_TYPE = "op_type"
SORT_TENSORS_BY_TENSOR_NAME = "tensor_name"


def _add_main_menu(output,
                   node_name=None,
                   enable_list_tensors=True,
                   enable_node_info=True,
                   enable_print_tensor=True,
                   enable_list_inputs=True,
                   enable_list_outputs=True):
  """Generate main menu for the screen output from a command.

  Args:
    output: (debugger_cli_common.RichTextLines) the output object to modify.
    node_name: (str or None) name of the node involved (if any). If None,
      the menu items node_info, list_inputs and list_outputs will be
      automatically disabled, overriding the values of arguments
      enable_node_info, enable_list_inputs and enable_list_outputs.
    enable_list_tensors: (bool) whether the list_tensor menu item will be
      enabled.
    enable_node_info: (bool) whether the node_info item will be enabled.
    enable_print_tensor: (bool) whether the print_tensor item will be enabled.
    enable_list_inputs: (bool) whether the item list_inputs will be enabled.
    enable_list_outputs: (bool) whether the item list_outputs will be enabled.
  """

  menu = debugger_cli_common.Menu()

  menu.append(
      debugger_cli_common.MenuItem(
          "list_tensors", "list_tensors", enabled=enable_list_tensors))

  if node_name:
    menu.append(
        debugger_cli_common.MenuItem(
            "node_info",
            "node_info -a -d -t %s" % node_name,
            enabled=enable_node_info))
    menu.append(
        debugger_cli_common.MenuItem(
            "print_tensor",
            "print_tensor %s" % node_name,
            enabled=enable_print_tensor))
    menu.append(
        debugger_cli_common.MenuItem(
            "list_inputs",
            "list_inputs -c -r %s" % node_name,
            enabled=enable_list_inputs))
    menu.append(
        debugger_cli_common.MenuItem(
            "list_outputs",
            "list_outputs -c -r %s" % node_name,
            enabled=enable_list_outputs))
  else:
    menu.append(
        debugger_cli_common.MenuItem(
            "node_info", None, enabled=False))
    menu.append(
        debugger_cli_common.MenuItem("print_tensor", None, enabled=False))
    menu.append(
        debugger_cli_common.MenuItem("list_inputs", None, enabled=False))
    menu.append(
        debugger_cli_common.MenuItem("list_outputs", None, enabled=False))

  menu.append(
      debugger_cli_common.MenuItem("run_info", "run_info"))
  menu.append(
      debugger_cli_common.MenuItem("help", "help"))

  output.annotations[debugger_cli_common.MAIN_MENU_KEY] = menu


class DebugAnalyzer(object):
  """Analyzer for debug data from dump directories."""

  _TIMESTAMP_COLUMN_HEAD = "t (ms)"
  _DUMP_SIZE_COLUMN_HEAD = "Size (B)"
  _OP_TYPE_COLUMN_HEAD = "Op type"
  _TENSOR_NAME_COLUMN_HEAD = "Tensor name"

  # Op types to be omitted when generating descriptions of graph structure.
  _GRAPH_STRUCT_OP_TYPE_BLACKLIST = (
      "_Send", "_Recv", "_HostSend", "_HostRecv", "_Retval")

  def __init__(self, debug_dump, config):
    """DebugAnalyzer constructor.

    Args:
      debug_dump: A DebugDumpDir object.
      config: A `cli_config.CLIConfig` object that carries user-facing
        configurations.
    """

    self._debug_dump = debug_dump
    self._evaluator = evaluator.ExpressionEvaluator(self._debug_dump)

    # Initialize tensor filters state.
    self._tensor_filters = {}

    self._build_argument_parsers(config)
    config.set_callback("graph_recursion_depth",
                        self._build_argument_parsers)

    # TODO(cais): Implement list_nodes.

  def _build_argument_parsers(self, config):
    """Build argument parsers for DebugAnalayzer.

    Args:
      config: A `cli_config.CLIConfig` object.

    Returns:
      A dict mapping command handler name to `ArgumentParser` instance.
    """
    # Argument parsers for command handlers.
    self._arg_parsers = {}

    # Parser for list_tensors.
    ap = argparse.ArgumentParser(
        description="List dumped intermediate tensors.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-f",
        "--tensor_filter",
        dest="tensor_filter",
        type=str,
        default="",
        help="List only Tensors passing the filter of the specified name")
    ap.add_argument(
        "-fenn",
        "--filter_exclude_node_names",
        dest="filter_exclude_node_names",
        type=str,
        default="",
        help="When applying the tensor filter, exclude node with names "
        "matching the regular expression. Applicable only if --tensor_filter "
        "or -f is used.")
    ap.add_argument(
        "-n",
        "--node_name_filter",
        dest="node_name_filter",
        type=str,
        default="",
        help="filter node name by regex.")
    ap.add_argument(
        "-t",
        "--op_type_filter",
        dest="op_type_filter",
        type=str,
        default="",
        help="filter op type by regex.")
    ap.add_argument(
        "-s",
        "--sort_by",
        dest="sort_by",
        type=str,
        default=SORT_TENSORS_BY_TIMESTAMP,
        help=("the field to sort the data by: (%s | %s | %s | %s)" %
              (SORT_TENSORS_BY_TIMESTAMP, SORT_TENSORS_BY_DUMP_SIZE,
               SORT_TENSORS_BY_OP_TYPE, SORT_TENSORS_BY_TENSOR_NAME)))
    ap.add_argument(
        "-r",
        "--reverse",
        dest="reverse",
        action="store_true",
        help="sort the data in reverse (descending) order")
    self._arg_parsers["list_tensors"] = ap

    # Parser for node_info.
    ap = argparse.ArgumentParser(
        description="Show information about a node.", usage=argparse.SUPPRESS)
    ap.add_argument(
        "node_name",
        type=str,
        help="Name of the node or an associated tensor, e.g., "
        "hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0")
    ap.add_argument(
        "-a",
        "--attributes",
        dest="attributes",
        action="store_true",
        help="Also list attributes of the node.")
    ap.add_argument(
        "-d",
        "--dumps",
        dest="dumps",
        action="store_true",
        help="Also list dumps available from the node.")
    ap.add_argument(
        "-t",
        "--traceback",
        dest="traceback",
        action="store_true",
        help="Also include the traceback of the node's creation "
        "(if available in Python).")
    self._arg_parsers["node_info"] = ap

    # Parser for list_inputs.
    ap = argparse.ArgumentParser(
        description="Show inputs to a node.", usage=argparse.SUPPRESS)
    ap.add_argument(
        "node_name",
        type=str,
        help="Name of the node or an output tensor from the node, e.g., "
        "hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0")
    ap.add_argument(
        "-c", "--control", action="store_true", help="Include control inputs.")
    ap.add_argument(
        "-d",
        "--depth",
        dest="depth",
        type=int,
        default=config.get("graph_recursion_depth"),
        help="Maximum depth of recursion used when showing the input tree.")
    ap.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Show inputs to the node recursively, i.e., the input tree.")
    ap.add_argument(
        "-t",
        "--op_type",
        action="store_true",
        help="Show op types of input nodes.")
    self._arg_parsers["list_inputs"] = ap

    # Parser for list_outputs.
    ap = argparse.ArgumentParser(
        description="Show the nodes that receive the outputs of given node.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "node_name",
        type=str,
        help="Name of the node or an output tensor from the node, e.g., "
        "hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0")
    ap.add_argument(
        "-c", "--control", action="store_true", help="Include control inputs.")
    ap.add_argument(
        "-d",
        "--depth",
        dest="depth",
        type=int,
        default=config.get("graph_recursion_depth"),
        help="Maximum depth of recursion used when showing the output tree.")
    ap.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Show recipients of the node recursively, i.e., the output "
        "tree.")
    ap.add_argument(
        "-t",
        "--op_type",
        action="store_true",
        help="Show op types of recipient nodes.")
    self._arg_parsers["list_outputs"] = ap

    # Parser for print_tensor.
    self._arg_parsers["print_tensor"] = (
        command_parser.get_print_tensor_argparser(
            "Print the value of a dumped tensor."))

    # Parser for print_source.
    ap = argparse.ArgumentParser(
        description="Print a Python source file with overlaid debug "
        "information, including the nodes (ops) or Tensors created at the "
        "source lines.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "source_file_path",
        type=str,
        help="Path to the source file.")
    ap.add_argument(
        "-t",
        "--tensors",
        dest="tensors",
        action="store_true",
        help="Label lines with dumped Tensors, instead of ops.")
    ap.add_argument(
        "-m",
        "--max_elements_per_line",
        type=int,
        default=10,
        help="Maximum number of elements (ops or Tensors) to show per source "
             "line.")
    ap.add_argument(
        "-b",
        "--line_begin",
        type=int,
        default=1,
        help="Print source beginning at line number (1-based.)")
    self._arg_parsers["print_source"] = ap

    # Parser for list_source.
    ap = argparse.ArgumentParser(
        description="List source files responsible for constructing nodes and "
        "tensors present in the run().",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-p",
        "--path_filter",
        type=str,
        default="",
        help="Regular expression filter for file path.")
    ap.add_argument(
        "-n",
        "--node_name_filter",
        type=str,
        default="",
        help="Regular expression filter for node name.")
    self._arg_parsers["list_source"] = ap

    # Parser for eval.
    ap = argparse.ArgumentParser(
        description="""Evaluate an arbitrary expression. Can use tensor values
        from the current debug dump. The debug tensor names should be enclosed
        in pairs of backticks. Expressions with spaces should be enclosed in
        a pair of double quotes or a pair of single quotes. By default, numpy
        is imported as np and can be used in the expressions. E.g.,
          1) eval np.argmax(`Softmax:0`),
          2) eval 'np.sum(`Softmax:0`, axis=1)',
          3) eval "np.matmul((`output/Identity:0`/`Softmax:0`).T, `Softmax:0`)".
        """,
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "expression",
        type=str,
        help="""Expression to be evaluated.
        1) in the simplest case, use <node_name>:<output_slot>, e.g.,
          hidden_0/MatMul:0.

        2) if the default debug op "DebugIdentity" is to be overridden, use
          <node_name>:<output_slot>:<debug_op>, e.g.,
          hidden_0/MatMul:0:DebugNumericSummary.

        3) if the tensor of the same name exists on more than one device, use
          <device_name>:<node_name>:<output_slot>[:<debug_op>], e.g.,
          /job:worker/replica:0/task:0/gpu:0:hidden_0/MatMul:0
          /job:worker/replica:0/task:2/cpu:0:hidden_0/MatMul:0:DebugNanCount.

        4) if the tensor is executed multiple times in a given `Session.run`
        call, specify the execution index with a 0-based integer enclose in a
        pair of brackets at the end, e.g.,
          RNN/tanh:0[0]
          /job:worker/replica:0/task:0/gpu:0:RNN/tanh:0[0].""")
    ap.add_argument(
        "-a",
        "--all",
        dest="print_all",
        action="store_true",
        help="Print the tensor in its entirety, i.e., do not use ellipses "
        "(may be slow for large results).")
    ap.add_argument(
        "-w",
        "--write_path",
        default="",
        help="Path of the numpy file to write the evaluation result to, "
        "using numpy.save()")
    self._arg_parsers["eval"] = ap

  def add_tensor_filter(self, filter_name, filter_callable):
    """Add a tensor filter.

    A tensor filter is a named callable of the signature:
      filter_callable(dump_datum, tensor),

    wherein dump_datum is an instance of debug_data.DebugTensorDatum carrying
    metadata about the dumped tensor, including tensor name, timestamps, etc.
    tensor is the value of the dumped tensor as an numpy.ndarray object.
    The return value of the function is a bool.
    This is the same signature as the input argument to
    debug_data.DebugDumpDir.find().

    Args:
      filter_name: (str) name of the filter. Cannot be empty.
      filter_callable: (callable) a filter function of the signature described
        as above.

    Raises:
      ValueError: If filter_name is an empty str.
      TypeError: If filter_name is not a str.
                 Or if filter_callable is not callable.
    """

    if not isinstance(filter_name, str):
      raise TypeError("Input argument filter_name is expected to be str, "
                      "but is not.")

    # Check that filter_name is not an empty str.
    if not filter_name:
      raise ValueError("Input argument filter_name cannot be empty.")

    # Check that filter_callable is callable.
    if not callable(filter_callable):
      raise TypeError(
          "Input argument filter_callable is expected to be callable, "
          "but is not.")

    self._tensor_filters[filter_name] = filter_callable

  def get_tensor_filter(self, filter_name):
    """Retrieve filter function by name.

    Args:
      filter_name: Name of the filter set during add_tensor_filter() call.

    Returns:
      The callable associated with the filter name.

    Raises:
      ValueError: If there is no tensor filter of the specified filter name.
    """

    if filter_name not in self._tensor_filters:
      raise ValueError("There is no tensor filter named \"%s\"" % filter_name)

    return self._tensor_filters[filter_name]

  def get_help(self, handler_name):
    return self._arg_parsers[handler_name].format_help()

  def list_tensors(self, args, screen_info=None):
    """Command handler for list_tensors.

    List tensors dumped during debugged Session.run() call.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.

    Raises:
      ValueError: If `--filter_exclude_node_names` is used without `-f` or
        `--tensor_filter` being used.
    """

    # TODO(cais): Add annotations of substrings for dumped tensor names, to
    # facilitate on-screen highlighting/selection of node names.
    _ = screen_info

    parsed = self._arg_parsers["list_tensors"].parse_args(args)

    output = []

    filter_strs = []
    if parsed.op_type_filter:
      op_type_regex = re.compile(parsed.op_type_filter)
      filter_strs.append("Op type regex filter: \"%s\"" % parsed.op_type_filter)
    else:
      op_type_regex = None

    if parsed.node_name_filter:
      node_name_regex = re.compile(parsed.node_name_filter)
      filter_strs.append("Node name regex filter: \"%s\"" %
                         parsed.node_name_filter)
    else:
      node_name_regex = None

    output = debugger_cli_common.RichTextLines(filter_strs)
    output.append("")

    if parsed.tensor_filter:
      try:
        filter_callable = self.get_tensor_filter(parsed.tensor_filter)
      except ValueError:
        output = cli_shared.error("There is no tensor filter named \"%s\"." %
                                  parsed.tensor_filter)
        _add_main_menu(output, node_name=None, enable_list_tensors=False)
        return output

      data_to_show = self._debug_dump.find(
          filter_callable,
          exclude_node_names=parsed.filter_exclude_node_names)
    else:
      if parsed.filter_exclude_node_names:
        raise ValueError(
            "The flag --filter_exclude_node_names is valid only when "
            "the flag -f or --tensor_filter is used.")

      data_to_show = self._debug_dump.dumped_tensor_data

    # TODO(cais): Implement filter by lambda on tensor value.

    max_timestamp_width, max_dump_size_width, max_op_type_width = (
        self._measure_tensor_list_column_widths(data_to_show))

    # Sort the data.
    data_to_show = self._sort_dump_data_by(
        data_to_show, parsed.sort_by, parsed.reverse)

    output.extend(
        self._tensor_list_column_heads(parsed, max_timestamp_width,
                                       max_dump_size_width, max_op_type_width))

    dump_count = 0
    for dump in data_to_show:
      if node_name_regex and not node_name_regex.match(dump.node_name):
        continue

      if op_type_regex:
        op_type = self._debug_dump.node_op_type(dump.node_name)
        if not op_type_regex.match(op_type):
          continue

      rel_time = (dump.timestamp - self._debug_dump.t0) / 1000.0
      dump_size_str = cli_shared.bytes_to_readable_str(dump.dump_size_bytes)
      dumped_tensor_name = "%s:%d" % (dump.node_name, dump.output_slot)
      op_type = self._debug_dump.node_op_type(dump.node_name)

      line = "[%.3f]" % rel_time
      line += " " * (max_timestamp_width - len(line))
      line += dump_size_str
      line += " " * (max_timestamp_width + max_dump_size_width - len(line))
      line += op_type
      line += " " * (max_timestamp_width + max_dump_size_width +
                     max_op_type_width - len(line))
      line += dumped_tensor_name

      output.append(
          line,
          font_attr_segs=[(
              len(line) - len(dumped_tensor_name), len(line),
              debugger_cli_common.MenuItem("", "pt %s" % dumped_tensor_name))])
      dump_count += 1

    if parsed.tensor_filter:
      output.prepend([
          "%d dumped tensor(s) passing filter \"%s\":" %
          (dump_count, parsed.tensor_filter)
      ])
    else:
      output.prepend(["%d dumped tensor(s):" % dump_count])

    _add_main_menu(output, node_name=None, enable_list_tensors=False)
    return output

  def _measure_tensor_list_column_widths(self, data):
    """Determine the maximum widths of the timestamp and op-type column.

    This method assumes that data is sorted in the default order, i.e.,
    by ascending timestamps.

    Args:
      data: (list of DebugTensorDaum) the data based on which the maximum
        column widths will be determined.

    Returns:
      (int) maximum width of the timestamp column. 0 if data is empty.
      (int) maximum width of the dump size column. 0 if data is empty.
      (int) maximum width of the op type column. 0 if data is empty.
    """

    max_timestamp_width = 0
    if data:
      max_rel_time_ms = (data[-1].timestamp - self._debug_dump.t0) / 1000.0
      max_timestamp_width = len("[%.3f] " % max_rel_time_ms) + 1
    max_timestamp_width = max(max_timestamp_width,
                              len(self._TIMESTAMP_COLUMN_HEAD) + 1)

    max_dump_size_width = 0
    for dump in data:
      dump_size_str = cli_shared.bytes_to_readable_str(dump.dump_size_bytes)
      if len(dump_size_str) + 1 > max_dump_size_width:
        max_dump_size_width = len(dump_size_str) + 1
    max_dump_size_width = max(max_dump_size_width,
                              len(self._DUMP_SIZE_COLUMN_HEAD) + 1)

    max_op_type_width = 0
    for dump in data:
      op_type = self._debug_dump.node_op_type(dump.node_name)
      if len(op_type) + 1 > max_op_type_width:
        max_op_type_width = len(op_type) + 1
    max_op_type_width = max(max_op_type_width,
                            len(self._OP_TYPE_COLUMN_HEAD) + 1)

    return max_timestamp_width, max_dump_size_width, max_op_type_width

  def _sort_dump_data_by(self, data, sort_by, reverse):
    """Sort a list of DebugTensorDatum in specified order.

    Args:
      data: (list of DebugTensorDatum) the data to be sorted.
      sort_by: The field to sort data by.
      reverse: (bool) Whether to use reversed (descending) order.

    Returns:
      (list of DebugTensorDatum) in sorted order.

    Raises:
      ValueError: given an invalid value of sort_by.
    """

    if sort_by == SORT_TENSORS_BY_TIMESTAMP:
      return sorted(
          data,
          reverse=reverse,
          key=lambda x: x.timestamp)
    elif sort_by == SORT_TENSORS_BY_DUMP_SIZE:
      return sorted(data, reverse=reverse, key=lambda x: x.dump_size_bytes)
    elif sort_by == SORT_TENSORS_BY_OP_TYPE:
      return sorted(
          data,
          reverse=reverse,
          key=lambda x: self._debug_dump.node_op_type(x.node_name))
    elif sort_by == SORT_TENSORS_BY_TENSOR_NAME:
      return sorted(
          data,
          reverse=reverse,
          key=lambda x: "%s:%d" % (x.node_name, x.output_slot))
    else:
      raise ValueError("Unsupported key to sort tensors by: %s" % sort_by)

  def _tensor_list_column_heads(self, parsed, max_timestamp_width,
                                max_dump_size_width, max_op_type_width):
    """Generate a line containing the column heads of the tensor list.

    Args:
      parsed: Parsed arguments (by argparse) of the list_tensors command.
      max_timestamp_width: (int) maximum width of the timestamp column.
      max_dump_size_width: (int) maximum width of the dump size column.
      max_op_type_width: (int) maximum width of the op type column.

    Returns:
      A RichTextLines object.
    """

    base_command = "list_tensors"
    if parsed.tensor_filter:
      base_command += " -f %s" % parsed.tensor_filter
    if parsed.op_type_filter:
      base_command += " -t %s" % parsed.op_type_filter
    if parsed.node_name_filter:
      base_command += " -n %s" % parsed.node_name_filter

    attr_segs = {0: []}
    row = self._TIMESTAMP_COLUMN_HEAD
    command = "%s -s %s" % (base_command, SORT_TENSORS_BY_TIMESTAMP)
    if parsed.sort_by == SORT_TENSORS_BY_TIMESTAMP and not parsed.reverse:
      command += " -r"
    attr_segs[0].append(
        (0, len(row), [debugger_cli_common.MenuItem(None, command), "bold"]))
    row += " " * (max_timestamp_width - len(row))

    prev_len = len(row)
    row += self._DUMP_SIZE_COLUMN_HEAD
    command = "%s -s %s" % (base_command, SORT_TENSORS_BY_DUMP_SIZE)
    if parsed.sort_by == SORT_TENSORS_BY_DUMP_SIZE and not parsed.reverse:
      command += " -r"
    attr_segs[0].append((prev_len, len(row),
                         [debugger_cli_common.MenuItem(None, command), "bold"]))
    row += " " * (max_dump_size_width + max_timestamp_width - len(row))

    prev_len = len(row)
    row += self._OP_TYPE_COLUMN_HEAD
    command = "%s -s %s" % (base_command, SORT_TENSORS_BY_OP_TYPE)
    if parsed.sort_by == SORT_TENSORS_BY_OP_TYPE and not parsed.reverse:
      command += " -r"
    attr_segs[0].append((prev_len, len(row),
                         [debugger_cli_common.MenuItem(None, command), "bold"]))
    row += " " * (
        max_op_type_width + max_dump_size_width + max_timestamp_width - len(row)
    )

    prev_len = len(row)
    row += self._TENSOR_NAME_COLUMN_HEAD
    command = "%s -s %s" % (base_command, SORT_TENSORS_BY_TENSOR_NAME)
    if parsed.sort_by == SORT_TENSORS_BY_TENSOR_NAME and not parsed.reverse:
      command += " -r"
    attr_segs[0].append((prev_len, len(row),
                         [debugger_cli_common.MenuItem("", command), "bold"]))
    row += " " * (
        max_op_type_width + max_dump_size_width + max_timestamp_width - len(row)
    )

    return debugger_cli_common.RichTextLines([row], font_attr_segs=attr_segs)

  def node_info(self, args, screen_info=None):
    """Command handler for node_info.

    Query information about a given node.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """

    # TODO(cais): Add annotation of substrings for node names, to facilitate
    # on-screen highlighting/selection of node names.
    _ = screen_info

    parsed = self._arg_parsers["node_info"].parse_args(args)

    # Get a node name, regardless of whether the input is a node name (without
    # output slot attached) or a tensor name (with output slot attached).
    node_name, unused_slot = debug_graphs.parse_node_or_tensor_name(
        parsed.node_name)

    if not self._debug_dump.node_exists(node_name):
      output = cli_shared.error(
          "There is no node named \"%s\" in the partition graphs" % node_name)
      _add_main_menu(
          output,
          node_name=None,
          enable_list_tensors=True,
          enable_node_info=False,
          enable_list_inputs=False,
          enable_list_outputs=False)
      return output

    # TODO(cais): Provide UI glossary feature to explain to users what the
    # term "partition graph" means and how it is related to TF graph objects
    # in Python. The information can be along the line of:
    # "A tensorflow graph defined in Python is stripped of unused ops
    # according to the feeds and fetches and divided into a number of
    # partition graphs that may be distributed among multiple devices and
    # hosts. The partition graphs are what's actually executed by the C++
    # runtime during a run() call."

    lines = ["Node %s" % node_name]
    font_attr_segs = {
        0: [(len(lines[-1]) - len(node_name), len(lines[-1]), "bold")]
    }
    lines.append("")
    lines.append("  Op: %s" % self._debug_dump.node_op_type(node_name))
    lines.append("  Device: %s" % self._debug_dump.node_device(node_name))
    output = debugger_cli_common.RichTextLines(
        lines, font_attr_segs=font_attr_segs)

    # List node inputs (non-control and control).
    inputs = self._exclude_blacklisted_ops(
        self._debug_dump.node_inputs(node_name))
    ctrl_inputs = self._exclude_blacklisted_ops(
        self._debug_dump.node_inputs(node_name, is_control=True))
    output.extend(self._format_neighbors("input", inputs, ctrl_inputs))

    # List node output recipients (non-control and control).
    recs = self._exclude_blacklisted_ops(
        self._debug_dump.node_recipients(node_name))
    ctrl_recs = self._exclude_blacklisted_ops(
        self._debug_dump.node_recipients(node_name, is_control=True))
    output.extend(self._format_neighbors("recipient", recs, ctrl_recs))

    # Optional: List attributes of the node.
    if parsed.attributes:
      output.extend(self._list_node_attributes(node_name))

    # Optional: List dumps available from the node.
    if parsed.dumps:
      output.extend(self._list_node_dumps(node_name))

    if parsed.traceback:
      output.extend(self._render_node_traceback(node_name))

    _add_main_menu(output, node_name=node_name, enable_node_info=False)
    return output

  def _exclude_blacklisted_ops(self, node_names):
    """Exclude all nodes whose op types are in _GRAPH_STRUCT_OP_TYPE_BLACKLIST.

    Args:
      node_names: An iterable of node or graph element names.

    Returns:
      A list of node names that are not blacklisted.
    """
    return [node_name for node_name in node_names
            if self._debug_dump.node_op_type(
                debug_graphs.get_node_name(node_name)) not in
            self._GRAPH_STRUCT_OP_TYPE_BLACKLIST]

  def _render_node_traceback(self, node_name):
    """Render traceback of a node's creation in Python, if available.

    Args:
      node_name: (str) name of the node.

    Returns:
      A RichTextLines object containing the stack trace of the node's
      construction.
    """

    lines = [RL(""), RL(""), RL("Traceback of node construction:", "bold")]

    try:
      node_stack = self._debug_dump.node_traceback(node_name)
      for depth, (file_path, line, function_name, text) in enumerate(
          node_stack):
        lines.append("%d: %s" % (depth, file_path))

        attribute = debugger_cli_common.MenuItem(
            "", "ps %s -b %d" % (file_path, line)) if text else None
        line_number_line = RL("  ")
        line_number_line += RL("Line:     %d" % line, attribute)
        lines.append(line_number_line)

        lines.append("  Function: %s" % function_name)
        lines.append("  Text:     " + (("\"%s\"" % text) if text else "None"))
        lines.append("")
    except KeyError:
      lines.append("(Node unavailable in the loaded Python graph)")
    except LookupError:
      lines.append("(Unavailable because no Python graph has been loaded)")

    return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

  def list_inputs(self, args, screen_info=None):
    """Command handler for inputs.

    Show inputs to a given node.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """

    # Screen info not currently used by this handler. Include this line to
    # mute pylint.
    _ = screen_info
    # TODO(cais): Use screen info to format the output lines more prettily,
    # e.g., hanging indent of long node names.

    parsed = self._arg_parsers["list_inputs"].parse_args(args)

    output = self._list_inputs_or_outputs(
        parsed.recursive,
        parsed.node_name,
        parsed.depth,
        parsed.control,
        parsed.op_type,
        do_outputs=False)

    node_name = debug_graphs.get_node_name(parsed.node_name)
    _add_main_menu(output, node_name=node_name, enable_list_inputs=False)

    return output

  def print_tensor(self, args, screen_info=None):
    """Command handler for print_tensor.

    Print value of a given dumped tensor.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """

    parsed = self._arg_parsers["print_tensor"].parse_args(args)

    np_printoptions = cli_shared.numpy_printoptions_from_screen_info(
        screen_info)

    # Determine if any range-highlighting is required.
    highlight_options = cli_shared.parse_ranges_highlight(parsed.ranges)

    tensor_name, tensor_slicing = (
        command_parser.parse_tensor_name_with_slicing(parsed.tensor_name))

    node_name, output_slot = debug_graphs.parse_node_or_tensor_name(tensor_name)
    if (self._debug_dump.loaded_partition_graphs() and
        not self._debug_dump.node_exists(node_name)):
      output = cli_shared.error(
          "Node \"%s\" does not exist in partition graphs" % node_name)
      _add_main_menu(
          output,
          node_name=None,
          enable_list_tensors=True,
          enable_print_tensor=False)
      return output

    watch_keys = self._debug_dump.debug_watch_keys(node_name)
    if output_slot is None:
      output_slots = set()
      for watch_key in watch_keys:
        output_slots.add(int(watch_key.split(":")[1]))

      if len(output_slots) == 1:
        # There is only one dumped tensor from this node, so there is no
        # ambiguity. Proceed to show the only dumped tensor.
        output_slot = list(output_slots)[0]
      else:
        # There are more than one dumped tensors from this node. Indicate as
        # such.
        # TODO(cais): Provide an output screen with command links for
        # convenience.
        lines = [
            "Node \"%s\" generated debug dumps from %s output slots:" %
            (node_name, len(output_slots)),
            "Please specify the output slot: %s:x." % node_name
        ]
        output = debugger_cli_common.RichTextLines(lines)
        _add_main_menu(
            output,
            node_name=node_name,
            enable_list_tensors=True,
            enable_print_tensor=False)
        return output

    # Find debug dump data that match the tensor name (node name + output
    # slot).
    matching_data = []
    for watch_key in watch_keys:
      debug_tensor_data = self._debug_dump.watch_key_to_data(watch_key)
      for datum in debug_tensor_data:
        if datum.output_slot == output_slot:
          matching_data.append(datum)

    if not matching_data:
      # No dump for this tensor.
      output = cli_shared.error("Tensor \"%s\" did not generate any dumps." %
                                parsed.tensor_name)
    elif len(matching_data) == 1:
      # There is only one dump for this tensor.
      if parsed.number <= 0:
        output = cli_shared.format_tensor(
            matching_data[0].get_tensor(),
            matching_data[0].watch_key,
            np_printoptions,
            print_all=parsed.print_all,
            tensor_slicing=tensor_slicing,
            highlight_options=highlight_options,
            include_numeric_summary=parsed.numeric_summary,
            write_path=parsed.write_path)
      else:
        output = cli_shared.error(
            "Invalid number (%d) for tensor %s, which generated one dump." %
            (parsed.number, parsed.tensor_name))

      _add_main_menu(output, node_name=node_name, enable_print_tensor=False)
    else:
      # There are more than one dumps for this tensor.
      if parsed.number < 0:
        lines = [
            "Tensor \"%s\" generated %d dumps:" % (parsed.tensor_name,
                                                   len(matching_data))
        ]
        font_attr_segs = {}

        for i, datum in enumerate(matching_data):
          rel_time = (datum.timestamp - self._debug_dump.t0) / 1000.0
          lines.append("#%d [%.3f ms] %s" % (i, rel_time, datum.watch_key))
          command = "print_tensor %s -n %d" % (parsed.tensor_name, i)
          font_attr_segs[len(lines) - 1] = [(
              len(lines[-1]) - len(datum.watch_key), len(lines[-1]),
              debugger_cli_common.MenuItem(None, command))]

        lines.append("")
        lines.append(
            "You can use the -n (--number) flag to specify which dump to "
            "print.")
        lines.append("For example:")
        lines.append("  print_tensor %s -n 0" % parsed.tensor_name)

        output = debugger_cli_common.RichTextLines(
            lines, font_attr_segs=font_attr_segs)
      elif parsed.number >= len(matching_data):
        output = cli_shared.error(
            "Specified number (%d) exceeds the number of available dumps "
            "(%d) for tensor %s" %
            (parsed.number, len(matching_data), parsed.tensor_name))
      else:
        output = cli_shared.format_tensor(
            matching_data[parsed.number].get_tensor(),
            matching_data[parsed.number].watch_key + " (dump #%d)" %
            parsed.number,
            np_printoptions,
            print_all=parsed.print_all,
            tensor_slicing=tensor_slicing,
            highlight_options=highlight_options,
            write_path=parsed.write_path)
      _add_main_menu(output, node_name=node_name, enable_print_tensor=False)

    return output

  def list_outputs(self, args, screen_info=None):
    """Command handler for inputs.

    Show inputs to a given node.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """

    # Screen info not currently used by this handler. Include this line to
    # mute pylint.
    _ = screen_info
    # TODO(cais): Use screen info to format the output lines more prettily,
    # e.g., hanging indent of long node names.

    parsed = self._arg_parsers["list_outputs"].parse_args(args)

    output = self._list_inputs_or_outputs(
        parsed.recursive,
        parsed.node_name,
        parsed.depth,
        parsed.control,
        parsed.op_type,
        do_outputs=True)

    node_name = debug_graphs.get_node_name(parsed.node_name)
    _add_main_menu(output, node_name=node_name, enable_list_outputs=False)

    return output

  def evaluate_expression(self, args, screen_info=None):
    parsed = self._arg_parsers["eval"].parse_args(args)

    eval_res = self._evaluator.evaluate(parsed.expression)

    np_printoptions = cli_shared.numpy_printoptions_from_screen_info(
        screen_info)
    return cli_shared.format_tensor(
        eval_res,
        "from eval of expression '%s'" % parsed.expression,
        np_printoptions,
        print_all=parsed.print_all,
        include_numeric_summary=True,
        write_path=parsed.write_path)

  def _reconstruct_print_source_command(self,
                                        parsed,
                                        line_begin,
                                        max_elements_per_line_increase=0):
    return "ps %s %s -b %d -m %d" % (
        parsed.source_file_path, "-t" if parsed.tensors else "", line_begin,
        parsed.max_elements_per_line + max_elements_per_line_increase)

  def print_source(self, args, screen_info=None):
    """Print the content of a source file."""
    del screen_info  # Unused.

    parsed = self._arg_parsers["print_source"].parse_args(args)

    source_annotation = source_utils.annotate_source(
        self._debug_dump,
        parsed.source_file_path,
        do_dumped_tensors=parsed.tensors)

    source_lines, line_num_width = source_utils.load_source(
        parsed.source_file_path)

    labeled_source_lines = []
    actual_initial_scroll_target = 0
    for i, line in enumerate(source_lines):
      annotated_line = RL("L%d" % (i + 1), cli_shared.COLOR_YELLOW)
      annotated_line += " " * (line_num_width - len(annotated_line))
      annotated_line += line
      labeled_source_lines.append(annotated_line)

      if i + 1 == parsed.line_begin:
        actual_initial_scroll_target = len(labeled_source_lines) - 1

      if i + 1 in source_annotation:
        sorted_elements = sorted(source_annotation[i + 1])
        for k, element in enumerate(sorted_elements):
          if k >= parsed.max_elements_per_line:
            omitted_info_line = RL("    (... Omitted %d of %d %s ...) " % (
                len(sorted_elements) - parsed.max_elements_per_line,
                len(sorted_elements),
                "tensor(s)" if parsed.tensors else "op(s)"))
            omitted_info_line += RL(
                "+5",
                debugger_cli_common.MenuItem(
                    None,
                    self._reconstruct_print_source_command(
                        parsed, i + 1, max_elements_per_line_increase=5)))
            labeled_source_lines.append(omitted_info_line)
            break

          label = RL(" " * 4)
          if self._debug_dump.debug_watch_keys(
              debug_graphs.get_node_name(element)):
            attribute = debugger_cli_common.MenuItem("", "pt %s" % element)
          else:
            attribute = cli_shared.COLOR_BLUE

          label += RL(element, attribute)
          labeled_source_lines.append(label)

    output = debugger_cli_common.rich_text_lines_from_rich_line_list(
        labeled_source_lines,
        annotations={debugger_cli_common.INIT_SCROLL_POS_KEY:
                     actual_initial_scroll_target})
    _add_main_menu(output, node_name=None)
    return output

  def _make_source_table(self, source_list, is_tf_py_library):
    """Make a table summarizing the source files that create nodes and tensors.

    Args:
      source_list: List of source files and related information as a list of
        tuples (file_path, is_tf_library, num_nodes, num_tensors, num_dumps,
        first_line).
      is_tf_py_library: (`bool`) whether this table is for files that belong
        to the TensorFlow Python library.

    Returns:
      The table as a `debugger_cli_common.RichTextLines` object.
    """
    path_head = "Source file path"
    num_nodes_head = "#(nodes)"
    num_tensors_head = "#(tensors)"
    num_dumps_head = "#(tensor dumps)"

    if is_tf_py_library:
      # Use color to mark files that are guessed to belong to TensorFlow Python
      # library.
      color = cli_shared.COLOR_GRAY
      lines = [RL("TensorFlow Python library file(s):", color)]
    else:
      color = cli_shared.COLOR_WHITE
      lines = [RL("File(s) outside TensorFlow Python library:", color)]

    if not source_list:
      lines.append(RL("[No files.]"))
      lines.append(RL())
      return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

    path_column_width = max(
        max(len(item[0]) for item in source_list), len(path_head)) + 1
    num_nodes_column_width = max(
        max(len(str(item[2])) for item in source_list),
        len(num_nodes_head)) + 1
    num_tensors_column_width = max(
        max(len(str(item[3])) for item in source_list),
        len(num_tensors_head)) + 1

    head = RL(path_head + " " * (path_column_width - len(path_head)), color)
    head += RL(num_nodes_head + " " * (
        num_nodes_column_width - len(num_nodes_head)), color)
    head += RL(num_tensors_head + " " * (
        num_tensors_column_width - len(num_tensors_head)), color)
    head += RL(num_dumps_head, color)

    lines.append(head)

    for (file_path, _, num_nodes, num_tensors, num_dumps,
         first_line_num) in source_list:
      path_attributes = [color]
      if source_utils.is_extension_uncompiled_python_source(file_path):
        path_attributes.append(
            debugger_cli_common.MenuItem(None, "ps %s -b %d" %
                                         (file_path, first_line_num)))

      line = RL(file_path, path_attributes)
      line += " " * (path_column_width - len(line))
      line += RL(
          str(num_nodes) + " " * (num_nodes_column_width - len(str(num_nodes))),
          color)
      line += RL(
          str(num_tensors) + " " *
          (num_tensors_column_width - len(str(num_tensors))), color)
      line += RL(str(num_dumps), color)
      lines.append(line)
    lines.append(RL())

    return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

  def list_source(self, args, screen_info=None):
    """List Python source files that constructed nodes and tensors."""
    del screen_info  # Unused.

    parsed = self._arg_parsers["list_source"].parse_args(args)
    source_list = source_utils.list_source_files_against_dump(
        self._debug_dump,
        path_regex_whitelist=parsed.path_filter,
        node_name_regex_whitelist=parsed.node_name_filter)

    top_lines = [
        RL("List of source files that created nodes in this run", "bold")]
    if parsed.path_filter:
      top_lines.append(
          RL("File path regex filter: \"%s\"" % parsed.path_filter))
    if parsed.node_name_filter:
      top_lines.append(
          RL("Node name regex filter: \"%s\"" % parsed.node_name_filter))
    top_lines.append(RL())
    output = debugger_cli_common.rich_text_lines_from_rich_line_list(top_lines)
    if not source_list:
      output.append("[No source file information.]")
      return output

    output.extend(self._make_source_table(
        [item for item in source_list if not item[1]], False))
    output.extend(self._make_source_table(
        [item for item in source_list if item[1]], True))
    _add_main_menu(output, node_name=None)
    return output

  def _list_inputs_or_outputs(self,
                              recursive,
                              node_name,
                              depth,
                              control,
                              op_type,
                              do_outputs=False):
    """Helper function used by list_inputs and list_outputs.

    Format a list of lines to display the inputs or output recipients of a
    given node.

    Args:
      recursive: Whether the listing is to be done recursively, as a boolean.
      node_name: The name of the node in question, as a str.
      depth: Maximum recursion depth, applies only if recursive == True, as an
        int.
      control: Whether control inputs or control recipients are included, as a
        boolean.
      op_type: Whether the op types of the nodes are to be included, as a
        boolean.
      do_outputs: Whether recipients, instead of input nodes are to be
        listed, as a boolean.

    Returns:
      Input or recipient tree formatted as a RichTextLines object.
    """

    if do_outputs:
      tracker = self._debug_dump.node_recipients
      type_str = "Recipients of"
      short_type_str = "recipients"
    else:
      tracker = self._debug_dump.node_inputs
      type_str = "Inputs to"
      short_type_str = "inputs"

    lines = []
    font_attr_segs = {}

    # Check if this is a tensor name, instead of a node name.
    node_name, _ = debug_graphs.parse_node_or_tensor_name(node_name)

    # Check if node exists.
    if not self._debug_dump.node_exists(node_name):
      return cli_shared.error(
          "There is no node named \"%s\" in the partition graphs" % node_name)

    if recursive:
      max_depth = depth
    else:
      max_depth = 1

    if control:
      include_ctrls_str = ", control %s included" % short_type_str
    else:
      include_ctrls_str = ""

    line = "%s node \"%s\"" % (type_str, node_name)
    font_attr_segs[0] = [(len(line) - 1 - len(node_name), len(line) - 1, "bold")
                        ]
    lines.append(line + " (Depth limit = %d%s):" % (max_depth, include_ctrls_str
                                                   ))

    command_template = "lo -c -r %s" if do_outputs else "li -c -r %s"
    self._dfs_from_node(
        lines,
        font_attr_segs,
        node_name,
        tracker,
        max_depth,
        1, [],
        control,
        op_type,
        command_template=command_template)

    # Include legend.
    lines.append("")
    lines.append("Legend:")
    lines.append("  (d): recursion depth = d.")

    if control:
      lines.append("  (Ctrl): Control input.")
    if op_type:
      lines.append("  [Op]: Input node has op type Op.")

    # TODO(cais): Consider appending ":0" at the end of 1st outputs of nodes.

    return debugger_cli_common.RichTextLines(
        lines, font_attr_segs=font_attr_segs)

  def _dfs_from_node(self,
                     lines,
                     attr_segs,
                     node_name,
                     tracker,
                     max_depth,
                     depth,
                     unfinished,
                     include_control=False,
                     show_op_type=False,
                     command_template=None):
    """Perform depth-first search (DFS) traversal of a node's input tree.

    It recursively tracks the inputs (or output recipients) of the node called
    node_name, and append these inputs (or output recipients) to a list of text
    lines (lines) with proper indentation that reflects the recursion depth,
    together with some formatting attributes (to attr_segs). The formatting
    attributes can include command shortcuts, for example.

    Args:
      lines: Text lines to append to, as a list of str.
      attr_segs: (dict) Attribute segments dictionary to append to.
      node_name: Name of the node, as a str. This arg is updated during the
        recursion.
      tracker: A callable that takes one str as the node name input and
        returns a list of str as the inputs/outputs.
        This makes it this function general enough to be used with both
        node-input and node-output tracking.
      max_depth: Maximum recursion depth, as an int.
      depth: Current recursion depth. This arg is updated during the
        recursion.
      unfinished: A stack of unfinished recursion depths, as a list of int.
      include_control: Whether control dependencies are to be included as
        inputs (and marked as such).
      show_op_type: Whether op type of the input nodes are to be displayed
        alongside the nodes' names.
      command_template: (str) Template for command shortcut of the node names.
    """

    # Make a shallow copy of the list because it may be extended later.
    all_inputs = self._exclude_blacklisted_ops(
        copy.copy(tracker(node_name, is_control=False)))
    is_ctrl = [False] * len(all_inputs)
    if include_control:
      # Sort control inputs or recipients in alphabetical order of the node
      # names.
      ctrl_inputs = self._exclude_blacklisted_ops(
          sorted(tracker(node_name, is_control=True)))
      all_inputs.extend(ctrl_inputs)
      is_ctrl.extend([True] * len(ctrl_inputs))

    if not all_inputs:
      if depth == 1:
        lines.append("  [None]")

      return

    unfinished.append(depth)

    # Create depth-dependent hanging indent for the line.
    hang = ""
    for k in xrange(depth):
      if k < depth - 1:
        if k + 1 in unfinished:
          hang += HANG_UNFINISHED
        else:
          hang += HANG_FINISHED
      else:
        hang += HANG_SUFFIX

    if all_inputs and depth > max_depth:
      lines.append(hang + ELLIPSIS)
      unfinished.pop()
      return

    hang += DEPTH_TEMPLATE % depth

    for i, inp in enumerate(all_inputs):
      op_type = self._debug_dump.node_op_type(debug_graphs.get_node_name(inp))
      if op_type in self._GRAPH_STRUCT_OP_TYPE_BLACKLIST:
        continue

      if is_ctrl[i]:
        ctrl_str = CTRL_LABEL
      else:
        ctrl_str = ""

      op_type_str = ""
      if show_op_type:
        op_type_str = OP_TYPE_TEMPLATE % op_type

      if i == len(all_inputs) - 1:
        unfinished.pop()

      line = hang + ctrl_str + op_type_str + inp
      lines.append(line)
      if command_template:
        attr_segs[len(lines) - 1] = [(
            len(line) - len(inp), len(line),
            debugger_cli_common.MenuItem(None, command_template % inp))]

      # Recursive call.
      # The input's/output's name can be a tensor name, in the case of node
      # with >1 output slots.
      inp_node_name, _ = debug_graphs.parse_node_or_tensor_name(inp)
      self._dfs_from_node(
          lines,
          attr_segs,
          inp_node_name,
          tracker,
          max_depth,
          depth + 1,
          unfinished,
          include_control=include_control,
          show_op_type=show_op_type,
          command_template=command_template)

  def _format_neighbors(self, neighbor_type, non_ctrls, ctrls):
    """List neighbors (inputs or recipients) of a node.

    Args:
      neighbor_type: ("input" | "recipient")
      non_ctrls: Non-control neighbor node names, as a list of str.
      ctrls: Control neighbor node names, as a list of str.

    Returns:
      A RichTextLines object.
    """

    # TODO(cais): Return RichTextLines instead, to allow annotation of node
    # names.
    lines = []
    font_attr_segs = {}

    lines.append("")
    lines.append("  %d %s(s) + %d control %s(s):" %
                 (len(non_ctrls), neighbor_type, len(ctrls), neighbor_type))
    lines.append("    %d %s(s):" % (len(non_ctrls), neighbor_type))
    for non_ctrl in non_ctrls:
      line = "      [%s] %s" % (self._debug_dump.node_op_type(non_ctrl),
                                non_ctrl)
      lines.append(line)
      font_attr_segs[len(lines) - 1] = [(
          len(line) - len(non_ctrl), len(line),
          debugger_cli_common.MenuItem(None, "ni -a -d -t %s" % non_ctrl))]

    if ctrls:
      lines.append("")
      lines.append("    %d control %s(s):" % (len(ctrls), neighbor_type))
      for ctrl in ctrls:
        line = "      [%s] %s" % (self._debug_dump.node_op_type(ctrl), ctrl)
        lines.append(line)
        font_attr_segs[len(lines) - 1] = [(
            len(line) - len(ctrl), len(line),
            debugger_cli_common.MenuItem(None, "ni -a -d -t %s" % ctrl))]

    return debugger_cli_common.RichTextLines(
        lines, font_attr_segs=font_attr_segs)

  def _list_node_attributes(self, node_name):
    """List neighbors (inputs or recipients) of a node.

    Args:
      node_name: Name of the node of which the attributes are to be listed.

    Returns:
      A RichTextLines object.
    """

    lines = []
    lines.append("")
    lines.append("Node attributes:")

    attrs = self._debug_dump.node_attributes(node_name)
    for attr_key in attrs:
      lines.append("  %s:" % attr_key)
      attr_val_str = repr(attrs[attr_key]).strip().replace("\n", " ")
      lines.append("    %s" % attr_val_str)
      lines.append("")

    return debugger_cli_common.RichTextLines(lines)

  def _list_node_dumps(self, node_name):
    """List dumped tensor data from a node.

    Args:
      node_name: Name of the node of which the attributes are to be listed.

    Returns:
      A RichTextLines object.
    """

    lines = []
    font_attr_segs = {}

    watch_keys = self._debug_dump.debug_watch_keys(node_name)

    dump_count = 0
    for watch_key in watch_keys:
      debug_tensor_data = self._debug_dump.watch_key_to_data(watch_key)
      for datum in debug_tensor_data:
        line = "  Slot %d @ %s @ %.3f ms" % (
            datum.output_slot, datum.debug_op,
            (datum.timestamp - self._debug_dump.t0) / 1000.0)
        lines.append(line)
        command = "pt %s:%d -n %d" % (node_name, datum.output_slot, dump_count)
        font_attr_segs[len(lines) - 1] = [(
            2, len(line), debugger_cli_common.MenuItem(None, command))]
        dump_count += 1

    output = debugger_cli_common.RichTextLines(
        lines, font_attr_segs=font_attr_segs)
    output_with_header = debugger_cli_common.RichTextLines(
        ["%d dumped tensor(s):" % dump_count, ""])
    output_with_header.extend(output)
    return output_with_header


def create_analyzer_ui(debug_dump,
                       tensor_filters=None,
                       ui_type="curses",
                       on_ui_exit=None,
                       config=None):
  """Create an instance of CursesUI based on a DebugDumpDir object.

  Args:
    debug_dump: (debug_data.DebugDumpDir) The debug dump to use.
    tensor_filters: (dict) A dict mapping tensor filter name (str) to tensor
      filter (Callable).
    ui_type: (str) requested UI type, e.g., "curses", "readline".
    on_ui_exit: (`Callable`) the callback to be called when the UI exits.
    config: A `cli_config.CLIConfig` object.

  Returns:
    (base_ui.BaseUI) A BaseUI subtype object with a set of standard analyzer
      commands and tab-completions registered.
  """
  if config is None:
    config = cli_config.CLIConfig()

  analyzer = DebugAnalyzer(debug_dump, config=config)
  if tensor_filters:
    for tensor_filter_name in tensor_filters:
      analyzer.add_tensor_filter(
          tensor_filter_name, tensor_filters[tensor_filter_name])

  cli = ui_factory.get_ui(ui_type, on_ui_exit=on_ui_exit, config=config)
  cli.register_command_handler(
      "list_tensors",
      analyzer.list_tensors,
      analyzer.get_help("list_tensors"),
      prefix_aliases=["lt"])
  cli.register_command_handler(
      "node_info",
      analyzer.node_info,
      analyzer.get_help("node_info"),
      prefix_aliases=["ni"])
  cli.register_command_handler(
      "list_inputs",
      analyzer.list_inputs,
      analyzer.get_help("list_inputs"),
      prefix_aliases=["li"])
  cli.register_command_handler(
      "list_outputs",
      analyzer.list_outputs,
      analyzer.get_help("list_outputs"),
      prefix_aliases=["lo"])
  cli.register_command_handler(
      "print_tensor",
      analyzer.print_tensor,
      analyzer.get_help("print_tensor"),
      prefix_aliases=["pt"])
  cli.register_command_handler(
      "print_source",
      analyzer.print_source,
      analyzer.get_help("print_source"),
      prefix_aliases=["ps"])
  cli.register_command_handler(
      "list_source",
      analyzer.list_source,
      analyzer.get_help("list_source"),
      prefix_aliases=["ls"])
  cli.register_command_handler(
      "eval",
      analyzer.evaluate_expression,
      analyzer.get_help("eval"),
      prefix_aliases=["ev"])

  dumped_tensor_names = []
  for datum in debug_dump.dumped_tensor_data:
    dumped_tensor_names.append("%s:%d" % (datum.node_name, datum.output_slot))

  # Tab completions for command "print_tensors".
  cli.register_tab_comp_context(["print_tensor", "pt"], dumped_tensor_names)

  return cli
