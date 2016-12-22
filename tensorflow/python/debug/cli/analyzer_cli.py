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

The other part of the debugger is the stepper (c.f. stepper_cli.py).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import re

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.debug import debug_data
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import curses_ui
from tensorflow.python.debug.cli import debugger_cli_common


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


class DebugAnalyzer(object):
  """Analyzer for debug data from dump directories."""

  def __init__(self, debug_dump):
    """DebugAnalyzer constructor.

    Args:
      debug_dump: A DebugDumpDir object.
    """

    self._debug_dump = debug_dump

    # Initialize tensor filters state.
    self._tensor_filters = {}

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
        default=20,
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
        default=20,
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
    ap = argparse.ArgumentParser(
        description="Print the value of a dumped tensor.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "tensor_name",
        type=str,
        help="Name of the tensor, followed by any slicing indices, "
        "e.g., hidden1/Wx_plus_b/MatMul:0, "
        "hidden1/Wx_plus_b/MatMul:0[1, :]")
    ap.add_argument(
        "-n",
        "--number",
        dest="number",
        type=int,
        default=-1,
        help="0-based dump number for the specified tensor. "
        "Required for tensor with multiple dumps.")
    ap.add_argument(
        "-r",
        "--ranges",
        dest="ranges",
        type=str,
        default="",
        help="Numerical ranges to highlight tensor elements in. "
        "Examples: -r 0,1e-8, -r [-0.1,0.1], "
        "-r \"[[-inf, -0.1], [0.1, inf]]\"")

    ap.add_argument(
        "-a",
        "--all",
        dest="print_all",
        action="store_true",
        help="Print the tensor in its entirety, i.e., do not use ellipses.")
    self._arg_parsers["print_tensor"] = ap

    # TODO(cais): Implement list_nodes.

  def add_tensor_filter(self, filter_name, filter_callable):
    """Add a tensor filter.

    A tensor filter is a named callable of the siganture:
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

    if parsed.tensor_filter:
      try:
        filter_callable = self.get_tensor_filter(parsed.tensor_filter)
      except ValueError:
        return cli_shared.error(
            "There is no tensor filter named \"%s\"." % parsed.tensor_filter)

      data_to_show = self._debug_dump.find(filter_callable)
    else:
      data_to_show = self._debug_dump.dumped_tensor_data

    # TODO(cais): Implement filter by lambda on tensor value.

    dump_count = 0
    for dump in data_to_show:
      if node_name_regex and not node_name_regex.match(dump.node_name):
        continue

      if op_type_regex:
        op_type = self._debug_dump.node_op_type(dump.node_name)
        if not op_type_regex.match(op_type):
          continue

      rel_time = (dump.timestamp - self._debug_dump.t0) / 1000.0
      output.append("[%.3f ms] %s:%d" % (rel_time, dump.node_name,
                                         dump.output_slot))
      dump_count += 1

    output.insert(0, "")

    output = filter_strs + output

    if parsed.tensor_filter:
      output.insert(0, "%d dumped tensor(s) passing filter \"%s\":" %
                    (dump_count, parsed.tensor_filter))
    else:
      output.insert(0, "%d dumped tensor(s):" % dump_count)

    return debugger_cli_common.RichTextLines(output)

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
    node_name, unused_slot = debug_data.parse_node_or_tensor_name(
        parsed.node_name)

    if not self._debug_dump.node_exists(node_name):
      return cli_shared.error(
          "There is no node named \"%s\" in the partition graphs" % node_name)

    # TODO(cais): Provide UI glossary feature to explain to users what the
    # term "partition graph" means and how it is related to TF graph objects
    # in Python. The information can be along the line of:
    # "A tensorflow graph defined in Python is stripped of unused ops
    # according to the feeds and fetches and divided into a number of
    # partition graphs that may be distributed among multiple devices and
    # hosts. The partition graphs are what's actually executed by the C++
    # runtime during a run() call."

    lines = ["Node %s" % node_name]
    lines.append("")
    lines.append("  Op: %s" % self._debug_dump.node_op_type(node_name))
    lines.append("  Device: %s" % self._debug_dump.node_device(node_name))

    # List node inputs (non-control and control).
    inputs = self._debug_dump.node_inputs(node_name)
    ctrl_inputs = self._debug_dump.node_inputs(node_name, is_control=True)

    input_lines = self._format_neighbors("input", inputs, ctrl_inputs)
    lines.extend(input_lines)

    # List node output recipients (non-control and control).
    recs = self._debug_dump.node_recipients(node_name)
    ctrl_recs = self._debug_dump.node_recipients(node_name, is_control=True)

    rec_lines = self._format_neighbors("recipient", recs, ctrl_recs)
    lines.extend(rec_lines)

    # Optional: List attributes of the node.
    if parsed.attributes:
      lines.extend(self._list_node_attributes(node_name))

    # Optional: List dumps available from the node.
    if parsed.dumps:
      lines.extend(self._list_node_dumps(node_name))

    return debugger_cli_common.RichTextLines(lines)

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

    return self._list_inputs_or_outputs(
        parsed.recursive,
        parsed.node_name,
        parsed.depth,
        parsed.control,
        parsed.op_type,
        do_outputs=False)

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

    if screen_info and "cols" in screen_info:
      np_printoptions = {"linewidth": screen_info["cols"]}
    else:
      np_printoptions = {}

    # Determine if any range-highlighting is required.
    highlight_options = cli_shared.parse_ranges_highlight(parsed.ranges)

    tensor_name, tensor_slicing = (
        command_parser.parse_tensor_name_with_slicing(parsed.tensor_name))

    node_name, output_slot = debug_data.parse_node_or_tensor_name(tensor_name)
    if output_slot is None:
      return cli_shared.error("\"%s\" is not a valid tensor name" %
                              parsed.tensor_name)

    if (self._debug_dump.loaded_partition_graphs() and
        not self._debug_dump.node_exists(node_name)):
      return cli_shared.error(
          "Node \"%s\" does not exist in partition graphs" % node_name)

    watch_keys = self._debug_dump.debug_watch_keys(node_name)

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
      return cli_shared.error(
          "Tensor \"%s\" did not generate any dumps." % parsed.tensor_name)
    elif len(matching_data) == 1:
      # There is only one dump for this tensor.
      if parsed.number <= 0:
        return cli_shared.format_tensor(
            matching_data[0].get_tensor(),
            matching_data[0].watch_key,
            np_printoptions,
            print_all=parsed.print_all,
            tensor_slicing=tensor_slicing,
            highlight_options=highlight_options)
      else:
        return cli_shared.error(
            "Invalid number (%d) for tensor %s, which generated one dump." %
            (parsed.number, parsed.tensor_name))
    else:
      # There are more than one dumps for this tensor.
      if parsed.number < 0:
        lines = [
            "Tensor \"%s\" generated %d dumps:" % (parsed.tensor_name,
                                                   len(matching_data))
        ]

        for i, datum in enumerate(matching_data):
          rel_time = (datum.timestamp - self._debug_dump.t0) / 1000.0
          lines.append("#%d [%.3f ms] %s" % (i, rel_time, datum.watch_key))

        lines.append("")
        lines.append(
            "Use the -n (--number) flag to specify which dump to print.")
        lines.append("For example:")
        lines.append("  print_tensor %s -n 0" % parsed.tensor_name)

        return debugger_cli_common.RichTextLines(lines)
      elif parsed.number >= len(matching_data):
        return cli_shared.error(
            "Specified number (%d) exceeds the number of available dumps "
            "(%d) for tensor %s" %
            (parsed.number, len(matching_data), parsed.tensor_name))
      else:
        return cli_shared.format_tensor(
            matching_data[parsed.number].get_tensor(),
            matching_data[parsed.number].watch_key + " (dump #%d)" %
            parsed.number,
            np_printoptions,
            print_all=parsed.print_all,
            tensor_slicing=tensor_slicing,
            highlight_options=highlight_options)

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

    return self._list_inputs_or_outputs(
        parsed.recursive,
        parsed.node_name,
        parsed.depth,
        parsed.control,
        parsed.op_type,
        do_outputs=True)

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

    # Check if this is a tensor name, instead of a node name.
    node_name, _ = debug_data.parse_node_or_tensor_name(node_name)

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

    lines.append("%s node \"%s\" (Depth limit = %d%s):" %
                 (type_str, node_name, max_depth, include_ctrls_str))

    self._dfs_from_node(lines, node_name, tracker, max_depth, 1, [], control,
                        op_type)

    # Include legend.
    lines.append("")
    lines.append("Legend:")
    lines.append("  (d): recursion depth = d.")

    if control:
      lines.append("  (Ctrl): Control input.")
    if op_type:
      lines.append("  [Op]: Input node has op type Op.")

    # TODO(cais): Consider appending ":0" at the end of 1st outputs of nodes.

    return debugger_cli_common.RichTextLines(lines)

  def _dfs_from_node(self,
                     lines,
                     node_name,
                     tracker,
                     max_depth,
                     depth,
                     unfinished,
                     include_control=False,
                     show_op_type=False):
    """Perform depth-first search (DFS) traversal of a node's input tree.

    Args:
      lines: Text lines to append to, as a list of str.
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
    """

    # Make a shallow copy of the list because it may be extended later.
    all_inputs = copy.copy(tracker(node_name, is_control=False))
    is_ctrl = [False] * len(all_inputs)
    if include_control:
      # Sort control inputs or recipients in in alphabetical order of the node
      # names.
      ctrl_inputs = sorted(tracker(node_name, is_control=True))
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

    for i in xrange(len(all_inputs)):
      inp = all_inputs[i]
      if is_ctrl[i]:
        ctrl_str = CTRL_LABEL
      else:
        ctrl_str = ""

      op_type_str = ""
      if show_op_type:
        op_type_str = OP_TYPE_TEMPLATE % self._debug_dump.node_op_type(inp)

      if i == len(all_inputs) - 1:
        unfinished.pop()

      lines.append(hang + ctrl_str + op_type_str + inp)

      # Recursive call.
      # The input's/output's name can be a tensor name, in the case of node
      # with >1 output slots.
      inp_node_name, _ = debug_data.parse_node_or_tensor_name(inp)
      self._dfs_from_node(
          lines,
          inp_node_name,
          tracker,
          max_depth,
          depth + 1,
          unfinished,
          include_control=include_control,
          show_op_type=show_op_type)

  def _format_neighbors(self, neighbor_type, non_ctrls, ctrls):
    """List neighbors (inputs or recipients) of a node.

    Args:
      neighbor_type: ("input" | "recipient")
      non_ctrls: Non-control neighbor node names, as a list of str.
      ctrls: Control neighbor node names, as a list of str.

    Returns:
      A list of text lines, as a list of str.
    """

    # TODO(cais): Return RichTextLines instead, to allow annotation of node
    # names.
    lines = []
    lines.append("")
    lines.append("  %d %s(s) + %d control %s(s):" %
                 (len(non_ctrls), neighbor_type, len(ctrls), neighbor_type))
    lines.append("    %d %s(s):" % (len(non_ctrls), neighbor_type))
    for non_ctrl in non_ctrls:
      lines.append("      [%s] %s" %
                   (self._debug_dump.node_op_type(non_ctrl), non_ctrl))

    if ctrls:
      lines.append("")
      lines.append("    %d control %s(s):" % (len(ctrls), neighbor_type))
      for ctrl in ctrls:
        lines.append("      [%s] %s" %
                     (self._debug_dump.node_op_type(ctrl), ctrl))

    return lines

  def _list_node_attributes(self, node_name):
    """List neighbors (inputs or recipients) of a node.

    Args:
      node_name: Name of the node of which the attributes are to be listed.

    Returns:
      A list of text lines, as a list of str.
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

    return lines

  def _list_node_dumps(self, node_name):
    """List dumped tensor data from a node.

    Args:
      node_name: Name of the node of which the attributes are to be listed.

    Returns:
      A list of text lines, as a list of str.
    """

    lines = []
    lines.append("")

    watch_keys = self._debug_dump.debug_watch_keys(node_name)

    dump_count = 0
    for watch_key in watch_keys:
      debug_tensor_data = self._debug_dump.watch_key_to_data(watch_key)
      for datum in debug_tensor_data:
        dump_count += 1
        lines.append("  Slot %d @ %s @ %.3f ms" %
                     (datum.output_slot, datum.debug_op,
                      (datum.timestamp - self._debug_dump.t0) / 1000.0))

    lines.insert(1, "%d dumped tensor(s):" % dump_count)

    return lines


def create_analyzer_curses_cli(debug_dump, tensor_filters=None):
  """Create an instance of CursesUI based on a DebugDumpDir object.

  Args:
    debug_dump: (debug_data.DebugDumpDir) The debug dump to use.
    tensor_filters: (dict) A dict mapping tensor filter name (str) to tensor
      filter (Callable).

  Returns:
    (curses_ui.CursesUI) A curses CLI object with a set of standard analyzer
      commands and tab-completions registered.
  """

  analyzer = DebugAnalyzer(debug_dump)
  if tensor_filters:
    for tensor_filter_name in tensor_filters:
      analyzer.add_tensor_filter(
          tensor_filter_name, tensor_filters[tensor_filter_name])

  cli = curses_ui.CursesUI()
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

  dumped_tensor_names = []
  for datum in debug_dump.dumped_tensor_data:
    dumped_tensor_names.append("%s:%d" % (datum.node_name, datum.output_slot))

  # Tab completions for command "print_tensors".
  cli.register_tab_comp_context(["print_tensor", "pt"], dumped_tensor_names)

  return cli
