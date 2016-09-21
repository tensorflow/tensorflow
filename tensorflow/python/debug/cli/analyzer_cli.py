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
import re

from tensorflow.python.debug.cli import debugger_cli_common


class DebugAnalyzer(object):
  """Analyzer for debug data from dump directories."""

  def __init__(self, debug_dump):
    """DebugAnalyzer constructor.

    Args:
      debug_dump: A DebugDumpDir object.
    """

    self._debug_dump = debug_dump

    # Argument parsers for command handlers.
    self._arg_parsers = {}

    # Parser for list_tensors.
    ap = argparse.ArgumentParser(
        description="List dumped intermediate tensors.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-o",
        "--offending",
        dest="offending",
        action="store_true",
        help="List only offending tensors.")  # TODO(cais): Implement.
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

    # TODO(cais): Implement list_nodes.
    # TODO(cais): Implement print_tensor.
    # TODO(cais): Implement inputs (including recursive).
    # TODO(cais): Implement outputs (including recursive).

  def _error(self, msg):
    return debugger_cli_common.RichTextLines(
        [msg], font_attr_segs={0: [(0, len(msg), "red")]})

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

    # TODO(cais): Implement filter by lambda on tensor value.

    dump_count = 0
    for dump in self._debug_dump.dumped_tensor_data:
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

    if ":" in parsed.node_name:
      node_name = parsed.node_name[:parsed.node_name.rfind(":")]
    else:
      node_name = parsed.node_name

    if node_name not in self._debug_dump.nodes():
      return self._error(
          "Error: There is no node named \"%s\" in the partition graphs" %
          node_name)

    lines = ["Node %s" % node_name]
    lines.append("")
    lines.append("  Op: %s" % self._debug_dump.node_op_type(node_name))
    lines.append("  Device: %s" % self._debug_dump.node_device(node_name))

    # List node inputs (non-control and control).
    inputs = self._debug_dump.node_inputs(node_name)
    ctrl_inputs = self._debug_dump.node_inputs(node_name, is_control=True)

    input_lines = self._list_neighbors("input", inputs, ctrl_inputs)
    lines.extend(input_lines)

    # List node output recipients (non-control and control).
    recs = self._debug_dump.node_recipients(node_name)
    ctrl_recs = self._debug_dump.node_recipients(node_name, is_control=True)

    rec_lines = self._list_neighbors("recipient", recs, ctrl_recs)
    lines.extend(rec_lines)

    # Optional: List attributes of the node.
    if parsed.attributes:
      lines.extend(self._list_node_attributes(node_name))

    # Optional: List dumps available from the node.
    if parsed.dumps:
      lines.extend(self._list_node_dumps(node_name))

    return debugger_cli_common.RichTextLines(lines)

  def _list_neighbors(self, neighbor_type, non_ctrls, ctrls):
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
