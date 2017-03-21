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
"""CLI Backend for the Node Stepper Part of the Debugger."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np  # pylint: disable=unused-import
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import stepper

RL = debugger_cli_common.RichLine


class NodeStepperCLI(object):
  """Command-line-interface backend of Node Stepper."""

  # Possible states of an element in the transitive closure of the stepper's
  # fetch(es).
  # State where the element is already continued-to and a TensorHandle is
  # available for the tensor.
  STATE_CONT = "H"

  # State where an intermediate dump of the tensor is available.
  STATE_DUMPED_INTERMEDIATE = "I"

  # State where the element is already overridden.
  STATE_OVERRIDDEN = "O"

  # State where the element is a placeholder (and hence cannot be continued to)
  STATE_IS_PLACEHOLDER = "P"

  # State where a variable's value has been updated during the lifetime of
  # this NodeStepperCLI instance.
  STATE_DIRTY_VARIABLE = "D"

  STATE_UNFEEDABLE = "U"

  NEXT_NODE_POINTER_STR = "-->"

  _MESSAGE_TEMPLATES = {
      "NOT_IN_CLOSURE":
          "%s is not in the transitive closure of this stepper instance.",
      "MULTIPLE_TENSORS":
          "Node %s has more than one output tensor. "
          "Please use full tensor name.",
  }

  _UPDATED_ATTRIBUTE = "bold"

  _STATE_COLORS = {
      STATE_CONT: "green",
      STATE_DIRTY_VARIABLE: "magenta",
      STATE_DUMPED_INTERMEDIATE: "blue",
      STATE_OVERRIDDEN: "yellow",
      STATE_IS_PLACEHOLDER: "cyan",
      STATE_UNFEEDABLE: "red",
  }

  _FEED_COLORS = {
      stepper.NodeStepper.FEED_TYPE_CLIENT: "white",
      stepper.NodeStepper.FEED_TYPE_HANDLE: "green",
      stepper.NodeStepper.FEED_TYPE_OVERRIDE: "yellow",
      stepper.NodeStepper.FEED_TYPE_DUMPED_INTERMEDIATE: "blue",
  }

  def __init__(self, node_stepper):
    self._node_stepper = node_stepper

    # Command parsers for the stepper.
    self.arg_parsers = {}

    # Parser for "list_sorted_nodes".
    ap = argparse.ArgumentParser(
        description="List the state of the sorted transitive closure of the "
        "stepper.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-l",
        "--lower_bound",
        dest="lower_bound",
        type=int,
        default=-1,
        help="Lower-bound index (0-based)")
    ap.add_argument(
        "-u",
        "--upper_bound",
        dest="upper_bound",
        type=int,
        default=-1,
        help="Upper-bound index (0-based)")
    self.arg_parsers["list_sorted_nodes"] = ap

    # Parser for "cont".
    ap = argparse.ArgumentParser(
        description="Continue to a tensor or op.", usage=argparse.SUPPRESS)
    ap.add_argument(
        "target_name",
        type=str,
        help="Name of the Tensor or Op to continue to.")
    ap.add_argument(
        "-i",
        "--invalidate_from_updated_variables",
        dest="invalidate_from_updated_variables",
        action="store_true",
        help="Whether to invalidate the cached "
             "tensor handles and intermediate tensor handles affected "
             "by Variable updates in this continue call.")
    ap.add_argument(
        "-r",
        "--restore_variable_values",
        dest="restore_variable_values",
        action="store_true",
        help="Restore all variables in the transitive closure of the cont "
             "target to their initial values (i.e., values when this stepper "
             "instance was created.")
    self.arg_parsers["cont"] = ap

    # Parser for "step".
    ap = argparse.ArgumentParser(
        description="Step to the next tensor or op in the sorted transitive "
        "closure of the stepper's fetch(es).",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-t",
        "--num_times",
        dest="num_times",
        type=int,
        default=1,
        help="Number of times to step (>=1)")
    self.arg_parsers["step"] = ap

    # Parser for "print_tensor".
    ap = argparse.ArgumentParser(
        description="Print the value of a tensor, from cached TensorHandle or "
        "client-provided overrides.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "tensor_name",
        type=str,
        help="Name of the tensor, followed by any slicing indices, "
        "e.g., hidden1/Wx_plus_b/MatMul:0, "
        "hidden1/Wx_plus_b/MatMul:0[1, :]")
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
    self.arg_parsers["print_tensor"] = ap

    # Parser for inject_value.
    ap = argparse.ArgumentParser(
        description="Inject (override) the value of a Tensor.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "tensor_name",
        type=str,
        help="Name of the Tensor of which the value is to be overridden.")
    ap.add_argument(
        "tensor_value_str",
        type=str,
        help="A string representing the value of the tensor, without any "
        "whitespaces, e.g., np.zeros([10,100])")
    self.arg_parsers["inject_value"] = ap

    self._initialize_state()

  def _initialize_state(self):
    """Initialize the state of this stepper CLI."""

    # Get the elements in the sorted transitive closure, as a list of str.
    self._sorted_nodes = self._node_stepper.sorted_nodes()
    self._closure_elements = self._node_stepper.closure_elements()
    self._placeholders = self._node_stepper.placeholders()
    self._completed_nodes = set()

    self._calculate_next()

  def _calculate_next(self):
    """Calculate the next target for "step" action based on current state."""

    override_names = self._node_stepper.override_names()

    next_i = -1
    for i in xrange(len(self._sorted_nodes)):
      if (i > next_i and (self._sorted_nodes[i] in self._completed_nodes) or
          (self._sorted_nodes[i] in override_names)):
        next_i = i

    next_i += 1
    self._next = next_i

  def list_sorted_nodes(self, args, screen_info=None):
    """List the sorted transitive closure of the stepper's fetches."""

    # TODO(cais): Use pattern such as del args, del screen_info python/debug.
    _ = args
    _ = screen_info

    parsed = self.arg_parsers["list_sorted_nodes"].parse_args(args)

    if parsed.lower_bound != -1 and parsed.upper_bound != -1:
      index_range = [
          max(0, parsed.lower_bound),
          min(len(self._sorted_nodes), parsed.upper_bound)
      ]
      verbose = False
    else:
      index_range = [0, len(self._sorted_nodes)]
      verbose = True

    handle_node_names = self._node_stepper.handle_node_names()
    intermediate_tensor_names = self._node_stepper.intermediate_tensor_names()
    override_names = self._node_stepper.override_names()
    dirty_variable_names = [
        dirty_variable.split(":")[0]
        for dirty_variable in self._node_stepper.dirty_variables()
    ]

    lines = []
    if verbose:
      lines.extend(
          ["Topologically-sorted transitive input(s) and fetch(es):", ""])

    for i, element_name in enumerate(self._sorted_nodes):
      if i < index_range[0] or i >= index_range[1]:
        continue

      # TODO(cais): Use fixed-width text to show node index.
      if i == self._next:
        node_prefix = RL("  ") + RL(self.NEXT_NODE_POINTER_STR, "bold")
      else:
        node_prefix = RL("     ")

      node_prefix += "(%d / %d)" % (i + 1, len(self._sorted_nodes)) + "  ["
      node_prefix += self._get_status_labels(
          element_name,
          handle_node_names,
          intermediate_tensor_names,
          override_names,
          dirty_variable_names)

      lines.append(node_prefix + "] " + element_name)

    output = debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

    if verbose:
      output.extend(self._node_status_label_legend())

    return output

  def _get_status_labels(self,
                         element_name,
                         handle_node_names,
                         intermediate_tensor_names,
                         override_names,
                         dirty_variable_names):
    """Get a string of status labels for a graph element.

    A status label indicates that a node has a certain state in this
    node-stepper CLI invocation. For example, 1) that the node has been
    continued-to and a handle to its output tensor is available to the node
    stepper; 2) the node is a Variable and its value has been altered, e.g.,
    by continuing to a variable-updating node, since the beginning of this
    node-stepper invocation (i.e., "dirty variable").

    Args:
      element_name: (str) name of the graph element.
      handle_node_names: (list of str) Names of the nodes of which the output
        tensors' handles are available.
      intermediate_tensor_names: (list of str) Names of the intermediate tensor
        dumps generated from the graph element.
      override_names: (list of str) Names of the tensors of which the values
        are overridden.
      dirty_variable_names: (list of str) Names of the dirty variables.

    Returns:
      (RichLine) The rich text string of status labels that currently apply to
        the graph element.
    """

    status = RL()

    node_name = element_name.split(":")[0]
    status += (RL(self.STATE_IS_PLACEHOLDER,
                  self._STATE_COLORS[self.STATE_IS_PLACEHOLDER])
               if node_name in self._placeholders else " ")
    status += (RL(self.STATE_UNFEEDABLE,
                  self._STATE_COLORS[self.STATE_UNFEEDABLE])
               if not self._node_stepper.is_feedable(str(element_name))
               else " ")
    status += (RL(self.STATE_CONT, self._STATE_COLORS[self.STATE_CONT])
               if element_name in handle_node_names else " ")

    intermediate_node_names = [
        tensor_name.split(":")[0] for tensor_name in intermediate_tensor_names]
    status += (RL(self.STATE_DUMPED_INTERMEDIATE,
                  self._STATE_COLORS[self.STATE_DUMPED_INTERMEDIATE])
               if element_name in intermediate_node_names else " ")

    slots = self._node_stepper.output_slots_in_closure(element_name)
    has_override = any(element_name + ":%d" % slot in override_names
                       for slot in slots)
    status += (RL(self.STATE_OVERRIDDEN,
                  self._STATE_COLORS[self.STATE_OVERRIDDEN])
               if has_override else " ")
    status += (RL(self.STATE_DIRTY_VARIABLE,
                  self._STATE_COLORS[self.STATE_DIRTY_VARIABLE])
               if element_name in dirty_variable_names else " ")

    return status

  def _node_status_label_legend(self):
    """Get legend for node-status labels.

    Returns:
      (debugger_cli_common.RichTextLines) Legend text.
    """

    return debugger_cli_common.rich_text_lines_from_rich_line_list([
        "",
        "Legend:",
        (RL("  ") +
         RL(self.STATE_IS_PLACEHOLDER,
            self._STATE_COLORS[self.STATE_IS_PLACEHOLDER]) +
         " - Placeholder"),
        (RL("  ") +
         RL(self.STATE_UNFEEDABLE,
            self._STATE_COLORS[self.STATE_UNFEEDABLE]) +
         " - Unfeedable"),
        (RL("  ") +
         RL(self.STATE_CONT,
            self._STATE_COLORS[self.STATE_CONT]) +
         " - Already continued-to; Tensor handle available from output "
         "slot(s)"),
        (RL("  ") +
         RL(self.STATE_DUMPED_INTERMEDIATE,
            self._STATE_COLORS[self.STATE_DUMPED_INTERMEDIATE]) +
         " - Unfeedable"),
        (RL("  ") +
         RL(self.STATE_OVERRIDDEN,
            self._STATE_COLORS[self.STATE_OVERRIDDEN]) +
         " - Has overriding (injected) tensor value"),
        (RL("  ") +
         RL(self.STATE_DIRTY_VARIABLE,
            self._STATE_COLORS[self.STATE_DIRTY_VARIABLE]) +
         " - Dirty variable: Variable already updated this node stepper.")])

  def cont(self, args, screen_info=None):
    """Continue-to action on the graph."""

    _ = screen_info

    parsed = self.arg_parsers["cont"].parse_args(args)

    # Determine which node is being continued to, so the _next pointer can be
    # set properly.
    node_name = parsed.target_name.split(":")[0]
    if node_name not in self._sorted_nodes:
      return cli_shared.error(self._MESSAGE_TEMPLATES["NOT_IN_CLOSURE"] %
                              parsed.target_name)
    self._next = self._sorted_nodes.index(node_name)

    cont_result = self._node_stepper.cont(
        parsed.target_name,
        invalidate_from_updated_variables=(
            parsed.invalidate_from_updated_variables),
        restore_variable_values=parsed.restore_variable_values)
    self._completed_nodes.add(parsed.target_name.split(":")[0])

    screen_output = debugger_cli_common.RichTextLines(
        ["Continued to %s:" % parsed.target_name, ""])
    screen_output.extend(self._report_last_feed_types())
    screen_output.extend(self._report_last_updated())
    screen_output.extend(
        tensor_format.format_tensor(
            cont_result, parsed.target_name, include_metadata=True))

    # Generate windowed view of the sorted transitive closure on which the
    # stepping is occurring.
    lower_bound = max(0, self._next - 2)
    upper_bound = min(len(self._sorted_nodes), self._next + 3)

    final_output = self.list_sorted_nodes(
        ["-l", str(lower_bound), "-u", str(upper_bound)])
    final_output.extend(debugger_cli_common.RichTextLines([""]))
    final_output.extend(screen_output)

    # Re-calculate the target of the next "step" action.
    self._calculate_next()

    return final_output

  def _report_last_feed_types(self):
    """Generate a report of the feed types used in the cont/step call.

    Returns:
      (debugger_cli_common.RichTextLines) A RichTextLines representation of the
        feeds used in the last cont/step call.
    """
    feed_types = self._node_stepper.last_feed_types()

    out = ["Stepper used feeds:"]
    if feed_types:
      for feed_name in feed_types:
        feed_info = RL("  %s : " % feed_name)
        feed_info += RL(feed_types[feed_name],
                        self._FEED_COLORS[feed_types[feed_name]])
        out.append(feed_info)
    else:
      out.append("  (No feeds)")
    out.append("")

    return debugger_cli_common.rich_text_lines_from_rich_line_list(out)

  def _report_last_updated(self):
    """Generate a report of the variables updated in the last cont/step call.

    Returns:
      (debugger_cli_common.RichTextLines) A RichTextLines representation of the
        variables updated in the last cont/step call.
    """

    last_updated = self._node_stepper.last_updated()
    if not last_updated:
      return debugger_cli_common.RichTextLines([])

    rich_lines = [RL("Updated:", self._UPDATED_ATTRIBUTE)]
    sorted_last_updated = sorted(list(last_updated))
    for updated in sorted_last_updated:
      rich_lines.append("  %s" % updated)
    rich_lines.append("")
    return debugger_cli_common.rich_text_lines_from_rich_line_list(rich_lines)

  def step(self, args, screen_info=None):
    """Step once.

    Args:
      args: (list of str) command-line arguments for the "step" command.
      screen_info: Information about screen.

    Returns:
      (RichTextLines) Screen output for the result of the stepping action.
    """

    parsed = self.arg_parsers["step"].parse_args(args)

    if parsed.num_times < 0:
      return debugger_cli_common.RichTextLines(
          "ERROR: Invalid number of times to step: %d" % parsed.num_times)

    for _ in xrange(parsed.num_times):
      if self._next >= len(self._sorted_nodes):
        return debugger_cli_common.RichTextLines(
            "ERROR: Cannot step any further because the end of the sorted "
            "transitive closure has been reached.")
      else:
        screen_output = self.cont([self._sorted_nodes[self._next]], screen_info)

    return screen_output

  def print_tensor(self, args, screen_info=None):
    """Print the value of a tensor that the stepper has access to."""

    parsed = self.arg_parsers["print_tensor"].parse_args(args)

    if screen_info and "cols" in screen_info:
      np_printoptions = {"linewidth": screen_info["cols"]}
    else:
      np_printoptions = {}

    # Determine if any range-highlighting is required.
    highlight_options = cli_shared.parse_ranges_highlight(parsed.ranges)

    tensor_name, tensor_slicing = (
        command_parser.parse_tensor_name_with_slicing(parsed.tensor_name))

    tensor_names = self._resolve_tensor_names(tensor_name)
    if not tensor_names:
      return cli_shared.error(
          self._MESSAGE_TEMPLATES["NOT_IN_CLOSURE"] % tensor_name)
    elif len(tensor_names) > 1:
      return cli_shared.error(
          self._MESSAGE_TEMPLATES["MULTIPLE_TENSORS"] % tensor_name)
    else:
      tensor_name = tensor_names[0]

    try:
      tensor_value = self._node_stepper.get_tensor_value(tensor_name)
    except ValueError as e:
      return debugger_cli_common.RichTextLines([str(e)])

    return cli_shared.format_tensor(
        tensor_value,
        tensor_name,
        np_printoptions,
        print_all=parsed.print_all,
        tensor_slicing=tensor_slicing,
        highlight_options=highlight_options)

  def inject_value(self, args, screen_info=None):
    """Inject value to a given tensor.

    Args:
      args: (list of str) command-line arguments for the "step" command.
      screen_info: Information about screen.

    Returns:
      (RichTextLines) Screen output for the result of the stepping action.
    """

    _ = screen_info  # Currently unused.

    if screen_info and "cols" in screen_info:
      np_printoptions = {"linewidth": screen_info["cols"]}
    else:
      np_printoptions = {}

    parsed = self.arg_parsers["inject_value"].parse_args(args)

    tensor_names = self._resolve_tensor_names(parsed.tensor_name)
    if not tensor_names:
      return cli_shared.error(
          self._MESSAGE_TEMPLATES["NOT_IN_CLOSURE"] % parsed.tensor_name)
    elif len(tensor_names) > 1:
      return cli_shared.error(
          self._MESSAGE_TEMPLATES["MULTIPLE_TENSORS"] % parsed.tensor_name)
    else:
      tensor_name = tensor_names[0]

    tensor_value = eval(parsed.tensor_value_str)  # pylint: disable=eval-used

    try:
      self._node_stepper.override_tensor(tensor_name, tensor_value)
      lines = [
          "Injected value \"%s\"" % parsed.tensor_value_str,
          "  to tensor \"%s\":" % tensor_name, ""
      ]

      tensor_lines = tensor_format.format_tensor(
          tensor_value,
          tensor_name,
          include_metadata=True,
          np_printoptions=np_printoptions).lines
      lines.extend(tensor_lines)

    except ValueError:
      lines = [
          "ERROR: Failed to inject value to tensor %s" % parsed.tensor_name
      ]

    return debugger_cli_common.RichTextLines(lines)

  # TODO(cais): Implement list_inputs
  # TODO(cais): Implement list_outputs
  # TODO(cais): Implement node_info

  def _resolve_tensor_names(self, element_name):
    """Resolve tensor name from graph element name.

    Args:
      element_name: (str) Name of the graph element to resolve.

    Returns:
      (list) Name of the tensor(s). If element_name is the name of a tensor in
      the transitive closure, return [element_name]. If element_name is the
      name of a node in the transitive closure, return the list of output
      tensors from the node that are in the transitive closure. Otherwise,
      return empty list.
    """

    if element_name in self._closure_elements and ":" in element_name:
      return [element_name]
    if (element_name in self._sorted_nodes or
        (element_name in self._closure_elements and ":" not in element_name)):
      slots = self._node_stepper.output_slots_in_closure(element_name)
      return [(element_name + ":%d" % slot) for slot in slots]
    else:
      return []
