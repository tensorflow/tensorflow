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
"""Data structures and helpers for TensorFlow Debugger (tfdbg)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.util import event_pb2
from tensorflow.python.framework import tensor_util


def load_tensor_from_event_file(event_file_path):
  """Load a tensor from an event file.

  Assumes that the event file contains a Event protobuf and the Event protobuf
  contains a tensor.

  Args:
    event_file_path: Path to the event file.

  Returns:
    The tensor value loaded from the event file. For uninitialized tensors,
    return None.
  """
  event = event_pb2.Event()
  with open(event_file_path, "rb") as f:
    event.ParseFromString(f.read())

    if (event.summary.value[0].tensor.tensor_content or
        event.summary.value[0].tensor.string_val):
      # Initialized tensor.
      tensor_value = tensor_util.MakeNdarray(event.summary.value[0].tensor)
    else:
      # Uninitialized tensor.
      tensor_value = None

  return tensor_value


def parse_node_or_tensor_name(name):
  """Get the node name from a string that can be node or tensor name.

  Args:
    name: An input node name (e.g., "node_a") or tensor name (e.g.,
      "node_a:0"), as a str.

  Returns:
    1) The node name, as a str. If the input name is a tensor name, i.e.,
      consists of a colon, the final colon and the following output slot
      will be stripped.
    2) If the input name is a tensor name, the output slot, as an int. If
      the input name is not a tensor name, None.
  """

  if ":" in name and not name.endswith(":"):
    node_name = name[:name.rfind(":")]
    output_slot = int(name[name.rfind(":") + 1:])

    return node_name, output_slot
  else:
    return name, None


def _get_tensor_name(node_name, output_slot):
  """Get tensor name given node name and output slot index.

  Args:
    node_name: Name of the node that outputs the tensor, as a string.
    output_slot: Output slot index of the tensor, as an integer.

  Returns:
    Name of the tensor, as a string.
  """

  return "%s:%d" % (node_name, output_slot)


def _get_tensor_watch_key(node_name, output_slot, debug_op):
  """Get the string representation of a debug watch on a tensor.

  Args:
    node_name: Name of the node by which the watched tensor is produced, as a
        string.
    output_slot: Output slot index of the tensor, as an integer.
    debug_op: Name of the debug op that is used to watch the tensor, as a
        string.

  Returns:
    A string representing the debug watch on the tensor (i.e., the "watch
        key").
  """
  return "%s:%s" % (_get_tensor_name(node_name, output_slot), debug_op)


def _is_copy_node(node_name):
  """Determine whether a node name is that of a debug Copy node.

  Such nodes are inserted by TensorFlow core upon request in
  RunOptions.debug_options.debug_tensor_watch_opts.

  Args:
    node_name: Name of the node.

  Returns:
    A bool indicating whether the input argument is the name of a debug Copy
    node.
  """
  return node_name.startswith("__copy_")


def _is_debug_node(node_name):
  """Determine whether a node name is that of a debug node.

  Such nodes are inserted by TensorFlow core upon request in
  RunOptions.debug_options.debug_tensor_watch_opts.

  Args:
    node_name: Name of the node.

  Returns:
    A bool indicating whether the input argument is the name of a debug node.
  """
  return node_name.startswith("__dbg_")


def _parse_debug_node_name(node_name):
  """Parse the name of a debug node.

  Args:
    node_name: Name of the debug node.

  Returns:
    1. Name of the watched node, as a str.
    2. Output slot index of the watched tensor, as an int.
    3. Index of the debug node, as an int.
    4. Name of the debug op, as a str, e.g, "DebugIdentity".

  Raises:
    ValueError: If the input node name is not a valid debug node name.
  """
  prefix = "__dbg_"

  name = node_name
  if not name.startswith(prefix):
    raise ValueError("Invalid prefix in debug node name: '%s'" % node_name)

  name = name[len(prefix):]

  if name.count("_") < 2:
    raise ValueError("Invalid debug node name: '%s'" % node_name)

  debug_op = name[name.rindex("_") + 1:]
  name = name[:name.rindex("_")]

  debug_op_index = int(name[name.rindex("_") + 1:])
  name = name[:name.rindex("_")]

  if name.count(":") != 1:
    raise ValueError("Invalid tensor name in debug node name: '%s'" % node_name)

  watched_node_name = name[:name.index(":")]
  watched_output_slot = int(name[name.index(":") + 1:])

  return watched_node_name, watched_output_slot, debug_op_index, debug_op


def has_inf_or_nan(datum, tensor):
  """A predicate for whether a tensor consists of any bad numerical values.

  This predicate is common enough to merit definition in this module.
  Bad numerical values include nans and infs.
  The signature of this function follows the requiremnet of DebugDumpDir's
  find() method.

  Args:
    datum: (DebugTensorDatum) Datum metadata.
    tensor: (numpy.ndarray or None) Value of the tensor. None represents
      an uninitialized tensor.

  Returns:
    (bool) True if and only if tensor consists of any nan or inf values.
  """

  _ = datum  # Datum metadata is unused in this predicate.
  if tensor is None:
    # Uninitialized tensor doesn't have bad numerical values.
    return False
  elif (np.issubdtype(tensor.dtype, np.float) or
        np.issubdtype(tensor.dtype, np.complex) or
        np.issubdtype(tensor.dtype, np.integer)):
    return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
  else:
    return False


class DebugTensorDatum(object):
  """A single tensor dumped by tfdbg.

  Contains "metadata" for the dumped tensor, including node name, output slot,
  debug op and timestamp.

  This type does not contain the space-expensive tensor (numpy array) itself.
  It just points to the file path from which the tensor can be loaded if
  needed.
  """

  def __init__(self, dump_root, debug_dump_rel_path):
    """DebugTensorDatum constructor.

    Args:
      dump_root: Debug dump root directory.
      debug_dump_rel_path: Path to a debug dump file, relative to the debug
          dump root directory. For example, suppose the debug dump root
          directory is "/tmp/tfdbg_1" and the dump file is at
          "/tmp/tfdbg_1/ns_1/node_a_0_DebugIdentity_123456789", then
          the value of the debug_dump_rel_path should be
          "ns_1/node_a_0_DebugIdenity_1234456789".
    """
    base = os.path.basename(debug_dump_rel_path)

    # TODO(cais): Add hostname and pid to support dumps from distributed
    #             sessions.

    self._timestamp = int(base.split("_")[-1])
    self._debug_op = base.split("_")[-2]
    self._output_slot = int(base.split("_")[-3])

    namespace = os.path.dirname(debug_dump_rel_path)
    node_base_name = "_".join(base.split("_")[:-3])
    if not namespace or namespace == ".":
      self._node_name = node_base_name
    else:
      self._node_name = namespace + "/" + node_base_name

    self._file_path = os.path.join(dump_root, debug_dump_rel_path)

  def __str__(self):
    return "{DebugTensorDatum: %s:%d @ %s @ %d}" % (self.node_name,
                                                    self.output_slot,
                                                    self.debug_op,
                                                    self.timestamp)

  def __repr__(self):
    return self.__str__()

  def get_tensor(self):
    """Get tensor from the dump (Event) file.

    Returns:
      The tensor loaded from the dump (Event) file.
    """
    return load_tensor_from_event_file(self.file_path)

  @property
  def timestamp(self):
    return self._timestamp

  @property
  def debug_op(self):
    return self._debug_op

  @property
  def node_name(self):
    return self._node_name

  @property
  def output_slot(self):
    return self._output_slot

  @property
  def tensor_name(self):
    return _get_tensor_name(self.node_name, self.output_slot)

  @property
  def watch_key(self):
    """Watch key identities a debug watch on a tensor.

    Returns:
      A watch key, in the form of <tensor_name>:<debug_op>.
    """
    return _get_tensor_watch_key(self.node_name, self.output_slot,
                                 self.debug_op)

  @property
  def file_path(self):
    return self._file_path


class DebugDumpDir(object):
  """Data set from a debug dump directory on filesystem.

  An instance of DebugDumpDir contains all DebugTensorDatum in a tfdbg dump
  root directory. This is an immutable object, of which all constitute tensor
  dump files and partition_graphs are loaded during the __init__ call.
  """

  def __init__(self, dump_root, partition_graphs=None, validate=True):
    """DebugDumpDir constructor.

    Args:
      dump_root: Path to the dump root directory.
      partition_graphs: A repeated field of GraphDefs representing the
          partition graphs executed by the TensorFlow runtime.
      validate: Whether the dump files are to be validated against the
          partition graphs.

    Raises:
      IOError: If dump_root does not exist as a directory.
      ValueError: If the dump_root directory contains file path patterns
         that do not conform to the canonical dump file naming pattern.
    """

    if not os.path.isdir(dump_root):
      raise IOError("Dump root directory %s does not exist" % dump_root)

    self._dump_root = dump_root
    self._dump_tensor_data = []

    # A map from node name to debug watches.
    # The key is the watched node name.
    # The value is a dictionary.
    #   Of this dictionary, the key is the watched_output_slot.
    #   The value is a set of debug ops watching this output slot.
    self._debug_watches = collections.defaultdict(
        lambda: collections.defaultdict(set))

    for root, _, files in os.walk(self._dump_root):
      for f in files:
        if f.count("_") < 3:
          raise ValueError(
              "Dump file path does not conform to the naming pattern: %s" % f)

        debug_dump_rel_path = os.path.join(
            os.path.relpath(root, self._dump_root), f)
        datum = DebugTensorDatum(self._dump_root, debug_dump_rel_path)
        self._dump_tensor_data.append(datum)

        # Attempt to load the debug watches from the tensor dump files first,
        # before loading the full set of debug watches from the partition
        # graphs as done further below.
        # This is necessary because sometimes the partition graphs may not be
        # available, e.g., when the run errors out.
        self._debug_watches[datum.node_name][datum.output_slot].add(
            datum.debug_op)

    # Sort the data by ascending timestamp.
    # This sorting order reflects the order in which the TensorFlow
    # executor processed the nodes of the graph. It is (one of many
    # possible) topological sort of the nodes. This is useful for
    # displaying tensors in the debugger frontend as well as for the use
    # case in which the user wants to find a "culprit tensor", i.e., the
    # first tensor in the graph that exhibits certain problematic
    # properties, i.e., all zero values, or bad numerical values such as
    # nan and inf.
    self._dump_tensor_data = sorted(
        self._dump_tensor_data, key=lambda x: x.timestamp)

    # Time stamp of the first tensor dump.
    if self._dump_tensor_data:
      self._t0 = self._dump_tensor_data[0].timestamp
    else:
      self._t0 = None

    # Create a map from watch key (tensor name + debug op) to
    # DebugTensorDatum item.
    # Also make a map from watch key to relative timestamp.
    # "relative" means (absolute timestamp - t0).
    self._watch_key_to_datum = {}
    self._watch_key_to_rel_time = {}
    for datum in self._dump_tensor_data:
      if datum.watch_key not in self._watch_key_to_datum:
        self._watch_key_to_datum[datum.watch_key] = [datum]
        self._watch_key_to_rel_time[datum.watch_key] = [
            datum.timestamp - self._t0
        ]
      else:
        self._watch_key_to_datum[datum.watch_key].append(datum)
        self._watch_key_to_rel_time[datum.watch_key].append(datum.timestamp -
                                                            self._t0)

    # Initialize partition graph-related information.
    self._partition_graphs = None
    self._node_inputs = None
    self._node_ctrl_inputs = None
    self._node_recipients = None
    self._node_ctrl_recipients = None
    self._devices = None
    self._node_devices = None
    self._node_op_types = None

    # Check the dump data against partition executor graphs.
    if partition_graphs:
      self._load_partition_graphs(partition_graphs)

    if (partition_graphs is not None) and validate:
      self._validate_dump_with_graphs()

  @property
  def dumped_tensor_data(self):
    return self._dump_tensor_data

  @property
  def t0(self):
    """Absolute timestamp of the first dumped tensor.

    Returns:
      Absolute timestamp of the first dumped tensor.
    """
    return self._t0

  @property
  def size(self):
    """Total number of dumped tensors in the dump root directory.

    Returns:
      Total number of dumped tensors in the dump root directory.
    """
    return len(self._dump_tensor_data)

  def _load_partition_graphs(self, partition_graphs):
    """Load and process partition graphs.

    Load the graphs; parse the input and control input structure; obtain the
    device and op type of each node; remove the Copy and debug ops inserted
    by the debugger. The gathered information can be used to validate the
    tensor dumps.

    Args:
      partition_graphs: Partition graphs executed by the TensorFlow runtime,
        represented as repeated fields of GraphDef.

    Raises:
      ValueError: If duplicate node names are encountered.
    """

    self._partition_graphs = partition_graphs

    # A map from node name to node attributes.
    self._node_attributes = {}

    # A map from node name to the node's non-control inputs, for non-debug &
    # non-copy nodes only.
    self._node_inputs = {}

    # A map from node name to the node's control inputs.
    self._node_ctrl_inputs = {}

    # A map from node name to non-control recipients of the node's output(s).
    self._node_recipients = {}

    # A map from node name to control recipients of the node.
    self._node_ctrl_recipients = {}

    # A map from node name to devices (as indices to self._devices)
    self._devices = []
    self._node_devices = {}

    # A map from node name to node type.
    self._node_op_types = {}

    # A list of _Send that send Copy node outputs across devices.
    copy_send_nodes = []

    for pg in self._partition_graphs:
      for node in pg.node:
        if _is_debug_node(node.name):
          # This is a debug node. Parse the node name and retrieve the
          # information about debug watches on tensors. But do not include
          # the node in the graph.
          (watched_node_name, watched_output_slot, _,
           debug_op) = _parse_debug_node_name(node.name)

          self._debug_watches[watched_node_name][watched_output_slot].add(
              debug_op)

          continue

        if node.name in self._node_inputs:
          raise ValueError("Duplicate node name: '%s'" % node.name)

        # Collect node attributes.
        self._node_attributes[node.name] = node.attr

        # Keep track of devices.
        if node.device not in self._devices and node.device:
          self._devices.append(node.device)

        self._node_inputs[node.name] = []
        self._node_ctrl_inputs[node.name] = []
        self._node_recipients[node.name] = []
        self._node_ctrl_recipients[node.name] = []

        self._node_devices[node.name] = node.device
        self._node_op_types[node.name] = node.op

        for inp in node.input:
          if _is_copy_node(inp) and node.op == "_Send":
            copy_send_nodes.append(node.name)

          if inp.startswith("^"):
            cinp = inp[1:]
            self._node_ctrl_inputs[node.name].append(cinp)
          else:
            self._node_inputs[node.name].append(inp)

    # Prune the Copy ops and associated _Send ops inserted by the debugger out
    # from the non-control inputs and output recipients map. Replace the inputs
    # and recipients with original ones.
    copy_nodes = []
    for node in self._node_inputs:
      if node in copy_send_nodes:
        continue

      if _is_copy_node(node):
        copy_nodes.append(node)

      inputs = self._node_inputs[node]

      for i in xrange(len(inputs)):
        inp = inputs[i]
        if _is_copy_node(inp):
          # Find the input to the Copy node, which should be the original
          # input to the node.
          orig_inp = self._node_inputs[inp][0]
          inputs[i] = orig_inp

    # Remove the Copy ops inserted by the debugger from the maps.
    for copy_node in copy_nodes:
      del self._node_inputs[copy_node]
      del self._node_ctrl_inputs[copy_node]
      del self._node_recipients[copy_node]
      del self._node_ctrl_recipients[copy_node]

    # Remove the _Send ops associated with the Copy ops.
    for copy_send_node in copy_send_nodes:
      del self._node_inputs[copy_send_node]
      del self._node_ctrl_inputs[copy_send_node]
      del self._node_recipients[copy_send_node]
      del self._node_ctrl_recipients[copy_send_node]

    # Prune the edges from debug ops from the control edge map.
    for node in self._node_ctrl_inputs:
      ctrl_inputs = self._node_ctrl_inputs[node]
      debug_op_inputs = []
      for ctrl_inp in ctrl_inputs:
        if _is_debug_node(ctrl_inp):
          debug_op_inputs.append(ctrl_inp)
      for debug_op_inp in debug_op_inputs:
        ctrl_inputs.remove(debug_op_inp)

    # Create the recipients maps.
    for node in self._node_inputs:
      inputs = self._node_inputs[node]
      for inp in inputs:
        # A tensor name: replace it with the node name.
        if inp.count(":") == 1:
          inp = inp.split(":")[0]

        self._node_recipients[inp].append(node)

    for node in self._node_ctrl_inputs:
      ctrl_inputs = self._node_ctrl_inputs[node]
      for ctrl_inp in ctrl_inputs:
        if ctrl_inp in copy_send_nodes:
          # Skip _Send ops associated with Copy nodes.
          continue

        self._node_ctrl_recipients[ctrl_inp].append(node)

  def _validate_dump_with_graphs(self):
    """Validate the dumped tensor data against the partition graphs.

    Raises:
      RuntimeError: If the partition graphs have not been loaded yet.
      ValueError: If dumps contain node names not found in partition graph.
        Or if the temporal order of the dump's timestamps violate the
        input relations on the partition graphs.
    """

    if not self._partition_graphs:
      raise RuntimeError("No partition graphs loaded.")

    # Verify that the node names in the dump data are all present in the
    # partittion graphs.
    for datum in self._dump_tensor_data:
      if datum.node_name not in self._node_inputs:
        raise ValueError("Node name '%s' is not found in partition graphs." %
                         datum.node_name)

    pending_inputs = {}
    for node in self._node_inputs:
      pending_inputs[node] = []

      # TODO(cais): tfdbg currently does not watch control edges. Add control
      # edges to pending_inputs when it does.
      inputs = self._node_inputs[node]
      for inp in inputs:
        if inp.count(":") == 1:
          inp = inp.split(":")[0]

        # Keep track of only the watched nodes, as the debugger allows clients
        # to watch a subset of the nodes.
        if inp in self._debug_watches:
          pending_inputs[node].append(inp)

    for datum in self._dump_tensor_data:
      node = datum.node_name
      if pending_inputs[node]:
        raise ValueError("Causality violated in timing relations of debug "
                         "dumps: %s (%d): "
                         "these input(s) are not satisfied: %s" %
                         (node, datum.timestamp, repr(pending_inputs[node])))

      # Get the recipients of the node's output
      recipients = self._node_recipients[node]
      for recipient in recipients:
        recipient_pending_inputs = pending_inputs[recipient]
        if node in recipient_pending_inputs:
          if self.node_op_type(recipient) == "Merge":
            # If this is a Merge op, we automatically clear the list because
            # a Merge node only requires one of its two inputs.
            del recipient_pending_inputs[:]
          else:
            del recipient_pending_inputs[recipient_pending_inputs.index(node)]

  def loaded_partition_graphs(self):
    """Test whether partition graphs have been loaded."""
    return self._partition_graphs is not None

  def partition_graphs(self):
    """Get the partition graphs.

    Returns:
      Partition graphs as repeated fields of GraphDef.

    Raises:
      RuntimeError: If no partition graphs have been loaded.
    """
    if self._partition_graphs is None:
      raise RuntimeError("No partition graphs have been loaded.")

    return self._partition_graphs

  def nodes(self):
    """Get a list of all nodes from the partition graphs.

    Returns:
      All nodes' names, as a list of str.

    Raises:
      RuntimeError: If no partition graphs have been loaded.
    """
    if self._partition_graphs is None:
      raise RuntimeError("No partition graphs have been loaded.")

    return [node_name for node_name in self._node_inputs]

  def node_attributes(self, node_name):
    """Get attributes of a node.

    Args:
      node_name: Name of the node in question.

    Returns:
      Attributes of the node.

    Raises:
      RuntimeError: If no partition graphs have been loaded.
      ValueError: If no node named node_name exists.
    """
    if self._partition_graphs is None:
      raise RuntimeError("No partition graphs have been loaded.")

    if node_name in self._node_attributes:
      return self._node_attributes[node_name]
    else:
      raise ValueError("No node named \"%s\" exists." % node_name)

  def node_inputs(self, node_name, is_control=False):
    """Get the inputs of given node according to partition graphs.

    Args:
      node_name: Name of the node.
      is_control: Whether control inputs, rather than non-control inputs, are
      to be returned.

    Returns:
      All non-control inputs to the node, as a list of node names.

    Raises:
      RuntimeError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """

    if self._node_inputs is None or self._node_ctrl_inputs is None:
      raise RuntimeError(
          "Node inputs are not loaded from partition graphs yet.")

    if node_name not in self._node_inputs:
      raise ValueError("Node '%s' does not exist in partition graphs." %
                       node_name)

    if is_control:
      return self._node_ctrl_inputs[node_name]
    else:
      return self._node_inputs[node_name]

  def transitive_inputs(self, node_name, include_control=True):
    """Get the transitive inputs of given node according to partition graphs.

    Args:
      node_name: Name of the node
      include_control: Include control inputs (True by default).

    Returns:
      All transitive inputs to the node, as a list of node names.

    Raises:
      RuntimeError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """

    if not self._node_inputs or not self._node_ctrl_inputs:
      raise RuntimeError(
          "Node inputs are not loaded from partition graphs yet.")

    if node_name not in self._node_inputs:
      raise ValueError("Node '%s' does not exist in partition graphs." %
                       node_name)

    inputs = []

    # Keep track of visited nodes to avoid infinite loops during input
    # tracing.
    visited_nodes = []

    def trace_inputs(node):
      """Inner function for recursive tracing of node inputs.

      The transitive input names are appended to the list captured list
      "inputs".

      Args:
        node: Name of the node, as a str.
      """
      if node.count(":") == 1:
        # This check is necessary for cases in which an input is not from the
        # 0-th output slot, e.g., from a Switch op.
        node = node[:node.rindex(":")]

      # Stop the tracing at a Merge op, as it is generally impossible to infer
      # outside the runtime which input to the Merge op is alive.
      if self._node_op_types[node] == "Merge":
        return

      if node in visited_nodes:
        # Avoid infinite loops.
        return
      visited_nodes.append(node)

      for inp in self._node_inputs[node]:
        if inp == node_name:
          continue
        inputs.append(inp)
        trace_inputs(inp)  # Recursive call.

      if include_control:
        for ctrl_inp in self._node_ctrl_inputs[node]:
          if ctrl_inp == node_name:
            continue
          inputs.append(ctrl_inp)
          trace_inputs(ctrl_inp)  # Recursive call.

    trace_inputs(node_name)

    return inputs

  def node_recipients(self, node_name, is_control=False):
    """Get recipient of the given node's output according to partition graphs.

    Args:
      node_name: Name of the node.
      is_control: Whether control outputs, rather than non-control outputs,
      are to be returned.

    Returns:
      All non-control inputs to the node, as a list of node names.

    Raises:
      RuntimeError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """

    if self._node_recipients is None or self._node_ctrl_recipients is None:
      raise RuntimeError(
          "Node recipients are not loaded from partition graphs yet.")

    if node_name not in self._node_recipients:
      raise ValueError("Node '%s' does not exist in partition graphs." %
                       node_name)

    if is_control:
      return self._node_ctrl_recipients[node_name]
    else:
      return self._node_recipients[node_name]

  def devices(self):
    """Get the list of devices.

    Returns:
      Number of devices.

    Raises:
      RuntimeError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
    """

    if self._devices is None:
      raise RuntimeError("Devices are not loaded from partition graphs yet.")

    return self._devices

  def node_exists(self, node_name):
    """Test if a node exists in the partition graphs.

    Args:
      node_name: Name of the node to be checked, as a str.

    Returns:
      A boolean indicating whether the node exists.

    Raises:
      RuntimeError: If no partition graphs have been loaded yet.
    """

    if self._node_inputs is None:
      raise RuntimeError(
          "Nodes have not been loaded from partition graphs yet.")

    return node_name in self._node_inputs

  def node_device(self, node_name):
    """Get the device of a node.

    Args:
      node_name: Name of the node.

    Returns:
      Name of the device on which the node is placed, as a str.

    Raises:
      RuntimeError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """
    if self._node_devices is None:
      raise RuntimeError(
          "Node devices are not loaded from partition graphs yet.")

    if node_name not in self._node_devices:
      raise ValueError("Node '%s' does not exist in partition graphs." %
                       node_name)

    return self._node_devices[node_name]

  def node_op_type(self, node_name):
    """Get the op type of given node.

    Args:
      node_name: Name of the node.

    Returns:
      Type of the node's op, as a str.

    Raises:
      RuntimeError: If node op types have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """
    if self._node_op_types is None:
      raise RuntimeError(
          "Node op types are not loaded from partition graphs yet.")

    if node_name not in self._node_op_types:
      raise ValueError("Node '%s' does not exist in partition graphs." %
                       node_name)

    return self._node_op_types[node_name]

  def debug_watch_keys(self, node_name):
    """Get all tensor watch keys of given node according to partition graphs.

    Args:
      node_name: Name of the node.

    Returns:
      All debug tensor watch keys, as a list of strings. Returns an empty list
      if the node name does not correspond to any debug watch keys.

    Raises:
      RuntimeError: If debug watch information has not been loaded from
        partition graphs yet.
    """

    if node_name not in self._debug_watches:
      return []

    watch_keys = []
    for watched_slot in self._debug_watches[node_name]:
      debug_ops = self._debug_watches[node_name][watched_slot]
      for debug_op in debug_ops:
        watch_keys.append(
            _get_tensor_watch_key(node_name, watched_slot, debug_op))

    return watch_keys

  def watch_key_to_data(self, debug_watch_key):
    """Get all DebugTensorDatum instances corresponding to a debug watch key.

    Args:
      debug_watch_key: A debug watch key, as a str.

    Returns:
      A list of DebugTensorDatuminstances that correspond to the debug watch
      key. If the watch key does not exist, returns an empty list.

    Raises:
      ValueError: If the debug watch key does not exist.
    """

    return self._watch_key_to_datum.get(debug_watch_key, [])

  def find(self, predicate, first_n=0):
    """Find dumped tensor data by a certain predicate.

    Args:
      predicate: A callable that takes two input arguments:
          predicate(debug_tensor_datum, tensor),
          where "debug_tensor_datum" is an instance of DebugTensorDatum, which
          carries "metadata", such as the name of the node, the tensor's slot
          index on the node, timestamp, debug op name, etc; and "tensor" is
          the dumped tensor value as a numpy array.
      first_n: Return only the first n dumped tensor data (in time order) for
          which the predicate is True. To return all such data, let first_n be
          <= 0.

    Returns:
      A list of all DebugTensorDatum objects in this DebugDumpDir object for
      which predicate returns True, sorted in ascending order of the timestamp.
    """

    matched_data = []
    for datum in self._dump_tensor_data:
      if predicate(datum, datum.get_tensor()):
        matched_data.append(datum)

        if first_n > 0 and len(matched_data) >= first_n:
          break

    return matched_data

  def get_tensor_file_paths(self, node_name, output_slot, debug_op):
    """Get the file paths from a debug-dumped tensor.

    Args:
      node_name: Name of the node that the tensor is produced by.
      output_slot: Output slot index of tensor.
      debug_op: Name of the debug op.

    Returns:
      List of file path(s) loaded. This is a list because each debugged tensor
        may be dumped multiple times.

    Raises:
      ValueError: If the tensor does not exist in the debub dump data.
    """

    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum:
      raise ValueError("Watch key \"%s\" does not exist in the debug dump" %
                       watch_key)

    return [datum.file_path for datum in self._watch_key_to_datum[watch_key]]

  def get_tensors(self, node_name, output_slot, debug_op):
    """Get the tensor value from for a debug-dumped tensor.

    The tensor may be dumped multiple times in the dump root directory, so a
    list of tensors (numpy arrays) is returned.

    Args:
      node_name: Name of the node that the tensor is produced by.
      output_slot: Output slot index of tensor.
      debug_op: Name of the debug op.

    Returns:
      List of tensor(s) loaded from the tensor dump file(s).

    Raises:
      ValueError: If the tensor does not exist in the debub dump data.
    """

    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum:
      raise ValueError("Watch key \"%s\" does not exist in the debug dump" %
                       watch_key)

    return [datum.get_tensor() for datum in self._watch_key_to_datum[watch_key]]

  def get_rel_timestamps(self, node_name, output_slot, debug_op):
    """Get the relative timestamp from for a debug-dumped tensor.

    Relative timestamp means (absolute timestamp - t0), t0 being the absolute
    timestamp of the first dumped tensor in the dump root. The tensor may be
    dumped multiple times in the dump root directory, so a list of relative
    timestamp (numpy arrays) is returned.

    Args:
      node_name: Name of the node that the tensor is produced by.
      output_slot: Output slot index of tensor.
      debug_op: Name of the debug op.

    Returns:
      List of relative timestamps.

    Raises:
      ValueError: If the tensor does not exist in the debub dump data.
    """

    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum:
      raise ValueError("Watch key \"%s\" does not exist in the debug dump" %
                       watch_key)

    return self._watch_key_to_rel_time[watch_key]
