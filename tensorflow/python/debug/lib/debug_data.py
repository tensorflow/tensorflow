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
"""Classes and functions to handle debug-dump data of TensorFlow Debugger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile


METADATA_FILE_PREFIX = "_tfdbg_"
CORE_METADATA_TAG = "core_metadata_"
GRAPH_FILE_TAG = "graph_"
FETCHES_INFO_FILE_TAG = "fetches_info_"
FEED_KEYS_INFO_FILE_TAG = "feed_keys_info_"


def load_tensor_from_event_file(event_file_path):
  """Load a tensor from an event file.

  Assumes that the event file contains a `Event` protobuf and the `Event`
  protobuf contains a `Tensor` value.

  Args:
    event_file_path: (`str`) path to the event file.

  Returns:
    The tensor value loaded from the event file, as a `numpy.ndarray`. For
    uninitialized Tensors, returns `None`. For Tensors of data types that
    cannot be converted to `numpy.ndarray` (e.g., `tf.resource`), return
    `None`.
  """

  event = event_pb2.Event()
  with gfile.Open(event_file_path, "rb") as f:
    event.ParseFromString(f.read())
    return load_tensor_from_event(event)


def load_tensor_from_event(event):
  """Load a tensor from an Event proto.

  Args:
    event: The Event proto, assumed to hold a tensor value in its
        summary.value[0] field.

  Returns:
    The tensor value loaded from the event file, as a `numpy.ndarray`. For
    uninitialized Tensors, returns `None`. For Tensors of data types that
    cannot be converted to `numpy.ndarray` (e.g., `tf.resource`), return
    `None`.
  """

  if (event.summary.value[0].tensor.tensor_content or
      event.summary.value[0].tensor.string_val):
    # Initialized tensor.
    tensor_proto = event.summary.value[0].tensor
    if tensor_proto.dtype == types_pb2.DT_RESOURCE:
      return None
    else:
      try:
        tensor_value = tensor_util.MakeNdarray(tensor_proto)
      except KeyError:
        tensor_value = None
  else:
    # Uninitialized tensor or tensor of unconvertible data type.
    tensor_value = None

  return tensor_value


def _load_graph_def_from_event_file(event_file_path):
  event = event_pb2.Event()
  with gfile.Open(event_file_path, "rb") as f:
    event.ParseFromString(f.read())

  return graph_pb2.GraphDef.FromString(event.graph_def)


def _load_log_message_from_event_file(event_file_path):
  event = event_pb2.Event()
  with gfile.Open(event_file_path, "rb") as f:
    event.ParseFromString(f.read())

  return event.log_message.message


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


def _is_core_metadata_file(file_name):
  return file_name.startswith(METADATA_FILE_PREFIX + CORE_METADATA_TAG)


def _is_graph_file(file_name):
  return file_name.startswith(METADATA_FILE_PREFIX + GRAPH_FILE_TAG)


def _is_run_fetches_info_file(file_name):
  return file_name == METADATA_FILE_PREFIX + FETCHES_INFO_FILE_TAG


def _is_run_feed_keys_info_file(file_name):
  return file_name == METADATA_FILE_PREFIX + FEED_KEYS_INFO_FILE_TAG


def get_node_name(element_name):
  return element_name.split(":")[0] if ":" in element_name else element_name


def get_output_slot(element_name):
  """Get the output slot number from the name of a graph element.

  If element_name is a node name without output slot at the end, 0 will be
  assumed.

  Args:
    element_name: (`str`) name of the graph element in question.

  Returns:
    (`int`) output slot number.
  """
  return int(element_name.split(":")[-1]) if ":" in element_name else 0


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
  Bad numerical values include `nan`s and `inf`s.
  The signature of this function follows the requirement of the method
  `DebugDumpDir.find()`.

  Args:
    datum: (`DebugTensorDatum`) Datum metadata.
    tensor: (`numpy.ndarray` or None) Value of the tensor. None represents
      an uninitialized tensor.

  Returns:
    (`bool`) True if and only if tensor consists of any nan or inf values.
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


def extract_core_metadata_from_event_proto(event):
  json_metadata = json.loads(event.log_message.message)
  core_metadata = collections.namedtuple("CoreMetadata", [
      "global_step", "session_run_count", "executor_step_count", "input_names",
      "output_names", "target_nodes"
  ])
  return core_metadata(json_metadata["global_step"],
                       json_metadata["session_run_count"],
                       json_metadata["executor_step_count"],
                       json_metadata["input_names"],
                       json_metadata["output_names"],
                       json_metadata["target_nodes"])


class DebugTensorDatum(object):
  """A single tensor dumped by TensorFlow Debugger (tfdbg).

  Contains metadata about the dumped tensor, including `timestamp`,
  `node_name`, `output_slot`, `debug_op`, and path to the dump file
  (`file_path`).

  This type does not hold the generally space-expensive tensor value (numpy
  array). Instead, it points to the file from which the tensor value can be
  loaded (with the `get_tensor` method) if needed.
  """

  def __init__(self, dump_root, debug_dump_rel_path):
    """`DebugTensorDatum` constructor.

    Args:
      dump_root: (`str`) Debug dump root directory.
      debug_dump_rel_path: (`str`) Path to a debug dump file, relative to the
          `dump_root`. For example, suppose the debug dump root
          directory is `/tmp/tfdbg_1` and the dump file is at
          `/tmp/tfdbg_1/ns_1/node_a_0_DebugIdentity_123456789`, then
          the value of the debug_dump_rel_path should be
          `ns_1/node_a_0_DebugIdenity_1234456789`.

    Raises:
      ValueError: If the base file name of the dump file does not conform to
        the dump file naming pattern:
        `node_name`_`output_slot`_`debug_op`_`timestamp`
    """

    base = os.path.basename(debug_dump_rel_path)

    if base.count("_") < 3:
      raise ValueError(
          "Dump file path does not conform to the naming pattern: %s" % base)

    # TODO(cais): Add hostname and pid to support dumps from distributed
    #             sessions.

    self._extended_timestamp = base.split("_")[-1]
    # It may include an index suffix at the end if file path collision happened
    # due to identical timestamps.
    if "-" in self._extended_timestamp:
      self._timestamp = int(
          self._extended_timestamp[:self._extended_timestamp.find("-")])
    else:
      self._timestamp = int(self._extended_timestamp)

    self._debug_op = base.split("_")[-2]
    self._output_slot = int(base.split("_")[-3])

    namespace = os.path.dirname(debug_dump_rel_path).replace("\\", "/")
    node_base_name = "_".join(base.split("_")[:-3])
    if not namespace or namespace == ".":
      self._node_name = node_base_name
    else:
      self._node_name = namespace + "/" + node_base_name

    self._file_path = os.path.join(dump_root, debug_dump_rel_path)
    self._dump_size_bytes = (gfile.Stat(self._file_path).length if
                             gfile.Exists(self._file_path) else None)

    self._run_fetches_info = None
    self._run_feed_keys_info = None

  def __str__(self):
    return "{DebugTensorDatum: %s:%d @ %s @ %d}" % (self.node_name,
                                                    self.output_slot,
                                                    self.debug_op,
                                                    self.timestamp)

  def __repr__(self):
    return self.__str__()

  def get_tensor(self):
    """Get tensor from the dump (`Event`) file.

    Returns:
      The tensor loaded from the dump (`Event`) file.
    """

    return load_tensor_from_event_file(self.file_path)

  # TODO(cais): Add time unit suffix to timestamp and t0 (us).
  @property
  def timestamp(self):
    """Timestamp of when this tensor value was dumped.

    Returns:
      (`int`) The timestamp in microseconds.
    """

    return self._timestamp

  @property
  def extended_timestamp(self):
    """Extended timestamp, possibly with an index suffix.

    The index suffix, e.g., "-1", is for disambiguating multiple dumps of the
    same tensor with the same timestamp, which can occur if the dumping events
    are spaced by shorter than the temporal resolution of the timestamps.

    Returns:
      (`str`) The extended timestamp.
    """

    return self._extended_timestamp

  @property
  def debug_op(self):
    """Name of the debug op.

    Returns:
      (`str`) debug op name (e.g., `DebugIdentity`).
    """

    return self._debug_op

  @property
  def node_name(self):
    """Name of the node from which the tensor value was dumped.

    Returns:
      (`str`) name of the node watched by the debug op.
    """

    return self._node_name

  @property
  def output_slot(self):
    """Output slot index from which the tensor value was dumped.

    Returns:
      (`int`) output slot index watched by the debug op.
    """

    return self._output_slot

  @property
  def tensor_name(self):
    """Name of the tensor watched by the debug op.

    Returns:
      (`str`) `Tensor` name, in the form of `node_name`:`output_slot`
    """

    return _get_tensor_name(self.node_name, self.output_slot)

  @property
  def watch_key(self):
    """Watch key identities a debug watch on a tensor.

    Returns:
      (`str`) A watch key, in the form of `tensor_name`:`debug_op`.
    """

    return _get_tensor_watch_key(self.node_name, self.output_slot,
                                 self.debug_op)

  @property
  def file_path(self):
    """Path to the file which stores the value of the dumped tensor."""

    return self._file_path

  @property
  def dump_size_bytes(self):
    """Size of the dump file.

    Unit: byte.

    Returns:
      If the dump file exists, size of the dump file, in bytes.
      If the dump file does not exist, None.
    """

    return self._dump_size_bytes


class DebugDumpDir(object):
  """Data set from a debug-dump directory on filesystem.

  An instance of `DebugDumpDir` contains all `DebugTensorDatum` instances
  in a tfdbg dump root directory.
  """

  def __init__(self, dump_root, partition_graphs=None, validate=True):
    """`DebugDumpDir` constructor.

    Args:
      dump_root: (`str`) path to the dump root directory.
      partition_graphs: A repeated field of GraphDefs representing the
          partition graphs executed by the TensorFlow runtime.
      validate: (`bool`) whether the dump files are to be validated against the
          partition graphs.

    Raises:
      IOError: If dump_root does not exist as a directory.
    """

    if not gfile.IsDirectory(dump_root):
      raise IOError("Dump root directory %s does not exist" % dump_root)

    self._core_metadata = None
    self._load_dumps(dump_root)
    self._create_tensor_watch_maps()
    self._load_partition_graphs(partition_graphs, validate)

    self._python_graph = None

  def _load_dumps(self, dump_root):
    """Load `DebugTensorDatum` instances from the dump root.

    Populates a list of `DebugTensorDatum` instance and sorts the list by
    ascending timestamp.

    This sorting order reflects the order in which the TensorFlow executor
    processed the nodes of the graph. It is (one of many possible) topological
    sort of the nodes. This is useful for displaying tensors in the debugger
    frontend as well as for the use case in which the user wants to find a
    "culprit tensor", i.e., the first tensor in the graph that exhibits certain
    problematic properties, i.e., all zero values, or bad numerical values such
    as nan and inf.

    In addition, creates a map from node name to debug watches. In this Map,
    the key is the watched node name; the value is a dictionary.
    Of this dictionary, the key is the watched_output_slot.

    This method attempts to load the debug watches from the tensor dump files
    first, before loading the full set of debug watches from the partition
    graphs as done later. This is necessary because sometimes the partition
    graphs may not be available, e.g., when the run errors out.

    Args:
      dump_root: (`str`) Dump root directory.
    """

    self._dump_root = dump_root
    self._dump_tensor_data = []
    self._dump_graph_file_paths = []

    self._debug_watches = collections.defaultdict(
        lambda: collections.defaultdict(set))

    for root, _, files in gfile.Walk(self._dump_root):
      for f in files:
        if f.startswith(METADATA_FILE_PREFIX):
          if _is_core_metadata_file(f):
            self._load_core_metadata(os.path.join(self._dump_root, root, f))

          if _is_graph_file(f):
            self._dump_graph_file_paths.append(
                os.path.join(self._dump_root, root, f))

          if _is_run_fetches_info_file(f):
            self._run_fetches_info = _load_log_message_from_event_file(
                os.path.join(root, f))

          if _is_run_feed_keys_info_file(f):
            self._run_feed_keys_info = _load_log_message_from_event_file(
                os.path.join(root, f))

          continue

        datum = self._dump_file_name_to_datum(root, f)
        self._dump_tensor_data.append(datum)

        self._debug_watches[datum.node_name][datum.output_slot].add(
            datum.debug_op)

    self._dump_tensor_data = sorted(
        self._dump_tensor_data, key=lambda x: x.extended_timestamp)

    if self._dump_tensor_data:
      self._t0 = self._dump_tensor_data[0].timestamp
    else:
      self._t0 = None

  def _load_core_metadata(self, event_file_path):
    event = event_pb2.Event()
    with gfile.Open(event_file_path, "rb") as f:
      event.ParseFromString(f.read())
      self._core_metadata = extract_core_metadata_from_event_proto(event)

  def _dump_file_name_to_datum(self, dir_name, file_name):
    """Obtain a DebugTensorDatum from the directory and file name.

    Args:
      dir_name: (`str`) Name of the directory in which the dump file resides.
      file_name: (`str`) Base name of the dump file.

    Returns:
      (`DebugTensorDatum`) The `DebugTensorDatum` loaded from the dump file.
    """

    # Calculate the relative path of the dump file with respect to the root.
    debug_dump_rel_path = os.path.join(
        os.path.relpath(dir_name, self._dump_root), file_name)

    return DebugTensorDatum(self._dump_root, debug_dump_rel_path)

  def _create_tensor_watch_maps(self):
    """Create maps from tensor watch keys to datum and to timestamps.

    Create a map from watch key (tensor name + debug op) to `DebugTensorDatum`
    item. Also make a map from watch key to relative timestamp.
    "relative" means (absolute timestamp - t0).
    """

    self._watch_key_to_datum = {}
    self._watch_key_to_rel_time = {}
    self._watch_key_to_dump_size_bytes = {}
    for datum in self._dump_tensor_data:
      if datum.watch_key not in self._watch_key_to_datum:
        self._watch_key_to_datum[datum.watch_key] = [datum]
        self._watch_key_to_rel_time[datum.watch_key] = [
            datum.timestamp - self._t0
        ]
        self._watch_key_to_dump_size_bytes[datum.watch_key] = [
            datum.dump_size_bytes
        ]
      else:
        self._watch_key_to_datum[datum.watch_key].append(datum)
        self._watch_key_to_rel_time[datum.watch_key].append(datum.timestamp -
                                                            self._t0)
        self._watch_key_to_dump_size_bytes[datum.watch_key].append(
            datum.dump_size_bytes)

  def set_python_graph(self, python_graph):
    """Provide Python `Graph` object to the wrapper.

    Unlike the partition graphs, which are protobuf `GraphDef` objects, `Graph`
    is a Python object and carries additional information such as the traceback
    of the construction of the nodes in the graph.

    Args:
      python_graph: (ops.Graph) The Python Graph object.
    """

    self._python_graph = python_graph
    self._node_traceback = {}
    if self._python_graph:
      for op in self._python_graph.get_operations():
        self._node_traceback[op.name] = op.traceback

  @property
  def python_graph(self):
    """Get the Python graph.

    Returns:
      If the Python graph has been set, returns a `tf.Graph` object. Otherwise,
      returns None.
    """

    return self._python_graph

  @property
  def core_metadata(self):
    """Metadata about the `Session.run()` call from the core runtime.

    Of the three counters available in the return value, `global_step` is
    supplied by the caller of the debugged `Session.run()`, while
    `session_run_count` and `executor_step_count` are determined by the state
    of the core runtime, automatically. For the same fetch list, feed keys and
    debug tensor watch options, the same executor will be used and
    `executor_step_count` should increase by one at a time. However, runs with
    different fetch lists, feed keys and debug_tensor watch options that all
    share the same `Session` object can lead to gaps in `session_run_count`.

    Returns:
      If core metadata are loaded, a `namedtuple` with the fields:
        `global_step`: A global step count supplied by the caller of
          `Session.run()`. It is optional to the caller. If the caller did not
          supply this parameter, its value will be -1.
        `session_run_count`: A counter for Run() calls to the underlying
          TensorFlow `Session` object.
        `executor_step_count`: A counter for invocations of a given runtime
          executor. The same executor is re-used for the same fetched tensors,
          target nodes, input feed keys and debug tensor watch options.
        `input_names`: Names of the input (feed) Tensors.
        `output_names`: Names of the output (fetched) Tensors.
        `target_nodes`: Names of the target nodes.
      If the core metadata have not been loaded, `None`.
    """

    return self._core_metadata

  @property
  def dumped_tensor_data(self):
    return self._dump_tensor_data

  @property
  def t0(self):
    """Absolute timestamp of the first dumped tensor.

    Returns:
      (`int`) absolute timestamp of the first dumped tensor, in microseconds.
    """

    return self._t0

  @property
  def size(self):
    """Total number of dumped tensors in the dump root directory.

    Returns:
      (`int`) total number of dumped tensors in the dump root directory.
    """

    return len(self._dump_tensor_data)

  def _load_partition_graphs(self, partition_graphs, validate):
    """Load and process partition graphs.

    Load the graphs; parse the input and control input structure; obtain the
    device and op type of each node; remove the Copy and debug ops inserted
    by the debugger. The gathered information can be used to validate the
    tensor dumps.

    Args:
      partition_graphs: Partition graphs executed by the TensorFlow runtime,
        represented as repeated fields of GraphDef.
        If no partition_graph is available, use None.
      validate: (`bool`) Whether the dump files are to be validated against the
        partition graphs.
    """

    if partition_graphs:
      self._partition_graphs = partition_graphs
    elif self._dump_graph_file_paths:
      # In case partition graphs are not available from arguments, load them
      # from the dump directory.
      self._partition_graphs = [
          _load_graph_def_from_event_file(dump_file_path)
          for dump_file_path in self._dump_graph_file_paths
      ]
    else:
      self._partition_graphs = None
      return

    self._node_attributes = {}

    self._node_inputs = {}
    self._node_ctrl_inputs = {}

    self._node_recipients = {}
    self._node_ctrl_recipients = {}

    self._devices = []
    self._node_devices = {}
    self._node_op_types = {}

    self._copy_send_nodes = []

    for pg in self._partition_graphs:
      for node in pg.node:
        self._process_partition_graph_node(node)

    self._prune_non_control_edges_of_debug_ops()
    self._prune_control_edges_of_debug_ops()

    self._populate_recipient_maps()

    if validate:
      self._validate_dump_with_graphs()

  def _process_partition_graph_node(self, node):
    """Process a node from the partition graphs.

    Args:
      node: (NodeDef) A partition-graph node to be processed.

    Raises:
      ValueError: If duplicate node names are encountered.
    """

    if _is_debug_node(node.name):
      # This is a debug node. Parse the node name and retrieve the
      # information about debug watches on tensors. But do not include
      # the node in the graph.
      (watched_node_name, watched_output_slot, _,
       debug_op) = _parse_debug_node_name(node.name)

      self._debug_watches[watched_node_name][watched_output_slot].add(
          debug_op)

      return

    if node.name in self._node_inputs:
      raise ValueError("Duplicate node name: '%s'" % node.name)

    self._node_attributes[node.name] = node.attr

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
        self._copy_send_nodes.append(node.name)

      if inp.startswith("^"):
        cinp = inp[1:]
        self._node_ctrl_inputs[node.name].append(cinp)
      else:
        self._node_inputs[node.name].append(inp)

  def _prune_nodes_from_input_and_recipient_maps(self, nodes_to_prune):
    """Prune nodes out of input and recipient maps.

    Args:
      nodes_to_prune: (`list` of `str`) Names of the nodes to be pruned.
    """

    for node in nodes_to_prune:
      del self._node_inputs[node]
      del self._node_ctrl_inputs[node]
      del self._node_recipients[node]
      del self._node_ctrl_recipients[node]

  def _prune_non_control_edges_of_debug_ops(self):
    """Prune (non-control) edges related to debug ops.

    Prune the Copy ops and associated _Send ops inserted by the debugger out
    from the non-control inputs and output recipients map. Replace the inputs
    and recipients with original ones.
    """

    copy_nodes = []
    for node in self._node_inputs:
      if node in self._copy_send_nodes:
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

    self._prune_nodes_from_input_and_recipient_maps(copy_nodes)
    self._prune_nodes_from_input_and_recipient_maps(self._copy_send_nodes)

  def _prune_control_edges_of_debug_ops(self):
    """Prune control edges related to the debug ops."""

    for node in self._node_ctrl_inputs:
      ctrl_inputs = self._node_ctrl_inputs[node]
      debug_op_inputs = []
      for ctrl_inp in ctrl_inputs:
        if _is_debug_node(ctrl_inp):
          debug_op_inputs.append(ctrl_inp)
      for debug_op_inp in debug_op_inputs:
        ctrl_inputs.remove(debug_op_inp)

  def _populate_recipient_maps(self):
    """Populate the map from node name to recipient(s) of its output(s)."""

    for node in self._node_inputs:
      inputs = self._node_inputs[node]
      for inp in inputs:
        inp = get_node_name(inp)
        if inp not in self._node_recipients:
          self._node_recipients[inp] = []
        self._node_recipients[inp].append(node)

    for node in self._node_ctrl_inputs:
      ctrl_inputs = self._node_ctrl_inputs[node]
      for ctrl_inp in ctrl_inputs:
        if ctrl_inp in self._copy_send_nodes:
          continue

        if ctrl_inp not in self._node_ctrl_recipients:
          self._node_ctrl_recipients[ctrl_inp] = []
        self._node_ctrl_recipients[ctrl_inp].append(node)

  def _validate_dump_with_graphs(self):
    """Validate the dumped tensor data against the partition graphs.

    Only the watched nodes are validated by this method, because tfdbg allows
    clients to watch only a subset of the nodes.

    Raises:
      LookupError: If the partition graphs have not been loaded yet.
      ValueError: If dumps contain node names not found in partition graph.
        Or if the temporal order of the dump's timestamps violate the
        input relations on the partition graphs.
    """

    if not self._partition_graphs:
      raise LookupError("No partition graphs loaded.")

    # Verify that the node names in the dump data are all present in the
    # partition graphs.
    for datum in self._dump_tensor_data:
      if datum.node_name not in self._node_inputs:
        raise ValueError("Node name '%s' is not found in partition graphs." %
                         datum.node_name)

    pending_inputs = {}
    for node in self._node_inputs:
      pending_inputs[node] = []
      inputs = self._node_inputs[node]
      for inp in inputs:
        inp_node = get_node_name(inp)
        inp_output_slot = get_output_slot(inp)
        if (inp_node in self._debug_watches and
            inp_output_slot in self._debug_watches[inp_node] and
            (inp_node, inp_output_slot) not in pending_inputs[node]):
          pending_inputs[node].append((inp_node, inp_output_slot))

    for i, datum in enumerate(self._dump_tensor_data):
      node = datum.node_name
      slot = datum.output_slot
      # In some cases (e.g., system clocks with insufficient precision),
      # the upstream and downstream tensors may have identical timestamps, the
      # following check examines this possibilty and avoids raising an error if
      # that is the case.
      if not self._satisfied_at_timestamp(
          pending_inputs[node], datum.timestamp, start_i=i + 1):
        raise ValueError("Causality violated in timing relations of debug "
                         "dumps: %s (%d): "
                         "these input(s) are not satisfied: %s" %
                         (node, datum.timestamp, repr(pending_inputs[node])))

      recipients = self._node_recipients[node]
      for recipient in recipients:
        recipient_pending_inputs = pending_inputs[recipient]
        if (node, slot) in recipient_pending_inputs:
          if self.node_op_type(recipient) == "Merge":
            # If this is a Merge op, we automatically clear the list because
            # a Merge node only requires one of its two inputs.
            del recipient_pending_inputs[:]
          else:
            del recipient_pending_inputs[
                recipient_pending_inputs.index((node, slot))]

  def _satisfied_at_timestamp(self, pending, timestamp, start_i=0):
    """Determine whether pending inputs are satisfied at given timestamp.

    Note: This method mutates the input argument "pending".

    Args:
      pending: A list of 2-tuple (node_name, output_slot): the dependencies to
        check.
      timestamp: (int) the timestamp in question.
      start_i: (int) the index in self._dump_tensor_data to start searching for
        the timestamp.

    Returns:
      (bool) Whether all the dependencies in pending are satisfied at the
        timestamp. If pending is empty to begin with, return True.
    """
    if not pending:
      return True

    for datum in self._dump_tensor_data[start_i:]:
      if datum.timestamp > timestamp:
        break
      if (datum.timestamp == timestamp and
          (datum.node_name, datum.output_slot) in pending):
        pending.remove((datum.node_name, datum.output_slot))
        if not pending:
          return True

    return not pending

  def loaded_partition_graphs(self):
    """Test whether partition graphs have been loaded."""
    return self._partition_graphs is not None

  def partition_graphs(self):
    """Get the partition graphs.

    Returns:
      Partition graphs as repeated fields of GraphDef.

    Raises:
      LookupError: If no partition graphs have been loaded.
    """

    if self._partition_graphs is None:
      raise LookupError("No partition graphs have been loaded.")

    return self._partition_graphs

  @property
  def run_fetches_info(self):
    """Get a str representation of the fetches used in the Session.run() call.

    Returns:
      If the information is available, a `str` obtained from `repr(fetches)`.
      If the information is not available, `None`.
    """

    return self._run_fetches_info

  @property
  def run_feed_keys_info(self):
    """Get a str representation of the feed_dict used in the Session.run() call.

    Returns:
      If the information is available, a `str` obtained from `repr(feed_dict)`.
      If the information is not available, `None`.
    """

    return self._run_feed_keys_info

  def nodes(self):
    """Get a list of all nodes from the partition graphs.

    Returns:
      All nodes' names, as a list of str.

    Raises:
      LookupError: If no partition graphs have been loaded.
    """

    if self._partition_graphs is None:
      raise LookupError("No partition graphs have been loaded.")

    return [node_name for node_name in self._node_inputs]

  def node_attributes(self, node_name):
    """Get the attributes of a node.

    Args:
      node_name: Name of the node in question.

    Returns:
      Attributes of the node.

    Raises:
      LookupError: If no partition graphs have been loaded.
      ValueError: If no node named node_name exists.
    """

    if self._partition_graphs is None:
      raise LookupError("No partition graphs have been loaded.")

    if node_name in self._node_attributes:
      return self._node_attributes[node_name]
    else:
      raise ValueError("No node named \"%s\" exists." % node_name)

  def node_inputs(self, node_name, is_control=False):
    """Get the inputs of given node according to partition graphs.

    Args:
      node_name: Name of the node.
      is_control: (`bool`) Whether control inputs, rather than non-control
        inputs, are to be returned.

    Returns:
      (`list` of `str`) inputs to the node, as a list of node names.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """

    if self._partition_graphs is None:
      raise LookupError(
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
      (`list` of `str`) all transitive inputs to the node, as a list of node
        names.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """

    if self._partition_graphs is None:
      raise LookupError(
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
      node = get_node_name(node)

      # Stop the tracing at a Merge op, as it is generally impossible to infer
      # outside the runtime which input to the Merge op is alive.
      if self._node_op_types[node] == "Merge":
        return

      if node in visited_nodes:
        return
      visited_nodes.append(node)

      for inp in self._node_inputs[node]:
        if inp == node_name:
          continue
        inputs.append(inp)
        trace_inputs(inp)

      if include_control:
        for ctrl_inp in self._node_ctrl_inputs[node]:
          if ctrl_inp == node_name:
            continue
          inputs.append(ctrl_inp)
          trace_inputs(ctrl_inp)

    trace_inputs(node_name)

    return inputs

  def node_recipients(self, node_name, is_control=False):
    """Get recipient of the given node's output according to partition graphs.

    Args:
      node_name: (`str`) name of the node.
      is_control: (`bool`) whether control outputs, rather than non-control
        outputs, are to be returned.

    Returns:
      (`list` of `str`) all inputs to the node, as a list of node names.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """

    if self._partition_graphs is None:
      raise LookupError(
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
      (`list` of `str`) names of the devices.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
    """

    if self._partition_graphs is None:
      raise LookupError("Devices are not loaded from partition graphs yet.")

    return self._devices

  def node_exists(self, node_name):
    """Test if a node exists in the partition graphs.

    Args:
      node_name: (`str`) name of the node to be checked.

    Returns:
      A boolean indicating whether the node exists.

    Raises:
      LookupError: If no partition graphs have been loaded yet.
    """

    if self._node_inputs is None:
      raise LookupError(
          "Nodes have not been loaded from partition graphs yet.")

    return node_name in self._node_inputs

  def node_device(self, node_name):
    """Get the device of a node.

    Args:
      node_name: (`str`) name of the node.

    Returns:
      (`str`) name of the device on which the node is placed.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """

    if self._partition_graphs is None:
      raise LookupError(
          "Node devices are not loaded from partition graphs yet.")

    if node_name not in self._node_devices:
      raise ValueError("Node '%s' does not exist in partition graphs." %
                       node_name)

    return self._node_devices[node_name]

  def node_op_type(self, node_name):
    """Get the op type of given node.

    Args:
      node_name: (`str`) name of the node.

    Returns:
      (`str`) op type of the node.

    Raises:
      LookupError: If node op types have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """

    if self._partition_graphs is None:
      raise LookupError(
          "Node op types are not loaded from partition graphs yet.")

    if node_name not in self._node_op_types:
      raise ValueError("Node '%s' does not exist in partition graphs." %
                       node_name)

    return self._node_op_types[node_name]

  def debug_watch_keys(self, node_name):
    """Get all tensor watch keys of given node according to partition graphs.

    Args:
      node_name: (`str`) name of the node.

    Returns:
      (`list` of `str`) all debug tensor watch keys. Returns an empty list if
        the node name does not correspond to any debug watch keys.

    Raises:
      `LookupError`: If debug watch information has not been loaded from
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
    """Get all `DebugTensorDatum` instances corresponding to a debug watch key.

    Args:
      debug_watch_key: (`str`) debug watch key.

    Returns:
      A list of `DebugTensorDatum` instances that correspond to the debug watch
      key. If the watch key does not exist, returns an empty list.

    Raises:
      ValueError: If the debug watch key does not exist.
    """

    return self._watch_key_to_datum.get(debug_watch_key, [])

  def find(self, predicate, first_n=0):
    """Find dumped tensor data by a certain predicate.

    Args:
      predicate: A callable that takes two input arguments:

        ```python
        def predicate(debug_tensor_datum, tensor):
          # returns a bool
        ```

        where `debug_tensor_datum` is an instance of `DebugTensorDatum`, which
        carries the metadata, such as the `Tensor`'s node name, output slot
        timestamp, debug op name, etc.; and `tensor` is the dumped tensor value
        as a `numpy.ndarray`.
      first_n: (`int`) return only the first n `DebugTensotDatum` instances (in
        time order) for which the predicate returns True. To return all the
        `DebugTensotDatum` instances, let first_n be <= 0.

    Returns:
      A list of all `DebugTensorDatum` objects in this `DebugDumpDir` object
       for which predicate returns True, sorted in ascending order of the
       timestamp.
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
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.

    Returns:
      List of file path(s) loaded. This is a list because each debugged tensor
        may be dumped multiple times.

    Raises:
      ValueError: If the tensor does not exist in the debug-dump data.
    """

    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum:
      raise ValueError("Watch key \"%s\" does not exist in the debug dump" %
                       watch_key)

    return [datum.file_path for datum in self._watch_key_to_datum[watch_key]]

  def get_tensors(self, node_name, output_slot, debug_op):
    """Get the tensor value from for a debug-dumped tensor.

    The tensor may be dumped multiple times in the dump root directory, so a
    list of tensors (`numpy.ndarray`) is returned.

    Args:
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.

    Returns:
      List of tensors (`numpy.ndarray`) loaded from the debug-dump file(s).

    Raises:
      ValueError: If the tensor does not exist in the debug-dump data.
    """

    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum:
      raise ValueError("Watch key \"%s\" does not exist in the debug dump" %
                       watch_key)

    return [datum.get_tensor() for datum in self._watch_key_to_datum[watch_key]]

  def get_rel_timestamps(self, node_name, output_slot, debug_op):
    """Get the relative timestamp from for a debug-dumped tensor.

    Relative timestamp means (absolute timestamp - `t0`), where `t0` is the
    absolute timestamp of the first dumped tensor in the dump root. The tensor
    may be dumped multiple times in the dump root directory, so a list of
    relative timestamps (`numpy.ndarray`) is returned.

    Args:
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.

    Returns:
      (`list` of `int`) list of relative timestamps.

    Raises:
      ValueError: If the tensor watch key does not exist in the debug dump data.
    """

    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum:
      raise ValueError("Watch key \"%s\" does not exist in the debug dump" %
                       watch_key)

    return self._watch_key_to_rel_time[watch_key]

  def get_dump_sizes_bytes(self, node_name, output_slot, debug_op):
    """Get the sizes of the dump files for a debug-dumped tensor.

    Unit of the file size: byte.

    Args:
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.

    Returns:
      (`list` of `int`): list of dump file sizes in bytes.

    Raises:
      ValueError: If the tensor watch key does not exist in the debug dump data.
    """

    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum:
      raise ValueError("Watch key \"%s\" does not exist in the debug dump" %
                       watch_key)

    return self._watch_key_to_dump_size_bytes[watch_key]

  def node_traceback(self, element_name):
    """Try to retrieve the Python traceback of node's construction.

    Args:
      element_name: (`str`) Name of a graph element (node or tensor).

    Returns:
      (list) The traceback list object as returned by the `extract_trace`
        method of Python's traceback module.

    Raises:
      LookupError: If Python graph is not available for traceback lookup.
      KeyError: If the node cannot be found in the Python graph loaded.
    """

    if self._python_graph is None:
      raise LookupError("Python graph is not available for traceback lookup")

    node_name = get_node_name(element_name)
    if node_name not in self._node_traceback:
      raise KeyError("Cannot find node \"%s\" in Python graph" % node_name)

    return self._node_traceback[node_name]
