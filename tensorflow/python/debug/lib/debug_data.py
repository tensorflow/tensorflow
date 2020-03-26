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
import glob
import json
import os
import platform
import re

import numpy as np
import six

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


# TODO(cais): Tie these string constants in with C++?
METADATA_FILE_PREFIX = "_tfdbg_"
CORE_METADATA_TAG = "core_metadata_"
GRAPH_FILE_TAG = "graph_"
DEVICE_TAG = "device_"
HASH_TAG = "hash"

FETCHES_INFO_FILE_TAG = "fetches_info_"
FEED_KEYS_INFO_FILE_TAG = "feed_keys_info_"


def _glob(glob_pattern):
  if platform.system() == "Windows":
    return glob.glob(glob_pattern)
  else:
    return gfile.Glob(glob_pattern)


class InconvertibleTensorProto(object):
  """Represents a TensorProto that cannot be converted to np.ndarray."""

  def __init__(self, tensor_proto, initialized=True):
    """Constructor.

    Args:
      tensor_proto: the `TensorProto` object that cannot be represented as a
        `np.ndarray` object.
      initialized: (`bool`) whether the Tensor is initialized.
    """
    self._tensor_proto = tensor_proto
    self._initialized = initialized

  def __str__(self):
    output = "" if self._initialized else "Uninitialized tensor:\n"
    output += str(self._tensor_proto)
    return output

  @property
  def initialized(self):
    return self._initialized


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
    The tensor value loaded from the event file, as a `numpy.ndarray`, if
    representation of the tensor value by a `numpy.ndarray` is possible.
    For uninitialized Tensors, returns `None`. For Tensors of data types that
    cannot be represented as `numpy.ndarray` (e.g., `tf.resource`), return
    the `TensorProto` protobuf object without converting it to a
    `numpy.ndarray`.
  """

  tensor_proto = event.summary.value[0].tensor
  shape = tensor_util.TensorShapeProtoToList(tensor_proto.tensor_shape)
  num_elements = 1
  for shape_dim in shape:
    num_elements *= shape_dim

  if tensor_proto.tensor_content or tensor_proto.string_val or not num_elements:
    # Initialized tensor or empty tensor.
    if tensor_proto.dtype == types_pb2.DT_RESOURCE:
      tensor_value = InconvertibleTensorProto(tensor_proto)
    else:
      try:
        tensor_value = tensor_util.MakeNdarray(tensor_proto)
      except KeyError:
        tensor_value = InconvertibleTensorProto(tensor_proto)
  else:
    # Uninitialized tensor or tensor of unconvertible data type.
    tensor_value = InconvertibleTensorProto(tensor_proto, False)

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


def _is_graph_file(file_name):
  return file_name.startswith(METADATA_FILE_PREFIX + GRAPH_FILE_TAG)


def _is_run_fetches_info_file(file_name):
  return file_name == METADATA_FILE_PREFIX + FETCHES_INFO_FILE_TAG


def _is_run_feed_keys_info_file(file_name):
  return file_name == METADATA_FILE_PREFIX + FEED_KEYS_INFO_FILE_TAG


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

  if isinstance(tensor, InconvertibleTensorProto):
    # Uninitialized tensor doesn't have bad numerical values.
    # Also return False for data types that cannot be represented as numpy
    # arrays.
    return False
  elif (np.issubdtype(tensor.dtype, np.floating) or
        np.issubdtype(tensor.dtype, np.complex) or
        np.issubdtype(tensor.dtype, np.integer)):
    return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
  else:
    return False


_CoreMetadata = collections.namedtuple("CoreMetadata", [
    "global_step", "session_run_index", "executor_step_index", "input_names",
    "output_names", "target_nodes"
])


def extract_core_metadata_from_event_proto(event):
  json_metadata = json.loads(event.log_message.message)
  return _CoreMetadata(json_metadata["global_step"],
                       json_metadata["session_run_index"],
                       json_metadata["executor_step_index"],
                       json_metadata["input_names"],
                       json_metadata["output_names"],
                       json_metadata["target_nodes"])


def device_name_to_device_path(device_name):
  """Convert device name to device path."""
  device_name_items = compat.as_text(device_name).split("/")
  device_name_items = [item.replace(":", "_") for item in device_name_items]
  return METADATA_FILE_PREFIX + DEVICE_TAG + ",".join(device_name_items)


def device_path_to_device_name(device_dir):
  """Parse device name from device path.

  Args:
    device_dir: (str) a directory name for the device.

  Returns:
    (str) parsed device name.
  """
  path_items = os.path.basename(device_dir)[
      len(METADATA_FILE_PREFIX) + len(DEVICE_TAG):].split(",")
  return "/".join([
      path_item.replace("device_", "device:").replace("_", ":", 1)
      for path_item in path_items])


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
      dump_root: (`str`) Debug dump root directory. This path should not include
        the path component that represents the device name (see also below).
      debug_dump_rel_path: (`str`) Path to a debug dump file, relative to the
        `dump_root`. The first item of this relative path is assumed to be
        a path representing the name of the device that the Tensor belongs to.
        See `device_path_to_device_name` for more details on the device path.
        For example, suppose the debug dump root
        directory is `/tmp/tfdbg_1` and the dump file is at
        `/tmp/tfdbg_1/<device_path>/>ns_1/node_a_0_DebugIdentity_123456789`,
        then the value of the debug_dump_rel_path should be
        `<device_path>/ns_1/node_a_0_DebugIdentity_1234456789`.

    Raises:
      ValueError: If the base file name of the dump file does not conform to
        the dump file naming pattern:
        `node_name`_`output_slot`_`debug_op`_`timestamp`
    """

    path_components = os.path.normpath(debug_dump_rel_path).split(os.sep)
    self._device_name = device_path_to_device_name(path_components[0])
    base = path_components[-1]
    if base.count("_") < 3:
      raise ValueError(
          "Dump file path does not conform to the naming pattern: %s" % base)

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

    node_base_name = "_".join(base.split("_")[:-3])
    self._node_name = "/".join(path_components[1:-1] + [node_base_name])

    self._file_path = os.path.join(dump_root, debug_dump_rel_path)
    self._dump_size_bytes = (gfile.Stat(self._file_path).length if
                             gfile.Exists(self._file_path) else None)

  def __str__(self):
    return "{DebugTensorDatum (%s) %s:%d @ %s @ %d}" % (self.device_name,
                                                        self.node_name,
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
  def device_name(self):
    """Name of the device that the tensor belongs to.

    Returns:
      (`str`) device name.
    """

    return self._device_name

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


class WatchKeyDoesNotExistInDebugDumpDirError(ValueError):
  pass


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
      ValueError: If more than one core metadata file is found under the dump
        root directory.
    """

    if not gfile.IsDirectory(dump_root):
      raise IOError("Dump root directory %s does not exist" % dump_root)

    self._core_metadata = []

    # Find the list of devices.
    self._dump_root = dump_root

    self._load_core_metadata()
    self._load_fetches_info()
    self._load_feeds_info()
    self._load_all_device_dumps(partition_graphs, validate)

    self._python_graph = None

  def _load_all_device_dumps(self, partition_graphs, validate):
    """Load the dump data for all devices."""
    device_dirs = _glob(os.path.join(
        self._dump_root, METADATA_FILE_PREFIX + DEVICE_TAG + "*"))

    self._device_names = []
    self._t0s = {}
    self._dump_tensor_data = {}
    self._dump_graph_file_paths = {}
    self._debug_watches = {}
    self._watch_key_to_devices = {}
    self._watch_key_to_datum = {}
    self._watch_key_to_rel_time = {}
    self._watch_key_to_dump_size_bytes = {}
    for device_dir in device_dirs:
      device_name = device_path_to_device_name(device_dir)
      self._device_names.append(device_name)
      self._load_device_dumps(device_name, device_dir)
    self._load_partition_graphs(partition_graphs, validate)
    self._calculate_t0()

    for device_name in self._device_names:
      self._create_tensor_watch_maps(device_name)

  def _load_device_dumps(self, device_name, device_root):
    """Load `DebugTensorDatum` instances from the dump root of a given device.

    Populates a map {device_name: a list of `DebugTensorDatum`}, where the list
    is sorted by ascending timestamp.

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
      device_name: (`str`) name of the device.
      device_root: (`str`) dump root directory of the given device.

    Raises:
      ValueError: If GraphDef for the device is not available.
    """

    self._dump_tensor_data[device_name] = []
    self._debug_watches[device_name] = collections.defaultdict(
        lambda: collections.defaultdict(set))

    for root, _, files in gfile.Walk(device_root):
      for f in files:
        if _is_graph_file(f):
          self._dump_graph_file_paths[device_name] = os.path.join(root, f)
        else:
          datum = self._dump_file_name_to_datum(root, f)
          self._dump_tensor_data[device_name].append(datum)
          self._debug_watches[device_name][datum.node_name][
              datum.output_slot].add(datum.debug_op)

    self._dump_tensor_data[device_name] = sorted(
        self._dump_tensor_data[device_name],
        key=lambda x: x.extended_timestamp)

    if self._dump_tensor_data[device_name]:
      self._t0s[device_name] = self._dump_tensor_data[device_name][0].timestamp
    else:
      self._t0s[device_name] = None

  def _calculate_t0(self):
    """Calculate the first timestamp across all devices."""
    t0s = [t0 for t0 in six.itervalues(self._t0s) if t0 is not None]
    self._t0 = min(t0s) if t0s else None

  def _load_core_metadata(self):
    core_metadata_files = _glob(os.path.join(
        self._dump_root, METADATA_FILE_PREFIX + CORE_METADATA_TAG + "*"))
    for core_metadata_file in core_metadata_files:
      with gfile.Open(core_metadata_file, "rb") as f:
        event = event_pb2.Event()
        event.ParseFromString(f.read())
        self._core_metadata.append(
            extract_core_metadata_from_event_proto(event))

  def _load_fetches_info(self):
    fetches_info_files = _glob(os.path.join(
        self._dump_root, METADATA_FILE_PREFIX + FETCHES_INFO_FILE_TAG + "*"))
    self._run_fetches_info = []
    for fetches_info_file in fetches_info_files:
      self._run_fetches_info.append(
          _load_log_message_from_event_file(fetches_info_file))

  def _load_feeds_info(self):
    feeds_info_files = _glob(os.path.join(
        self._dump_root, METADATA_FILE_PREFIX + FEED_KEYS_INFO_FILE_TAG + "*"))
    self._run_feed_keys_info = []
    for feeds_info_file in feeds_info_files:
      self._run_feed_keys_info.append(
          _load_log_message_from_event_file(feeds_info_file))

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

  def _create_tensor_watch_maps(self, device_name):
    """Create maps from tensor watch keys to datum and to timestamps.

    Create a map from watch key (tensor name + debug op) to `DebugTensorDatum`
    item. Also make a map from watch key to relative timestamp.
    "relative" means (absolute timestamp - t0).

    Args:
      device_name: (str) name of the device.
    """

    self._watch_key_to_datum[device_name] = {}
    self._watch_key_to_rel_time[device_name] = {}
    self._watch_key_to_dump_size_bytes[device_name] = {}
    for datum in self._dump_tensor_data[device_name]:
      if datum.watch_key not in self._watch_key_to_devices:
        self._watch_key_to_devices[datum.watch_key] = {device_name}
      else:
        self._watch_key_to_devices[datum.watch_key].add(device_name)

      if datum.watch_key not in self._watch_key_to_datum[device_name]:
        self._watch_key_to_datum[device_name][datum.watch_key] = [datum]
        self._watch_key_to_rel_time[device_name][datum.watch_key] = [
            datum.timestamp - self._t0]
        self._watch_key_to_dump_size_bytes[device_name][datum.watch_key] = [
            datum.dump_size_bytes]
      else:
        self._watch_key_to_datum[device_name][datum.watch_key].append(datum)
        self._watch_key_to_rel_time[device_name][datum.watch_key].append(
            datum.timestamp - self._t0)
        self._watch_key_to_dump_size_bytes[device_name][datum.watch_key].append(
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
        self._node_traceback[op.name] = tuple(map(tuple, op.traceback))

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
    `session_run_index` and `executor_step_index` are determined by the state
    of the core runtime, automatically. For the same fetch list, feed keys and
    debug tensor watch options, the same executor will be used and
    `executor_step_index` should increase by one at a time. However, runs with
    different fetch lists, feed keys and debug_tensor watch options that all
    share the same `Session` object can lead to gaps in `session_run_index`.

    Returns:
      If core metadata are loaded, a `namedtuple` with the fields:
        `global_step`: A global step count supplied by the caller of
          `Session.run()`. It is optional to the caller. If the caller did not
          supply this parameter, its value will be -1.
        `session_run_index`: A sorted index for Run() calls to the underlying
          TensorFlow `Session` object.
        `executor_step_index`: A counter for invocations of a given runtime
          executor. The same executor is re-used for the same fetched tensors,
          target nodes, input feed keys and debug tensor watch options.
        `input_names`: Names of the input (feed) Tensors.
        `output_names`: Names of the output (fetched) Tensors.
        `target_nodes`: Names of the target nodes.
      If the core metadata have not been loaded, `None`.
      If more than one core metadata files exist, return a list of the
        `nametuple` described above.
    """

    output = self._core_metadata
    return output[0] if len(output) == 1 else output

  @property
  def dumped_tensor_data(self):
    """Retrieve dumped tensor data."""
    if len(self.devices()) == 1:
      return self._dump_tensor_data[self.devices()[0]]
    else:
      all_devices_data = six.itervalues(self._dump_tensor_data)
      data = []
      for device_data in all_devices_data:
        data.extend(device_data)
      return sorted(data, key=lambda x: x.extended_timestamp)

  @property
  def t0(self):
    """Absolute timestamp of the first dumped tensor across all devices.

    Returns:
      (`int`) absolute timestamp of the first dumped tensor, in microseconds.
    """
    return self._t0

  @property
  def size(self):
    """Total number of dumped tensors in the dump root directory.

    Returns:
      (`int`) The total number of dumped tensors in the dump root directory.
    """
    return sum(len(self._dump_tensor_data[device_name])
               for device_name in self._dump_tensor_data)

  def _load_partition_graphs(self, client_partition_graphs, validate):
    """Load and process partition graphs.

    Load the graphs; parse the input and control input structure; obtain the
    device and op type of each node; remove the Copy and debug ops inserted
    by the debugger. The gathered information can be used to validate the
    tensor dumps.

    Args:
      client_partition_graphs: A repeated field of GraphDefs representing the
        partition graphs executed by the TensorFlow runtime, from the Python
        client. These partition graphs are used only if partition graphs
        cannot be loaded from the dump directory on the file system.
      validate: (`bool`) Whether the dump files are to be validated against the
        partition graphs.

    Raises:
      ValueError: If the partition GraphDef of one or more devices fail to be
        loaded.
    """
    self._debug_graphs = {}
    self._node_devices = {}

    partition_graphs_and_device_names = []
    for device_name in self._device_names:
      partition_graph = None
      if device_name in self._dump_graph_file_paths:
        partition_graph = _load_graph_def_from_event_file(
            self._dump_graph_file_paths[device_name])
      else:
        logging.warn(
            "Failed to load partition graphs for device %s from disk. "
            "As a fallback, the client graphs will be used. This "
            "may cause mismatches in device names." % device_name)
        partition_graph = self._find_partition_graph(client_partition_graphs,
                                                     device_name)

      if partition_graph:
        partition_graphs_and_device_names.append((partition_graph,
                                                  device_name))

    for partition_graph, maybe_device_name in partition_graphs_and_device_names:
      debug_graph = debug_graphs.DebugGraph(partition_graph,
                                            device_name=maybe_device_name)
      self._debug_graphs[debug_graph.device_name] = debug_graph
      self._collect_node_devices(debug_graph)

      if validate and debug_graph.device_name in self._dump_tensor_data:
        self._validate_dump_with_graphs(debug_graph.device_name)

  def _find_partition_graph(self, partition_graphs, device_name):
    if partition_graphs is None:
      return None
    else:
      for graph_def in partition_graphs:
        for node_def in graph_def.node:
          if node_def.device == device_name:
            return graph_def
      return None

  def _collect_node_devices(self, debug_graph):
    for node_name in debug_graph.node_devices:
      if node_name in self._node_devices:
        self._node_devices[node_name] = self._node_devices[node_name].union(
            debug_graph.node_devices[node_name])
      else:
        self._node_devices[node_name] = debug_graph.node_devices[node_name]

  def _validate_dump_with_graphs(self, device_name):
    """Validate the dumped tensor data against the partition graphs.

    Only the watched nodes are validated by this method, because tfdbg allows
    clients to watch only a subset of the nodes.

    Args:
      device_name: (`str`) device name.

    Raises:
      LookupError: If the partition graphs have not been loaded yet.
      ValueError: If dumps contain node names not found in partition graph.
        Or if the temporal order of the dump's timestamps violate the
        input relations on the partition graphs.
    """
    if not self._debug_graphs:
      raise LookupError(
          "No partition graphs loaded for device %s" % device_name)
    debug_graph = self._debug_graphs[device_name]

    # Verify that the node names in the dump data are all present in the
    # partition graphs.
    for datum in self._dump_tensor_data[device_name]:
      if datum.node_name not in debug_graph.node_inputs:
        raise ValueError("Node name '%s' is not found in partition graphs of "
                         "device %s." % (datum.node_name, device_name))

    pending_inputs = {}
    for node in debug_graph.node_inputs:
      pending_inputs[node] = []
      inputs = debug_graph.node_inputs[node]
      for inp in inputs:
        inp_node = debug_graphs.get_node_name(inp)
        inp_output_slot = debug_graphs.get_output_slot(inp)
        # Inputs from Enter and NextIteration nodes are not validated because
        # DebugNodeInserter::InsertNodes() in the debugger core skips creating
        # control edges from debug ops watching these types of nodes.
        if (inp_node in self._debug_watches[device_name] and
            inp_output_slot in self._debug_watches[device_name][inp_node] and
            debug_graph.node_op_types.get(inp) not in (
                "Enter", "NextIteration") and
            (inp_node, inp_output_slot) not in pending_inputs[node]):
          pending_inputs[node].append((inp_node, inp_output_slot))

    for i, datum in enumerate(self._dump_tensor_data[device_name]):
      node = datum.node_name
      slot = datum.output_slot
      # In some cases (e.g., system clocks with insufficient precision),
      # the upstream and downstream tensors may have identical timestamps, the
      # following check examines this possibility and avoids raising an error if
      # that is the case.
      if not self._satisfied_at_timestamp(
          device_name, pending_inputs[node], datum.timestamp, start_i=i + 1):
        raise ValueError("Causality violated in timing relations of debug "
                         "dumps: %s (%d): "
                         "these input(s) are not satisfied: %s" %
                         (node, datum.timestamp, repr(pending_inputs[node])))

      recipients = debug_graph.node_recipients[node]
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

  def _satisfied_at_timestamp(self, device_name, pending, timestamp, start_i=0):
    """Determine whether pending inputs are satisfied at given timestamp.

    Note: This method mutates the input argument "pending".

    Args:
      device_name: (str) device name.
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

    for datum in self._dump_tensor_data[device_name][start_i:]:
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
    return bool(self._debug_graphs)

  def partition_graphs(self):
    """Get the partition graphs.

    Returns:
      Partition graphs as a list of GraphDef.

    Raises:
      LookupError: If no partition graphs have been loaded.
    """
    if not self._debug_graphs:
      raise LookupError("No partition graphs have been loaded.")
    return [self._debug_graphs[key].debug_graph_def
            for key in self._debug_graphs]

  def reconstructed_non_debug_partition_graphs(self):
    """Reconstruct partition graphs with the debugger-inserted ops stripped.

    The reconstructed partition graphs are identical to the original (i.e.,
    non-debugger-decorated) partition graphs except in the following respects:
      1) The exact names of the runtime-inserted internal nodes may differ.
         These include _Send, _Recv, _HostSend, _HostRecv, _Retval ops.
      2) As a consequence of 1, the nodes that receive input directly from such
         send- and recv-type ops will have different input names.
      3) The parallel_iteration attribute of while-loop Enter ops are set to 1.

    Returns:
      A dict mapping device names (`str`s) to reconstructed
      `tf.compat.v1.GraphDef`s.
    """
    non_debug_graphs = {}
    for key in self._debug_graphs:
      non_debug_graphs[key] = self._debug_graphs[key].non_debug_graph_def
    return non_debug_graphs

  @property
  def run_fetches_info(self):
    """Get a str representation of the fetches used in the Session.run() call.

    Returns:
      If the information is available from one `Session.run` call, a `str`
        obtained from `repr(fetches)`.
      If the information is available from multiple `Session.run` calls, a
        `list` of `str` from `repr(fetches)`.
      If the information is not available, `None`.
    """

    output = self._run_fetches_info
    return output[0] if len(output) == 1 else output

  @property
  def run_feed_keys_info(self):
    """Get a str representation of the feed_dict used in the Session.run() call.

    Returns:
      If the information is available from one `Session.run` call, a `str`
        obtained from `repr(feed_dict)`.
      If the information is available from multiple `Session.run` calls, a
        `list` of `str` obtained from `repr(feed_dict)`.
      If the information is not available, `None`.
    """

    output = self._run_feed_keys_info
    return output[0] if len(output) == 1 else output

  def _infer_device_name(self, device_name, node_name):
    """Infer the device name given node name.

    If device_name is provided (i.e., not None), it'll be simply returned right
    away.

    Args:
      device_name: (str or None) name of the device. If None, will try to infer
        the device name by looking at the available nodes.
      node_name: (str) name of the node.

    Returns:
      (str) Inferred name of the device, if available.

    Raises:
      ValueError: If the node name does not exist on any of the available
        devices or if there are multiple devices that contain the node with
        the given name.
    """
    if device_name is None:
      if node_name in self._node_devices:
        if len(self._node_devices[node_name]) == 1:
          return list(self._node_devices[node_name])[0]
        else:
          raise ValueError(
              "There are multiple (%d) devices with nodes named '%s' but "
              "device_name is not specified." %
              (len(self._node_devices[node_name]), node_name))
      else:
        raise ValueError("None of the %d device(s) has a node named '%s'." %
                         (len(self._device_names), node_name))
    else:
      return device_name

  def nodes(self, device_name=None):
    """Get a list of all nodes from the partition graphs.

    Args:
      device_name: (`str`) name of device. If None, all nodes from all available
        devices will be included.

    Returns:
      All nodes' names, as a list of str.

    Raises:
      LookupError: If no partition graphs have been loaded.
      ValueError: If specified node name does not exist.
    """
    if not self._debug_graphs:
      raise LookupError("No partition graphs have been loaded.")
    if device_name is None:
      nodes = []
      for device_name in self._debug_graphs:
        nodes.extend(self._debug_graphs[device_name].node_inputs.keys())
      return nodes
    else:
      if device_name not in self._debug_graphs:
        raise ValueError("Invalid device name: %s" % device_name)
      return self._debug_graphs[device_name].node_inputs.keys()

  def node_attributes(self, node_name, device_name=None):
    """Get the attributes of a node.

    Args:
      node_name: Name of the node in question.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      Attributes of the node.

    Raises:
      LookupError: If no partition graphs have been loaded.
    """
    if not self._debug_graphs:
      raise LookupError("No partition graphs have been loaded.")

    device_name = self._infer_device_name(device_name, node_name)
    return self._debug_graphs[device_name].node_attributes[node_name]

  def node_inputs(self, node_name, is_control=False, device_name=None):
    """Get the inputs of given node according to partition graphs.

    Args:
      node_name: Name of the node.
      is_control: (`bool`) Whether control inputs, rather than non-control
        inputs, are to be returned.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`list` of `str`) inputs to the node, as a list of node names.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
    """
    if not self._debug_graphs:
      raise LookupError(
          "Node inputs are not loaded from partition graphs yet.")

    device_name = self._infer_device_name(device_name, node_name)
    if is_control:
      return self._debug_graphs[device_name].node_ctrl_inputs[node_name]
    else:
      return self._debug_graphs[device_name].node_inputs[node_name]

  def transitive_inputs(self,
                        node_name,
                        include_control=True,
                        include_reversed_ref=False,
                        device_name=None,):
    """Get the transitive inputs of given node according to partition graphs.

    Args:
      node_name: Name of the node.
      include_control: Include control inputs (True by default).
      include_reversed_ref: Whether a ref input, say from A to B, is to be also
        considered as an input from B to A. The rationale is that ref inputs
        generally let the recipient (e.g., B in this case) mutate the value of
        the source (e.g., A in this case). So the reverse direction of the ref
        edge reflects the direction of information flow.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`list` of `str`) all transitive inputs to the node, as a list of node
        names.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
    """
    if not self._debug_graphs:
      raise LookupError(
          "Node inputs are not loaded from partition graphs yet.")

    device_name = self._infer_device_name(device_name, node_name)

    input_lists = [self._debug_graphs[device_name].node_inputs]
    if include_control:
      input_lists.append(self._debug_graphs[device_name].node_ctrl_inputs)
    if include_reversed_ref:
      input_lists.append(
          self._debug_graphs[device_name].node_reversed_ref_inputs)
    tracer = debug_graphs.DFSGraphTracer(
        input_lists,
        skip_node_names=self._get_merge_node_names(device_name))
    tracer.trace(node_name)
    return tracer.inputs()

  def _get_merge_node_names(self, device_name):
    """Lazily get a list of Merge nodes on a given device."""
    if device_name not in self._device_names:
      raise ValueError("Invalid device name: %s" % device_name)

    if not hasattr(self, "_merge_node_names"):
      self._merge_node_names = {}
    if device_name not in self._merge_node_names:
      debug_graph = self._debug_graphs[device_name]
      self._merge_node_names[device_name] = [
          node for node in debug_graph.node_op_types
          if debug_graph.node_op_types[node] == "Merge"]
    return self._merge_node_names[device_name]

  def find_some_path(self,
                     src_node_name,
                     dst_node_name,
                     include_control=True,
                     include_reversed_ref=False,
                     device_name=None):
    """Find a path between a source node and a destination node.

    Limitation: the source and destination are required to be on the same
    device, i.e., this method does not yet take into account Send/Recv nodes
    across devices.

    TODO(cais): Make this method work across device edges by tracing Send/Recv
      nodes.

    Args:
      src_node_name: (`str`) name of the source node or name of an output tensor
        of the node.
      dst_node_name: (`str`) name of the destination node or name of an output
        tensor of the node.
      include_control: (`bool`) whrther control edges are considered in the
        graph tracing.
      include_reversed_ref: Whether a ref input, say from A to B, is to be also
        considered as an input from B to A. The rationale is that ref inputs
        generally let the recipient (e.g., B in this case) mutate the value of
        the source (e.g., A in this case). So the reverse direction of the ref
        edge reflects the direction of information flow.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      A path from the src_node_name to dst_node_name, as a `list` of `str`, if
      it exists. The list includes src_node_name as the first item and
      dst_node_name as the last.
      If such a path does not exist, `None`.

    Raises:
      ValueError: If the source and destination nodes are not on the same
        device.
    """
    src_device_name = self._infer_device_name(device_name, src_node_name)
    dst_device_name = self._infer_device_name(device_name, dst_node_name)

    if src_device_name != dst_device_name:
      raise ValueError(
          "Source (%s) and destination (%s) are not on the same device: "
          "%s vs. %s" % (src_node_name, dst_node_name, src_device_name,
                         dst_device_name))

    input_lists = [self._debug_graphs[dst_device_name].node_inputs]
    debug_graph = self._debug_graphs[dst_device_name]
    if include_control:
      input_lists.append(debug_graph.node_ctrl_inputs)
    if include_reversed_ref:
      input_lists.append(debug_graph.node_reversed_ref_inputs)
    tracer = debug_graphs.DFSGraphTracer(
        input_lists,
        skip_node_names=self._get_merge_node_names(dst_device_name),
        destination_node_name=src_node_name)
    # Here the value of destination_node_name is src_node_name, because we
    # are tracing the graph from output to its inputs (i.e., going backwards
    # on the graph).

    try:
      tracer.trace(dst_node_name)
    except debug_graphs.GraphTracingReachedDestination:
      # Prune nodes not on the path.
      inputs = [dst_node_name] + tracer.inputs()
      depth_list = [0] + tracer.depth_list()

      path = []
      curr_depth = depth_list[-1]
      for inp, depth in zip(reversed(inputs), reversed(depth_list)):
        if depth == curr_depth:
          path.append(inp)
          curr_depth -= 1
      return path

  def node_recipients(self, node_name, is_control=False, device_name=None):
    """Get recipient of the given node's output according to partition graphs.

    Args:
      node_name: (`str`) name of the node.
      is_control: (`bool`) whether control outputs, rather than non-control
        outputs, are to be returned.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`list` of `str`) all inputs to the node, as a list of node names.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
    """

    if not self._debug_graphs:
      raise LookupError(
          "Node recipients are not loaded from partition graphs yet.")

    device_name = self._infer_device_name(device_name, node_name)
    debug_graph = self._debug_graphs[device_name]
    if is_control:
      return debug_graph.node_ctrl_recipients[node_name]
    else:
      return debug_graph.node_recipients[node_name]

  def devices(self):
    """Get the list of device names.

    Returns:
      (`list` of `str`) names of the devices.
    """
    return self._device_names

  def node_exists(self, node_name, device_name=None):
    """Test if a node exists in the partition graphs.

    Args:
      node_name: (`str`) name of the node to be checked.
      device_name: optional device name. If None, will search for the node
        on all available devices. Otherwise, search for the node only on
        the given device.

    Returns:
      A boolean indicating whether the node exists.

    Raises:
      LookupError: If no partition graphs have been loaded yet.
      ValueError: If device_name is specified but cannot be found.
    """
    if not self._debug_graphs:
      raise LookupError(
          "Nodes have not been loaded from partition graphs yet.")

    if (device_name is not None) and device_name not in self._debug_graphs:
      raise ValueError(
          "The specified device_name '%s' cannot be found." % device_name)

    for _, debug_graph in self._debug_graphs.items():
      if node_name in debug_graph.node_inputs:
        return True
    return False

  def node_device(self, node_name):
    """Get the names of the devices that has nodes of the specified name.

    Args:
      node_name: (`str`) name of the node.

    Returns:
      (`str` or `list` of `str`) name of the device(s) on which the node of the
        given name is found. Returns a `str` if there is only one such device,
        otherwise return a `list` of `str`.

    Raises:
      LookupError: If node inputs and control inputs have not been loaded
         from partition graphs yet.
      ValueError: If the node does not exist in partition graphs.
    """
    if not self._debug_graphs:
      raise LookupError(
          "Node devices are not loaded from partition graphs yet.")

    if node_name not in self._node_devices:
      raise ValueError("Node '%s' does not exist in partition graphs." %
                       node_name)

    output = list(self._node_devices[node_name])
    return output[0] if len(output) == 1 else output

  def node_op_type(self, node_name, device_name=None):
    """Get the op type of given node.

    Args:
      node_name: (`str`) name of the node.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`str`) op type of the node.

    Raises:
      LookupError: If node op types have not been loaded
         from partition graphs yet.
    """
    if not self._debug_graphs:
      raise LookupError(
          "Node op types are not loaded from partition graphs yet.")

    device_name = self._infer_device_name(device_name, node_name)
    return self._debug_graphs[device_name].node_op_types[node_name]

  def debug_watch_keys(self, node_name, device_name=None):
    """Get all tensor watch keys of given node according to partition graphs.

    Args:
      node_name: (`str`) name of the node.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`list` of `str`) all debug tensor watch keys. Returns an empty list if
        the node name does not correspond to any debug watch keys.

    Raises:
      `LookupError`: If debug watch information has not been loaded from
        partition graphs yet.
    """

    try:
      device_name = self._infer_device_name(device_name, node_name)
    except ValueError:
      return []

    if node_name not in self._debug_watches[device_name]:
      return []

    watch_keys = []
    for watched_slot in self._debug_watches[device_name][node_name]:
      debug_ops = self._debug_watches[device_name][node_name][watched_slot]
      for debug_op in debug_ops:
        watch_keys.append(
            _get_tensor_watch_key(node_name, watched_slot, debug_op))

    return watch_keys

  def watch_key_to_data(self, debug_watch_key, device_name=None):
    """Get all `DebugTensorDatum` instances corresponding to a debug watch key.

    Args:
      debug_watch_key: (`str`) debug watch key.
      device_name: (`str`) name of the device. If there is only one device or if
        the specified debug_watch_key exists on only one device, this argument
        is optional.

    Returns:
      A list of `DebugTensorDatum` instances that correspond to the debug watch
      key. If the watch key does not exist, returns an empty list.

    Raises:
      ValueError: If there are multiple devices that have the debug_watch_key,
        but device_name is not specified.
    """
    if device_name is None:
      matching_device_names = [
          name for name in self._watch_key_to_datum
          if debug_watch_key in self._watch_key_to_datum[name]]
      if not matching_device_names:
        return []
      elif len(matching_device_names) == 1:
        device_name = matching_device_names[0]
      else:
        raise ValueError(
            "The debug watch key '%s' exists on multiple (%d) devices, but "
            "device name is not specified." %
            (debug_watch_key, len(matching_device_names)))
    elif device_name not in self._debug_key_to_datum:
      raise ValueError(
          "There is no device named '%s' consisting of debug watch keys." %
          device_name)

    return self._watch_key_to_datum[device_name].get(debug_watch_key, [])

  def find(self,
           predicate,
           first_n=0,
           device_name=None,
           exclude_node_names=None):
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
      device_name: optional device name.
      exclude_node_names: Optional regular expression to exclude nodes with
        names matching the regular expression.

    Returns:
      A list of all `DebugTensorDatum` objects in this `DebugDumpDir` object
       for which predicate returns True, sorted in ascending order of the
       timestamp.
    """
    if exclude_node_names:
      exclude_node_names = re.compile(exclude_node_names)

    matched_data = []
    for device in (self._dump_tensor_data if device_name is None
                   else (self._dump_tensor_data[device_name],)):
      for datum in self._dump_tensor_data[device]:
        if exclude_node_names and exclude_node_names.match(datum.node_name):
          continue

        if predicate(datum, datum.get_tensor()):
          matched_data.append(datum)

          if first_n > 0 and len(matched_data) >= first_n:
            return matched_data

    return matched_data

  def get_tensor_file_paths(self,
                            node_name,
                            output_slot,
                            debug_op,
                            device_name=None):
    """Get the file paths from a debug-dumped tensor.

    Args:
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.
      device_name: (`str`) name of the device. If there is only one device or if
        the specified debug_watch_key exists on only one device, this argument
        is optional.

    Returns:
      List of file path(s) loaded. This is a list because each debugged tensor
        may be dumped multiple times.

    Raises:
      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor does not exist in
        the debug-dump data.
    """

    device_name = self._infer_device_name(device_name, node_name)
    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum[device_name]:
      raise WatchKeyDoesNotExistInDebugDumpDirError(
          "Watch key \"%s\" does not exist in the debug dump of device %s" %
          (watch_key, device_name))

    return [datum.file_path for datum in
            self._watch_key_to_datum[device_name][watch_key]]

  def get_tensors(self, node_name, output_slot, debug_op, device_name=None):
    """Get the tensor value from for a debug-dumped tensor.

    The tensor may be dumped multiple times in the dump root directory, so a
    list of tensors (`numpy.ndarray`) is returned.

    Args:
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.
      device_name: (`str`) name of the device. If there is only one device or if
        the specified debug_watch_key exists on only one device, this argument
        is optional.

    Returns:
      List of tensors (`numpy.ndarray`) loaded from the debug-dump file(s).

    Raises:
      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor does not exist in
        the debug-dump data.
    """

    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    try:
      device_name = self._infer_device_name(device_name, node_name)
      return [datum.get_tensor() for datum in
              self._watch_key_to_datum[device_name][watch_key]]
    except (ValueError, KeyError):
      raise WatchKeyDoesNotExistInDebugDumpDirError(
          "Watch key \"%s\" does not exist in the debug dump of device %s" %
          (watch_key, device_name))

  def get_rel_timestamps(self,
                         node_name,
                         output_slot,
                         debug_op,
                         device_name=None):
    """Get the relative timestamp from for a debug-dumped tensor.

    Relative timestamp means (absolute timestamp - `t0`), where `t0` is the
    absolute timestamp of the first dumped tensor in the dump root. The tensor
    may be dumped multiple times in the dump root directory, so a list of
    relative timestamps (`numpy.ndarray`) is returned.

    Args:
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.
      device_name: (`str`) name of the device. If there is only one device or if
        the specified debug_watch_key exists on only one device, this argument
        is optional.

    Returns:
      (`list` of `int`) list of relative timestamps.

    Raises:
      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor watch key does not
        exist in the debug dump data.
    """

    device_name = self._infer_device_name(device_name, node_name)
    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum[device_name]:
      raise WatchKeyDoesNotExistInDebugDumpDirError(
          "Watch key \"%s\" does not exist in the debug dump" % watch_key)

    # TODO(cais): Figure out whether this should be relative to the global t0.
    return self._watch_key_to_rel_time[device_name][watch_key]

  def get_dump_sizes_bytes(self,
                           node_name,
                           output_slot,
                           debug_op,
                           device_name=None):
    """Get the sizes of the dump files for a debug-dumped tensor.

    Unit of the file size: byte.

    Args:
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.
      device_name: (`str`) name of the device. If there is only one device or if
        the specified debug_watch_key exists on only one device, this argument
        is optional.

    Returns:
      (`list` of `int`): list of dump file sizes in bytes.

    Raises:
      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor watch key does not
        exist in the debug dump data.
    """

    device_name = self._infer_device_name(device_name, node_name)
    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum[device_name]:
      raise WatchKeyDoesNotExistInDebugDumpDirError(
          "Watch key \"%s\" does not exist in the debug dump of device %s" %
          (watch_key, device_name))

    return self._watch_key_to_dump_size_bytes[device_name][watch_key]

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

    node_name = debug_graphs.get_node_name(element_name)
    if node_name not in self._node_traceback:
      raise KeyError("Cannot find node \"%s\" in Python graph" % node_name)

    return self._node_traceback[node_name]
