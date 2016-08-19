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

import os

from tensorflow.core.util import event_pb2
from tensorflow.python.framework import tensor_util


def load_tensor_from_event_file(event_file_path):
  """Load a tensor from an event file.

  Assumes that the event file contains a Event protobuf and the Event protobuf
  contains a tensor.

  Args:
    event_file_path: Path to the event file.

  Returns:
    The tensor value loaded from the event file.
  """
  event = event_pb2.Event()
  with open(event_file_path, "rb") as f:
    event.ParseFromString(f.read())
    tensor_value = tensor_util.MakeNdarray(event.summary.value[0].tensor)

  return tensor_value


def get_tensor_name(node_name, output_slot):
  """Get tensor name given node name and output slot index.

  Args:
    node_name: Name of the node that outputs the tensor, as a string.
    output_slot: Output slot index of the tensor, as an integer.

  Returns:
    Name of the tensor, as a string.
  """
  return "%s:%d" % (node_name, output_slot)


def get_tensor_watch_key(node_name, output_slot, debug_op):
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
  return "%s:%s" % (get_tensor_name(node_name, output_slot), debug_op)


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

    node_base_name = "_".join(base.split("_")[:-3])
    self._node_name = os.path.dirname(
        debug_dump_rel_path) + "/" + node_base_name

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
    return get_tensor_name(self.node_name, self.output_slot)

  @property
  def watch_key(self):
    """Watch key identities a debug watch on a tensor.

    Returns:
      A watch key, in the form of <tensor_name>:<debug_op>.
    """
    return get_tensor_watch_key(self.node_name, self.output_slot, self.debug_op)

  @property
  def file_path(self):
    return self._file_path


class DebugDumpDir(object):
  """Data set from a debug dump directory on filesystem.

  An instance of DebugDumpDir contains all DebugTensorDatum in a tfdbg dump
  root directory.
  """

  def __init__(self, dump_root):
    """DebugDumpDir constructor.

    Args:
      dump_root: Path to the dump root directory.

    Raises:
      IOError: If dump_root does not exist as a directory.
      ValueError: If the dump_root directory contains file path patterns
         that do not conform to the canonical dump file naming pattern.
    """

    if not os.path.isdir(dump_root):
      raise IOError("Dump root directory %s does not exist" % dump_root)

    self._dump_root = dump_root
    self._dump_tensor_data = []

    for root, _, files in os.walk(self._dump_root):
      for f in files:
        if f.count("_") < 3:
          raise ValueError(
              "Dump file path does not conform to the naming pattern: %s" % f)

        debug_dump_rel_path = os.path.join(
            os.path.relpath(root, self._dump_root), f)
        self._dump_tensor_data.append(
            DebugTensorDatum(self._dump_root, debug_dump_rel_path))

        # Sort the data by ascending timestamp.
        # This sorting order reflects the order in which the TensorFlow
        # executor processed the nodes of the graph. It is (one of many
        # possible) topological sort of the nodes. This is useful for
        # displaying tensors in the debugger frontend as well as for the use
        # case in which the user wants to find a "culprit tensor", i.e., the
        # first tensor in the graph that exhibits certain problematic
        # properties, i.e., all zero values, or bad numerical values such as
        # nan and inf.
        # TODO(cais): Check the time-sorted order against partition executor
        #     GraphDefs from RunMetadata.
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
            self._watch_key_to_rel_time[datum.watch_key].append(datum.timestamp
                                                                - self._t0)

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

    watch_key = get_tensor_watch_key(node_name, output_slot, debug_op)
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

    watch_key = get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum:
      raise ValueError("Watch key \"%s\" does not exist in the debug dump" %
                       watch_key)

    return self._watch_key_to_rel_time[watch_key]
