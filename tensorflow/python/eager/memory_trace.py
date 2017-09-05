# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utility to trace per-device memory consumption across time over execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

TraceEntry = collections.namedtuple(
    "TraceEntry", ["op_name", "tensor_id", "mem_usage", "device", "size"])
TensorData = collections.namedtuple(
    "TensorData", ["op_name", "tensor_size", "device"])


class MemoryTrace(object):
  """Records a trace of memory usage over operation execution."""

  def __init__(self, n_devices):

    self.trace = []
    self.tensor_to_data = {}
    self.current_device_mem_usage = [0] * n_devices

  def record_tensor(self, op_name, tensor_id, device, size):
    self.current_device_mem_usage[device] += size
    self.tensor_to_data[tensor_id] = TensorData(op_name, size, device)
    self.trace.append(TraceEntry(op_name,
                                 tensor_id,
                                 self.current_device_mem_usage[:],
                                 device,
                                 size))

  def delete_tensor(self, tensor_id):
    if tensor_id not in self.tensor_to_data:
      return
    data = self.tensor_to_data.pop(tensor_id)
    self.current_device_mem_usage[data.device] -= data.tensor_size
    self.trace.append(TraceEntry(data.op_name,
                                 tensor_id,
                                 self.current_device_mem_usage[:],
                                 data.device,
                                 -data.tensor_size))

  def flush_trace(self):
    """Prints the formatted trace recorded so far."""
    longest_op_name = max(len(t.op_name) for t in self.trace)
    longest_op_name = max(longest_op_name, len("op_name"))
    longest_heap_size = max(max(len(str(d)) for d in t.mem_usage)
                            for t in self.trace)
    longest_heap_size = max(longest_heap_size, len("d0"))
    longest_id_len = max(len(str(t.tensor_id)) for t in self.trace)
    longest_id_len = max(longest_id_len, 2)
    first_line = []
    first_line.append("+/-")
    first_line.append("op_name".ljust(longest_op_name))
    first_line.append("id".ljust(longest_id_len))
    for i in range(len(self.current_device_mem_usage)):
      first_line.append(("d"+str(i)).ljust(longest_heap_size))
    first_line.append("size")
    print(" | ".join(first_line))
    for t in self.trace:
      line = []
      if t.size > 0:
        line.append("+  ")
      else:
        line.append("-  ")
      line.append(t.op_name.ljust(longest_op_name))
      line.append(str(t.tensor_id).ljust(longest_id_len))
      for d in t.mem_usage:
        line.append(str(d).ljust(longest_heap_size))
      line.append(str(t.size))
      print(" | ".join(line))
    self.trace = []
    print()
