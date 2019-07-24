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
"""Utility for batching remote OPs together to reduce RPC overhead."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


@six.add_metaclass(abc.ABCMeta)
class ScheduledOp(object):
  """Represents a scheduled remote operation."""

  @abc.abstractmethod
  def batching_key(self):
    """Returns the key for batching operations."""

  @abc.abstractmethod
  def batch_runner_fn(self):
    """Returns the function that executes the operation on the batch."""


class ScheduledStampedResourceOp(ScheduledOp):
  """Wrapper class for batched operations on stamped resources."""

  def __init__(self, resource_handle, op, **kwargs):
    self.resource_handle = resource_handle
    self.op = op
    self.args = kwargs

  def batching_key(self):
    # We want to group the same operations on the same device and run them in
    # one batch. So we use (device, operation) as the key.
    return self.resource_handle.device, self.op

  def batch_runner_fn(self):
    return _scheduled_stamp_resource_op_runner


def _move_tensors(tensors, device):
  """Moves a list of tensors to a device by concatenating/splitting them."""
  # Reset the device setting to avoid weird interactions with device merging
  # logic.
  zero = constant_op.constant(0, dtype=dtypes.int32)
  with ops.device(None):
    if all(tensor.shape.rank == 0 for tensor in tensors):
      with ops.device(tensors[0].device):
        values = array_ops.stack(tensors)
      with ops.device(device):
        return array_ops.unstack(values)
    else:
      with ops.device(tensors[0].device):
        sizes = array_ops.stack(array_ops.shape_n(tensors))[:, 0]
        values = array_ops.concat(tensors, axis=zero)
      with ops.device(device):
        sizes = array_ops.unstack(sizes)
        return list(array_ops.split(values, sizes, axis=zero))


def _scheduled_stamp_resource_op_runner(batch, stamp):
  """Runs a batch operation on a stamped resource."""
  if not batch:
    return
  arg_keys = set(batch[0].args.keys())
  grouped_args = collections.OrderedDict()
  resource_handles = []
  # Check that the set of arguments is the same across all the scheduled ops.
  for op in batch:
    if set(op.args.keys()) != arg_keys:
      raise ValueError("Mismatching arguments: %s, %s.", op.args, arg_keys)
    for key in arg_keys:
      grouped_args.setdefault(key, []).append(op.args[key])
    resource_handles.append(op.resource_handle)
  # Move all the inputs to the op device in one RPC.
  grouped_args = collections.OrderedDict(
      (k, _move_tensors(v, resource_handles[0].device))
      for k, v in sorted(grouped_args.items()))
  with ops.device(resource_handles[0].device):
    return batch[0].op(resource_handles, stamp, **grouped_args)


def run_handler_scheduled_ops(per_handler_ops, stamp, worker_device):
  """Given a dictionary of ops for each handler, runs them in batch."""
  batched_ops = collections.OrderedDict()
  # Group the ops by their batching_key. Ops that share the same batching key
  # can be executed together.
  for handler in per_handler_ops.keys():
    for op in per_handler_ops[handler]:
      key = (op.batching_key(), op.batch_runner_fn())
      batched_ops.setdefault(key, []).append(op)
  op_results = {}
  for batch in batched_ops.values():
    # Run each of the batched ops using its runner.
    results = batch[0].batch_runner_fn()(batch, stamp)
    # If the result is a tuple, move each entry in the tuple in one RPC.
    if isinstance(results, tuple):
      results = tuple(
          _move_tensors(result, worker_device) for result in results)
      # Once all the results are on the worker, create individual tuple for
      # each scheduled op request.
      for i in range(len(batch)):
        op_results[batch[i]] = tuple(result[i] for result in results)
    # If the result is a tuple, it didn't have any outputs, so use the
    # `ops.Operation` as the result for all the scheduled ops.
    elif isinstance(results, ops.Operation):
      for i in range(len(batch)):
        op_results[batch[i]] = results
    else:
      raise ValueError("Unknown type of result %s.", results)
  handler_results = collections.defaultdict(list)
  # Dispatch the results of the ScheduledOps to the handlers that requested
  # them.
  for handler in per_handler_ops.keys():
    for op in per_handler_ops[handler]:
      handler_results[handler].append(op_results[op])
  return handler_results
