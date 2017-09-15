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
"""Gradient tape utilites."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import threading

from tensorflow.python.util import tf_contextlib


def tid(tensor):
  return tensor._id  # pylint: disable=protected-access


class TapeEntry(
    collections.namedtuple("TapeEntry", [
        "output_ids", "inputs", "side_outputs", "backward_function"
    ])):
  """Entry in the gradient tape.

  Represents the execution of one op or function, with instructions for doing
  its backward pass and useful information for it.

  Args:
   output_ids: tensor_id(t) for each output tensor T
   inputs: input tensors
   side_outputs: optional tensors which need to be provided to the backward
    function.
   backward_function: function to be called with the downstream gradients and
    side outputs as arguments which computes the backward pass.
  """


def _tensor_shape(t):
  return t._shape_tuple()  # pylint: disable=protected-access


class Tape(object):
  """Represents a gradient propagation trace."""

  def __init__(self):
    # _tensor_tape maps from tensor IDs to their operation IDs
    self._tensor_tape = {}
    # maps output tensor IDs to their shapes and dtypes
    self._shape_dtype = {}
    # maps from operation ID to TapeEntry
    self._op_tape = {}
    # next operation ID
    self._next_op_id = 0
    # List of directly watched tensors
    self._watched = []
    # Set of directly watched variables
    self._watched_variables = set()

  def should_record(self, tensors):
    """Returns true if any tensor should be recorded.

    Args:
      tensors: some tensors.

    Returns:
      True if any of the tensors is in the tape.
    """
    return any(x._id in self._tensor_tape for x in tensors)  # pylint: disable=protected-access

  def watch(self, tensor):
    """Adds a tensor to the tape."""
    if tid(tensor) not in self._tensor_tape:
      self._tensor_tape[tid(tensor)] = None
      self._watched.append(tensor)

  def watch_variable(self, v):
    self._watched_variables.add(v)
    self.watch(v.handle)

  def record_operation(self, output_tensors, input_tensors, side_outputs,
                       backward_function):
    """Records an operation in the tape."""
    if not self.should_record(input_tensors):
      return output_tensors
    for t in output_tensors:
      self._tensor_tape[tid(t)] = self._next_op_id
      self._shape_dtype[tid(t)] = (_tensor_shape(t), t.dtype)

    self._op_tape[self._next_op_id] = TapeEntry(
        [tid(t) for t in output_tensors],
        input_tensors,
        side_outputs,
        backward_function)
    self._next_op_id += 1

  def delete_trace(self, tensor):
    """Deletes any trace we have for this tensor."""
    if tid(tensor) in self._tensor_tape:
      op = self._tensor_tape[tid(tensor)]
      del self._tensor_tape[tid(tensor)]
      if op in self._op_tape:
        if not any(
            x in self._tensor_tape for x in self._op_tape[op].output_ids):
          del self._op_tape[op]

  def export(self):
    """Exports the internal state of this tape.

    Returns:
      tensor_tape: a map from tensor_id(tensor) to <identifier for op>
       responsible for generating that tensor.
      op_tape: a map from <identifier for op> to TapeEntry for that op.
      output_to_shape_dtype: a map from tensor_id(tensor) to its shape and
        dtype, for tensors which are outputs
    """
    return self._tensor_tape, self._op_tape, self._shape_dtype


class _TapeStack(threading.local):

  def __init__(self):
    super(_TapeStack, self).__init__()
    self._stack = []

  @property
  def stack(self):
    return self._stack

  @tf_contextlib.contextmanager
  def replace_stack(self, new_stack):
    old = self._stack
    self._stack = new_stack
    yield
    self._stack = old


# The global tape stack.
_tape_stack = _TapeStack()


def push_new_tape():
  """Pushes a new tape onto the tape stack."""
  _tape_stack.stack.append(Tape())


def watch(tensor):
  """Marks this tensor to be watched by all tapes in the stack.

  Args:
    tensor: tensor to be watched.

  Returns:
    The tensor, potentially wrapped by all tapes in the stack.
  """
  for t in _tape_stack.stack:
    t.watch(tensor)


def watch_variable(variable):
  """Marks this variable to be watched by all tapes in the stack.

  Args:
    variable: variable to be watched.

  Returns:
    The tensor, potentially wrapped by all tapes in the stack.
  """
  for t in _tape_stack.stack:
    t.watch_variable(variable)


def pop_tape():
  """Pops the top tape in the stack, if any."""
  if _tape_stack.stack:
    return _tape_stack.stack.pop()
  return None


def should_record(tensors):
  """Returns true if any tape in the stach watches any of these tensors."""
  if not _tape_stack.stack:
    return False
  return any(x.should_record(tensors) for x in _tape_stack.stack)


def record_operation(output_tensors, input_tensors, side_outputs,
                     backward_function):
  """Records the operation on all tapes in the stack."""
  for t in _tape_stack.stack:
    t.record_operation(output_tensors,
                       input_tensors,
                       side_outputs,
                       backward_function)


def delete_trace(tensor):
  """Deletes traces for this Tensor from all tapes in the stack."""
  for t in _tape_stack.stack:
    t.delete_trace(tensor)


def top_tape_watched_tensors():
  t = _tape_stack.stack[-1]
  return t._watched  # pylint: disable=protected-access


def top_tape_watched_variables():
  t = _tape_stack.stack[-1]
  return t._watched_variables  # pylint: disable=protected-access
