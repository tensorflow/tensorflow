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
import contextlib
import threading

from tensorflow.python.util import tf_contextlib


def tid(tensor):
  return tensor._id  # pylint: disable=protected-access


class TapeEntry(
    collections.namedtuple("TapeEntry", [
        "op_type",
        "output_ids", "input_ids", "side_outputs", "backward_function",
        "output_shape_and_dtype",
    ])):
  """Entry in the gradient tape.

  Represents the execution of one op or function, with instructions for doing
  its backward pass and useful information for it.

  Args:
   output_ids: tensor_id(t) for each output tensor T
   input_ids: tensor_id(t) for each input tensor T
   side_outputs: optional tensors (not IDs) which need to be provided to the
    backward function.
   backward_function: function to be called with the downstream gradients and
    side outputs as arguments which computes the backward pass.
   output_shape_and_dtype: a list of (shape_tuple, dtype) for every output
    tensor_id
  """


def _tensor_shape(t):
  return t._shape_tuple()  # pylint: disable=protected-access


class Tape(object):
  """Represents a gradient propagation trace."""

  def __init__(self):
    # _tensor_tape maps from tensor IDs to their operation IDs
    self._tensor_tape = {}
    # maps from tensor ID to usage count. Triggers garbage collection when this
    # goes to zero.
    self._tensor_usage = {}
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
    i = tid(tensor)
    if i not in self._tensor_tape:
      self._tensor_tape[i] = None
      self._tensor_usage[i] = 1
      self._watched.append(tensor)

  def watch_variable(self, v):
    self._watched_variables.add(v)
    self.watch(v.handle)

  def record_operation(self, op_type, output_tensors, input_tensors,
                       side_outputs, backward_function):
    """Records an operation in the tape."""
    if not self.should_record(input_tensors):
      return output_tensors
    for t in output_tensors:
      i = tid(t)
      self._tensor_tape[i] = self._next_op_id
      self._tensor_usage[i] = 1
    for t in input_tensors:
      i = tid(t)
      self._tensor_usage[i] = self._tensor_usage.get(i, 0) + 1
    self._op_tape[self._next_op_id] = TapeEntry(
        op_type,
        [tid(t) for t in output_tensors],
        [tid(t) for t in input_tensors],
        side_outputs,
        backward_function,
        [(_tensor_shape(t), t.dtype) for t in output_tensors])
    self._next_op_id += 1

  def _delete_tensor_id(self, i):
    if i in self._tensor_usage:
      self._tensor_usage[i] -= 1
      if self._tensor_usage[i] == 0:
        del self._tensor_usage[i]
        op_id = self._tensor_tape.pop(i, None)
        if op_id is None:
          return
        op = self._op_tape[op_id]
        if not any(tensor_id in self._tensor_usage
                   for tensor_id in op.output_ids):
          del self._op_tape[op_id]
          for tensor_id in op.input_ids:
            # TODO(apassos) this recursion might come to bite us. Consider
            # adding an explicit stack if this ever gets out of hand
            self._delete_tensor_id(tensor_id)

  def delete_trace(self, tensor_id):
    """Deletes any trace we have for this tensor."""
    self._delete_tensor_id(tensor_id)

  def export(self):
    """Exports the internal state of this tape.

    Returns:
      tensor_tape: a map from tensor_id(tensor) to <identifier for op>
       responsible for generating that tensor.
      op_tape: a map from <identifier for op> to TapeEntry for that op.
    """
    return self._tensor_tape, self._op_tape


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


@contextlib.contextmanager
def stop_recording():
  old = _tape_stack.stack
  _tape_stack._stack = []  # pylint: disable=protected-access
  try:
    yield
  finally:
    _tape_stack._stack = old  # pylint: disable=protected-access


def should_record(tensors):
  """Returns true if any tape in the stach watches any of these tensors."""
  if not _tape_stack.stack:
    return False
  return any(x.should_record(tensors) for x in _tape_stack.stack)


def record_operation(op_type, output_tensors, input_tensors, side_outputs,
                     backward_function):
  """Records the operation on all tapes in the stack."""
  for t in _tape_stack.stack:
    t.record_operation(op_type, output_tensors,
                       input_tensors,
                       side_outputs,
                       backward_function)


def delete_trace(tensor_id):
  """Deletes traces for this Tensor from all tapes in the stack."""
  for t in _tape_stack.stack:
    t.delete_trace(tensor_id)


def top_tape_watched_tensors():
  t = _tape_stack.stack[-1]
  return t._watched  # pylint: disable=protected-access


def top_tape_watched_variables():
  t = _tape_stack.stack[-1]
  return t._watched_variables  # pylint: disable=protected-access


def could_possibly_record():
  """Returns True if any tape is active."""
  return len(_tape_stack.stack) > 0  # pylint: disable=g-explicit-length-test
