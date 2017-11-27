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

import contextlib

from tensorflow.python import pywrap_tensorflow


class Tape(object):
  """Represents a gradient propagation trace."""

  def __init__(self, tape):
    self._tape = tape

  def watched_variables(self):
    return pywrap_tensorflow.TFE_Py_TapeWatchedVariables(self._tape)


def push_new_tape():
  """Pushes a new tape onto the tape stack."""
  pywrap_tensorflow.TFE_Py_TapeStackPushNew()


def watch(tensor):
  """Marks this tensor to be watched by all tapes in the stack.

  Args:
    tensor: tensor to be watched.
  """
  pywrap_tensorflow.TFE_Py_TapeStackWatch(tensor)


def watch_variable(variable):
  """Marks this variable to be watched by all tapes in the stack.

  Args:
    variable: variable to be watched.
  """
  pywrap_tensorflow.TFE_Py_TapeStackWatchVariable(variable)


def pop_tape():
  """Pops the top tape in the stack, if any."""
  return Tape(pywrap_tensorflow.TFE_Py_TapeStackPop())


@contextlib.contextmanager
def stop_recording():
  stack = []
  while not pywrap_tensorflow.TFE_Py_TapeStackIsEmpty():
    stack.append(pop_tape()._tape)  # pylint: disable=protected-access
  try:
    yield
  finally:
    for tape in reversed(stack):
      pywrap_tensorflow.TFE_Py_TapeStackPush(tape)


def should_record(tensors):
  """Returns true if any tape in the stack watches any of these tensors."""
  return pywrap_tensorflow.TFE_Py_TapeStackShouldRecord(tensors)


def record_operation(op_type, output_tensors, input_tensors, backward_function):
  """Records the operation on all tapes in the stack."""
  pywrap_tensorflow.TFE_Py_TapeStackRecordOperation(
      op_type, output_tensors, input_tensors, backward_function)


def delete_trace(tensor_id):
  """Deletes traces for this Tensor from all tapes in the stack."""
  pywrap_tensorflow.TFE_Py_TapeStackDeleteTrace(tensor_id)


def could_possibly_record():
  """Returns True if any tape is active."""
  return not pywrap_tensorflow.TFE_Py_TapeStackIsEmpty()
