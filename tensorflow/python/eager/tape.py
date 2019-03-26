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
from tensorflow.python.util.lazy_loader import LazyLoader

# There is a circular dependency between this, ops.py, and
# distribution_strategy_context.
# TODO(b/117329403): Remove this circular dependency.
distribution_strategy_context = LazyLoader(
    "distribution_strategy_context", globals(),
    "tensorflow.python.distribute."
    "distribution_strategy_context")


class Tape(object):
  """Represents a gradient propagation trace."""

  def __init__(self, tape):
    self._tape = tape

  def watched_variables(self):
    return pywrap_tensorflow.TFE_Py_TapeWatchedVariables(self._tape)


def push_new_tape(persistent=False, watch_accessed_variables=True):
  """Pushes a new tape onto the tape stack."""
  tape = pywrap_tensorflow.TFE_Py_TapeSetNew(persistent,
                                             watch_accessed_variables)
  return Tape(tape)


def push_tape(tape):
  """Pushes an existing tape onto the tape stack."""
  pywrap_tensorflow.TFE_Py_TapeSetAdd(tape._tape)  # pylint: disable=protected-access


def watch(tape, tensor):
  """Marks this tensor to be watched by the given tape."""
  pywrap_tensorflow.TFE_Py_TapeWatch(tape._tape, tensor)  # pylint: disable=protected-access


def watch_variable(tape, variable):
  """Marks this variable to be watched by the given tape."""
  strategy, context = (
      distribution_strategy_context.get_strategy_and_replica_context())
  if context:
    variables = [strategy.extended.value_container(variable)]
  else:
    variables = strategy.unwrap(variable)
  for var in variables:
    pywrap_tensorflow.TFE_Py_TapeWatchVariable(tape._tape, var)  # pylint: disable=protected-access


def variable_accessed(variable):
  """Notifies all tapes in the stack that a variable has been accessed.

  Args:
    variable: variable to be watched.
  """
  strategy, context = (
      distribution_strategy_context.get_strategy_and_replica_context())
  if context:
    variables = [strategy.extended.value_container(variable)]
  else:
    variables = strategy.unwrap(variable)
  for var in variables:
    pywrap_tensorflow.TFE_Py_TapeVariableAccessed(var)


def variables_accessed(variables):
  """Notifies all tapes in the stack that variables have been accessed.

  Only trainable variables are marked as accessed.

  Args:
    variables: iterable of variables to mark as accessed.
  """
  strategy, context = (
      distribution_strategy_context.get_strategy_and_replica_context())
  accessed = []
  if context:
    accessed = [strategy.extended.value_container(variable)
                for variable in variables if variable.trainable]
  else:
    for variable in variables:
      if variable.trainable:
        accessed.extend(strategy.unwrap(variable))

  for var in accessed:
    pywrap_tensorflow.TFE_Py_TapeVariableAccessed(var)


def pop_tape(tape):
  """Pops the top tape in the stack, if any."""
  pywrap_tensorflow.TFE_Py_TapeSetRemove(tape._tape)  # pylint: disable=protected-access


@contextlib.contextmanager
def stop_recording():
  try:
    pywrap_tensorflow.TFE_Py_TapeSetStopOnThread()
    yield
  finally:
    pywrap_tensorflow.TFE_Py_TapeSetRestartOnThread()


def should_record(tensors):
  """Returns true if any tape in the stack watches any of these tensors."""
  return pywrap_tensorflow.TFE_Py_TapeSetShouldRecord(tensors)


def record_operation(op_type, output_tensors, input_tensors, backward_function):
  """Records the operation on all tapes in the stack."""
  pywrap_tensorflow.TFE_Py_TapeSetRecordOperation(
      op_type, output_tensors, input_tensors, backward_function)


def delete_trace(tensor_id):
  """Deletes traces for this Tensor from all tapes in the stack."""
  pywrap_tensorflow.TFE_Py_TapeSetDeleteTrace(tensor_id)


def could_possibly_record():
  """Returns True if any tape is active."""
  return not pywrap_tensorflow.TFE_Py_TapeSetIsEmpty()
