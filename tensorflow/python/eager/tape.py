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
"""Gradient tape utilities."""

import contextlib

from tensorflow.python import pywrap_tfe


class Tape(object):
  """Represents a gradient propagation trace."""

  __slots__ = ["_tape"]

  def __init__(self, tape):
    self._tape = tape

  def watched_variables(self):
    return pywrap_tfe.TFE_Py_TapeWatchedVariables(self._tape)


def push_new_tape(persistent=False, watch_accessed_variables=True):
  """Pushes a new tape onto the tape stack."""
  tape = pywrap_tfe.TFE_Py_TapeSetNew(persistent, watch_accessed_variables)
  return Tape(tape)


def push_tape(tape):
  """Pushes an existing tape onto the tape stack."""
  pywrap_tfe.TFE_Py_TapeSetAdd(tape._tape)  # pylint: disable=protected-access


def watch(tape, tensor):
  """Marks this tensor to be watched by the given tape."""
  pywrap_tfe.TFE_Py_TapeWatch(tape._tape, tensor)  # pylint: disable=protected-access


class VariableWatcher(object):
  """A scope that tracks all trainable variable accesses within it.

  This explicitly ignores variables that are not marked as trainable.

  Sample usage:

  var = tf.Variable(0.0)
  with VariableWatcher() as variable_watcher:
    var.assign_add(1.0)

  assert variable_watcher.watched_variables == [var]
  """

  __slots__ = ["_variable_watcher"]

  def __init__(self):
    self._variable_watcher = None

  def __enter__(self):
    self._variable_watcher = pywrap_tfe.TFE_Py_VariableWatcherNew()
    return self

  def __exit__(self, typ, value, traceback):
    pywrap_tfe.TFE_Py_VariableWatcherRemove(self._variable_watcher)

  def watched_variables(self):
    """Returns a tuple of variables accessed under this scope."""
    return pywrap_tfe.TFE_Py_VariableWatcherWatchedVariables(
        self._variable_watcher)


def default_get_variables(variable):
  return [variable]

# Gets a list of changed variables. Can be overriden using
# register_variables_override. An example of overriding is for getting the
# varibles within a distributed context.
_variables_override = default_get_variables


def register_watched_variable_resolver(resolver):
  """Registers the resolver to be used to get the list of variables to watch.

  Args:
    resolver: callable, takes a Variable and returns a list of Variables that
      shall be watched.
  """
  global _variables_override
  assert _variables_override is default_get_variables
  _variables_override = resolver


def watch_variable(tape, variable):
  """Marks this variable to be watched by the given tape."""
  variables = _variables_override(variable)
  for var in variables:
    pywrap_tfe.TFE_Py_TapeWatchVariable(tape._tape, var)  # pylint: disable=protected-access
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def variable_accessed(variable):
  """Notifies all tapes in the stack that a variable has been accessed.

  Args:
    variable: variable to be watched.
  """
  variables = _variables_override(variable)
  for var in variables:
    pywrap_tfe.TFE_Py_TapeVariableAccessed(var)
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def variables_accessed(variables):
  """Notifies all tapes in the stack that variables have been accessed.

  Only trainable variables are marked as accessed.

  Args:
    variables: iterable of variables to mark as accessed.
  """
  accessed = []
  for variable in variables:
    if variable.trainable:
      accessed.extend(_variables_override(variable))

  for var in accessed:
    pywrap_tfe.TFE_Py_TapeVariableAccessed(var)
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def pop_tape(tape):
  """Pops the given tape in the stack."""
  pywrap_tfe.TFE_Py_TapeSetRemove(tape._tape)  # pylint: disable=protected-access


@contextlib.contextmanager
def stop_recording():
  """Stop all gradient recording (backprop and forwardprop)."""
  is_stopped = pywrap_tfe.TFE_Py_TapeSetIsStopped()
  try:
    if not is_stopped:
      pywrap_tfe.TFE_Py_TapeSetStopOnThread()
    yield
  finally:
    if not is_stopped:
      pywrap_tfe.TFE_Py_TapeSetRestartOnThread()


def should_record_backprop(tensors):
  """Returns true if any tape in the stack watches any of these tensors.

  Only takes GradientTapes into account, not forward accumulators.

  Args:
    tensors: Tensors to check, typically inputs to an operation.

  Returns:
    Boolean, whether any tape watches any of `tensors`.
  """
  return pywrap_tfe.TFE_Py_TapeSetShouldRecordBackprop(tensors)


def record_operation(op_type, output_tensors, input_tensors, backward_function,
                     forward_function=None):
  """Records the operation on all tapes in the stack."""
  pywrap_tfe.TFE_Py_TapeSetRecordOperation(op_type, output_tensors,
                                           input_tensors, backward_function,
                                           forward_function)


def record_operation_backprop_only(op_type, output_tensors, input_tensors,
                                   backward_function):
  """Records the operation on all backward tapes in the stack."""
  pywrap_tfe.TFE_Py_TapeSetRecordOperationBackprop(op_type, output_tensors,
                                                   input_tensors,
                                                   backward_function)


def record_operation_forwardprop_only(op_type, output_tensors, input_tensors,
                                      backward_function,
                                      forwardprop_output_indices):
  """Records the operation on all forward accumulators in the stack.

  Args:
    op_type: a string for the operation type, used in the backprop code
    output_tensors: a list of Python Tensor objects output by the operation
    input_tensors: a list of input Tensors to the recorded operation
    backward_function: the function to be called to, given the gradients of the
      output tensors, produce the gradients of the input tensors. This function
      is automatically transposed to produce output gradients given input
      gradients.
    forwardprop_output_indices: indicates any output_tensors which contain JVPs.
      Typically these will have come from TFE_Py_PackForwardGradients. May be
      None or an empty sequence if there are no JVP outputs from the operation.
  """
  pywrap_tfe.TFE_Py_TapeSetRecordOperationForwardprop(
      op_type, output_tensors, input_tensors, backward_function,
      forwardprop_output_indices)


def delete_trace(tensor_id):
  """Deletes traces for this Tensor from all tapes in the stack."""
  pywrap_tfe.TFE_Py_TapeSetDeleteTrace(tensor_id)


def could_possibly_record():
  """Returns True if any tape is active."""
  return not pywrap_tfe.TFE_Py_TapeSetIsEmpty()
