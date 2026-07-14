# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Gradient record utilities."""

import contextlib

from tensorflow.python import pywrap_tfe


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


def could_possibly_record():
  """Returns True if any tape is active."""
  return not pywrap_tfe.TFE_Py_TapeSetIsEmpty()
