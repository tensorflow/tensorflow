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
"""Execution Callbacks for Eager Mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import enum  # pylint: disable=g-bad-import-order

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import execute
from tensorflow.python.platform import tf_logging as logging


class ExecutionCallback(enum.Enum):
  """Valid callback actions.

  These can be passed to `seterr` or `errstate` to create callbacks when
  specific events occur (e.g. an operation produces `NaN`s).

  IGNORE: take no action.
  PRINT:  print a warning to `stdout`.
  RAISE:  raise an error (e.g. `InfOrNanError`).
  WARN:   print a warning using `tf.logging.warn`.
  """

  IGNORE = "ignore"
  PRINT = "print"
  RAISE = "raise"
  WARN = "warn"

_DEFAULT_CALLBACK_ACTION = ExecutionCallback.RAISE


# TODO(cais): Consider moving this exception class to errors_impl.py.
class InfOrNanError(Exception):
  """Exception for inf and/or nan being present in tensor."""

  def __init__(self,
               op_type,
               op_name,
               output_index,
               num_outputs,
               value):
    """Constructor of InfOrNanError.

    Args:
      op_type: Type name of the op that generated the tensor that generated the
        `inf`(s) or `nan`(s) (e.g., `Div`).
      op_name: Name of the op that generated the tensor with `inf`(s) or
        `nan`(s). This name is set by client and can be `None` if it is unset.
      output_index: The 0-based output index of the tensor that contains
        `inf`(s) or `nan`(s).
      num_outputs: Total number of outputs of the operation.
      value: The tensor value that contains `inf`(s) or `nan`(s).
    """
    self._op_type = op_type
    self._op_name = op_name
    self._output_index = output_index
    self._num_outputs = num_outputs
    self._value = value

    self._total_count = np.size(value)
    self._inf_count = np.count_nonzero(np.isinf(value))
    self._nan_count = np.count_nonzero(np.isnan(value))

    super(InfOrNanError, self).__init__(self._get_error_message())

  def _get_error_message(self):
    """Get the error message describing this InfOrNanError object."""
    name_str = (("'%s'" % self._op_name) if self._op_name is not None
                else str(self._op_name))
    msg = "Output %d of %d of TFE operation %s (name: %s) contains " % (
        self._output_index + 1, self._num_outputs, self._op_type, name_str)
    if self._inf_count and self._nan_count:
      msg += "%d inf(s) and %d nan(s) " % (self._inf_count, self._nan_count)
    elif self._inf_count:
      msg += "%d inf(s) " % self._inf_count
    else:
      msg += "%d nan(s) " % self._nan_count
    msg += "out of a total of %d element(s). Tensor value: %s" % (
        self._total_count, self._value)
    return msg

  @property
  def op_type(self):
    return self._op_type

  @property
  def op_name(self):
    return self._op_name

  @property
  def output_index(self):
    return self._output_index

  @property
  def num_outputs(self):
    return self._num_outputs

  @property
  def value(self):
    return self._value


def inf_nan_callback(op_type,
                     inputs,
                     attrs,
                     outputs,
                     op_name,
                     check_inf=True,
                     check_nan=True,
                     action=_DEFAULT_CALLBACK_ACTION):
  """An execution callback that checks for `inf`s and `nan`s in output tensors.

  This callback can be used with `tfe.add_execute_callback` to check for invalid
  numeric values. E.g.,
  ```python
  tfe.add_execute_callback(tfe.inf_nan_callback)
  ```

  Args:
    op_type: Name of the TFE operation type (e.g., `MatMul`).
    inputs: The `list` of input tensors to the operation, currently unused by
      this callback.
    attrs: Attributes of the TFE operation, as a tuple of alternating attribute
      names and attribute values.
    outputs: The `list` of output tensors from the operation, checked by this
      callback for `inf` and `nan` values.
    op_name: Name of the TFE operation. This name is set by client and can be
      `None` if it unset.
    check_inf: (`bool`) Whether this callback should check for `inf` values in
      the output tensor values.
    check_nan: (`bool`) Whether this callback should check for `nan` values in
      the output tensor values.
    action: (`ExecutionCallback`) Action to be taken by the callback when
      `inf` or `nan` values are detected.

  Raises:
    InfOrNanError: iff `inf` or `nan` values are seen in any of `outputs` and
      `action` is `"raise"`.
    ValueError: iff the value of `action` is invalid.
  """
  del attrs, inputs  # Not used.

  action = ExecutionCallback(action)
  ctx = context.context()

  for index, output in enumerate(outputs):
    if not output.dtype.is_numpy_compatible:
      continue

    numpy_dtype = output.dtype.as_numpy_dtype
    if (np.issubdtype(numpy_dtype, np.floating) or
        np.issubdtype(numpy_dtype, np.complex) or
        np.issubdtype(numpy_dtype, np.integer)):
      try:
        check_numerics_op_attrs = (
            "message", "Eager-mode inf/nan check",
            "T", outputs[0].dtype.as_datatype_enum)
        # TODO(cais): Consider moving this into execute.py.
        # pylint: disable=protected-access
        pywrap_tensorflow.TFE_Py_Execute(
            ctx._handle, output.device, "CheckNumerics", [output],
            check_numerics_op_attrs, 1)
        # pylint: enable=protected-access
      except core._NotOkStatusException:  # pylint: disable=protected-access
        value = output.numpy()
        inf_detected = np.any(np.isinf(value)) and check_inf
        nan_detected = np.any(np.isnan(value)) and check_nan
        if not inf_detected and not nan_detected:
          continue

        error = InfOrNanError(op_type, op_name, index, len(outputs), value)
        if action == ExecutionCallback.PRINT:
          print("Warning: %s" % str(error))
        elif action == ExecutionCallback.WARN:
          logging.warn(str(error))
        elif action == ExecutionCallback.RAISE:
          raise error
        else:
          raise ValueError(
              "Invalid action for inf_nan_callback: %s. Valid actions are: "
              "{PRINT | WARN | RAISE}" % action)


def inf_callback(op_type,
                 inputs,
                 attrs,
                 outputs,
                 op_name,
                 action=_DEFAULT_CALLBACK_ACTION):
  """A specialization of `inf_nan_callback` that checks for `inf`s only."""
  inf_nan_callback(
      op_type,
      inputs,
      attrs,
      outputs,
      op_name,
      check_inf=True,
      check_nan=False,
      action=action)


def nan_callback(op_type,
                 inputs,
                 attrs,
                 outputs,
                 op_name,
                 action=_DEFAULT_CALLBACK_ACTION):
  """A specialization of `inf_nan_callback` that checks for `nan`s only."""
  inf_nan_callback(
      op_type,
      inputs,
      attrs,
      outputs,
      op_name,
      check_inf=False,
      check_nan=True,
      action=action)


def add_execution_callback(callback):
  """Add an execution callback to the default eager context.

  An execution callback is invoked immediately after an eager operation or
  function has finished execution, providing access to the op's type, name
  input and output tensors. Multiple execution callbacks can be added, in
  which case the callbacks will be invoked in the order in which they are
  added. To clear all execution callbacks that have been added, use
  `clear_execution_callbacks()`.

  Example:
  ```python
  def print_even_callback(op_type, op_name, attrs, inputs, outputs):
    # A callback that prints only the even output values.
    if outputs[0].numpy() % 2 == 0:
      print("Even output from %s: %s" % (op_name or op_type,  outputs))
  tfe.add_execution_callback(print_even_callback)

  x = tf.pow(2.0, 3.0) - 3.0
  y = tf.multiply(x, tf.add(1.0, 5.0))
  # When the line above is run, you will see all intermediate outputs that are
  # even numbers printed to the console.

  tfe.clear_execution_callbacks()
  ```

  Args:
    callback: a callable of the signature
      `f(op_type, op_name, attrs, inputs, outputs)`.
      `op_type` is the type of the operation that was just executed (e.g.,
        `MatMul`).
      `op_name` is the name of the operation that was just executed. This
        name is set by the client who created the operation and can be `None` if
        it is unset.
      `attrs` contains the attributes of the operation as a `tuple` of
        alternating attribute name and attribute value.
      `inputs` is the `list` of input `Tensor`(s) to the op.
      `outputs` is the `list` of output `Tensor`(s) from the op.
       Return value(s) from the callback are ignored.
  """
  execute.execute = execute.execute_with_callbacks
  context.context().add_post_execution_callback(callback)


def clear_execution_callbacks():
  """Clear all execution callbacks from the default eager context."""
  context.context().clear_post_execution_callbacks()


def seterr(inf_or_nan=None):
  """Set how abnormal conditions are handled by the default eager context.

  Example:
  ```python
  tfe.seterr(inf_or_nan=ExecutionCallback.RAISE)
  a = tf.constant(10.0)
  b = tf.constant(0.0)
  try:
    c = a / b  # <-- Raises InfOrNanError.
  except Exception as e:
    print("Caught Exception: %s" % e)

  tfe.seterr(inf_or_nan=ExecutionCallback.IGNORE)
  c = a / b  # <-- Does NOT raise exception anymore.
  ```

  Args:
    inf_or_nan: An `ExecutionCallback` determining the action for infinity
      (`inf`) and NaN (`nan`) values. A value of `None` leads to no change in
      the action of the condition.

  Returns:
    A dictionary of old actions.

  Raises:
    ValueError: If the value of any keyword arguments is invalid.
  """
  inf_or_nan = ExecutionCallback(inf_or_nan) if inf_or_nan is not None else None
  old_settings = {"inf_or_nan": ExecutionCallback.IGNORE}
  default_context = context.context()

  carryover_callbacks = []
  for callback in default_context.post_execution_callbacks:
    # Check whether the callback is inf_nan_callback or a partial object of
    # inf_nan_callback.
    if (callback == inf_nan_callback or
        isinstance(callback, functools.partial) and
        callback.func == inf_nan_callback):
      if callback == inf_nan_callback:
        old_settings["inf_or_nan"] = _DEFAULT_CALLBACK_ACTION
      else:
        old_settings["inf_or_nan"] = callback.keywords.get(
            "action", _DEFAULT_CALLBACK_ACTION)
    elif inf_or_nan is not None:
      carryover_callbacks.append(callback)

  if inf_or_nan is not None:
    default_context.clear_post_execution_callbacks()
    for callback in carryover_callbacks:
      default_context.add_post_execution_callback(callback)
    if inf_or_nan != ExecutionCallback.IGNORE:
      default_context.add_post_execution_callback(
          functools.partial(inf_nan_callback, action=inf_or_nan))

  return old_settings


@contextlib.contextmanager
def errstate(inf_or_nan=None):
  """Context manager setting error state.

  Example:
  ```
  c = tf.log(0.)  # -inf

  with errstate(inf_or_nan=ExecutionCallback.RAISE):
    tf.log(0.)  # <-- Raises InfOrNanError.
  ```

  Args:
    inf_or_nan: An `ExecutionCallback` determining the action for infinity
      (`inf`) and NaN (`nan`) values. A value of `None` leads to no change in
      the action of the condition.

  Yields:
    None.

  Raises:
    ValueError: If the value of any keyword arguments is invalid.
  """
  if not context.executing_eagerly():
    yield
  else:
    old_settings = seterr(inf_or_nan=inf_or_nan)
    yield
    seterr(**old_settings)
