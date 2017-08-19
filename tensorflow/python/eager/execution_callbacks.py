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

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.platform import tf_logging as logging


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
                     op_name,
                     attrs,
                     inputs,
                     outputs,
                     check_inf=True,
                     check_nan=True,
                     action="raise"):
  """An execution callback that checks for `inf`s and `nan`s in output tensors.

  This callback can be used with `tfe.add_execute_callback` to check for invalid
  numeric values. E.g.,
  ```python
  tfe.add_execute_callback(tfe.inf_nan_callback)
  ```

  Args:
    op_type: Name of the TFE operation type (e.g., `MatMul`).
    op_name: Name of the TFE operation. This name is set by client and can be
      `None` if it unset.
    attrs: Attributes of the TFE operation, as a tuple of alternating attribute
      names and attribute values.
    inputs: The `list` of input tensors to the operation, currently unused by
      this callback.
    outputs: The `list` of output tensors from the operation, checked by this
      callback for `inf` and `nan` values.
    check_inf: (`bool`) Whether this callback should check for `inf` values in
      the output tensor values.
    check_nan: (`bool`) Whether this callback should check for `nan` values in
      the output tensor values.
    action: (`str`) Action to be taken by the callback when `inf` or `nan`
      values are detected. Possible values {"raise", "log", "print"}
      `"raise"`: Raise a `InfOrNanError`.
      `"log"`: Log a warning using `tf.logging.warn`.
      `"print"`: Print a message to `sys.stdout`.

  Raises:
    InfOrNanError: iff `inf` or `nan` values are seen in any of `outputs` and
      `action` is `"raise"`.
    ValueError: iff the value of `action` is invalid.
  """
  del attrs, inputs  # Not used.

  ctx = context.get_default_context()

  for index, output in enumerate(outputs):
    if not output.dtype.is_numpy_compatible:
      continue

    numpy_dtype = output.dtype.as_numpy_dtype
    if (np.issubdtype(numpy_dtype, np.float) or
        np.issubdtype(numpy_dtype, np.complex) or
        np.issubdtype(numpy_dtype, np.integer)):
      try:
        check_numerics_op_attrs = (
            "message", "Eager-mode inf/nan check",
            "T", outputs[0].dtype.as_datatype_enum)
        # TODO(cais): Consider moving this into execute.py.
        # pylint: disable=protected-access
        pywrap_tensorflow.TFE_Py_Execute(
            ctx._handle, output.device, "CheckNumerics", [output._handle],
            check_numerics_op_attrs, 1)
        # pylint: enable=protected-access
      except core._NotOkStatusException:  # pylint: disable=protected-access
        value = output.numpy()
        inf_detected = np.any(np.isinf(value)) and check_inf
        nan_detected = np.any(np.isnan(value)) and check_nan
        if not inf_detected and not nan_detected:
          continue

        error = InfOrNanError(op_type, op_name, index, len(outputs), value)
        if action == "print":
          print("Warning: %s" % str(error))
        elif action == "log":
          logging.warn(str(error))
        elif action == "raise":
          raise error
        else:
          raise ValueError(
              "Invalid action for inf_nan_callback: %s. Valid actions are: "
              "{print | log | raise}" % action)


def inf_callback(op_type, op_name, attrs, inputs, outputs, action="raise"):
  """A specialization of `inf_nan_callback` that checks for `inf`s only."""
  inf_nan_callback(
      op_type, op_name, attrs, inputs, outputs, check_inf=True, check_nan=False,
      action=action)


def nan_callback(op_type, op_name, attrs, inputs, outputs, action="raise"):
  """A specialization of `inf_nan_callback` that checks for `nan`s only."""
  inf_nan_callback(
      op_type, op_name, attrs, inputs, outputs, check_inf=False, check_nan=True,
      action=action)


# TODO(cais): (b/64674139) Provide an alias, perhaps called seterr(), for
# add_execute_callback(inf_nan_hook).
