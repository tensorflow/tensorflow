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
"""Experimental API for TensorFlow's "Eager" mode of execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors

# Trace of execution and memory usage.
_active_trace = None


def _status_to_exception(code, message):
  try:
    error_class = errors.exception_type_from_error_code(code)
    return error_class(None, None, message)
  except KeyError:
    return errors.UnknownError(None, None, message, code)


class _NotOkStatusException(Exception):
  """Exception class to handle not ok Status."""

  def __init__(self, message, code):
    super(_NotOkStatusException, self).__init__()
    self.message = message
    self.code = code

  def __str__(self):
    e = _status_to_exception(self.code, self.message)
    return "%s: %s" % (e.__class__.__name__, e)


pywrap_tensorflow.TFE_Py_RegisterExceptionClass(_NotOkStatusException)


class _FallbackException(Exception):
  """Exception class to handle fallback from the fastpath.

  The fastpath that we refer to here is the one implemented to reduce per-op
  overheads (TFE_Py_FastPathExecute_C). If the conditions for executing the op
  on the fastpath are not met, we fallback to a safer (and more complete)
  slowpath, and this Exception is raised to signal that transition.
  """
  pass


class _SymbolicException(Exception):
  """Exception class to handle use of symbolic tensors when executing eagerly.

  `keras.Input()` creates symbolic tensors (in a FuncGraph managed by the
  Keras backend) while in eager execution. This exception is used to
  identify this case (raised in `convert_to_tensor` cause generated functions
  for ops to construct graphs instead of executing the kernel).
  """
  pass


pywrap_tensorflow.TFE_Py_RegisterFallbackExceptionClass(_FallbackException)
