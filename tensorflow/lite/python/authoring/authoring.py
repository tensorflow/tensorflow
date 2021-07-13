# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Authoring tool package for TFLite compatibility.

WARNING: The package is experimental and subject to change.

This package provides a way to check TFLite compatibility at model authoring
time.

Example:
    @lite.authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ])
    def f(x):
      return tf.cosh(x)

    f(1.0)

    > CompatibilityWarning: op 'tf.Cosh' requires "Select TF Ops" for model
    conversion for TensorFlow Lite.
"""
import functools
import sys

# pylint: disable=g-direct-tensorflow-import
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics_wrapper import converter_error_data_pb2
from tensorflow.python.platform import tf_logging as logging


_CUSTOM_OPS_HDR = "Custom ops: "
_TF_OPS_HDR = "TF Select ops: "


class CompatibilityError(Exception):
  """Raised when an error occurs with TFLite compatibility."""
  pass


class _Compatible:
  """A decorator to check TFLite compatibility."""

  def __init__(self,
               target,
               raise_exception=False,
               converter_target_spec=None,
               converter_allow_custom_ops=None,
               debug=False):
    """Initialize the decorator object.

    Here is the description of the object variables.
    - _func     : decorated function.
    - _obj_func : for class object, we need to use this object to provide `self`
                  instance as 1 first argument.
    - _verified : whether the compatibility is checked or not.

    Args:
      target: decorated function.
      raise_exception : to raise an exception on compatibility issues.
          User need to use get_compatibility_log() to check details.
      converter_target_spec : target_spec of TFLite converter parameter.
      converter_allow_custom_ops : allow_custom_ops of TFLite converter
          parameter.
      debug: to dump execution details of decorated function.
    """
    functools.update_wrapper(self, target)
    self._func = target
    self._obj_func = None
    self._verified = False
    self._log_messages = []
    self._raise_exception = raise_exception
    self._converter_target_spec = converter_target_spec
    self._converter_allow_custom_ops = converter_allow_custom_ops
    self._debug = debug

  def __get__(self, instance, cls):
    """A Python descriptor interface."""
    self._obj_func = self._func.__get__(instance, cls)
    return self

  def _get_func(self):
    """Returns decorated function object.

    For a class method, use self._obj_func to provide `self` instance.
    """
    if self._obj_func is not None:
      return self._obj_func
    else:
      return self._func

  def __call__(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Calls decorated function object.

    Also verifies if the function is compatible with TFLite.

    Returns:
      A execution result of the decorated function.
    """
    if self._debug:
      args_repr = [repr(a) for a in args]
      kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
      signature = ", ".join(args_repr + kwargs_repr)
      print(
          f"DEBUG: Calling {self._get_func().__name__}({signature})",
          file=sys.stderr)

    if not self._verified:
      concrete_func = self._get_func().get_concrete_function(*args, **kwargs)
      converter = lite.TFLiteConverterV2.from_concrete_functions(
          [concrete_func])
      # Set provided converter parameters
      if self._converter_target_spec is not None:
        converter.target_spec = self._converter_target_spec
      if self._converter_allow_custom_ops is not None:
        converter.allow_custom_ops = self._converter_allow_custom_ops
      try:
        converter.convert()
      except convert.ConverterError as err:
        self._decode_error(err)
      finally:
        self._verified = True

    return self._get_func()(*args, **kwargs)

  def get_concrete_function(self, *args, **kwargs):
    """Returns a concrete function of the decorated function."""
    return self._get_func().get_concrete_function(*args, **kwargs)

  def _dump_error_details(self, ops, locations):
    """Dump the list of ops and locations."""
    for i in range(0, len(ops)):
      callstack = []
      for single_call in locations[i].call:
        if (locations[i].type ==
            converter_error_data_pb2.ConverterErrorData.CALLSITELOC):
          callstack.append(
              f"  - {single_call.source.filename}:{single_call.source.line}")
        else:
          callstack.append(str(single_call))
      callstack_dump = "\n".join(callstack)
      err_string = f"Op: {ops[i]}\n{callstack_dump}\n"
      self._log_messages.append(err_string)
      logging.warning(err_string)

  def _decode_error_legacy(self, err):
    """Parses the given legacy ConverterError for OSS."""
    for line in str(err).splitlines():
      # Check custom op usage error.
      if line.startswith(_CUSTOM_OPS_HDR):
        custom_ops = line[len(_CUSTOM_OPS_HDR):]
        err_string = (
            f"CompatibilityError: op '{custom_ops}' is(are) not natively "
            "supported by TensorFlow Lite. You need to provide a custom "
            "operator. https://www.tensorflow.org/lite/guide/ops_custom")
        self._log_messages.append(err_string)
        logging.warning(err_string)
      # Check TensorFlow op usage error.
      elif line.startswith(_TF_OPS_HDR):
        tf_ops = line[len(_TF_OPS_HDR):]
        err_string = (
            f"CompatibilityWarning: op '{tf_ops}' require(s) \"Select TF Ops\" "
            "for model conversion for TensorFlow Lite. "
            "https://www.tensorflow.org/lite/guide/ops_select")
        self._log_messages.append(err_string)
        logging.warning(err_string)

  def _decode_converter_error(self, err):
    """Parses the given ConverterError which has detailed error information."""
    custom_ops = []
    custom_ops_location = []
    tf_ops = []
    tf_ops_location = []
    for err in err.errors:
      # Check custom op usage error.
      if err.error_code == converter_error_data_pb2.ConverterErrorData.ERROR_NEEDS_CUSTOM_OPS:
        custom_ops.append(err.operator.name)
        custom_ops_location.append(err.location)
      # Check TensorFlow op usage error.
      elif err.error_code == converter_error_data_pb2.ConverterErrorData.ERROR_NEEDS_FLEX_OPS:
        tf_ops.append(err.operator.name)
        tf_ops_location.append(err.location)

    if custom_ops:
      custom_ops_str = ", ".join(sorted(custom_ops))
      err_string = (
          f"CompatibilityError: op '{custom_ops_str}' is(are) not natively "
          "supported by TensorFlow Lite. You need to provide a custom "
          "operator. https://www.tensorflow.org/lite/guide/ops_custom")
      self._log_messages.append(err_string)
      logging.warning(err_string)
      self._dump_error_details(custom_ops, custom_ops_location)

    if tf_ops:
      tf_ops_str = ", ".join(sorted(tf_ops))
      err_string = (
          f"CompatibilityWarning: op '{tf_ops_str}' require(s) \"Select TF Ops"
          "\" for model conversion for TensorFlow Lite. "
          "https://www.tensorflow.org/lite/guide/ops_select")
      self._log_messages.append(err_string)
      logging.warning(err_string)
      self._dump_error_details(tf_ops, tf_ops_location)

  def _decode_error(self, err):
    """Parses the given ConverterError and generates compatibility warnings."""
    if hasattr(err, "errors"):
      self._decode_converter_error(err)
    else:
      self._decode_error_legacy(err)

    if self._raise_exception and self._log_messages:
      raise CompatibilityError(f"CompatibilityException at {repr(self._func)}")

  def get_compatibility_log(self):
    """Returns list of compatibility log messages.

    WARNING: This method should only be used for unit tests.

    Returns:
      The list of log messages by the recent compatibility check.
    Raises:
      RuntimeError: when the compatibility was NOT checked.
    """
    if not self._verified:
      raise RuntimeError("target compatibility isn't verified yet")
    return self._log_messages


def compatible(target=None, **kwargs):
  """Wraps _Compatible to allow for deferred calling."""
  if target is None:
    def wrapper(target):
      return _Compatible(target, **kwargs)

    return wrapper
  else:
    return _Compatible(target, **kwargs)
