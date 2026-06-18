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
    @tf.lite.experimental.authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      return tf.cosh(x)

    result = f(tf.constant([0.0]))

    > COMPATIBILITY WARNING: op 'tf.Cosh' require(s) "Select TF Ops" for model
    > conversion for TensorFlow Lite.
    > Op: tf.Cosh
    >   - tensorflow/python/framework/op_def_library.py:xxx
    >   - tensorflow/python/ops/gen_math_ops.py:xxx
    >   - simple_authoring.py:xxx
"""
import functools
from tensorflow.compiler.mlir.lite.metrics import converter_error_data_pb2
# pylint: disable=g-import-not-at-top
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.python.util.tf_export import tf_export as _tf_export


_CUSTOM_OPS_HDR = "Custom ops: "
_TF_OPS_HDR = "TF Select ops: "
_AUTHORING_ERROR_HDR = "COMPATIBILITY ERROR"
_AUTHORING_WARNING_HDR = "COMPATIBILITY WARNING"
_FUNC_GRAPH_SRC_PATH = "tensorflow/python/framework/func_graph.py"


class CompatibilityError(Exception):
  """Raised when an error occurs with TFLite compatibility."""
  pass


class _Compatible:
  """A decorator class to check TFLite compatibility created by `lite.experimental.authoring.compatible`."""

  def __init__(self,
               target,
               converter_target_spec=None,
               converter_allow_custom_ops=None,
               raise_exception=False):
    """Initialize the decorator object.

    Here is the description of the object variables.
    - _func     : decorated function.
    - _obj_func : for class object, we need to use this object to provide `self`
                  instance as 1 first argument.
    - _verified : whether the compatibility is checked or not.

    Args:
      target: decorated function.
      converter_target_spec : target_spec of TFLite converter parameter.
      converter_allow_custom_ops : allow_custom_ops of TFLite converter
          parameter.
      raise_exception : to raise an exception on compatibility issues.
          User need to use get_compatibility_log() to check details.
    """
    functools.update_wrapper(self, target)
    self._func = target
    self._obj_func = None
    self._verified = False
    self._log_messages = []
    self._raise_exception = raise_exception
    self._converter_target_spec = converter_target_spec
    self._converter_allow_custom_ops = converter_allow_custom_ops

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

    if not self._verified:
      model = self._get_func()
      concrete_func = model.get_concrete_function(*args, **kwargs)
      converter = lite.TFLiteConverterV2.from_concrete_functions(
          [concrete_func], model)
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

  def _get_location_string(self, location):
    """Dump location of ConveterError.errors.location."""
    callstack = []
    for single_call in reversed(location.call):
      if (location.type ==
          converter_error_data_pb2.ConverterErrorData.CALLSITELOC):
        callstack.append(
            f"  - {single_call.source.filename}:{single_call.source.line}")
      else:
        callstack.append(str(single_call))
    callstack_dump = "\n".join(callstack)
    return callstack_dump

  def _dump_error_details(self, ops, locations):
    """Dump the list of ops and locations."""
    for i in range(0, len(ops)):
      callstack_dump = self._get_location_string(locations[i])
      err_string = f"Op: {ops[i]}\n{callstack_dump}\n"
      self._log(err_string)

  def _decode_error_legacy(self, err):
    """Parses the given legacy ConverterError for OSS."""
    for line in str(err).splitlines():
      # Check custom op usage error.
      if line.startswith(_CUSTOM_OPS_HDR):
        custom_ops = line[len(_CUSTOM_OPS_HDR):]
        err_string = (
            f"{_AUTHORING_ERROR_HDR}: op '{custom_ops}' is(are) not natively "
            "supported by TensorFlow Lite. You need to provide a custom "
            "operator. https://www.tensorflow.org/lite/guide/ops_custom")
        self._log(err_string)
      # Check TensorFlow op usage error.
      elif line.startswith(_TF_OPS_HDR):
        tf_ops = line[len(_TF_OPS_HDR):]
        err_string = (
            f"{_AUTHORING_WARNING_HDR}: op '{tf_ops}' require(s) \"Select TF "
            "Ops\" for model conversion for TensorFlow Lite. "
            "https://www.tensorflow.org/lite/guide/ops_select")
        self._log(err_string)

  def _decode_converter_error(self, err):
    """Parses the given ConverterError which has detailed error information."""
    custom_ops = []
    custom_ops_location = []
    tf_ops = []
    tf_ops_location = []
    gpu_not_compatible_ops = []
    for err in err.errors:
      # Check custom op usage error.
      if err.error_code == converter_error_data_pb2.ConverterErrorData.ERROR_NEEDS_CUSTOM_OPS:
        custom_ops.append(err.operator.name)
        custom_ops_location.append(err.location)
      # Check TensorFlow op usage error.
      elif err.error_code == converter_error_data_pb2.ConverterErrorData.ERROR_NEEDS_FLEX_OPS:
        tf_ops.append(err.operator.name)
        tf_ops_location.append(err.location)
      # Check GPU delegate compatibility error.
      elif err.error_code == converter_error_data_pb2.ConverterErrorData.ERROR_GPU_NOT_COMPATIBLE:
        gpu_not_compatible_ops.append(err.operator.name)
        # Log the first line of ConveterError.errors.error_message only
        # since the seond line is "Error code: xxxx"
        self._log(err.error_message.splitlines()[0])
        self._log(self._get_location_string(err.location) + "\n")
      else:
        # Log other errors.
        self._log(f"{_AUTHORING_ERROR_HDR}: {err.error_message}")
        self._log(self._get_location_string(err.location) + "\n")

    if custom_ops:
      custom_ops_str = ", ".join(sorted(custom_ops))
      err_string = (
          f"{_AUTHORING_ERROR_HDR}: op '{custom_ops_str}' is(are) not natively "
          "supported by TensorFlow Lite. You need to provide a custom "
          "operator. https://www.tensorflow.org/lite/guide/ops_custom")
      self._log(err_string)
      self._dump_error_details(custom_ops, custom_ops_location)

    if tf_ops:
      tf_ops_str = ", ".join(sorted(tf_ops))
      err_string = (
          f"{_AUTHORING_WARNING_HDR}: op '{tf_ops_str}' require(s) \"Select TF"
          " Ops\" for model conversion for TensorFlow Lite. "
          "https://www.tensorflow.org/lite/guide/ops_select")
      self._log(err_string)
      self._dump_error_details(tf_ops, tf_ops_location)

    if gpu_not_compatible_ops:
      not_compatible_ops_str = ", ".join(sorted(gpu_not_compatible_ops))
      err_string = (
          f"{_AUTHORING_WARNING_HDR}: op '{not_compatible_ops_str}' aren't "
          "compatible with TensorFlow Lite GPU delegate. "
          "https://www.tensorflow.org/lite/performance/gpu")
      self._log(err_string)

  def _decode_error(self, err):
    """Parses the given ConverterError and generates compatibility warnings."""
    if hasattr(err, "errors"):
      self._decode_converter_error(err)
    else:
      self._decode_error_legacy(err)

    if self._raise_exception and self._log_messages:
      raise CompatibilityError(f"CompatibilityException at {repr(self._func)}")

  def _log(self, message):
    """Log and print authoring warning / error message."""
    self._log_messages.append(message)
    print(message)

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


@_tf_export("lite.experimental.authoring.compatible")
def compatible(target=None, converter_target_spec=None, **kwargs):
  """Wraps `tf.function` into a callable function with TFLite compatibility checking.

  Example:

  ```python
  @tf.lite.experimental.authoring.compatible
  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.float32)
  ])
  def f(x):
      return tf.cosh(x)

  result = f(tf.constant([0.0]))
  # COMPATIBILITY WARNING: op 'tf.Cosh' require(s) "Select TF Ops" for model
  # conversion for TensorFlow Lite.
  # Op: tf.Cosh
  #   - tensorflow/python/framework/op_def_library.py:748
  #   - tensorflow/python/ops/gen_math_ops.py:2458
  #   - <stdin>:6
  ```

  WARNING: Experimental interface, subject to change.

  Args:
    target: A `tf.function` to decorate.
    converter_target_spec : target_spec of TFLite converter parameter.
    **kwargs: The keyword arguments of the decorator class _Compatible.

  Returns:
     A callable object of `tf.lite.experimental.authoring._Compatible`.
  """
  if target is None:
    def wrapper(target):
      return _Compatible(target, converter_target_spec, **kwargs)
    return wrapper
  else:
    return _Compatible(target, converter_target_spec, **kwargs)
