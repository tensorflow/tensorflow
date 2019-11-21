# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Python TF-Lite interpreter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import platform
import sys

import numpy as np

# pylint: disable=g-import-not-at-top
if not __file__.endswith('tflite_runtime/interpreter.py'):
  # This file is part of tensorflow package.
  from tensorflow.python.util.lazy_loader import LazyLoader
  from tensorflow.python.util.tf_export import tf_export as _tf_export

  # Lazy load since some of the performance benchmark skylark rules
  # break dependencies. Must use double quotes to match code internal rewrite
  # rule.
  # pylint: disable=g-inconsistent-quotes
  _interpreter_wrapper = LazyLoader(
      "_interpreter_wrapper", globals(),
      "tensorflow.lite.python.interpreter_wrapper."
      "tensorflow_wrap_interpreter_wrapper")
  # pylint: enable=g-inconsistent-quotes

  del LazyLoader
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import interpreter_wrapper as _interpreter_wrapper

  def _tf_export(*x, **kwargs):
    del x, kwargs
    return lambda x: x


class Delegate(object):
  """Python wrapper class to manage TfLiteDelegate objects.

  The shared library is expected to have two functions:
    TfLiteDelegate* tflite_plugin_create_delegate(
        char**, char**, size_t, void (*report_error)(const char *))
    void tflite_plugin_destroy_delegate(TfLiteDelegate*)

  The first one creates a delegate object. It may return NULL to indicate an
  error (with a suitable error message reported by calling report_error()).
  The second one destroys delegate object and must be called for every
  created delegate object. Passing NULL as argument value is allowed, i.e.

    tflite_plugin_destroy_delegate(tflite_plugin_create_delegate(...))

  always works.
  """

  def __init__(self, library, options=None):
    """Loads delegate from the shared library.

    Args:
      library: Shared library name.
      options: Dictionary of options that are required to load the delegate. All
        keys and values in the dictionary should be serializable. Consult the
        documentation of the specific delegate for required and legal options.
        (default None)
    Raises:
      RuntimeError: This is raised if the Python implementation is not CPython.
    """

    # TODO(b/136468453): Remove need for __del__ ordering needs of CPython
    # by using explicit closes(). See implementation of Interpreter __del__.
    if platform.python_implementation() != 'CPython':
      raise RuntimeError('Delegates are currently only supported into CPython'
                         'due to missing immediate reference counting.')

    self._library = ctypes.pydll.LoadLibrary(library)
    self._library.tflite_plugin_create_delegate.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
        ctypes.CFUNCTYPE(None, ctypes.c_char_p)
    ]
    self._library.tflite_plugin_create_delegate.restype = ctypes.c_void_p

    # Convert the options from a dictionary to lists of char pointers.
    options = options or {}
    options_keys = (ctypes.c_char_p * len(options))()
    options_values = (ctypes.c_char_p * len(options))()
    for idx, (key, value) in enumerate(options.items()):
      options_keys[idx] = str(key).encode('utf-8')
      options_values[idx] = str(value).encode('utf-8')

    class ErrorMessageCapture(object):

      def __init__(self):
        self.message = ''

      def report(self, x):
        self.message += x if isinstance(x, str) else x.decode('utf-8')

    capture = ErrorMessageCapture()
    error_capturer_cb = ctypes.CFUNCTYPE(None, ctypes.c_char_p)(capture.report)
    # Do not make a copy of _delegate_ptr. It is freed by Delegate's finalizer.
    self._delegate_ptr = self._library.tflite_plugin_create_delegate(
        options_keys, options_values, len(options), error_capturer_cb)
    if self._delegate_ptr is None:
      raise ValueError(capture.message)

  def __del__(self):
    # __del__ can be called multiple times, so if the delegate is destroyed.
    # don't try to destroy it twice.
    if self._library is not None:
      self._library.tflite_plugin_destroy_delegate.argtypes = [ctypes.c_void_p]
      self._library.tflite_plugin_destroy_delegate(self._delegate_ptr)
      self._library = None

  def _get_native_delegate_pointer(self):
    """Returns the native TfLiteDelegate pointer.

    It is not safe to copy this pointer because it needs to be freed.

    Returns:
      TfLiteDelegate *
    """
    return self._delegate_ptr


@_tf_export('lite.experimental.load_delegate')
def load_delegate(library, options=None):
  """Returns loaded Delegate object.

  Args:
    library: Name of shared library containing the
      [TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates).
    options: Dictionary of options that are required to load the delegate. All
      keys and values in the dictionary should be convertible to str. Consult
      the documentation of the specific delegate for required and legal options.
      (default None)

  Returns:
    Delegate object.

  Raises:
    ValueError: Delegate failed to load.
    RuntimeError: If delegate loading is used on unsupported platform.
  """

  # TODO(b/137299813): Fix darwin support for delegates.
  if sys.platform == 'darwin':
    raise RuntimeError('Dynamic loading of delegates on Darwin not supported.')

  try:
    delegate = Delegate(library, options)
  except ValueError as e:
    raise ValueError('Failed to load delegate from {}\n{}'.format(
        library, str(e)))
  return delegate


@_tf_export('lite.Interpreter')
class Interpreter(object):
  """Interpreter interface for TensorFlow Lite Models.

  This makes the TensorFlow Lite interpreter accessible in Python.
  It is possible to use this interpreter in a multithreaded Python environment,
  but you must be sure to call functions of a particular instance from only
  one thread at a time. So if you want to have 4 threads running different
  inferences simultaneously, create  an interpreter for each one as thread-local
  data. Similarly, if you are calling invoke() in one thread on a single
  interpreter but you want to use tensor() on another thread once it is done,
  you must use a synchronization primitive between the threads to ensure invoke
  has returned before calling tensor().
  """

  def __init__(self,
               model_path=None,
               model_content=None,
               experimental_delegates=None):
    """Constructor.

    Args:
      model_path: Path to TF-Lite Flatbuffer file.
      model_content: Content of model.
      experimental_delegates: Experimental. Subject to change. List of
        [TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates)
        objects returned by lite.load_delegate().

    Raises:
      ValueError: If the interpreter was unable to create.
    """
    if not hasattr(self, '_custom_op_registerers'):
      self._custom_op_registerers = []
    if model_path and not model_content:
      self._interpreter = (
          _interpreter_wrapper.InterpreterWrapper_CreateWrapperCPPFromFile(
              model_path, self._custom_op_registerers))
      if not self._interpreter:
        raise ValueError('Failed to open {}'.format(model_path))
    elif model_content and not model_path:
      # Take a reference, so the pointer remains valid.
      # Since python strings are immutable then PyString_XX functions
      # will always return the same pointer.
      self._model_content = model_content
      self._interpreter = (
          _interpreter_wrapper.InterpreterWrapper_CreateWrapperCPPFromBuffer(
              model_content, self._custom_op_registerers))
    elif not model_path and not model_path:
      raise ValueError('`model_path` or `model_content` must be specified.')
    else:
      raise ValueError('Can\'t both provide `model_path` and `model_content`')

    # Each delegate is a wrapper that owns the delegates that have been loaded
    # as plugins. The interpreter wrapper will be using them, but we need to
    # hold them in a list so that the lifetime is preserved at least as long as
    # the interpreter wrapper.
    self._delegates = []
    if experimental_delegates:
      self._delegates = experimental_delegates
      for delegate in self._delegates:
        self._interpreter.ModifyGraphWithDelegate(
            delegate._get_native_delegate_pointer())  # pylint: disable=protected-access

  def __del__(self):
    # Must make sure the interpreter is destroyed before things that
    # are used by it like the delegates. NOTE this only works on CPython
    # probably.
    # TODO(b/136468453): Remove need for __del__ ordering needs of CPython
    # by using explicit closes(). See implementation of Interpreter __del__.
    self._interpreter = None
    self._delegates = None

  def allocate_tensors(self):
    self._ensure_safe()
    return self._interpreter.AllocateTensors()

  def _safe_to_run(self):
    """Returns true if there exist no numpy array buffers.

    This means it is safe to run tflite calls that may destroy internally
    allocated memory. This works, because in the wrapper.cc we have made
    the numpy base be the self._interpreter.
    """
    # NOTE, our tensor() call in cpp will use _interpreter as a base pointer.
    # If this environment is the only _interpreter, then the ref count should be
    # 2 (1 in self and 1 in temporary of sys.getrefcount).
    return sys.getrefcount(self._interpreter) == 2

  def _ensure_safe(self):
    """Makes sure no numpy arrays pointing to internal buffers are active.

    This should be called from any function that will call a function on
    _interpreter that may reallocate memory e.g. invoke(), ...

    Raises:
      RuntimeError: If there exist numpy objects pointing to internal memory
        then we throw.
    """
    if not self._safe_to_run():
      raise RuntimeError("""There is at least 1 reference to internal data
      in the interpreter in the form of a numpy array or slice. Be sure to
      only hold the function returned from tensor() if you are using raw
      data access.""")

  # Experimental and subject to change
  def _get_op_details(self, op_index):
    """Gets a dictionary with arrays of ids for tensors involved with an op.

    Args:
      op_index: Operation/node index of node to query.

    Returns:
      a dictionary containing the index, op name, and arrays with lists of the
      indices for the inputs and outputs of the op/node.
    """
    op_index = int(op_index)
    op_name = self._interpreter.NodeName(op_index)
    op_inputs = self._interpreter.NodeInputs(op_index)
    op_outputs = self._interpreter.NodeOutputs(op_index)

    details = {
        'index': op_index,
        'op_name': op_name,
        'inputs': op_inputs,
        'outputs': op_outputs,
    }

    return details

  def _get_tensor_details(self, tensor_index):
    """Gets tensor details.

    Args:
      tensor_index: Tensor index of tensor to query.

    Returns:
      A dictionary containing the following fields of the tensor:
        'name': The tensor name.
        'index': The tensor index in the interpreter.
        'shape': The shape of the tensor.
        'quantization': Deprecated, use 'quantization_parameters'. This field
            only works for per-tensor quantization, whereas
            'quantization_parameters' works in all cases.
        'quantization_parameters': The parameters used to quantize the tensor:
          'scales': List of scales (one if per-tensor quantization)
          'zero_points': List of zero_points (one if per-tensor quantization)
          'quantized_dimension': Specifies the dimension of per-axis
              quantization, in the case of multiple scales/zero_points.

    Raises:
      ValueError: If tensor_index is invalid.
    """
    tensor_index = int(tensor_index)
    tensor_name = self._interpreter.TensorName(tensor_index)
    tensor_size = self._interpreter.TensorSize(tensor_index)
    tensor_type = self._interpreter.TensorType(tensor_index)
    tensor_quantization = self._interpreter.TensorQuantization(tensor_index)
    tensor_quantization_params = self._interpreter.TensorQuantizationParameters(
        tensor_index)

    if not tensor_name or not tensor_type:
      raise ValueError('Could not get tensor details')

    details = {
        'name': tensor_name,
        'index': tensor_index,
        'shape': tensor_size,
        'dtype': tensor_type,
        'quantization': tensor_quantization,
        'quantization_parameters': {
            'scales': tensor_quantization_params[0],
            'zero_points': tensor_quantization_params[1],
            'quantized_dimension': tensor_quantization_params[2],
        }
    }

    return details

  # Experimental and subject to change
  def _get_ops_details(self):
    """Gets op details for every node.

    Returns:
      A list of dictionaries containing arrays with lists of tensor ids for
      tensors involved in the op.
    """
    return [
        self._get_op_details(idx) for idx in range(self._interpreter.NumNodes())
    ]

  def get_tensor_details(self):
    """Gets tensor details for every tensor with valid tensor details.

    Tensors where required information about the tensor is not found are not
    added to the list. This includes temporary tensors without a name.

    Returns:
      A list of dictionaries containing tensor information.
    """
    tensor_details = []
    for idx in range(self._interpreter.NumTensors()):
      try:
        tensor_details.append(self._get_tensor_details(idx))
      except ValueError:
        pass
    return tensor_details

  def get_input_details(self):
    """Gets model input details.

    Returns:
      A list of input details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.InputIndices()
    ]

  def set_tensor(self, tensor_index, value):
    """Sets the value of the input tensor. Note this copies data in `value`.

    If you want to avoid copying, you can use the `tensor()` function to get a
    numpy buffer pointing to the input buffer in the tflite interpreter.

    Args:
      tensor_index: Tensor index of tensor to set. This value can be gotten from
                    the 'index' field in get_input_details.
      value: Value of tensor to set.

    Raises:
      ValueError: If the interpreter could not set the tensor.
    """
    self._interpreter.SetTensor(tensor_index, value)

  def resize_tensor_input(self, input_index, tensor_size):
    """Resizes an input tensor.

    Args:
      input_index: Tensor index of input to set. This value can be gotten from
                   the 'index' field in get_input_details.
      tensor_size: The tensor_shape to resize the input to.

    Raises:
      ValueError: If the interpreter could not resize the input tensor.
    """
    self._ensure_safe()
    # `ResizeInputTensor` now only accepts int32 numpy array as `tensor_size
    # parameter.
    tensor_size = np.array(tensor_size, dtype=np.int32)
    self._interpreter.ResizeInputTensor(input_index, tensor_size)

  def get_output_details(self):
    """Gets model output details.

    Returns:
      A list of output details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.OutputIndices()
    ]

  def get_tensor(self, tensor_index):
    """Gets the value of the input tensor (get a copy).

    If you wish to avoid the copy, use `tensor()`. This function cannot be used
    to read intermediate results.

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
                    the 'index' field in get_output_details.

    Returns:
      a numpy array.
    """
    return self._interpreter.GetTensor(tensor_index)

  def tensor(self, tensor_index):
    """Returns function that gives a numpy view of the current tensor buffer.

    This allows reading and writing to this tensors w/o copies. This more
    closely mirrors the C++ Interpreter class interface's tensor() member, hence
    the name. Be careful to not hold these output references through calls
    to `allocate_tensors()` and `invoke()`. This function cannot be used to read
    intermediate results.

    Usage:

    ```
    interpreter.allocate_tensors()
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    for i in range(10):
      input().fill(3.)
      interpreter.invoke()
      print("inference %s" % output())
    ```

    Notice how this function avoids making a numpy array directly. This is
    because it is important to not hold actual numpy views to the data longer
    than necessary. If you do, then the interpreter can no longer be invoked,
    because it is possible the interpreter would resize and invalidate the
    referenced tensors. The NumPy API doesn't allow any mutability of the
    the underlying buffers.

    WRONG:

    ```
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])()
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    interpreter.allocate_tensors()  # This will throw RuntimeError
    for i in range(10):
      input.fill(3.)
      interpreter.invoke()  # this will throw RuntimeError since input,output
    ```

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
                    the 'index' field in get_output_details.

    Returns:
      A function that can return a new numpy array pointing to the internal
      TFLite tensor state at any point. It is safe to hold the function forever,
      but it is not safe to hold the numpy array forever.
    """
    return lambda: self._interpreter.tensor(self._interpreter, tensor_index)

  def invoke(self):
    """Invoke the interpreter.

    Be sure to set the input sizes, allocate tensors and fill values before
    calling this. Also, note that this function releases the GIL so heavy
    computation can be done in the background while the Python interpreter
    continues. No other function on this object should be called while the
    invoke() call has not finished.

    Raises:
      ValueError: When the underlying interpreter fails raise ValueError.
    """
    self._ensure_safe()
    self._interpreter.Invoke()

  def reset_all_variables(self):
    return self._interpreter.ResetVariableTensors()


class InterpreterWithCustomOps(Interpreter):
  """Interpreter interface for TensorFlow Lite Models that accepts custom ops.

  The interface provided by this class is experimenal and therefore not exposed
  as part of the public API.

  Wraps the tf.lite.Interpreter class and adds the ability to load custom ops
  by providing the names of functions that take a pointer to a BuiltinOpResolver
  and add a custom op.
  """

  def __init__(self,
               model_path=None,
               model_content=None,
               experimental_delegates=None,
               custom_op_registerers=None):
    """Constructor.

    Args:
      model_path: Path to TF-Lite Flatbuffer file.
      model_content: Content of model.
      experimental_delegates: Experimental. Subject to change. List of
        [TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates)
          objects returned by lite.load_delegate().
      custom_op_registerers: List of str, symbol names of functions that take a
        pointer to a MutableOpResolver and register a custom op.

    Raises:
      ValueError: If the interpreter was unable to create.
    """
    self._custom_op_registerers = custom_op_registerers
    super(InterpreterWithCustomOps, self).__init__(
        model_path=model_path,
        model_content=model_content,
        experimental_delegates=experimental_delegates)
