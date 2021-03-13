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
import os

import enum
import numpy as np

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join('tflite_runtime', 'interpreter')):
  # This file is part of tensorflow package.
  from tensorflow.lite.python.interpreter_wrapper import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper
  from tensorflow.python.util.tf_export import tf_export as _tf_export
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper

  def _tf_export(*x, **kwargs):
    del x, kwargs
    return lambda x: x


try:
  from tensorflow.lite.python import metrics_portable as metrics
except ImportError:
  from tensorflow.lite.python import metrics_nonportable as metrics
# pylint: enable=g-import-not-at-top


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
    # __del__ can not be called multiple times, so if the delegate is destroyed.
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
  try:
    delegate = Delegate(library, options)
  except ValueError as e:
    raise ValueError('Failed to load delegate from {}\n{}'.format(
        library, str(e)))
  return delegate


class SignatureRunner(object):
  """SignatureRunner class for running TFLite models using SignatureDef.

  This class should be instantiated through TFLite Interpreter only using
  get_signature_runner method on Interpreter.
  Example,
  signature = interpreter.get_signature_runner("my_signature")
  result = signature(input_1=my_input_1, input_2=my_input_2)
  print(result["my_output"])
  print(result["my_second_output"])
  All names used are this specific SignatureDef names.

  Notes:
    No other function on this object or on the interpreter provided should be
    called while this object call has not finished.
  """

  def __init__(self, interpreter=None, signature_def_name=None):
    """Constructor.

    Args:
      interpreter: Interpreter object that is already initialized with the
        requested model.
      signature_def_name: SignatureDef names to be used.
    """
    if not interpreter:
      raise ValueError('None interpreter provided.')
    if not signature_def_name:
      raise ValueError('None signature_def_name provided.')
    self._interpreter = interpreter
    self._signature_def_name = signature_def_name
    signature_defs = interpreter._get_full_signature_list()
    if signature_def_name not in signature_defs:
      raise ValueError('Invalid signature_def_name provided.')
    self._signature_def = signature_defs[signature_def_name]
    self._outputs = self._signature_def['outputs'].items()
    self._inputs = self._signature_def['inputs']

  def __call__(self, **kwargs):
    """Runs the SignatureDef given the provided inputs in arguments.

    Args:
      **kwargs: key,value for inputs to the model. Key is the SignatureDef input
        name. Value is numpy array with the value.

    Returns:
      dictionary of the results from the model invoke.
      Key in the dictionary is SignatureDef output name.
      Value is the result Tensor.
    """

    if len(kwargs) != len(self._inputs):
      raise ValueError(
          'Invalid number of inputs provided for running a SignatureDef, '
          'expected %s vs provided %s' % (len(kwargs), len(self._inputs)))
    # Resize input tensors
    for input_name, value in kwargs.items():
      if input_name not in self._inputs:
        raise ValueError('Invalid Input name (%s) for SignatureDef' %
                         input_name)
      self._interpreter.resize_tensor_input(self._inputs[input_name],
                                            value.shape)
    # Allocate tensors.
    self._interpreter.allocate_tensors()
    # Set the input values.
    for input_name, value in kwargs.items():
      self._interpreter._set_input_tensor(
          input_name, value=value, method_name=self._signature_def_name)
    self._interpreter.invoke()
    result = {}
    for output_name, output_index in self._outputs:
      result[output_name] = self._interpreter.get_tensor(output_index)
    return result


@_tf_export('lite.experimental.OpResolver')
@enum.unique
class OpResolver(enum.Enum):
  """Different types of op resolvers for Tensorflow Lite.

  * `AUTO`: Indicates the op resolver that is chosen by default in TfLite
     Python, which is the "BUILTIN" as described below.
  * `BUILTIN`: Indicates the op resolver for built-in ops with optimized kernel
    implementation.
  * `BUILTIN_REF`: Indicates the op resolver for built-in ops with reference
    kernel implementation. It's generally used for testing and debugging.
  * `BUILTIN_WITHOUT_DEFAULT_DELEGATES`: Indicates the op resolver for
    built-in ops with optimized kernel implementation, but it will disable
    the application of default TfLite delegates (like the XNNPACK delegate) to
    the model graph. Generally this should not be used unless there are issues
    with the default configuration.
  """
  # Corresponds to an op resolver chosen by default in TfLite Python.
  AUTO = 0

  # Corresponds to tflite::ops::builtin::BuiltinOpResolver in C++.
  BUILTIN = 1

  # Corresponds to tflite::ops::builtin::BuiltinRefOpResolver in C++.
  BUILTIN_REF = 2

  # Corresponds to
  # tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates in C++.
  BUILTIN_WITHOUT_DEFAULT_DELEGATES = 3


def _get_op_resolver_id(op_resolver=OpResolver.AUTO):
  """Get a integer identifier for the op resolver."""

  # Note: the integer identifier value needs to be same w/ op resolver ids
  # defined in interpreter_wrapper/interpreter_wrapper.cc.
  return {
      OpResolver.AUTO: 1,  # Note the identifier is same with that of BUILTIN
      OpResolver.BUILTIN: 1,
      OpResolver.BUILTIN_REF: 2,
      OpResolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES: 3
  }.get(op_resolver, None)


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
               experimental_delegates=None,
               num_threads=None,
               experimental_op_resolver=OpResolver.AUTO):
    """Constructor.

    Args:
      model_path: Path to TF-Lite Flatbuffer file.
      model_content: Content of model.
      experimental_delegates: Experimental. Subject to change. List of
        [TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates)
          objects returned by lite.load_delegate().
      num_threads: Sets the number of threads used by the interpreter and
        available to CPU kernels. If not set, the interpreter will use an
        implementation-dependent default number of threads. Currently, only a
        subset of kernels, such as conv, support multi-threading.
      experimental_op_resolver: The op resolver used by the interpreter. It must
        be an instance of OpResolver. By default, we use the built-in op
        resolver which corresponds to tflite::ops::builtin::BuiltinOpResolver
        in C++.

    Raises:
      ValueError: If the interpreter was unable to create.
    """
    if not hasattr(self, '_custom_op_registerers'):
      self._custom_op_registerers = []

    op_resolver_id = _get_op_resolver_id(experimental_op_resolver)
    if op_resolver_id is None:
      raise ValueError('Unrecognized passed in op resolver: {}'.format(
          experimental_op_resolver))

    if model_path and not model_content:
      custom_op_registerers_by_name = [
          x for x in self._custom_op_registerers if isinstance(x, str)
      ]
      custom_op_registerers_by_func = [
          x for x in self._custom_op_registerers if not isinstance(x, str)
      ]
      self._interpreter = (
          _interpreter_wrapper.CreateWrapperFromFile(
              model_path, op_resolver_id, custom_op_registerers_by_name,
              custom_op_registerers_by_func))
      if not self._interpreter:
        raise ValueError('Failed to open {}'.format(model_path))
    elif model_content and not model_path:
      custom_op_registerers_by_name = [
          x for x in self._custom_op_registerers if isinstance(x, str)
      ]
      custom_op_registerers_by_func = [
          x for x in self._custom_op_registerers if not isinstance(x, str)
      ]
      # Take a reference, so the pointer remains valid.
      # Since python strings are immutable then PyString_XX functions
      # will always return the same pointer.
      self._model_content = model_content
      self._interpreter = (
          _interpreter_wrapper.CreateWrapperFromBuffer(
              model_content, op_resolver_id, custom_op_registerers_by_name,
              custom_op_registerers_by_func))
    elif not model_content and not model_path:
      raise ValueError('`model_path` or `model_content` must be specified.')
    else:
      raise ValueError('Can\'t both provide `model_path` and `model_content`')

    if num_threads is not None:
      if not isinstance(num_threads, int):
        raise ValueError('type of num_threads should be int')
      if num_threads < 1:
        raise ValueError('num_threads should >= 1')
      self._interpreter.SetNumThreads(num_threads)

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
    self._signature_defs = self.get_signature_list()

    self._metrics = metrics.TFLiteMetrics()
    self._metrics.increase_counter_interpreter_creation()

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
    tensor_size_signature = self._interpreter.TensorSizeSignature(tensor_index)
    tensor_type = self._interpreter.TensorType(tensor_index)
    tensor_quantization = self._interpreter.TensorQuantization(tensor_index)
    tensor_quantization_params = self._interpreter.TensorQuantizationParameters(
        tensor_index)
    tensor_sparsity_params = self._interpreter.TensorSparsityParameters(
        tensor_index)

    if not tensor_type:
      raise ValueError('Could not get tensor details')

    details = {
        'name': tensor_name,
        'index': tensor_index,
        'shape': tensor_size,
        'shape_signature': tensor_size_signature,
        'dtype': tensor_type,
        'quantization': tensor_quantization,
        'quantization_parameters': {
            'scales': tensor_quantization_params[0],
            'zero_points': tensor_quantization_params[1],
            'quantized_dimension': tensor_quantization_params[2],
        },
        'sparsity_parameters': tensor_sparsity_params
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
    """Sets the value of the input tensor.

    Note this copies data in `value`.

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

  def resize_tensor_input(self, input_index, tensor_size, strict=False):
    """Resizes an input tensor.

    Args:
      input_index: Tensor index of input to set. This value can be gotten from
        the 'index' field in get_input_details.
      tensor_size: The tensor_shape to resize the input to.
      strict: Only unknown dimensions can be resized when `strict` is True.
        Unknown dimensions are indicated as `-1` in the `shape_signature`
        attribute of a given tensor. (default False)

    Raises:
      ValueError: If the interpreter could not resize the input tensor.

    Usage:
    ```
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.resize_tensor_input(0, [num_test_images, 224, 224, 3])
    interpreter.allocate_tensors()
    interpreter.set_tensor(0, test_images)
    interpreter.invoke()
    ```
    """
    self._ensure_safe()
    # `ResizeInputTensor` now only accepts int32 numpy array as `tensor_size
    # parameter.
    tensor_size = np.array(tensor_size, dtype=np.int32)
    self._interpreter.ResizeInputTensor(input_index, tensor_size, strict)

  def get_output_details(self):
    """Gets model output details.

    Returns:
      A list of output details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.OutputIndices()
    ]

  def get_signature_list(self):
    """Gets list of SignatureDefs in the model.

    Example,
    ```
    signatures = interpreter.get_signature_list()
    print(signatures)

    # {
    #   'add': {'inputs': ['x', 'y'], 'outputs': ['output_0']}
    # }

    Then using the names in the signature list you can get a callable from
    get_signature_runner().
    ```

    Returns:
      A list of SignatureDef details in a dictionary structure.
      It is keyed on the SignatureDef method name, and the value holds
      dictionary of inputs and outputs.
    """
    full_signature_defs = self._interpreter.GetSignatureDefs()
    for _, signature_def in full_signature_defs.items():
      signature_def['inputs'] = list(signature_def['inputs'].keys())
      signature_def['outputs'] = list(signature_def['outputs'].keys())
    return full_signature_defs

  def _get_full_signature_list(self):
    """Gets list of SignatureDefs in the model.

    Example,
    ```
    signatures = interpreter._get_full_signature_list()
    print(signatures)

    # {
    #   'add': {'inputs': {'x': 1, 'y': 0}, 'outputs': {'output_0': 4}}
    # }

    Then using the names in the signature list you can get a callable from
    get_signature_runner().
    ```

    Returns:
      A list of SignatureDef details in a dictionary structure.
      It is keyed on the SignatureDef method name, and the value holds
      dictionary of inputs and outputs.
    """
    return self._interpreter.GetSignatureDefs()

  def _set_input_tensor(self, input_name, value, method_name=None):
    """Sets the value of the input tensor.

    Input tensor is identified by `input_name` in the SignatureDef identified
    by `method_name`.
    If the model has a single SignatureDef then you can pass None as
    `method_name`.

    Note this copies data in `value`.

    Example,
    ```
    input_data = np.array([1.2, 1.4], np.float32)
    signatures = interpreter.get_signature_list()
    print(signatures)
    # {
    #   'add': {'inputs': {'x': 1, 'y': 0}, 'outputs': {'output_0': 4}}
    # }
    interpreter._set_input_tensor(input_name='x', value=input_data,
    method_name='add_fn')
    ```

    Args:
      input_name: Name of the output tensor in the SignatureDef.
      value: Value of tensor to set as a numpy array.
      method_name: The exported method name for the SignatureDef, it can be None
        if and only if the model has a single SignatureDef. Default value is
        None.

    Raises:
      ValueError: If the interpreter could not set the tensor. Or
      if `method_name` is None and model doesn't have a single
      Signature.
    """
    if method_name is None:
      if len(self._signature_defs) != 1:
        raise ValueError(
            'SignatureDef method_name is None and model has {0} Signatures. '
            'None is only allowed when the model has 1 SignatureDef'.format(
                len(self._signature_defs)))
      else:
        method_name = next(iter(self._signature_defs))
    self._interpreter.SetInputTensorFromSignatureDefName(
        input_name, method_name, value)

  def get_signature_runner(self, method_name=None):
    """Gets callable for inference of specific SignatureDef.

    Example usage,
    ```
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    fn = interpreter.get_signature_runner('div_with_remainder')
    output = fn(x=np.array([3]), y=np.array([2]))
    print(output)
    # {
    #   'quotient': array([1.], dtype=float32)
    #   'remainder': array([1.], dtype=float32)
    # }
    ```

    None can be passed for method_name if the model has a single Signature only.

    All names used are this specific SignatureDef names.


    Args:
      method_name: The exported method name for the SignatureDef, it can be None
        if and only if the model has a single SignatureDef. Default value is
        None.

    Returns:
      This returns a callable that can run inference for SignatureDef defined
      by argument 'method_name'.
      The callable will take key arguments corresponding to the arguments of the
      SignatureDef, that should have numpy values.
      The callable will returns dictionary that maps from output names to numpy
      values of the computed results.

    Raises:
      ValueError: If passed method_name is invalid.
    """
    if method_name is None:
      if len(self._signature_defs) != 1:
        raise ValueError(
            'SignatureDef method_name is None and model has {0} Signatures. '
            'None is only allowed when the model has 1 SignatureDef'.format(
                len(self._signature_defs)))
      else:
        method_name = next(iter(self._signature_defs))
    return SignatureRunner(interpreter=self, signature_def_name=method_name)

  def get_tensor(self, tensor_index):
    """Gets the value of the output tensor (get a copy).

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

  # Experimental and subject to change.
  def _native_handle(self):
    """Returns a pointer to the underlying tflite::Interpreter instance.

    This allows extending tflite.Interpreter's functionality in a custom C++
    function. Consider how that may work in a custom pybind wrapper:

      m.def("SomeNewFeature", ([](py::object handle) {
        auto* interpreter =
          reinterpret_cast<tflite::Interpreter*>(handle.cast<intptr_t>());
        ...
      }))

    and corresponding Python call:

      SomeNewFeature(interpreter.native_handle())

    Note: This approach is fragile. Users must guarantee the C++ extension build
    is consistent with the tflite.Interpreter's underlying C++ build.
    """
    return self._interpreter.interpreter()


class InterpreterWithCustomOps(Interpreter):
  """Interpreter interface for TensorFlow Lite Models that accepts custom ops.

  The interface provided by this class is experimental and therefore not exposed
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
      custom_op_registerers: List of str (symbol names) or functions that take a
        pointer to a MutableOpResolver and register a custom op. When passing
        functions, use a pybind function that takes a uintptr_t that can be
        recast as a pointer to a MutableOpResolver.

    Raises:
      ValueError: If the interpreter was unable to create.
    """
    self._custom_op_registerers = custom_op_registerers or []
    super(InterpreterWithCustomOps, self).__init__(
        model_path=model_path,
        model_content=model_content,
        experimental_delegates=experimental_delegates)
