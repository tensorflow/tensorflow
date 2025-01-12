# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Contains layer utilities for input validation and format conversion."""
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables

# Constants for validation
VALID_DATA_FORMATS = {'channels_first', 'channels_last'}
VALID_PADDING = {'valid', 'same', 'full'}
VALID_NDIM = {3, 4, 5}

def convert_data_format(data_format: str, ndim: int) -> str:
    if ndim not in VALID_NDIM:
        raise ValueError(f"Invalid input rank {ndim}. Supported ranks are {VALID_NDIM}.")
    return {
        ('channels_last', 3): 'NWC',
        ('channels_last', 4): 'NHWC',
        ('channels_last', 5): 'NDHWC',
        ('channels_first', 3): 'NCW',
        ('channels_first', 4): 'NCHW',
        ('channels_first', 5): 'NCDHW',
    }.get((data_format, ndim), ValueError(f"Invalid data_format {data_format}."))


def normalize_tuple(value, n, name):
  """Transforms a single integer or iterable of integers into an integer tuple.

  Args:
    value: The value to validate and convert. Could an int, or any iterable
      of ints.
    n: The size of the tuple to be returned.
    name: The name of the argument being validated, e.g. "strides" or
      "kernel_size". This is only used to format error messages.

  Returns:
    A tuple of n integers.

  Raises:
    ValueError: If something else than an int/long or iterable thereof was
      passed.
  """
  try:
      value_tuple = tuple(value if isinstance(value, (list, tuple)) else (value,))
      if len(value_tuple) != n or not all(isinstance(v, int) for v in value_tuple):
          raise ValueError(f"`{name}` must be a tuple of {n} integers. Got {value_tuple}.")
  except:
      raise ValueError(f"`{name}` must be a tuple of {n} integers. Got {value}.")
  return value_tuple


def normalize_data_format(value):
  data_format = value.lower()
  if data_format not in VALID_DATA_FORMATS:
      raise ValueError(f'The `data_format` argument must be one of {VALID_DATA_FORMATS}. Received: {value}.')
  return data_format


def normalize_padding(value):
    padding = value.lower()
    if padding not in VALID_PADDING:
        raise ValueError(f'The `padding` argument must be one of {VALID_PADDING}. Received: {value}.')
    return padding


def get_dilated_filter_size(filter_size, dilation):
    return filter_size + (filter_size - 1) * (dilation - 1)


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.

  Args:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  dilated_filter_size = get_dilated_filter_size(filter_size, dilation) # avoids redundant computations.
  output_length = input_length if padding == 'same' else input_length - dilated_filter_size + 1 if padding == 'valid' else input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, padding, stride):
  """Determines input length of a convolution given output length.

  Args:
      output_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.

  Returns:
      The input length (integer).
  """
  if output_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  if padding == 'same':
    pad = filter_size // 2
  elif padding == 'valid':
    pad = 0
  elif padding == 'full':
    pad = filter_size - 1
  return (output_length - 1) * stride - 2 * pad + filter_size


def deconv_output_length(input_length, filter_size, padding, stride):
  """Determines output length of a transposed convolution given input length.

  Args:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  input_length *= stride
  if padding == 'valid':
    input_length += max(filter_size - stride, 0)
  elif padding == 'full':
    input_length -= (stride + filter_size - 2)
  return input_length


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if not callable(true_fn) or not callable(false_fn):
      raise TypeError("`true_fn` and `false_fn` must be callable.")
  if isinstance(pred, variables.Variable):
      return cond.cond(
          pred, true_fn=lambda: true_fn(), false_fn=lambda: false_fn(), name=name)
  return smart_module.smart_cond(
      pred, true_fn=lambda: true_fn(), false_fn=lambda: false_fn(), name=name)


def constant_value(pred):
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

    Args:
      pred: A scalar, either a Python bool or a TensorFlow boolean variable
        or tensor, or the Python integer 1 or 0.

    Returns:
      True or False if `pred` has a constant boolean value, None otherwise.

    Raises:
      TypeError: If `pred` is not a Variable, Tensor or bool, or Python
        integer 1 or 0.
    """
  # Allow integer booleans.
  if isinstance(pred, int):
      return bool(pred)  # Convert integer 1/0 to True/False
  if isinstance(pred, variables.Variable):
      return None  # Dynamic value, cannot infer constant
  return smart_module.smart_constant_value(pred)
