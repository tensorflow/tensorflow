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
# ==============================================================================
"""Utilities used by convolution layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin

from tensorflow.python.keras import backend


def convert_data_format(data_format, ndim):
  if data_format == 'channels_last':
    if ndim == 3:
      return 'NWC'
    elif ndim == 4:
      return 'NHWC'
    elif ndim == 5:
      return 'NDHWC'
    else:
      raise ValueError('Input rank not supported:', ndim)
  elif data_format == 'channels_first':
    if ndim == 3:
      return 'NCW'
    elif ndim == 4:
      return 'NCHW'
    elif ndim == 5:
      return 'NCDHW'
    else:
      raise ValueError('Input rank not supported:', ndim)
  else:
    raise ValueError('Invalid data_format:', data_format)


def normalize_tuple(value, n, name):
  """Transforms a single integer or iterable of integers into an integer tuple.

  Arguments:
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
  if isinstance(value, int):
    return (value,) * n
  else:
    try:
      value_tuple = tuple(value)
    except TypeError:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(n) + ' integers. Received: ' + str(value))
    if len(value_tuple) != n:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(n) + ' integers. Received: ' + str(value))
    for single_value in value_tuple:
      try:
        int(single_value)
      except (ValueError, TypeError):
        raise ValueError('The `' + name + '` argument must be a tuple of ' +
                         str(n) + ' integers. Received: ' + str(value) + ' '
                         'including element ' + str(single_value) + ' of type' +
                         ' ' + str(type(single_value)))
    return value_tuple


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.

  Arguments:
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
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding == 'same':
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, padding, stride):
  """Determines input length of a convolution given output length.

  Arguments:
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

  Arguments:
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


def normalize_data_format(value):
  if value is None:
    value = backend.image_data_format()
  data_format = value.lower()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('The `data_format` argument must be one of '
                     '"channels_first", "channels_last". Received: ' +
                     str(value))
  return data_format


def normalize_padding(value):
  padding = value.lower()
  if padding not in {'valid', 'same', 'causal'}:
    raise ValueError('The `padding` argument must be one of '
                     '"valid", "same" (or "causal", only for `Conv1D). '
                     'Received: ' + str(padding))
  return padding


def convert_kernel(kernel):
  """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

  Also works reciprocally, since the transformation is its own inverse.

  Arguments:
      kernel: Numpy array (3D, 4D or 5D).

  Returns:
      The converted kernel.

  Raises:
      ValueError: in case of invalid kernel shape or invalid data_format.
  """
  kernel = np.asarray(kernel)
  if not 3 <= kernel.ndim <= 5:
    raise ValueError('Invalid kernel shape:', kernel.shape)
  slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
  no_flip = (slice(None, None), slice(None, None))
  slices[-2:] = no_flip
  return np.copy(kernel[slices])
