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

import itertools
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
      padding: one of "same", "valid", "full", "causal"
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full', 'causal'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding in ['same', 'causal']:
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


def deconv_output_length(input_length, filter_size, padding,
                         output_padding=None, stride=0, dilation=1):
  """Determines output length of a transposed convolution given input length.

  Arguments:
      input_length: Integer.
      filter_size: Integer.
      padding: one of `"same"`, `"valid"`, `"full"`.
      output_padding: Integer, amount of padding along the output dimension.
          Can be set to `None` in which case the output length is inferred.
      stride: Integer.
      dilation: Integer.

  Returns:
      The output length (integer).
  """
  assert padding in {'same', 'valid', 'full'}
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'valid':
      length = input_length * stride + max(filter_size - stride, 0)
    elif padding == 'full':
      length = input_length * stride - (stride + filter_size - 2)
    elif padding == 'same':
      length = input_length * stride

  else:
    if padding == 'same':
      pad = filter_size // 2
    elif padding == 'valid':
      pad = 0
    elif padding == 'full':
      pad = filter_size - 1

    length = ((input_length - 1) * stride + filter_size - 2 * pad +
              output_padding)
  return length


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


def conv_kernel_mask(input_shape, kernel_shape, strides, padding):
  """Compute a mask representing the connectivity of a convolution operation.

  Assume a convolution with given parameters is applied to an input having N
  spatial dimensions with `input_shape = (d_in1, ..., d_inN)` to produce an
  output with shape `(d_out1, ..., d_outN)`. This method returns a boolean array
  of shape `(d_in1, ..., d_inN, d_out1, ..., d_outN)` with `True` entries
  indicating pairs of input and output locations that are connected by a weight.

  Example:
    ```python
        >>> input_shape = (4,)
        >>> kernel_shape = (2,)
        >>> strides = (1,)
        >>> padding = "valid"
        >>> conv_kernel_mask(input_shape, kernel_shape, strides, padding)
        array([[ True, False, False],
               [ True,  True, False],
               [False,  True,  True],
               [False, False,  True]], dtype=bool)
    ```
    where rows and columns correspond to inputs and outputs respectively.


  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`,
                 spatial shape of the input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel
                  / receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.

  Returns:
    A boolean 2N-D `np.ndarray` of shape
    `(d_in1, ..., d_inN, d_out1, ..., d_outN)`, where `(d_out1, ..., d_outN)`
    is the spatial shape of the output. `True` entries in the mask represent
    pairs of input-output locations that are connected by a weight.

  Raises:
    ValueError: if `input_shape`, `kernel_shape` and `strides` don't have the
        same number of dimensions.
    NotImplementedError: if `padding` is not in {`"same"`, `"valid"`}.
  """
  if padding not in {'same', 'valid'}:
    raise NotImplementedError('Padding type %s not supported. '
                              'Only "valid" and "same" '
                              'are implemented.' % padding)

  in_dims = len(input_shape)
  if isinstance(kernel_shape, int):
    kernel_shape = (kernel_shape,) * in_dims
  if isinstance(strides, int):
    strides = (strides,) * in_dims

  kernel_dims = len(kernel_shape)
  stride_dims = len(strides)
  if kernel_dims != in_dims or stride_dims != in_dims:
    raise ValueError('Number of strides, input and kernel dimensions must all '
                     'match. Received: %d, %d, %d.' %
                     (stride_dims, in_dims, kernel_dims))

  output_shape = conv_output_shape(input_shape, kernel_shape, strides, padding)

  mask_shape = input_shape + output_shape
  mask = np.zeros(mask_shape, np.bool)

  output_axes_ticks = [range(dim) for dim in output_shape]
  for output_position in itertools.product(*output_axes_ticks):
    input_axes_ticks = conv_connected_inputs(input_shape,
                                             kernel_shape,
                                             output_position,
                                             strides,
                                             padding)
    for input_position in itertools.product(*input_axes_ticks):
      mask[input_position + output_position] = True

  return mask


def conv_connected_inputs(input_shape,
                          kernel_shape,
                          output_position,
                          strides,
                          padding):
  """Return locations of the input connected to an output position.

  Assume a convolution with given parameters is applied to an input having N
  spatial dimensions with `input_shape = (d_in1, ..., d_inN)`. This method
  returns N ranges specifying the input region that was convolved with the
  kernel to produce the output at position
  `output_position = (p_out1, ..., p_outN)`.

  Example:
    ```python
        >>> input_shape = (4, 4)
        >>> kernel_shape = (2, 1)
        >>> output_position = (1, 1)
        >>> strides = (1, 1)
        >>> padding = "valid"
        >>> conv_connected_inputs(input_shape, kernel_shape, output_position,
        >>>                       strides, padding)
        [xrange(1, 3), xrange(1, 2)]
    ```
  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`,
                 spatial shape of the input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel
                  / receptive field.
    output_position: tuple of size N: `(p_out1, ..., p_outN)`,
                     a single position in the output of the convolution.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.

  Returns:
    N ranges `[[p_in_left1, ..., p_in_right1], ...,
              [p_in_leftN, ..., p_in_rightN]]` specifying the region in the
    input connected to output_position.
  """
  ranges = []

  ndims = len(input_shape)
  for d in range(ndims):
    left_shift = int(kernel_shape[d] / 2)
    right_shift = kernel_shape[d] - left_shift

    center = output_position[d] * strides[d]

    if padding == 'valid':
      center += left_shift

    start = max(0, center - left_shift)
    end = min(input_shape[d], center + right_shift)

    ranges.append(range(start, end))

  return ranges


def conv_output_shape(input_shape, kernel_shape, strides, padding):
  """Return the output shape of an N-D convolution.

  Forces dimensions where input is empty (size 0) to remain empty.

  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`,
                 spatial shape of the input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel
                  / receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.

  Returns:
    tuple of size N: `(d_out1, ..., d_outN)`, spatial shape of the output.
  """
  dims = range(len(kernel_shape))
  output_shape = [conv_output_length(input_shape[d],
                                     kernel_shape[d],
                                     padding,
                                     strides[d])
                  for d in dims]
  output_shape = tuple([0 if input_shape[d] == 0 else output_shape[d]
                        for d in dims])
  return output_shape
