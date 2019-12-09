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
"""Wrappers for primitive Neural Net (NN) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numbers
import os

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import device_context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

from tensorflow.python.util.tf_export import tf_export

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn

# pylint: disable=protected-access


def _get_sequence(value, n, channel_index, name):
  """Formats a value input for gen_nn_ops."""
  if value is None:
    value = [1]
  elif not isinstance(value, collections_abc.Sized):
    value = [value]

  current_n = len(value)
  if current_n == n + 2:
    return value
  elif current_n == 1:
    value = list((value[0],) * n)
  elif current_n == n:
    value = list(value)
  else:
    raise ValueError("{} should be of length 1, {} or {} but was {}".format(
        name, n, n + 2, current_n))

  if channel_index == 1:
    return [1, 1] + value
  else:
    return [1] + value + [1]


def _non_atrous_convolution(
    input,  # pylint: disable=redefined-builtin
    filter,  # pylint: disable=redefined-builtin
    padding,
    data_format=None,  # pylint: disable=redefined-builtin
    strides=None,
    name=None):
  """Computes sums of N-D convolutions (actually cross correlation).

  It is required that 1 <= N <= 3.

  This is used to implement the more generic `convolution` function, which
  extends the interface of this function with a `dilation_rate` parameter.

  Args:

    input: Rank N+2 tensor of type T of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if `data_format`
      does not start with `"NC"`, or
      `[batch_size, in_channels] + input_spatial_shape` if `data_format` starts
      with `"NC"`.
    filter: Rank N+2 tensor of type T of shape
      `filter_spatial_shape + [in_channels, out_channels]`.  Rank of either
      `input` or `filter` must be known.
    padding: Padding method to use, must be either "VALID" or "SAME".
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    strides: Sequence of N positive integers, defaults to `[1] * N`.
    name: Name prefix to use.

  Returns:
    Rank N+2 tensor of type T of shape
    `[batch_size] + output_spatial_shape + [out_channels]`, where
    if padding == "SAME":
      output_spatial_shape = input_spatial_shape
    if padding == "VALID":
      output_spatial_shape = input_spatial_shape - filter_spatial_shape + 1.

  Raises:
    ValueError: if ranks are incompatible.

  """
  with ops.name_scope(name, "non_atrous_convolution", [input, filter]) as scope:
    input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin
    input_shape = input.get_shape()
    filter = ops.convert_to_tensor(filter, name="filter")  # pylint: disable=redefined-builtin
    filter_shape = filter.get_shape()
    op = _NonAtrousConvolution(
        input_shape,
        filter_shape=filter_shape,
        padding=padding,
        data_format=data_format,
        strides=strides,
        name=scope)
    return op(input, filter)


class _NonAtrousConvolution(object):
  """Helper class for _non_atrous_convolution.

  Note that this class assumes that shapes of input and filter passed to
  __call__ are compatible with input_shape and filter_shape passed to the
  constructor.

  Arguments:
    input_shape: static input shape, i.e. input.get_shape().
    filter_shape: static filter shape, i.e. filter.get_shape().
    padding: see _non_atrous_convolution.
    data_format: see _non_atrous_convolution.
    strides: see _non_atrous_convolution.
    name: see _non_atrous_convolution.
  """

  def __init__(
      self,
      input_shape,
      filter_shape,  # pylint: disable=redefined-builtin
      padding,
      data_format=None,
      strides=None,
      name=None):
    filter_shape = filter_shape.with_rank(input_shape.ndims)
    self.padding = padding
    self.name = name
    input_shape = input_shape.with_rank(filter_shape.ndims)
    if input_shape.ndims is None:
      raise ValueError("Rank of convolution must be known")
    if input_shape.ndims < 3 or input_shape.ndims > 5:
      raise ValueError(
          "`input` and `filter` must have rank at least 3 and at most 5")
    conv_dims = input_shape.ndims - 2
    if strides is None:
      strides = [1] * conv_dims
    elif len(strides) != conv_dims:
      raise ValueError("len(strides)=%d, but should be %d" % (len(strides),
                                                              conv_dims))
    if conv_dims == 1:
      # conv1d uses the 2-d data format names
      if data_format is None:
        data_format = "NWC"
      elif data_format not in {"NCW", "NWC", "NCHW", "NHWC"}:
        raise ValueError("data_format must be \"NWC\" or \"NCW\".")
      self.strides = strides[0]
      self.data_format = data_format
      self.conv_op = self._conv1d
    elif conv_dims == 2:
      if data_format is None or data_format == "NHWC":
        data_format = "NHWC"
        strides = [1] + list(strides) + [1]
      elif data_format == "NCHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
      self.strides = strides
      self.data_format = data_format
      self.conv_op = conv2d
    elif conv_dims == 3:
      if data_format is None or data_format == "NDHWC":
        strides = [1] + list(strides) + [1]
      elif data_format == "NCDHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NDHWC\" or \"NCDHW\". Have: %s"
                         % data_format)
      self.strides = strides
      self.data_format = data_format
      self.conv_op = gen_nn_ops.conv3d

  # Note that we need this adapter since argument names for conv1d don't match
  # those for gen_nn_ops.conv2d and gen_nn_ops.conv3d.
  # pylint: disable=redefined-builtin
  def _conv1d(self, input, filter, strides, padding, data_format, name):
    return conv1d(
        value=input,
        filters=filter,
        stride=strides,
        padding=padding,
        data_format=data_format,
        name=name)

  # pylint: enable=redefined-builtin

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.conv_op(
        input=inp,
        filter=filter,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        name=self.name)


@tf_export("nn.dilation2d", v1=[])
def dilation2d_v2(
    input,   # pylint: disable=redefined-builtin
    filters,  # pylint: disable=redefined-builtin
    strides,
    padding,
    data_format,
    dilations,
    name=None):
  """Computes the grayscale dilation of 4-D `input` and 3-D `filters` tensors.

  The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filters` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
  input channel is processed independently of the others with its own
  structuring function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
  tensor depend on the `padding` algorithm. We currently only support the
  default "NHWC" `data_format`.

  In detail, the grayscale morphological 2-D dilation is the max-sum correlation
  (for consistency with `conv2d`, we use unmirrored filters):

      output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filters[dy, dx, c]

  Max-pooling is a special case when the filter has size equal to the pooling
  kernel size and contains all zeros.

  Note on duality: The dilation of `input` by the `filters` is equal to the
  negation of the erosion of `-input` by the reflected `filters`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`,
      `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filters: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input
      tensor. Must be: `[1, stride_height, stride_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: A `string`, only `"NHWC"` is currently supported.
    dilations: A list of `ints` that has length `>= 4`.
      The input stride for atrous morphological dilation. Must be:
      `[1, rate_height, rate_width, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  if data_format != "NHWC":
    raise ValueError("Data formats other than NHWC are not yet supported")

  return gen_nn_ops.dilation2d(input=input,
                               filter=filters,
                               strides=strides,
                               rates=dilations,
                               padding=padding,
                               name=name)


@tf_export(v1=["nn.dilation2d"])
def dilation2d_v1(  # pylint: disable=missing-docstring
    input,  # pylint: disable=redefined-builtin
    filter=None,  # pylint: disable=redefined-builtin
    strides=None,
    rates=None,
    padding=None,
    name=None,
    filters=None,
    dilations=None):
  filter = deprecated_argument_lookup("filters", filters, "filter", filter)
  rates = deprecated_argument_lookup("dilations", dilations, "rates", rates)
  return gen_nn_ops.dilation2d(input, filter, strides, rates, padding, name)


dilation2d_v1.__doc__ = gen_nn_ops.dilation2d.__doc__


@tf_export("nn.with_space_to_batch")
def with_space_to_batch(
    input,  # pylint: disable=redefined-builtin
    dilation_rate,
    padding,
    op,
    filter_shape=None,
    spatial_dims=None,
    data_format=None):
  """Performs `op` on the space-to-batch representation of `input`.

  This has the effect of transforming sliding window operations into the
  corresponding "atrous" operation in which the input is sampled at the
  specified `dilation_rate`.

  In the special case that `dilation_rate` is uniformly 1, this simply returns:

    op(input, num_spatial_dims, padding)

  Otherwise, it returns:

    batch_to_space_nd(
      op(space_to_batch_nd(input, adjusted_dilation_rate, adjusted_paddings),
         num_spatial_dims,
         "VALID")
      adjusted_dilation_rate,
      adjusted_crops),

  where:

    adjusted_dilation_rate is an int64 tensor of shape [max(spatial_dims)],
    adjusted_{paddings,crops} are int64 tensors of shape [max(spatial_dims), 2]

  defined as follows:

  We first define two int64 tensors `paddings` and `crops` of shape
  `[num_spatial_dims, 2]` based on the value of `padding` and the spatial
  dimensions of the `input`:

  If `padding = "VALID"`, then:

    paddings, crops = required_space_to_batch_paddings(
      input_shape[spatial_dims],
      dilation_rate)

  If `padding = "SAME"`, then:

    dilated_filter_shape =
      filter_shape + (filter_shape - 1) * (dilation_rate - 1)

    paddings, crops = required_space_to_batch_paddings(
      input_shape[spatial_dims],
      dilation_rate,
      [(dilated_filter_shape - 1) // 2,
       dilated_filter_shape - 1 - (dilated_filter_shape - 1) // 2])

  Because `space_to_batch_nd` and `batch_to_space_nd` assume that the spatial
  dimensions are contiguous starting at the second dimension, but the specified
  `spatial_dims` may not be, we must adjust `dilation_rate`, `paddings` and
  `crops` in order to be usable with these operations.  For a given dimension,
  if the block size is 1, and both the starting and ending padding and crop
  amounts are 0, then space_to_batch_nd effectively leaves that dimension alone,
  which is what is needed for dimensions not part of `spatial_dims`.
  Furthermore, `space_to_batch_nd` and `batch_to_space_nd` handle this case
  efficiently for any number of leading and trailing dimensions.

  For 0 <= i < len(spatial_dims), we assign:

    adjusted_dilation_rate[spatial_dims[i] - 1] = dilation_rate[i]
    adjusted_paddings[spatial_dims[i] - 1, :] = paddings[i, :]
    adjusted_crops[spatial_dims[i] - 1, :] = crops[i, :]

  All unassigned values of `adjusted_dilation_rate` default to 1, while all
  unassigned values of `adjusted_paddings` and `adjusted_crops` default to 0.

  Note in the case that `dilation_rate` is not uniformly 1, specifying "VALID"
  padding is equivalent to specifying `padding = "SAME"` with a filter_shape of
  `[1]*N`.

  Advanced usage. Note the following optimization: A sequence of
  `with_space_to_batch` operations with identical (not uniformly 1)
  `dilation_rate` parameters and "VALID" padding

    net = with_space_to_batch(net, dilation_rate, "VALID", op_1)
    ...
    net = with_space_to_batch(net, dilation_rate, "VALID", op_k)

  can be combined into a single `with_space_to_batch` operation as follows:

    def combined_op(converted_input, num_spatial_dims, _):
      result = op_1(converted_input, num_spatial_dims, "VALID")
      ...
      result = op_k(result, num_spatial_dims, "VALID")

    net = with_space_to_batch(net, dilation_rate, "VALID", combined_op)

  This eliminates the overhead of `k-1` calls to `space_to_batch_nd` and
  `batch_to_space_nd`.

  Similarly, a sequence of `with_space_to_batch` operations with identical (not
  uniformly 1) `dilation_rate` parameters, "SAME" padding, and odd filter
  dimensions

    net = with_space_to_batch(net, dilation_rate, "SAME", op_1, filter_shape_1)
    ...
    net = with_space_to_batch(net, dilation_rate, "SAME", op_k, filter_shape_k)

  can be combined into a single `with_space_to_batch` operation as follows:

    def combined_op(converted_input, num_spatial_dims, _):
      result = op_1(converted_input, num_spatial_dims, "SAME")
      ...
      result = op_k(result, num_spatial_dims, "SAME")

    net = with_space_to_batch(net, dilation_rate, "VALID", combined_op)

  Args:
    input: Tensor of rank > max(spatial_dims).
    dilation_rate: int32 Tensor of *known* shape [num_spatial_dims].
    padding: str constant equal to "VALID" or "SAME"
    op: Function that maps (input, num_spatial_dims, padding) -> output
    filter_shape: If padding = "SAME", specifies the shape of the convolution
      kernel/pooling window as an integer Tensor of shape [>=num_spatial_dims].
      If padding = "VALID", filter_shape is ignored and need not be specified.
    spatial_dims: Monotonically increasing sequence of `num_spatial_dims`
      integers (which are >= 1) specifying the spatial dimensions of `input`
      and output.  Defaults to: `range(1, num_spatial_dims+1)`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".

  Returns:
    The output Tensor as described above, dimensions will vary based on the op
    provided.

  Raises:
    ValueError: if `padding` is invalid or the arguments are incompatible.
    ValueError: if `spatial_dims` are invalid.

  """
  input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin
  input_shape = input.get_shape()

  def build_op(num_spatial_dims, padding):
    return lambda inp, _: op(inp, num_spatial_dims, padding)

  new_op = _WithSpaceToBatch(
      input_shape,
      dilation_rate,
      padding,
      build_op,
      filter_shape=filter_shape,
      spatial_dims=spatial_dims,
      data_format=data_format)
  return new_op(input, None)


class _WithSpaceToBatch(object):
  """Helper class for with_space_to_batch.

  Note that this class assumes that shapes of input and filter passed to
  __call__ are compatible with input_shape and filter_shape passed to the
  constructor.

  Arguments
    input_shape: static shape of input. i.e. input.get_shape().
    dilation_rate: see with_space_to_batch
    padding: see with_space_to_batch
    build_op: Function that maps (num_spatial_dims, paddings) -> (function that
      maps (input, filter) -> output).
    filter_shape: see with_space_to_batch
    spatial_dims: see with_space_to_batch
    data_format: see with_space_to_batch
  """

  def __init__(self,
               input_shape,
               dilation_rate,
               padding,
               build_op,
               filter_shape=None,
               spatial_dims=None,
               data_format=None):
    """Helper class for _with_space_to_batch."""
    dilation_rate = ops.convert_to_tensor(
        dilation_rate, dtypes.int32, name="dilation_rate")
    try:
      rate_shape = dilation_rate.get_shape().with_rank(1)
    except ValueError:
      raise ValueError("rate must be rank 1")

    if not dilation_rate.get_shape().is_fully_defined():
      raise ValueError("rate must have known shape")

    num_spatial_dims = rate_shape.dims[0].value

    if data_format is not None and data_format.startswith("NC"):
      starting_spatial_dim = 2
    else:
      starting_spatial_dim = 1

    if spatial_dims is None:
      spatial_dims = range(starting_spatial_dim,
                           num_spatial_dims + starting_spatial_dim)
    orig_spatial_dims = list(spatial_dims)
    spatial_dims = sorted(set(int(x) for x in orig_spatial_dims))
    if spatial_dims != orig_spatial_dims or any(x < 1 for x in spatial_dims):
      raise ValueError(
          "spatial_dims must be a montonically increasing sequence of positive "
          "integers")

    if data_format is not None and data_format.startswith("NC"):
      expected_input_rank = spatial_dims[-1]
    else:
      expected_input_rank = spatial_dims[-1] + 1

    try:
      input_shape.with_rank_at_least(expected_input_rank)
    except ValueError:
      raise ValueError(
          "input tensor must have rank %d at least" % (expected_input_rank))

    const_rate = tensor_util.constant_value(dilation_rate)
    rate_or_const_rate = dilation_rate
    if const_rate is not None:
      rate_or_const_rate = const_rate
      if np.any(const_rate < 1):
        raise ValueError("dilation_rate must be positive")
      if np.all(const_rate == 1):
        self.call = build_op(num_spatial_dims, padding)
        return

    # We have two padding contributions. The first is used for converting "SAME"
    # to "VALID". The second is required so that the height and width of the
    # zero-padded value tensor are multiples of rate.

    # Padding required to reduce to "VALID" convolution
    if padding == "SAME":
      if filter_shape is None:
        raise ValueError("filter_shape must be specified for SAME padding")
      filter_shape = ops.convert_to_tensor(filter_shape, name="filter_shape")
      const_filter_shape = tensor_util.constant_value(filter_shape)
      if const_filter_shape is not None:
        filter_shape = const_filter_shape
        self.base_paddings = _with_space_to_batch_base_paddings(
            const_filter_shape, num_spatial_dims, rate_or_const_rate)
      else:
        self.num_spatial_dims = num_spatial_dims
        self.rate_or_const_rate = rate_or_const_rate
        self.base_paddings = None
    elif padding == "VALID":
      self.base_paddings = np.zeros([num_spatial_dims, 2], np.int32)
    else:
      raise ValueError("Invalid padding method %r" % padding)

    self.input_shape = input_shape
    self.spatial_dims = spatial_dims
    self.dilation_rate = dilation_rate
    self.data_format = data_format
    self.op = build_op(num_spatial_dims, "VALID")
    self.call = self._with_space_to_batch_call

  def _with_space_to_batch_call(self, inp, filter):  # pylint: disable=redefined-builtin
    """Call functionality for with_space_to_batch."""
    # Handle input whose shape is unknown during graph creation.
    input_spatial_shape = None
    input_shape = self.input_shape
    spatial_dims = self.spatial_dims
    if input_shape.ndims is not None:
      input_shape_list = input_shape.as_list()
      input_spatial_shape = [input_shape_list[i] for i in spatial_dims]
    if input_spatial_shape is None or None in input_spatial_shape:
      input_shape_tensor = array_ops.shape(inp)
      input_spatial_shape = array_ops.stack(
          [input_shape_tensor[i] for i in spatial_dims])

    base_paddings = self.base_paddings
    if base_paddings is None:
      # base_paddings could not be computed at build time since static filter
      # shape was not fully defined.
      filter_shape = array_ops.shape(filter)
      base_paddings = _with_space_to_batch_base_paddings(
          filter_shape, self.num_spatial_dims, self.rate_or_const_rate)
    paddings, crops = array_ops.required_space_to_batch_paddings(
        input_shape=input_spatial_shape,
        base_paddings=base_paddings,
        block_shape=self.dilation_rate)

    dilation_rate = _with_space_to_batch_adjust(self.dilation_rate, 1,
                                                spatial_dims)
    paddings = _with_space_to_batch_adjust(paddings, 0, spatial_dims)
    crops = _with_space_to_batch_adjust(crops, 0, spatial_dims)
    input_converted = array_ops.space_to_batch_nd(
        input=inp, block_shape=dilation_rate, paddings=paddings)

    result = self.op(input_converted, filter)

    result_converted = array_ops.batch_to_space_nd(
        input=result, block_shape=dilation_rate, crops=crops)

    # Recover channel information for output shape if channels are not last.
    if self.data_format is not None and self.data_format.startswith("NC"):
      if not result_converted.shape.dims[1].value and filter is not None:
        output_shape = result_converted.shape.as_list()
        output_shape[1] = filter.shape[-1]
        result_converted.set_shape(output_shape)

    return result_converted

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.call(inp, filter)


def _with_space_to_batch_base_paddings(filter_shape, num_spatial_dims,
                                       rate_or_const_rate):
  """Helper function to compute base_paddings."""
  # Spatial dimensions of the filters and the upsampled filters in which we
  # introduce (rate - 1) zeros between consecutive filter values.
  filter_spatial_shape = filter_shape[:num_spatial_dims]
  dilated_filter_spatial_shape = (
      filter_spatial_shape + (filter_spatial_shape - 1) *
      (rate_or_const_rate - 1))
  pad_extra_shape = dilated_filter_spatial_shape - 1

  # When full_padding_shape is odd, we pad more at end, following the same
  # convention as conv2d.
  pad_extra_start = pad_extra_shape // 2
  pad_extra_end = pad_extra_shape - pad_extra_start
  base_paddings = array_ops.stack(
      [[pad_extra_start[i], pad_extra_end[i]] for i in range(num_spatial_dims)])
  return base_paddings


def _with_space_to_batch_adjust(orig, fill_value, spatial_dims):
  """Returns an `adjusted` version of `orig` based on `spatial_dims`.

  Tensor of the same type as `orig` and with shape
  `[max(spatial_dims), ...]` where:

    adjusted[spatial_dims[i] - 1, ...] = orig[i, ...]

  for 0 <= i < len(spatial_dims), and

    adjusted[j, ...] = fill_value

  for j != spatial_dims[i] - 1 for some i.

  If `orig` is a constant value, then the result will be a constant value.

  Args:
    orig: Tensor of rank > max(spatial_dims).
    fill_value: Numpy scalar (of same data type as `orig) specifying the fill
      value for non-spatial dimensions.
    spatial_dims: See with_space_to_batch.

  Returns:
    `adjusted` tensor.
  """
  fill_dims = orig.get_shape().as_list()[1:]
  dtype = orig.dtype.as_numpy_dtype
  parts = []
  const_orig = tensor_util.constant_value(orig)
  const_or_orig = const_orig if const_orig is not None else orig
  prev_spatial_dim = 0
  i = 0
  while i < len(spatial_dims):
    start_i = i
    start_spatial_dim = spatial_dims[i]
    if start_spatial_dim > 1:
      # Fill in any gap from the previous spatial dimension (or dimension 1 if
      # this is the first spatial dimension) with `fill_value`.
      parts.append(
          np.full(
              [start_spatial_dim - 1 - prev_spatial_dim] + fill_dims,
              fill_value,
              dtype=dtype))
    # Find the largest value of i such that:
    #   [spatial_dims[start_i], ..., spatial_dims[i]]
    #     == [start_spatial_dim, ..., start_spatial_dim + i - start_i],
    # i.e. the end of a contiguous group of spatial dimensions.
    while (i + 1 < len(spatial_dims) and
           spatial_dims[i + 1] == spatial_dims[i] + 1):
      i += 1
    parts.append(const_or_orig[start_i:i + 1])
    prev_spatial_dim = spatial_dims[i]
    i += 1
  if const_orig is not None:
    return np.concatenate(parts)
  else:
    return array_ops.concat(parts, 0)


def _get_strides_and_dilation_rate(num_spatial_dims, strides, dilation_rate):
  """Helper function for verifying strides and dilation_rate arguments.

  This is used by `convolution` and `pool`.

  Args:
    num_spatial_dims: int
    strides: Optional.  List of N ints >= 1.  Defaults to [1]*N.  If any value
      of strides is > 1, then all values of dilation_rate must be 1.
    dilation_rate: Optional.  List of N ints >= 1.  Defaults to [1]*N.  If any
      value of dilation_rate is > 1, then all values of strides must be 1.

  Returns:
    Normalized (strides, dilation_rate) as int32 numpy arrays of shape
    [num_spatial_dims].

  Raises:
    ValueError: if the parameters are invalid.
  """
  if dilation_rate is None:
    dilation_rate = [1] * num_spatial_dims
  elif len(dilation_rate) != num_spatial_dims:
    raise ValueError("len(dilation_rate)=%d but should be %d" %
                     (len(dilation_rate), num_spatial_dims))
  dilation_rate = np.array(dilation_rate, dtype=np.int32)
  if np.any(dilation_rate < 1):
    raise ValueError("all values of dilation_rate must be positive")

  if strides is None:
    strides = [1] * num_spatial_dims
  elif len(strides) != num_spatial_dims:
    raise ValueError("len(strides)=%d but should be %d" % (len(strides),
                                                           num_spatial_dims))
  strides = np.array(strides, dtype=np.int32)
  if np.any(strides < 1):
    raise ValueError("all values of strides must be positive")

  if np.any(strides > 1) and np.any(dilation_rate > 1):
    raise ValueError(
        "strides > 1 not supported in conjunction with dilation_rate > 1")
  return strides, dilation_rate


@tf_export(v1=["nn.convolution"])
def convolution(
    input,  # pylint: disable=redefined-builtin
    filter,  # pylint: disable=redefined-builtin
    padding,
    strides=None,
    dilation_rate=None,
    name=None,
    data_format=None,
    filters=None,
    dilations=None):
  """Computes sums of N-D convolutions (actually cross-correlation).

  This also supports either output striding via the optional `strides` parameter
  or atrous convolution (also known as convolution with holes or dilated
  convolution, based on the French word "trous" meaning holes in English) via
  the optional `dilation_rate` parameter.  Currently, however, output striding
  is not supported for atrous convolutions.

  Specifically, in the case that `data_format` does not start with "NC", given
  a rank (N+2) `input` Tensor of shape

    [num_batches,
     input_spatial_shape[0],
     ...,
     input_spatial_shape[N-1],
     num_input_channels],

  a rank (N+2) `filter` Tensor of shape

    [spatial_filter_shape[0],
     ...,
     spatial_filter_shape[N-1],
     num_input_channels,
     num_output_channels],

  an optional `dilation_rate` tensor of shape [N] (defaulting to [1]*N)
  specifying the filter upsampling/input downsampling rate, and an optional list
  of N `strides` (defaulting [1]*N), this computes for each N-D spatial output
  position (x[0], ..., x[N-1]):

  ```
    output[b, x[0], ..., x[N-1], k] =
        sum_{z[0], ..., z[N-1], q}
            filter[z[0], ..., z[N-1], q, k] *
            padded_input[b,
                         x[0]*strides[0] + dilation_rate[0]*z[0],
                         ...,
                         x[N-1]*strides[N-1] + dilation_rate[N-1]*z[N-1],
                         q]
  ```
  where b is the index into the batch, k is the output channel number, q is the
  input channel number, and z is the N-D spatial offset within the filter. Here,
  `padded_input` is obtained by zero padding the input using an effective
  spatial filter shape of `(spatial_filter_shape-1) * dilation_rate + 1` and
  output striding `strides` as described in the
  [comment here](https://tensorflow.org/api_guides/python/nn#Convolution).

  In the case that `data_format` does start with `"NC"`, the `input` and output
  (but not the `filter`) are simply transposed as follows:

    convolution(input, data_format, **kwargs) =
      tf.transpose(convolution(tf.transpose(input, [0] + range(2,N+2) + [1]),
                               **kwargs),
                   [0, N+1] + range(1, N+1))

  It is required that 1 <= N <= 3.

  Args:
    input: An (N+2)-D `Tensor` of type `T`, of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    filter: An (N+2)-D `Tensor` with the same type as `input` and shape
      `spatial_filter_shape + [in_channels, out_channels]`.
    padding: A string, either `"VALID"` or `"SAME"`. The padding algorithm.
    strides: Optional.  Sequence of N ints >= 1.  Specifies the output stride.
      Defaults to [1]*N.  If any value of strides is > 1, then all values of
      dilation_rate must be 1.
    dilation_rate: Optional.  Sequence of N ints >= 1.  Specifies the filter
      upsampling/input downsampling rate.  In the literature, the same parameter
      is sometimes called `input stride` or `dilation`.  The effective filter
      size used for the convolution will be `spatial_filter_shape +
      (spatial_filter_shape - 1) * (rate - 1)`, obtained by inserting
      (dilation_rate[i]-1) zeros between consecutive elements of the original
      filter in each spatial dimension i.  If any value of dilation_rate is > 1,
      then all values of strides must be 1.
    name: Optional name for the returned tensor.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    filters: Alias of filter.
    dilations: Alias of dilation_rate.

  Returns:
    A `Tensor` with the same type as `input` of shape

        `[batch_size] + output_spatial_shape + [out_channels]`

    if data_format is None or does not start with "NC", or

        `[batch_size, out_channels] + output_spatial_shape`

    if data_format starts with "NC",
    where `output_spatial_shape` depends on the value of `padding`.

    If padding == "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])

    If padding == "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] -
              (spatial_filter_shape[i]-1) * dilation_rate[i])
             / strides[i]).

  Raises:
    ValueError: If input/output depth does not match `filter` shape, if padding
      is other than `"VALID"` or `"SAME"`, or if data_format is invalid.

  """
  filter = deprecated_argument_lookup("filters", filters, "filter", filter)
  dilation_rate = deprecated_argument_lookup(
      "dilations", dilations, "dilation_rate", dilation_rate)
  return convolution_internal(
      input,
      filter,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilations=dilation_rate,
      name=name)


@tf_export("nn.convolution", v1=[])
def convolution_v2(
    input,  # pylint: disable=redefined-builtin
    filters,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None):
  return convolution_internal(
      input,  # pylint: disable=redefined-builtin
      filters,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilations=dilations,
      name=name)


convolution_v2.__doc__ = deprecation.rewrite_argument_docstring(
    deprecation.rewrite_argument_docstring(
        convolution.__doc__, "dilation_rate", "dilations"),
    "filter", "filters")


def convolution_internal(
    input,  # pylint: disable=redefined-builtin
    filters,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None,
    call_from_convolution=True):
  """Internal function which performs rank agnostic convolution."""
  if isinstance(input.shape, tensor_shape.TensorShape) and \
        input.shape.rank is not None:
    n = len(input.shape) - 2
  elif not isinstance(input.shape, tensor_shape.TensorShape) and \
        input.shape is not None:
    n = len(input.shape) - 2
  elif isinstance(filters.shape, tensor_shape.TensorShape) and \
        filters.shape.rank is not None:
    n = len(filters.shape) - 2
  elif not isinstance(filters.shape, tensor_shape.TensorShape) and \
        filters.shape is not None:
    n = len(filters.shape) - 2
  else:
    raise ValueError("rank of input or filter must be known")

  if not 1 <= n <= 3:
    raise ValueError(
        "Input tensor must be of rank 3, 4 or 5 but was {}.".format(n + 2))

  if data_format is None:
    channel_index = n + 1
  else:
    channel_index = 1 if data_format.startswith("NC") else n + 1

  strides = _get_sequence(strides, n, channel_index, "strides")
  dilations = _get_sequence(dilations, n, channel_index, "dilations")

  scopes = {1: "conv1d", 2: "Conv2D", 3: "Conv3D"}
  if not call_from_convolution and device_context.enclosing_tpu_context(
  ) is not None:
    scope = scopes[n]
  else:
    scope = "convolution"

  with ops.name_scope(name, scope, [input, filters]) as name:
    conv_ops = {1: conv1d, 2: gen_nn_ops.conv2d, 3: gen_nn_ops.conv3d}

    if device_context.enclosing_tpu_context() is not None or all(
        i == 1 for i in dilations):
      # fast path for TPU or if no dilation as gradient only supported on GPU
      # for dilations
      op = conv_ops[n]
      return op(
          input,
          filters,
          strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations,
          name=name)
    else:
      if channel_index == 1:
        strides = strides[2:]
        dilations = dilations[2:]
      else:
        strides = strides[1:-1]
        dilations = dilations[1:-1]

      op = Convolution(
          tensor_shape.as_shape(input.shape),
          tensor_shape.as_shape(filters.shape),
          padding,
          strides=strides,
          dilation_rate=dilations,
          name=name,
          data_format=data_format)
      return op(input, filters)


class Convolution(object):
  """Helper class for convolution.

  Note that this class assumes that shapes of input and filter passed to
  __call__ are compatible with input_shape and filter_shape passed to the
  constructor.

  Arguments
    input_shape: static shape of input. i.e. input.get_shape().
    filter_shape: static shape of the filter. i.e. filter.get_shape().
    padding:  see convolution.
    strides: see convolution.
    dilation_rate: see convolution.
    name: see convolution.
    data_format: see convolution.
  """

  def __init__(self,
               input_shape,
               filter_shape,
               padding,
               strides=None,
               dilation_rate=None,
               name=None,
               data_format=None):
    """Helper function for convolution."""
    num_total_dims = filter_shape.ndims
    if num_total_dims is None:
      num_total_dims = input_shape.ndims
    if num_total_dims is None:
      raise ValueError("rank of input or filter must be known")

    num_spatial_dims = num_total_dims - 2

    try:
      input_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      raise ValueError(
          "input tensor must have rank %d" % (num_spatial_dims + 2))

    try:
      filter_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      raise ValueError(
          "filter tensor must have rank %d" % (num_spatial_dims + 2))

    if data_format is None or not data_format.startswith("NC"):
      input_channels_dim = tensor_shape.dimension_at_index(
          input_shape, num_spatial_dims + 1)
      spatial_dims = range(1, num_spatial_dims + 1)
    else:
      input_channels_dim = tensor_shape.dimension_at_index(input_shape, 1)
      spatial_dims = range(2, num_spatial_dims + 2)

    if not input_channels_dim.is_compatible_with(
        filter_shape[num_spatial_dims]):
      raise ValueError(
          "number of input channels does not match corresponding dimension of "
          "filter, {} != {}".format(input_channels_dim,
                                    filter_shape[num_spatial_dims]))

    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.data_format = data_format
    self.strides = strides
    self.padding = padding
    self.name = name
    self.dilation_rate = dilation_rate
    self.conv_op = _WithSpaceToBatch(
        input_shape,
        dilation_rate=dilation_rate,
        padding=padding,
        build_op=self._build_op,
        filter_shape=filter_shape,
        spatial_dims=spatial_dims,
        data_format=data_format)

  def _build_op(self, _, padding):
    return _NonAtrousConvolution(
        self.input_shape,
        filter_shape=self.filter_shape,
        padding=padding,
        data_format=self.data_format,
        strides=self.strides,
        name=self.name)

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    # TPU convolution supports dilations greater than 1.
    if device_context.enclosing_tpu_context() is not None:
      return convolution_internal(
          inp,
          filter,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          dilations=self.dilation_rate,
          name=self.name,
          call_from_convolution=False)
    else:
      return self.conv_op(inp, filter)


@tf_export(v1=["nn.pool"])
def pool(
    input,  # pylint: disable=redefined-builtin
    window_shape,
    pooling_type,
    padding,
    dilation_rate=None,
    strides=None,
    name=None,
    data_format=None,
    dilations=None):
  """Performs an N-D pooling operation.

  In the case that `data_format` does not start with "NC", computes for
      0 <= b < batch_size,
      0 <= x[i] < output_spatial_shape[i],
      0 <= c < num_channels:

  ```
    output[b, x[0], ..., x[N-1], c] =
      REDUCE_{z[0], ..., z[N-1]}
        input[b,
              x[0] * strides[0] - pad_before[0] + dilation_rate[0]*z[0],
              ...
              x[N-1]*strides[N-1] - pad_before[N-1] + dilation_rate[N-1]*z[N-1],
              c],
  ```

  where the reduction function REDUCE depends on the value of `pooling_type`,
  and pad_before is defined based on the value of `padding` as described in
  the "returns" section of `tf.nn.convolution` for details.
  The reduction never includes out-of-bounds positions.

  In the case that `data_format` starts with `"NC"`, the `input` and output are
  simply transposed as follows:

  ```
    pool(input, data_format, **kwargs) =
      tf.transpose(pool(tf.transpose(input, [0] + range(2,N+2) + [1]),
                        **kwargs),
                   [0, N+1] + range(1, N+1))
  ```

  Args:
    input: Tensor of rank N+2, of shape
      `[batch_size] + input_spatial_shape + [num_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC".  Pooling happens over the spatial dimensions only.
    window_shape: Sequence of N ints >= 1.
    pooling_type: Specifies pooling operation, must be "AVG" or "MAX".
    padding: The padding algorithm, must be "SAME" or "VALID".
      See the "returns" section of `tf.nn.convolution` for details.
    dilation_rate: Optional.  Dilation rate.  List of N ints >= 1.
      Defaults to [1]*N.  If any value of dilation_rate is > 1, then all values
      of strides must be 1.
    strides: Optional.  Sequence of N ints >= 1.  Defaults to [1]*N.
      If any value of strides is > 1, then all values of dilation_rate must be
      1.
    name: Optional. Name of the op.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations: Alias for dilation_rate

  Returns:
    Tensor of rank N+2, of shape
      [batch_size] + output_spatial_shape + [num_channels]

    if data_format is None or does not start with "NC", or

      [batch_size, num_channels] + output_spatial_shape

    if data_format starts with "NC",
    where `output_spatial_shape` depends on the value of padding:

    If padding = "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])

    If padding = "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] - (window_shape[i] - 1) * dilation_rate[i])
             / strides[i]).

  Raises:
    ValueError: if arguments are invalid.

  """
  dilation_rate = deprecated_argument_lookup(
      "dilations", dilations, "dilation_rate", dilation_rate)
  # pylint: enable=line-too-long
  with ops.name_scope(name, "%s_pool" % (pooling_type.lower()),
                      [input]) as scope:
    input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin

    num_spatial_dims = len(window_shape)
    if num_spatial_dims < 1 or num_spatial_dims > 3:
      raise ValueError("It is required that 1 <= num_spatial_dims <= 3.")

    input.get_shape().with_rank(num_spatial_dims + 2)

    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    if padding == "SAME" and np.any(dilation_rate > 1):
      raise ValueError(
          "pooling with SAME padding is not implemented for dilation_rate > 1")

    if np.any(strides > window_shape):
      raise ValueError(
          "strides > window_shape not supported due to inconsistency between "
          "CPU and GPU implementations")

    pooling_ops = {
        ("MAX", 1): max_pool,
        ("MAX", 2): max_pool,
        ("MAX", 3): max_pool3d,  # pylint: disable=undefined-variable
        ("AVG", 1): avg_pool,
        ("AVG", 2): avg_pool,
        ("AVG", 3): avg_pool3d,  # pylint: disable=undefined-variable
    }
    op_key = (pooling_type, num_spatial_dims)
    if op_key not in pooling_ops:
      raise ValueError("%d-D %s pooling is not supported." % (op_key[1],
                                                              op_key[0]))

    if data_format is None or not data_format.startswith("NC"):
      adjusted_window_shape = [1] + list(window_shape) + [1]
      adjusted_strides = [1] + list(strides) + [1]
      spatial_dims = range(1, num_spatial_dims + 1)
    else:
      adjusted_window_shape = [1, 1] + list(window_shape)
      adjusted_strides = [1, 1] + list(strides)
      spatial_dims = range(2, num_spatial_dims + 2)

    if num_spatial_dims == 1:
      if data_format is None or data_format == "NWC":
        data_format_kwargs = dict(data_format="NHWC")
      elif data_format == "NCW":
        data_format_kwargs = dict(data_format="NCHW")
      else:
        raise ValueError("data_format must be either \"NWC\" or \"NCW\".")
      adjusted_window_shape = [1] + adjusted_window_shape
      adjusted_strides = [1] + adjusted_strides
    else:
      data_format_kwargs = dict(data_format=data_format)

    def op(converted_input, _, converted_padding):  # pylint: disable=missing-docstring
      if num_spatial_dims == 1:
        converted_input = array_ops.expand_dims(converted_input,
                                                spatial_dims[0])
      result = pooling_ops[op_key](
          converted_input,
          adjusted_window_shape,
          adjusted_strides,
          converted_padding,
          name=scope,
          **data_format_kwargs)
      if num_spatial_dims == 1:
        result = array_ops.squeeze(result, [spatial_dims[0]])
      return result

    return with_space_to_batch(
        input=input,
        dilation_rate=dilation_rate,
        padding=padding,
        op=op,
        spatial_dims=spatial_dims,
        filter_shape=window_shape)


@tf_export("nn.pool", v1=[])
def pool_v2(
    input,  # pylint: disable=redefined-builtin
    window_shape,
    pooling_type,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None):
  # pylint: disable=line-too-long
  """Performs an N-D pooling operation.

  In the case that `data_format` does not start with "NC", computes for
      0 <= b < batch_size,
      0 <= x[i] < output_spatial_shape[i],
      0 <= c < num_channels:

  ```
    output[b, x[0], ..., x[N-1], c] =
      REDUCE_{z[0], ..., z[N-1]}
        input[b,
              x[0] * strides[0] - pad_before[0] + dilation_rate[0]*z[0],
              ...
              x[N-1]*strides[N-1] - pad_before[N-1] + dilation_rate[N-1]*z[N-1],
              c],
  ```

  where the reduction function REDUCE depends on the value of `pooling_type`,
  and pad_before is defined based on the value of `padding` as described in
  the "returns" section of `tf.nn.convolution` for details.
  The reduction never includes out-of-bounds positions.

  In the case that `data_format` starts with `"NC"`, the `input` and output are
  simply transposed as follows:

  ```
    pool(input, data_format, **kwargs) =
      tf.transpose(pool(tf.transpose(input, [0] + range(2,N+2) + [1]),
                        **kwargs),
                   [0, N+1] + range(1, N+1))
  ```

  Args:
    input: Tensor of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]` if data_format does not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC".  Pooling happens over the spatial dimensions only.
    window_shape: Sequence of N ints >= 1.
    pooling_type: Specifies pooling operation, must be "AVG" or "MAX".
    strides: Optional. Sequence of N ints >= 1.  Defaults to [1]*N. If any value of
      strides is > 1, then all values of dilation_rate must be 1.
    padding: The padding algorithm, must be "SAME" or "VALID". Defaults to "SAME".
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW". For
      N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations: Optional.  Dilation rate.  List of N ints >= 1. Defaults to
      [1]*N.  If any value of dilation_rate is > 1, then all values of strides
      must be 1.
    name: Optional. Name of the op.

  Returns:
    Tensor of rank N+2, of shape
      [batch_size] + output_spatial_shape + [num_channels]

    if data_format is None or does not start with "NC", or

      [batch_size, num_channels] + output_spatial_shape

    if data_format starts with "NC",
    where `output_spatial_shape` depends on the value of padding:

    If padding = "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])

    If padding = "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] - (window_shape[i] - 1) * dilation_rate[i])
             / strides[i]).

  Raises:
    ValueError: if arguments are invalid.

  """
  return pool(
      input=input,
      window_shape=window_shape,
      pooling_type=pooling_type,
      padding=padding,
      dilation_rate=dilations,
      strides=strides,
      name=name,
      data_format=data_format)


@tf_export("nn.atrous_conv2d")
def atrous_conv2d(value, filters, rate, padding, name=None):
  """Atrous convolution (a.k.a. convolution with holes or dilated convolution).

  This function is a simpler wrapper around the more general
  `tf.nn.convolution`, and exists only for backwards compatibility. You can
  use `tf.nn.convolution` to perform 1-D, 2-D, or 3-D atrous convolution.


  Computes a 2-D atrous convolution, also known as convolution with holes or
  dilated convolution, given 4-D `value` and `filters` tensors. If the `rate`
  parameter is equal to one, it performs regular 2-D convolution. If the `rate`
  parameter is greater than one, it performs convolution with holes, sampling
  the input values every `rate` pixels in the `height` and `width` dimensions.
  This is equivalent to convolving the input with a set of upsampled filters,
  produced by inserting `rate - 1` zeros between two consecutive values of the
  filters along the `height` and `width` dimensions, hence the name atrous
  convolution or convolution with holes (the French word trous means holes in
  English).

  More specifically:

  ```
  output[batch, height, width, out_channel] =
      sum_{dheight, dwidth, in_channel} (
          filters[dheight, dwidth, in_channel, out_channel] *
          value[batch, height + rate*dheight, width + rate*dwidth, in_channel]
      )
  ```

  Atrous convolution allows us to explicitly control how densely to compute
  feature responses in fully convolutional networks. Used in conjunction with
  bilinear interpolation, it offers an alternative to `conv2d_transpose` in
  dense prediction tasks such as semantic image segmentation, optical flow
  computation, or depth estimation. It also allows us to effectively enlarge
  the field of view of filters without increasing the number of parameters or
  the amount of computation.

  For a description of atrous convolution and how it can be used for dense
  feature extraction, please see: (Chen et al., 2015). The same operation is
  investigated further in (Yu et al., 2016). Previous works that effectively
  use atrous convolution in different ways are, among others,
  (Sermanet et al., 2014) and (Giusti et al., 2013).
  Atrous convolution is also closely related to the so-called noble identities
  in multi-rate signal processing.

  There are many different ways to implement atrous convolution (see the refs
  above). The implementation here reduces

  ```python
      atrous_conv2d(value, filters, rate, padding=padding)
  ```

  to the following three operations:

  ```python
      paddings = ...
      net = space_to_batch(value, paddings, block_size=rate)
      net = conv2d(net, filters, strides=[1, 1, 1, 1], padding="VALID")
      crops = ...
      net = batch_to_space(net, crops, block_size=rate)
  ```

  Advanced usage. Note the following optimization: A sequence of `atrous_conv2d`
  operations with identical `rate` parameters, 'SAME' `padding`, and filters
  with odd heights/ widths:

  ```python
      net = atrous_conv2d(net, filters1, rate, padding="SAME")
      net = atrous_conv2d(net, filters2, rate, padding="SAME")
      ...
      net = atrous_conv2d(net, filtersK, rate, padding="SAME")
  ```

  can be equivalently performed cheaper in terms of computation and memory as:

  ```python
      pad = ...  # padding so that the input dims are multiples of rate
      net = space_to_batch(net, paddings=pad, block_size=rate)
      net = conv2d(net, filters1, strides=[1, 1, 1, 1], padding="SAME")
      net = conv2d(net, filters2, strides=[1, 1, 1, 1], padding="SAME")
      ...
      net = conv2d(net, filtersK, strides=[1, 1, 1, 1], padding="SAME")
      net = batch_to_space(net, crops=pad, block_size=rate)
  ```

  because a pair of consecutive `space_to_batch` and `batch_to_space` ops with
  the same `block_size` cancel out when their respective `paddings` and `crops`
  inputs are identical.

  Args:
    value: A 4-D `Tensor` of type `float`. It needs to be in the default "NHWC"
      format. Its shape is `[batch, in_height, in_width, in_channels]`.
    filters: A 4-D `Tensor` with the same type as `value` and shape
      `[filter_height, filter_width, in_channels, out_channels]`. `filters`'
      `in_channels` dimension must match that of `value`. Atrous convolution is
      equivalent to standard convolution with upsampled filters with effective
      height `filter_height + (filter_height - 1) * (rate - 1)` and effective
      width `filter_width + (filter_width - 1) * (rate - 1)`, produced by
      inserting `rate - 1` zeros along consecutive elements across the
      `filters`' spatial dimensions.
    rate: A positive int32. The stride with which we sample input values across
      the `height` and `width` dimensions. Equivalently, the rate by which we
      upsample the filter values by inserting zeros across the `height` and
      `width` dimensions. In the literature, the same parameter is sometimes
      called `input stride` or `dilation`.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.
    Output shape with `'VALID'` padding is:

        [batch, height - 2 * (filter_width - 1),
         width - 2 * (filter_height - 1), out_channels].

    Output shape with `'SAME'` padding is:

        [batch, height, width, out_channels].

  Raises:
    ValueError: If input/output depth does not match `filters`' shape, or if
      padding is other than `'VALID'` or `'SAME'`.

  References:
    Multi-Scale Context Aggregation by Dilated Convolutions:
      [Yu et al., 2016](https://arxiv.org/abs/1511.07122)
      ([pdf](https://arxiv.org/pdf/1511.07122.pdf))
    Semantic Image Segmentation with Deep Convolutional Nets and Fully
    Connected CRFs:
      [Chen et al., 2015](http://arxiv.org/abs/1412.7062)
      ([pdf](https://arxiv.org/pdf/1412.7062))
    OverFeat - Integrated Recognition, Localization and Detection using
    Convolutional Networks:
      [Sermanet et al., 2014](https://arxiv.org/abs/1312.6229)
      ([pdf](https://arxiv.org/pdf/1312.6229.pdf))
    Fast Image Scanning with Deep Max-Pooling Convolutional Neural Networks:
      [Giusti et al., 2013]
      (https://ieeexplore.ieee.org/abstract/document/6738831)
      ([pdf](https://arxiv.org/pdf/1302.1700.pdf))
  """
  return convolution(
      input=value,
      filter=filters,
      padding=padding,
      dilation_rate=np.broadcast_to(rate, (2,)),
      name=name)


def _convert_padding(padding):
  """Converts Python padding to C++ padding for ops which take EXPLICIT padding.

  Args:
    padding: the `padding` argument for a Python op which supports EXPLICIT
      padding.

  Returns:
    (padding, explicit_paddings) pair, which should be passed as attributes to a
    C++ op.

  Raises:
    ValueError: If padding is invalid.
  """
  explicit_paddings = []
  if padding == "EXPLICIT":
    # Give a better error message if EXPLICIT is passed.
    raise ValueError('"EXPLICIT" is not a valid value for the padding '
                     "parameter. To use explicit padding, the padding "
                     "parameter must be a list.")
  if isinstance(padding, (list, tuple)):
    for i, dim_paddings in enumerate(padding):
      if not isinstance(dim_paddings, (list, tuple)):
        raise ValueError("When padding is a list, each element of padding must "
                         "be a list/tuple of size 2. Element with index %d of "
                         "padding is not a list/tuple" % i)
      if len(dim_paddings) != 2:
        raise ValueError("When padding is a list, each element of padding must "
                         "be a list/tuple of size 2. Element with index %d of "
                         "padding has size %d" % (i, len(dim_paddings)))
      explicit_paddings.extend(dim_paddings)
    if len(padding) != 4:
      raise ValueError("When padding is a list, it must be of size 4. Got "
                       "padding of size: %d" % len(padding))
    padding = "EXPLICIT"
  return padding, explicit_paddings


@tf_export(v1=["nn.conv1d"])
@deprecation.deprecated_arg_values(
    None,
    "`NCHW` for data_format is deprecated, use `NCW` instead",
    warn_once=True,
    data_format="NCHW")
@deprecation.deprecated_arg_values(
    None,
    "`NHWC` for data_format is deprecated, use `NWC` instead",
    warn_once=True,
    data_format="NHWC")
def conv1d(
    value=None,
    filters=None,
    stride=None,
    padding=None,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None,
    input=None,  # pylint: disable=redefined-builtin
    dilations=None):
  r"""Computes a 1-D convolution given 3-D input and filter tensors.

  Given an input tensor of shape
    [batch, in_width, in_channels]
  if data_format is "NWC", or
    [batch, in_channels, in_width]
  if data_format is "NCW",
  and a filter / kernel tensor of shape
  [filter_width, in_channels, out_channels], this op reshapes
  the arguments to pass them to conv2d to perform the equivalent
  convolution operation.

  Internally, this op reshapes the input tensors and invokes `tf.nn.conv2d`.
  For example, if `data_format` does not start with "NC", a tensor of shape
    [batch, in_width, in_channels]
  is reshaped to
    [batch, 1, in_width, in_channels],
  and the filter is reshaped to
    [1, filter_width, in_channels, out_channels].
  The result is then reshaped back to
    [batch, out_width, out_channels]
  \(where out_width is a function of the stride and padding as in conv2d\) and
  returned to the caller.

  Args:
    value: A 3D `Tensor`.  Must be of type `float16`, `float32`, or `float64`.
    filters: A 3D `Tensor`.  Must have the same type as `value`.
    stride: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: 'SAME' or 'VALID'
    use_cudnn_on_gpu: An optional `bool`.  Defaults to `True`.
    data_format: An optional `string` from `"NWC", "NCW"`.  Defaults to `"NWC"`,
      the data is stored in the order of [batch, in_width, in_channels].  The
      `"NCW"` format stores data as [batch, in_channels, in_width].
    name: A name for the operation (optional).
    input: Alias for value.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.

  Returns:
    A `Tensor`.  Has the same type as input.

  Raises:
    ValueError: if `data_format` is invalid.
  """
  value = deprecation.deprecated_argument_lookup("input", input, "value", value)
  with ops.name_scope(name, "conv1d", [value, filters]) as name:
    # Reshape the input tensor to [batch, 1, in_width, in_channels]
    if data_format is None or data_format == "NHWC" or data_format == "NWC":
      data_format = "NHWC"
      spatial_start_dim = 1
      channel_index = 2
    elif data_format == "NCHW" or data_format == "NCW":
      data_format = "NCHW"
      spatial_start_dim = 2
      channel_index = 1
    else:
      raise ValueError("data_format must be \"NWC\" or \"NCW\".")
    strides = [1] + _get_sequence(stride, 1, channel_index, "stride")
    dilations = [1] + _get_sequence(dilations, 1, channel_index, "dilations")

    value = array_ops.expand_dims(value, spatial_start_dim)
    filters = array_ops.expand_dims(filters, 0)
    result = gen_nn_ops.conv2d(
        value,
        filters,
        strides,
        padding,
        use_cudnn_on_gpu=use_cudnn_on_gpu,
        data_format=data_format,
        dilations=dilations,
        name=name)
    return array_ops.squeeze(result, [spatial_start_dim])


@tf_export("nn.conv1d", v1=[])
def conv1d_v2(
    input,  # pylint: disable=redefined-builtin
    filters,
    stride,
    padding,
    data_format="NWC",
    dilations=None,
    name=None):
  r"""Computes a 1-D convolution given 3-D input and filter tensors.

  Given an input tensor of shape
    [batch, in_width, in_channels]
  if data_format is "NWC", or
    [batch, in_channels, in_width]
  if data_format is "NCW",
  and a filter / kernel tensor of shape
  [filter_width, in_channels, out_channels], this op reshapes
  the arguments to pass them to conv2d to perform the equivalent
  convolution operation.

  Internally, this op reshapes the input tensors and invokes `tf.nn.conv2d`.
  For example, if `data_format` does not start with "NC", a tensor of shape
    [batch, in_width, in_channels]
  is reshaped to
    [batch, 1, in_width, in_channels],
  and the filter is reshaped to
    [1, filter_width, in_channels, out_channels].
  The result is then reshaped back to
    [batch, out_width, out_channels]
  \(where out_width is a function of the stride and padding as in conv2d\) and
  returned to the caller.

  Args:
    input: A 3D `Tensor`.  Must be of type `float16`, `float32`, or `float64`.
    filters: A 3D `Tensor`.  Must have the same type as `input`.
    stride: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: 'SAME' or 'VALID'
    data_format: An optional `string` from `"NWC", "NCW"`.  Defaults to `"NWC"`,
      the data is stored in the order of [batch, in_width, in_channels].  The
      `"NCW"` format stores data as [batch, in_channels, in_width].
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.  Has the same type as input.

  Raises:
    ValueError: if `data_format` is invalid.
  """
  return conv1d(
      input,  # pylint: disable=redefined-builtin
      filters,
      stride,
      padding,
      use_cudnn_on_gpu=True,
      data_format=data_format,
      name=name,
      dilations=dilations)


@tf_export("nn.conv1d_transpose")
def conv1d_transpose(
    input,  # pylint: disable=redefined-builtin
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NWC",
    dilations=None,
    name=None):
  """The transpose of `conv1d`.

  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is actually the transpose (gradient) of `conv1d`
  rather than an actual deconvolution.

  Args:
    input: A 3-D `Tensor` of type `float` and shape
      `[batch, in_width, in_channels]` for `NWC` data format or
      `[batch, in_channels, in_width]` for `NCW` data format.
    filters: A 3-D `Tensor` with the same type as `value` and shape
      `[filter_width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor`, containing three elements, representing the
      output shape of the deconvolution op.
    strides: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. `'NWC'` and `'NCW'` are supported.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, if
      `output_shape` is not at 3-element vector, if `padding` is other than
      `'VALID'` or `'SAME'`, or if `data_format` is invalid.

  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "conv1d_transpose",
                      [input, filters, output_shape]) as name:
    # The format could be either NWC or NCW, map to NHWC or NCHW
    if data_format is None or data_format == "NWC":
      data_format = "NHWC"
      spatial_start_dim = 1
      channel_index = 2
    elif data_format == "NCW":
      data_format = "NCHW"
      spatial_start_dim = 2
      channel_index = 1
    else:
      raise ValueError("data_format must be \"NWC\" or \"NCW\".")

    # Reshape the input tensor to [batch, 1, in_width, in_channels]
    strides = [1] + _get_sequence(strides, 1, channel_index, "stride")
    dilations = [1] + _get_sequence(dilations, 1, channel_index, "dilations")

    input = array_ops.expand_dims(input, spatial_start_dim)
    filters = array_ops.expand_dims(filters, 0)
    output_shape = list(output_shape) if not isinstance(
        output_shape, ops.Tensor) else output_shape
    output_shape = array_ops.concat([output_shape[: spatial_start_dim], [1],
                                     output_shape[spatial_start_dim:]], 0)

    result = gen_nn_ops.conv2d_backprop_input(
        input_sizes=output_shape,
        filter=filters,
        out_backprop=input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)
    return array_ops.squeeze(result, spatial_start_dim)


@tf_export("nn.conv2d", v1=[])
def conv2d_v2(input,  # pylint: disable=redefined-builtin
              filters,
              strides,
              padding,
              data_format="NHWC",
              dilations=None,
              name=None):
  # pylint: disable=line-too-long
  r"""Computes a 2-D convolution given 4-D `input` and `filters` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types:
      `half`, `bfloat16`, `float32`, `float64`.
      A 4-D tensor. The dimension order is interpreted according to the value
      of `data_format`, see below for details.
    filters: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `H` and `W` dimension. By default
      the `N` and `C` dimensions are set to 1. The dimension order is determined
      by the value of `data_format`, see below for details.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. When explicit padding is used and
      data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
      pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
      and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`.
      Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    dilations: An int or list of `ints` that has length `1`, `2` or `4`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 4-d tensor
      must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  # pylint: enable=line-too-long
  return conv2d(input,  # pylint: disable=redefined-builtin
                filters,
                strides,
                padding,
                use_cudnn_on_gpu=True,
                data_format=data_format,
                dilations=dilations,
                name=name)


@tf_export(v1=["nn.conv2d"])
def conv2d(  # pylint: disable=redefined-builtin,dangerous-default-value
    input,
    filter=None,
    strides=None,
    padding=None,
    use_cudnn_on_gpu=True,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name=None,
    filters=None):
  r"""Computes a 2-D convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q]
                          * filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types:
      `half`, `bfloat16`, `float32`, `float64`.
      A 4-D tensor. The dimension order is interpreted according to the value
      of `data_format`, see below for details.
    filter: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `H` and `W` dimension. By default
      the `N` and `C` dimensions are set to 1. The dimension order is determined
      by the value of `data_format`, see below for details.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. When explicit padding is used and
      data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
      pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
      and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`.
      Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    dilations: An int or list of `ints` that has length `1`, `2` or `4`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 4-d tensor
      must be 1.
    name: A name for the operation (optional).
    filters: Alias for filter.

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  filter = deprecation.deprecated_argument_lookup(
      "filters", filters, "filter", filter)
  padding, explicit_paddings = _convert_padding(padding)
  if data_format is None:
    data_format = "NHWC"
  channel_index = 1 if data_format.startswith("NC") else 3

  strides = _get_sequence(strides, 2, channel_index, "strides")
  dilations = _get_sequence(dilations, 2, channel_index, "dilations")
  return gen_nn_ops.conv2d(input,  # pylint: disable=redefined-builtin
                           filter,
                           strides,
                           padding,
                           use_cudnn_on_gpu=use_cudnn_on_gpu,
                           explicit_paddings=explicit_paddings,
                           data_format=data_format,
                           dilations=dilations,
                           name=name)


@tf_export(v1=["nn.conv2d_backprop_filter"])
def conv2d_backprop_filter(  # pylint: disable=redefined-builtin,dangerous-default-value
    input,
    filter_sizes,
    out_backprop,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name=None):
  r"""Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types:
      `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, out_channels]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified
      with format.
    padding: Either the `string `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. When explicit padding is used and
      data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
      pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
      and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`.
      Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by
      the value of `data_format`, see above for details. Dilations in the batch
      and depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  padding, explicit_paddings = _convert_padding(padding)
  return gen_nn_ops.conv2d_backprop_filter(
      input, filter_sizes, out_backprop, strides, padding, use_cudnn_on_gpu,
      explicit_paddings, data_format, dilations, name)


@tf_export(v1=["nn.conv2d_backprop_input"])
def conv2d_backprop_input(  # pylint: disable=redefined-builtin,dangerous-default-value
    input_sizes,
    filter=None,
    out_backprop=None,
    strides=None,
    padding=None,
    use_cudnn_on_gpu=True,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name=None,
    filters=None):
  r"""Computes the gradients of convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types:
      `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified
      with format.
    padding: Either the `string `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. When explicit padding is used and
      data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
      pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
      and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`.
      Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by
      the value of `data_format`, see above for details. Dilations in the batch
      and depth dimensions must be 1.
    name: A name for the operation (optional).
    filters: Alias for filter.

  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  filter = deprecation.deprecated_argument_lookup(
      "filters", filters, "filter", filter)
  padding, explicit_paddings = _convert_padding(padding)
  return gen_nn_ops.conv2d_backprop_input(
      input_sizes, filter, out_backprop, strides, padding, use_cudnn_on_gpu,
      explicit_paddings, data_format, dilations, name)


@tf_export(v1=["nn.conv2d_transpose"])
def conv2d_transpose(
    value=None,
    filter=None,  # pylint: disable=redefined-builtin
    output_shape=None,
    strides=None,
    padding="SAME",
    data_format="NHWC",
    name=None,
    input=None,  # pylint: disable=redefined-builtin
    filters=None,
    dilations=None):
  """The transpose of `conv2d`.

  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of `conv2d`
  rather than an actual deconvolution.

  Args:
    value: A 4-D `Tensor` of type `float` and shape
      `[batch, height, width, in_channels]` for `NHWC` data format or
      `[batch, in_channels, height, width]` for `NCHW` data format.
    filter: A 4-D `Tensor` with the same type as `value` and shape
      `[height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `H` and `W` dimension. By default
      the `N` and `C` dimensions are set to 0. The dimension order is determined
      by the value of `data_format`, see below for details.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the returned tensor.
    input: Alias for value.
    filters: Alias for filter.
    dilations: An int or list of `ints` that has length `1`, `2` or `4`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 4-d tensor
      must be 1.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.

  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  value = deprecated_argument_lookup("input", input, "value", value)
  filter = deprecated_argument_lookup("filters", filters, "filter", filter)
  with ops.name_scope(name, "conv2d_transpose",
                      [value, filter, output_shape]) as name:
    return conv2d_transpose_v2(
        value,
        filter,
        output_shape,
        strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)


@tf_export("nn.conv2d_transpose", v1=[])
def conv2d_transpose_v2(
    input,  # pylint: disable=redefined-builtin
    filters,  # pylint: disable=redefined-builtin
    output_shape,
    strides,
    padding="SAME",
    data_format="NHWC",
    dilations=None,
    name=None):
  """The transpose of `conv2d`.

  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of
  `atrous_conv2d` rather than an actual deconvolution.

  Args:
    input: A 4-D `Tensor` of type `float` and shape `[batch, height, width,
      in_channels]` for `NHWC` data format or `[batch, in_channels, height,
      width]` for `NCHW` data format.
    filters: A 4-D `Tensor` with the same type as `input` and shape `[height,
      width, output_channels, in_channels]`.  `filter`'s `in_channels` dimension
      must match that of `input`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `H` and `W` dimension. By default
      the `N` and `C` dimensions are set to 0. The dimension order is determined
      by the value of `data_format`, see below for details.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    dilations: An int or list of `ints` that has length `1`, `2` or `4`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 4-d tensor
      must be 1.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `input`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.

  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "conv2d_transpose",
                      [input, filter, output_shape]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    strides = _get_sequence(strides, 2, channel_index, "strides")
    dilations = _get_sequence(dilations, 2, channel_index, "dilations")

    return gen_nn_ops.conv2d_backprop_input(
        input_sizes=output_shape,
        filter=filters,
        out_backprop=input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)


@tf_export("nn.atrous_conv2d_transpose")
def atrous_conv2d_transpose(value,
                            filters,
                            output_shape,
                            rate,
                            padding,
                            name=None):
  """The transpose of `atrous_conv2d`.

  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of
  `atrous_conv2d` rather than an actual deconvolution.

  Args:
    value: A 4-D `Tensor` of type `float`. It needs to be in the default `NHWC`
      format. Its shape is `[batch, in_height, in_width, in_channels]`.
    filters: A 4-D `Tensor` with the same type as `value` and shape
      `[filter_height, filter_width, out_channels, in_channels]`. `filters`'
      `in_channels` dimension must match that of `value`. Atrous convolution is
      equivalent to standard convolution with upsampled filters with effective
      height `filter_height + (filter_height - 1) * (rate - 1)` and effective
      width `filter_width + (filter_width - 1) * (rate - 1)`, produced by
      inserting `rate - 1` zeros along consecutive elements across the
      `filters`' spatial dimensions.
    output_shape: A 1-D `Tensor` of shape representing the output shape of the
      deconvolution op.
    rate: A positive int32. The stride with which we sample input values across
      the `height` and `width` dimensions. Equivalently, the rate by which we
      upsample the filter values by inserting zeros across the `height` and
      `width` dimensions. In the literature, the same parameter is sometimes
      called `input stride` or `dilation`.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filters`' shape, or if
      padding is other than `'VALID'` or `'SAME'`, or if the `rate` is less
      than one, or if the output_shape is not a tensor with 4 elements.

  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "atrous_conv2d_transpose",
                      [value, filters, output_shape]) as name:
    value = ops.convert_to_tensor(value, name="value")
    filters = ops.convert_to_tensor(filters, name="filters")
    if not value.get_shape().dims[3].is_compatible_with(filters.get_shape()[3]):
      raise ValueError(
          "value's input channels does not match filters' input channels, "
          "{} != {}".format(value.get_shape()[3],
                            filters.get_shape()[3]))
    if rate < 1:
      raise ValueError("rate {} cannot be less than one".format(rate))

    if rate == 1:
      return conv2d_transpose(
          value,
          filters,
          output_shape,
          strides=[1, 1, 1, 1],
          padding=padding,
          data_format="NHWC")

    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(
        tensor_shape.TensorShape([4])):
      raise ValueError("output_shape must have shape (4,), got {}".format(
          output_shape_.get_shape()))

    if isinstance(output_shape, tuple):
      output_shape = list(output_shape)

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [4] if reached this point.
      if not filters.get_shape().dims[2].is_compatible_with(output_shape[3]):
        raise ValueError(
            "output_shape does not match filter's output channels, "
            "{} != {}".format(output_shape[3],
                              filters.get_shape()[2]))

    # We have two padding contributions. The first is used for converting "SAME"
    # to "VALID". The second is required so that the height and width of the
    # zero-padded value tensor are multiples of rate.

    # Padding required to reduce to "VALID" convolution
    if padding == "SAME":
      # Handle filters whose shape is unknown during graph creation.
      if filters.get_shape().is_fully_defined():
        filter_shape = filters.get_shape().as_list()
      else:
        filter_shape = array_ops.shape(filters)
      filter_height, filter_width = filter_shape[0], filter_shape[1]

      # Spatial dimensions of the filters and the upsampled filters in which we
      # introduce (rate - 1) zeros between consecutive filter values.
      filter_height_up = filter_height + (filter_height - 1) * (rate - 1)
      filter_width_up = filter_width + (filter_width - 1) * (rate - 1)

      pad_height = filter_height_up - 1
      pad_width = filter_width_up - 1

      # When pad_height (pad_width) is odd, we pad more to bottom (right),
      # following the same convention as conv2d().
      pad_top = pad_height // 2
      pad_bottom = pad_height - pad_top
      pad_left = pad_width // 2
      pad_right = pad_width - pad_left
    elif padding == "VALID":
      pad_top = 0
      pad_bottom = 0
      pad_left = 0
      pad_right = 0
    else:
      raise ValueError("padding must be either VALID or SAME:"
                       " {}".format(padding))

    in_height = output_shape[1] + pad_top + pad_bottom
    in_width = output_shape[2] + pad_left + pad_right

    # More padding so that rate divides the height and width of the input.
    pad_bottom_extra = (rate - in_height % rate) % rate
    pad_right_extra = (rate - in_width % rate) % rate

    # The paddings argument to space_to_batch is just the extra padding
    # component.
    space_to_batch_pad = [[0, pad_bottom_extra], [0, pad_right_extra]]

    value = array_ops.space_to_batch(
        input=value, paddings=space_to_batch_pad, block_size=rate)

    input_sizes = [
        rate * rate * output_shape[0], (in_height + pad_bottom_extra) // rate,
        (in_width + pad_right_extra) // rate, output_shape[3]
    ]

    value = gen_nn_ops.conv2d_backprop_input(
        input_sizes=input_sizes,
        filter=filters,
        out_backprop=value,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC")

    # The crops argument to batch_to_space includes both padding components.
    batch_to_space_crop = [[pad_top, pad_bottom + pad_bottom_extra],
                           [pad_left, pad_right + pad_right_extra]]

    return array_ops.batch_to_space(
        input=value, crops=batch_to_space_crop, block_size=rate)


@tf_export("nn.conv3d", v1=[])
def conv3d_v2(input,  # pylint: disable=redefined-builtin,missing-docstring
              filters,
              strides,
              padding,
              data_format="NDHWC",
              dilations=None,
              name=None):
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  return gen_nn_ops.conv3d(input,
                           filters,
                           strides,
                           padding,
                           data_format=data_format,
                           dilations=dilations,
                           name=name)


@tf_export(v1=["nn.conv3d"])
def conv3d_v1(  # pylint: disable=missing-docstring,dangerous-default-value
    input,  # pylint: disable=redefined-builtin
    filter=None,  # pylint: disable=redefined-builtin
    strides=None,
    padding=None,
    data_format="NDHWC",
    dilations=[1, 1, 1, 1, 1],
    name=None,
    filters=None):
  filter = deprecated_argument_lookup("filters", filters, "filter", filter)
  return gen_nn_ops.conv3d(
      input, filter, strides, padding, data_format, dilations, name)


conv3d_v2.__doc__ = deprecation.rewrite_argument_docstring(
    gen_nn_ops.conv3d.__doc__, "filter", "filters")
conv3d_v1.__doc__ = gen_nn_ops.conv3d.__doc__


@tf_export(v1=["nn.conv3d_transpose"])
def conv3d_transpose(
    value,
    filter=None,  # pylint: disable=redefined-builtin
    output_shape=None,
    strides=None,
    padding="SAME",
    data_format="NDHWC",
    name=None,
    input=None,  # pylint: disable=redefined-builtin
    filters=None,
    dilations=None):
  """The transpose of `conv3d`.

  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of `conv3d`
  rather than an actual deconvolution.

  Args:
    value: A 5-D `Tensor` of type `float` and shape
      `[batch, depth, height, width, in_channels]`.
    filter: A 5-D `Tensor` with the same type as `value` and shape
      `[depth, height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: A list of ints. The stride of the sliding window for each
      dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string, either `'NDHWC'` or `'NCDHW`' specifying the layout
      of the input and output tensors. Defaults to `'NDHWC'`.
    name: Optional name for the returned tensor.
    input: Alias of value.
    filters: Alias of filter.
    dilations: An int or list of `ints` that has length `1`, `3` or `5`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `D`, `H` and `W` dimension.
      By default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 5-d tensor
      must be 1.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.

  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  filter = deprecated_argument_lookup("filters", filters, "filter", filter)
  value = deprecated_argument_lookup("input", input, "value", value)
  return conv3d_transpose_v2(
      value,
      filter,
      output_shape,
      strides,
      padding=padding,
      data_format=data_format,
      dilations=dilations,
      name=name)


@tf_export("nn.conv3d_transpose", v1=[])
def conv3d_transpose_v2(input,  # pylint: disable=redefined-builtin
                        filters,
                        output_shape,
                        strides,
                        padding="SAME",
                        data_format="NDHWC",
                        dilations=None,
                        name=None):
  """The transpose of `conv3d`.

  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of `conv3d`
  rather than an actual deconvolution.

  Args:
    input: A 5-D `Tensor` of type `float` and shape `[batch, height, width,
      in_channels]` for `NHWC` data format or `[batch, in_channels, height,
      width]` for `NCHW` data format.
    filters: A 5-D `Tensor` with the same type as `value` and shape `[height,
      width, output_channels, in_channels]`.  `filter`'s `in_channels` dimension
      must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: An int or list of `ints` that has length `1`, `3` or `5`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `D`, `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 0. The dimension order is
      determined by the value of `data_format`, see below for details.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NDHWC' and 'NCDHW' are supported.
    dilations: An int or list of `ints` that has length `1`, `3` or `5`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `D`, `H` and `W` dimension.
      By default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 5-d tensor
      must be 1.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "conv3d_transpose",
                      [input, filter, output_shape]) as name:
    if data_format is None:
      data_format = "NDHWC"
    channel_index = 1 if data_format.startswith("NC") else 4

    strides = _get_sequence(strides, 3, channel_index, "strides")
    dilations = _get_sequence(dilations, 3, channel_index, "dilations")

    return gen_nn_ops.conv3d_backprop_input_v2(
        input_sizes=output_shape,
        filter=filters,
        out_backprop=input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)


CONV_TRANSPOSE_OPS = (
    conv1d_transpose,
    conv2d_transpose_v2,
    conv3d_transpose_v2,
)


@tf_export("nn.conv_transpose")
def conv_transpose(input,  # pylint: disable=redefined-builtin
                   filters,
                   output_shape,
                   strides,
                   padding="SAME",
                   data_format=None,
                   dilations=None,
                   name=None):
  """The transpose of `convolution`.

  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of `conv3d`
  rather than an actual deconvolution.

  Args:
    input: An N+2 dimensional `Tensor` of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC". It must be one of the following types:
      `half`, `bfloat16`, `float32`, `float64`.
    filters: An N+2 dimensional `Tensor` with the same type as `input` and
      shape `spatial_filter_shape + [in_channels, out_channels]`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: An int or list of `ints` that has length `1`, `N` or `N+2`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the spatial dimensions. By default
      the `N` and `C` dimensions are set to 0. The dimension order is determined
      by the value of `data_format`, see below for details.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations: An int or list of `ints` that has length `1`, `N` or `N+2`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the spatial dimensions. By
      default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details.
    name: A name for the operation (optional). If not specified "conv_transpose"
      is used.

  Returns:
    A `Tensor` with the same type as `value`.

  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "conv_transpose",
                      [input, filter, output_shape]) as name:
    if tensor_util.is_tensor(output_shape):
      n = output_shape.shape[0] - 2
    elif isinstance(output_shape, collections.Sized):
      n = len(output_shape) - 2
    else:
      raise ValueError("output_shape must be a tensor or sized collection.")

    if not 1 <= n <= 3:
      raise ValueError(
          "output_shape must be of length 3, 4 or 5 but was {}.".format(n + 2))

    op = CONV_TRANSPOSE_OPS[n-1]
    return op(
        input,
        filters,
        output_shape,
        strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)


def _tf_deterministic_ops():
  if _tf_deterministic_ops.value is None:
    tf_deterministic_ops = os.environ.get("TF_DETERMINISTIC_OPS")
    if tf_deterministic_ops is not None:
      tf_deterministic_ops = tf_deterministic_ops.lower()
    _tf_deterministic_ops.value = (
        tf_deterministic_ops == "true" or tf_deterministic_ops == "1")
  return _tf_deterministic_ops.value


_tf_deterministic_ops.value = None


@tf_export("nn.bias_add")
def bias_add(value, bias, data_format=None, name=None):
  """Adds `bias` to `value`.

  This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.
  Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
  case where both types are quantized.

  Args:
    value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
      `int16`, `int8`, `complex64`, or `complex128`.
    bias: A 1-D `Tensor` with size matching the channel dimension of `value`.
      Must be the same type as `value` unless `value` is a quantized type,
      in which case a different quantized type may be used.
    data_format: A string. 'N...C' and 'NC...' are supported. If `None` (the
      default) is specified then 'N..C' is assumed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError if data format is unrecognized, if `value` has less than two
    dimensions when `data_format` is 'N..C'/`None` or `value` has less
    then three dimensions when `data_format` is `NC..`, if `bias` does not
    have exactly one dimension (is a vector), or if the size of `bias`
    does not match the size of the channel dimension of `value`.
  """
  with ops.name_scope(name, "BiasAdd", [value, bias]) as name:
    if data_format is not None:
      if data_format.startswith("NC"):
        data_format = "NCHW"
      elif data_format.startswith("N") and data_format.endswith("C"):
        data_format = "NHWC"
      else:
        raise ValueError("data_format must be of the form `N...C` or `NC...`")

    if not context.executing_eagerly():
      value = ops.convert_to_tensor(value, name="input")
      bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")

    # TODO(duncanriach): Implement deterministic functionality at CUDA kernel
    #   level.
    if _tf_deterministic_ops():
      # Note that this code does not implement the same error checks as the
      # pre-existing C++ ops.
      if data_format == "NCHW":
        broadcast_shape_head = [1, array_ops.size(bias)]
        broadcast_shape_tail = array_ops.ones(
            array_ops.rank(value) - 2, dtype=dtypes.int32)
        broadcast_shape = array_ops.concat(
            [broadcast_shape_head, broadcast_shape_tail], 0)
        return math_ops.add(
            value, array_ops.reshape(bias, broadcast_shape), name=name)
      else:  # data_format == 'NHWC' or data_format == None
        return math_ops.add(value, bias, name=name)
    else:
      return gen_nn_ops.bias_add(
          value, bias, data_format=data_format, name=name)


def bias_add_v1(value, bias, name=None):
  """Adds `bias` to `value`.

  This is a deprecated version of bias_add and will soon to be removed.

  This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.
  Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
  case where both types are quantized.

  Args:
    value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
      `int16`, `int8`, `complex64`, or `complex128`.
    bias: A 1-D `Tensor` with size matching the last dimension of `value`.
      Must be the same type as `value` unless `value` is a quantized type,
      in which case a different quantized type may be used.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `value`.
  """
  with ops.name_scope(name, "BiasAddV1", [value, bias]) as name:
    value = ops.convert_to_tensor(value, name="input")
    bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")
    return gen_nn_ops.bias_add_v1(value, bias, name=name)


@tf_export(v1=["nn.crelu"])
def crelu(features, name=None, axis=-1):
  """Computes Concatenated ReLU.

  Concatenates a ReLU which selects only the positive part of the activation
  with a ReLU which selects only the *negative* part of the activation.
  Note that as a result this non-linearity doubles the depth of the activations.
  Source: [Understanding and Improving Convolutional Neural Networks via
  Concatenated Rectified Linear Units. W. Shang, et
  al.](https://arxiv.org/abs/1603.05201)

  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).
    axis: The axis that the output values are concatenated along. Default is -1.

  Returns:
    A `Tensor` with the same type as `features`.

  References:
    Understanding and Improving Convolutional Neural Networks via Concatenated
    Rectified Linear Units:
      [Shang et al., 2016](http://proceedings.mlr.press/v48/shang16)
      ([pdf](http://proceedings.mlr.press/v48/shang16.pdf))
  """
  with ops.name_scope(name, "CRelu", [features]) as name:
    features = ops.convert_to_tensor(features, name="features")
    c = array_ops.concat([features, -features], axis, name=name)
    return gen_nn_ops.relu(c)


@tf_export("nn.crelu", v1=[])
def crelu_v2(features, axis=-1, name=None):
  return crelu(features, name=name, axis=axis)
crelu_v2.__doc__ = crelu.__doc__


@tf_export("nn.relu6")
def relu6(features, name=None):
  """Computes Rectified Linear 6: `min(max(features, 0), 6)`.

  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `features`.

  References:
    Convolutional Deep Belief Networks on CIFAR-10:
      Krizhevsky et al., 2010
      ([pdf](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf))
  """
  with ops.name_scope(name, "Relu6", [features]) as name:
    features = ops.convert_to_tensor(features, name="features")
    return gen_nn_ops.relu6(features, name=name)


@tf_export("nn.leaky_relu")
def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.

  Source: [Rectifier Nonlinearities Improve Neural Network Acoustic Models.
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013]
  (https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf).
  Args:
    features: A `Tensor` representing preactivation values. Must be one of
      the following types: `float16`, `float32`, `float64`, `int32`, `int64`.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).

  Returns:
    The activation value.

  References:
    Rectifier Nonlinearities Improve Neural Network Acoustic Models:
      [Maas et al., 2013]
      (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.693.1422)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.1422&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]) as name:
    features = ops.convert_to_tensor(features, name="features")
    if features.dtype.is_integer:
      features = math_ops.cast(features, dtypes.float32)
    if isinstance(alpha, np.ndarray):
      alpha = alpha.item()
    return gen_nn_ops.leaky_relu(features, alpha=alpha, name=name)


def _flatten_outer_dims(logits):
  """Flattens logits' outer dimensions and keep its last dimension."""
  rank = array_ops.rank(logits)
  last_dim_size = array_ops.slice(
      array_ops.shape(logits), [math_ops.subtract(rank, 1)], [1])
  output = array_ops.reshape(logits, array_ops.concat([[-1], last_dim_size], 0))

  # Set output shape if known.
  if not context.executing_eagerly():
    shape = logits.get_shape()
    if shape is not None and shape.dims is not None:
      shape = shape.as_list()
      product = 1
      product_valid = True
      for d in shape[:-1]:
        if d is None:
          product_valid = False
          break
        else:
          product *= d
      if product_valid:
        output_shape = [product, shape[-1]]
        output.set_shape(output_shape)

  return output


def _softmax(logits, compute_op, dim=-1, name=None):
  """Helper function for softmax and log_softmax.

  It reshapes and transposes the input logits into a 2-D Tensor and then invokes
  the tf.nn._softmax or tf.nn._log_softmax function. The output would be
  transposed and reshaped back.

  Args:
    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    compute_op: Either gen_nn_ops.softmax or gen_nn_ops.log_softmax
    dim: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
  Raises:
    InvalidArgumentError: if `logits` is empty or `dim` is beyond the last
      dimension of `logits`.
  """

  def _swap_axis(logits, dim_index, last_index, name=None):
    """Swaps logits's dim_index and last_index."""
    return array_ops.transpose(
        logits,
        array_ops.concat([
            math_ops.range(dim_index), [last_index],
            math_ops.range(dim_index + 1, last_index), [dim_index]
        ], 0),
        name=name)

  logits = ops.convert_to_tensor(logits)

  # We need its original shape for shape inference.
  shape = logits.get_shape()
  is_last_dim = (dim == -1) or (dim == shape.ndims - 1)

  if is_last_dim:
    return compute_op(logits, name=name)

  dim_val = dim
  if isinstance(dim, ops.Tensor):
    dim_val = tensor_util.constant_value(dim)
  if dim_val is not None and not -shape.ndims <= dim_val < shape.ndims:
    raise errors_impl.InvalidArgumentError(
        None, None,
        "Dimension (%d) must be in the range [%d, %d) where %d is the number of"
        " dimensions in the input." % (dim_val, -shape.ndims, shape.ndims,
                                       shape.ndims))

  # If dim is not the last dimension, we have to do a transpose so that we can
  # still perform softmax on its last dimension.

  # In case dim is negative (and is not last dimension -1), add shape.ndims
  ndims = array_ops.rank(logits)
  if not isinstance(dim, ops.Tensor):
    if dim < 0:
      dim += ndims
  else:
    dim = array_ops.where(math_ops.less(dim, 0), dim + ndims, dim)

  # Swap logits' dimension of dim and its last dimension.
  input_rank = array_ops.rank(logits)
  dim_axis = dim % shape.ndims
  logits = _swap_axis(logits, dim_axis, math_ops.subtract(input_rank, 1))

  # Do the actual softmax on its last dimension.
  output = compute_op(logits)

  output = _swap_axis(
      output, dim_axis, math_ops.subtract(input_rank, 1), name=name)

  # Make shape inference work since transpose may erase its static shape.
  output.set_shape(shape)

  return output


@tf_export(v1=["nn.softmax", "math.softmax"])
@deprecation.deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def softmax(logits, axis=None, name=None, dim=None):
  """Computes softmax activations.

  This function performs the equivalent of

      softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

  See: https://en.wikipedia.org/wiki/Softmax_function

  Example usage:
  >>> tf.nn.softmax([-1, 0., 1.])
  <tf.Tensor: shape=(3,), dtype=float32,
  numpy=array([0.09003057, 0.24472848, 0.66524094], dtype=float32)>

  Args:
    logits: A non-empty `Tensor`, or an object whose type has a registered
      `Tensor` conversion function. Must be one of the following types:
      `half`,`float32`, `float64`. See also `convert_to_tensor`
    axis: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).
    dim: Deprecated alias for `axis`.

  Returns:
    A `Tensor`. Has the same type and shape as `logits`.

  Raises:
    InvalidArgumentError: if `logits` is empty or `axis` is beyond the last
      dimension of `logits`.
    TypeError: If no conversion function is registered for `logits` to
      Tensor.
    RuntimeError: If a registered conversion function returns an invalid
      value.
      
  """
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dim", dim)
  if axis is None:
    axis = -1
  return _softmax(logits, gen_nn_ops.softmax, axis, name)


@tf_export("nn.softmax", "math.softmax", v1=[])
def softmax_v2(logits, axis=None, name=None):
  """Computes softmax activations.

  This function performs the equivalent of

      softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

  Args:
    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    axis: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type and shape as `logits`.

  Raises:
    InvalidArgumentError: if `logits` is empty or `axis` is beyond the last
      dimension of `logits`.
  """
  if axis is None:
    axis = -1
  return _softmax(logits, gen_nn_ops.softmax, axis, name)


@tf_export(v1=["nn.log_softmax", "math.log_softmax"])
@deprecation.deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def log_softmax(logits, axis=None, name=None, dim=None):
  """Computes log softmax activations.

  For each batch `i` and class `j` we have

      logsoftmax = logits - log(reduce_sum(exp(logits), axis))

  Args:
    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    axis: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).
    dim: Deprecated alias for `axis`.

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.

  Raises:
    InvalidArgumentError: if `logits` is empty or `axis` is beyond the last
      dimension of `logits`.
  """
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dim", dim)
  if axis is None:
    axis = -1
  return _softmax(logits, gen_nn_ops.log_softmax, axis, name)


@tf_export("nn.log_softmax", "math.log_softmax", v1=[])
def log_softmax_v2(logits, axis=None, name=None):
  """Computes log softmax activations.

  For each batch `i` and class `j` we have

      logsoftmax = logits - log(reduce_sum(exp(logits), axis))

  Args:
    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    axis: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.

  Raises:
    InvalidArgumentError: if `logits` is empty or `axis` is beyond the last
      dimension of `logits`.
  """
  if axis is None:
    axis = -1
  return _softmax(logits, gen_nn_ops.log_softmax, axis, name)


def _ensure_xent_args(name, sentinel, labels, logits):
  # Make sure that all arguments were passed as named arguments.
  if sentinel is not None:
    raise ValueError("Only call `%s` with "
                     "named arguments (labels=..., logits=..., ...)" % name)
  if labels is None or logits is None:
    raise ValueError("Both labels and logits must be provided.")


@tf_export("nn.softmax_cross_entropy_with_logits", v1=[])
def softmax_cross_entropy_with_logits_v2(labels, logits, axis=-1, name=None):
  """Computes softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.

  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

  Usage:
  >>> logits = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]
  >>> labels = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]
  >>> tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([0.16984604, 0.82474494], dtype=float32)>

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits and labels of shape
  `[batch_size, num_classes]`, but higher dimensions are supported, with
  the `axis` argument specifying the class dimension.

  `logits` and `labels` must have the same dtype (either `float16`, `float32`,
  or `float64`).

  Backpropagation will happen into both `logits` and `labels`.  To disallow
  backpropagation into `labels`, pass label tensors through `tf.stop_gradient`
  before feeding it to this function.

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    labels: Each vector along the class dimension should hold a valid
      probability distribution e.g. for the case in which labels are of shape
      `[batch_size, num_classes]`, each row of `labels[i]` must be a valid
      probability distribution.
    logits: Per-label activations, typically a linear output. These activation
      energies are interpreted as unnormalized log probabilities.
    axis: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` that contains the softmax cross entropy loss. Its type is the
    same as `logits` and its shape is the same as `labels` except that it does
    not have the last dimension of `labels`.
  """
  return softmax_cross_entropy_with_logits_v2_helper(
      labels=labels, logits=logits, axis=axis, name=name)


@tf_export(v1=["nn.softmax_cross_entropy_with_logits_v2"])
@deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def softmax_cross_entropy_with_logits_v2_helper(
    labels, logits, axis=None, name=None, dim=None):
  """Computes softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.

  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits and labels of shape
  `[batch_size, num_classes]`, but higher dimensions are supported, with
  the `axis` argument specifying the class dimension.

  `logits` and `labels` must have the same dtype (either `float16`, `float32`,
  or `float64`).

  Backpropagation will happen into both `logits` and `labels`.  To disallow
  backpropagation into `labels`, pass label tensors through `tf.stop_gradient`
  before feeding it to this function.

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    labels: Each vector along the class dimension should hold a valid
      probability distribution e.g. for the case in which labels are of shape
      `[batch_size, num_classes]`, each row of `labels[i]` must be a valid
      probability distribution.
    logits: Unscaled log probabilities.
    axis: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).
    dim: Deprecated alias for axis.

  Returns:
    A `Tensor` that contains the softmax cross entropy loss. Its type is the
    same as `logits` and its shape is the same as `labels` except that it does
    not have the last dimension of `labels`.
  """
  # TODO(pcmurray) Raise an error when the labels do not sum to 1. Note: This
  # could break users who call this with bad labels, but disregard the bad
  # results.
  axis = deprecated_argument_lookup("axis", axis, "dim", dim)
  del dim
  if axis is None:
    axis = -1

  with ops.name_scope(name, "softmax_cross_entropy_with_logits",
                      [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    convert_to_float32 = (
        logits.dtype == dtypes.float16 or logits.dtype == dtypes.bfloat16)
    precise_logits = math_ops.cast(
        logits, dtypes.float32) if convert_to_float32 else logits
    # labels and logits must be of the same type
    labels = math_ops.cast(labels, precise_logits.dtype)
    input_rank = array_ops.rank(precise_logits)
    # For shape inference.
    shape = logits.get_shape()

    # Move the dim to the end if dim is not the last dimension.
    if axis != -1:

      def _move_dim_to_end(tensor, dim_index, rank):
        return array_ops.transpose(
            tensor,
            array_ops.concat([
                math_ops.range(dim_index),
                math_ops.range(dim_index + 1, rank), [dim_index]
            ], 0))

      precise_logits = _move_dim_to_end(precise_logits, axis, input_rank)
      labels = _move_dim_to_end(labels, axis, input_rank)

    input_shape = array_ops.shape(precise_logits)

    # Make precise_logits and labels into matrices.
    precise_logits = _flatten_outer_dims(precise_logits)
    labels = _flatten_outer_dims(labels)

    # Do the actual op computation.
    # The second output tensor contains the gradients.  We use it in
    # _CrossEntropyGrad() in nn_grad but not here.
    cost, unused_backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
        precise_logits, labels, name=name)

    # The output cost shape should be the input minus axis.
    output_shape = array_ops.slice(input_shape, [0],
                                   [math_ops.subtract(input_rank, 1)])
    cost = array_ops.reshape(cost, output_shape)

    # Make shape inference work since reshape and transpose may erase its static
    # shape.
    if not context.executing_eagerly(
    ) and shape is not None and shape.dims is not None:
      shape = shape.as_list()
      del shape[axis]
      cost.set_shape(shape)

    if convert_to_float32:
      return math_ops.cast(cost, logits.dtype)
    else:
      return cost


_XENT_DEPRECATION = """
Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See `tf.nn.softmax_cross_entropy_with_logits_v2`.
"""


@tf_export(v1=["nn.softmax_cross_entropy_with_logits"])
@deprecation.deprecated(date=None, instructions=_XENT_DEPRECATION)
def softmax_cross_entropy_with_logits(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    dim=-1,
    name=None,
    axis=None):
  """Computes softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.

  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits and labels of shape
  `[batch_size, num_classes]`, but higher dimensions are supported, with
  the `dim` argument specifying the class dimension.

  Backpropagation will happen only into `logits`.  To calculate a cross entropy
  loss that allows backpropagation into both `logits` and `labels`, see
  `tf.nn.softmax_cross_entropy_with_logits_v2`.

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: Each vector along the class dimension should hold a valid
      probability distribution e.g. for the case in which labels are of shape
      `[batch_size, num_classes]`, each row of `labels[i]` must be a valid
      probability distribution.
    logits: Per-label activations, typically a linear output. These activation
      energies are interpreted as unnormalized log probabilities.
    dim: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).
    axis: Alias for dim.

  Returns:
    A `Tensor` that contains the softmax cross entropy loss. Its type is the
    same as `logits` and its shape is the same as `labels` except that it does
    not have the last dimension of `labels`.
  """
  dim = deprecated_argument_lookup("axis", axis, "dim", dim)
  _ensure_xent_args("softmax_cross_entropy_with_logits", _sentinel, labels,
                    logits)

  with ops.name_scope(name, "softmax_cross_entropy_with_logits_sg",
                      [logits, labels]) as name:
    labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")

  return softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits, axis=dim, name=name)


@tf_export(v1=["nn.sparse_softmax_cross_entropy_with_logits"])
def sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    name=None):
  """Computes sparse softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **NOTE:**  For this operation, the probability of a given label is considered
  exclusive.  That is, soft classes are not allowed, and the `labels` vector
  must provide a single specific index for the true class for each row of
  `logits` (each minibatch entry).  For soft softmax classification with
  a probability distribution for each entry, see
  `softmax_cross_entropy_with_logits_v2`.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits of shape
  `[batch_size, num_classes]` and have labels of shape
  `[batch_size]`, but higher dimensions are supported, in which
  case the `dim`-th dimension is assumed to be of size `num_classes`.
  `logits` must have the dtype of `float16`, `float32`, or `float64`, and
  `labels` must have the dtype of `int32` or `int64`.

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Per-label activations (typically a linear output) of shape
      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32`, or
      `float64`. These activation energies are interpreted as unnormalized log
      probabilities.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `labels` and of the same type as `logits`
    with the softmax cross entropy loss.

  Raises:
    ValueError: If logits are scalars (need to have rank >= 1) or if the rank
      of the labels is not equal to the rank of the logits minus one.
  """
  _ensure_xent_args("sparse_softmax_cross_entropy_with_logits", _sentinel,
                    labels, logits)

  # TODO(pcmurray) Raise an error when the label is not an index in
  # [0, num_classes). Note: This could break users who call this with bad
  # labels, but disregard the bad results.

  # Reshape logits and labels to rank 2.
  with ops.name_scope(name, "SparseSoftmaxCrossEntropyWithLogits",
                      [labels, logits]):
    labels = ops.convert_to_tensor(labels)
    logits = ops.convert_to_tensor(logits)
    precise_logits = math_ops.cast(logits, dtypes.float32) if (dtypes.as_dtype(
        logits.dtype) == dtypes.float16) else logits

    # Store label shape for result later.
    labels_static_shape = labels.get_shape()
    labels_shape = array_ops.shape(labels)
    static_shapes_fully_defined = (
        labels_static_shape.is_fully_defined() and
        logits.get_shape()[:-1].is_fully_defined())
    if logits.get_shape().ndims is not None and logits.get_shape().ndims == 0:
      raise ValueError(
          "Logits cannot be scalars - received shape %s." % logits.get_shape())
    if logits.get_shape().ndims is not None and (
        labels_static_shape.ndims is not None and
        labels_static_shape.ndims != logits.get_shape().ndims - 1):
      raise ValueError("Rank mismatch: Rank of labels (received %s) should "
                       "equal rank of logits minus 1 (received %s)." %
                       (labels_static_shape.ndims, logits.get_shape().ndims))
    if (static_shapes_fully_defined and
        labels_static_shape != logits.get_shape()[:-1]):
      raise ValueError("Shape mismatch: The shape of labels (received %s) "
                       "should equal the shape of logits except for the last "
                       "dimension (received %s)." % (labels_static_shape,
                                                     logits.get_shape()))
    # Check if no reshapes are required.
    if logits.get_shape().ndims == 2:
      cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
          precise_logits, labels, name=name)
      if logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost

    # Perform a check of the dynamic shapes if the static shapes are not fully
    # defined.
    shape_checks = []
    if not static_shapes_fully_defined:
      shape_checks.append(
          check_ops.assert_equal(
              array_ops.shape(labels),
              array_ops.shape(logits)[:-1]))
    with ops.control_dependencies(shape_checks):
      # Reshape logits to 2 dim, labels to 1 dim.
      num_classes = array_ops.shape(logits)[array_ops.rank(logits) - 1]
      precise_logits = array_ops.reshape(precise_logits, [-1, num_classes])
      labels = array_ops.reshape(labels, [-1])
      # The second output tensor contains the gradients.  We use it in
      # _CrossEntropyGrad() in nn_grad but not here.
      cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
          precise_logits, labels, name=name)
      cost = array_ops.reshape(cost, labels_shape)
      cost.set_shape(labels_static_shape)
      if logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost


@tf_export("nn.sparse_softmax_cross_entropy_with_logits", v1=[])
def sparse_softmax_cross_entropy_with_logits_v2(labels, logits, name=None):
  """Computes sparse softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **NOTE:**  For this operation, the probability of a given label is considered
  exclusive.  That is, soft classes are not allowed, and the `labels` vector
  must provide a single specific index for the true class for each row of
  `logits` (each minibatch entry).  For soft softmax classification with
  a probability distribution for each entry, see
  `softmax_cross_entropy_with_logits_v2`.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits of shape
  `[batch_size, num_classes]` and have labels of shape
  `[batch_size]`, but higher dimensions are supported, in which
  case the `dim`-th dimension is assumed to be of size `num_classes`.
  `logits` must have the dtype of `float16`, `float32`, or `float64`, and
  `labels` must have the dtype of `int32` or `int64`.

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Unscaled log probabilities of shape `[d_0, d_1, ..., d_{r-1},
      num_classes]` and dtype `float16`, `float32`, or `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `labels` and of the same type as `logits`
    with the softmax cross entropy loss.

  Raises:
    ValueError: If logits are scalars (need to have rank >= 1) or if the rank
      of the labels is not equal to the rank of the logits minus one.
  """
  return sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name=name)


@tf_export("nn.avg_pool", v1=["nn.avg_pool_v2"])
def avg_pool_v2(input, ksize, strides, padding, data_format=None, name=None):  # pylint: disable=redefined-builtin
  """Performs the avg pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    input:  Tensor of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]` if `data_format` does not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC". Pooling happens over the spatial dimensions only.
    ksize: An int or list of `ints` that has length `1`, `N` or `N+2`. The size
      of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `N` or `N+2`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. Specifies the channel dimension. For N=1 it can be
      either "NWC" (default) or "NCW", for N=2 it can be either "NHWC" (default)
      or "NCHW" and for N=3 either "NDHWC" (default) or "NCDHW".
    name: Optional name for the operation.

  Returns:
    A `Tensor` of format specified by `data_format`.
    The average pooled output tensor.
  """
  if input.shape is not None:
    n = len(input.shape) - 2
  elif data_format is not None:
    n = len(data_format) - 2
  else:
    raise ValueError(
        "The input must have a rank or a data format must be given.")
  if not 1 <= n <= 3:
    raise ValueError(
        "Input tensor must be of rank 3, 4 or 5 but was {}.".format(n + 2))

  if data_format is None:
    channel_index = n + 1
  else:
    channel_index = 1 if data_format.startswith("NC") else n + 1

  ksize = _get_sequence(ksize, n, channel_index, "ksize")
  strides = _get_sequence(strides, n, channel_index, "strides")

  avg_pooling_ops = {
      1: avg_pool1d,
      2: gen_nn_ops.avg_pool,
      3: gen_nn_ops.avg_pool3d
  }

  op = avg_pooling_ops[n]
  return op(
      input,
      ksize=ksize,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=name)


@tf_export(v1=["nn.avg_pool", "nn.avg_pool2d"])
def avg_pool(value, ksize, strides, padding, data_format="NHWC",
             name=None, input=None):  # pylint: disable=redefined-builtin
  """Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the operation.
    input: Alias for value.

  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  with ops.name_scope(name, "AvgPool", [value]) as name:
    value = deprecation.deprecated_argument_lookup(
        "input", input, "value", value)

    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")

    return gen_nn_ops.avg_pool(
        value,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)


@tf_export("nn.avg_pool2d", v1=[])
def avg_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):  # pylint: disable=redefined-builtin
  """Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    input: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  with ops.name_scope(name, "AvgPool2D", [input]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")

    return gen_nn_ops.avg_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)


@tf_export("nn.avg_pool1d")
def avg_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):  # pylint: disable=redefined-builtin
  """Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Note internally this op reshapes and uses the underlying 2d operation.

  Args:
    input: A 3-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1` or `3`. The size of the
      window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1` or `3`. The stride of
      the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: An optional string from: "NWC", "NCW". Defaults to "NWC".
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  with ops.name_scope(name, "AvgPool1D", [input]) as name:
    if data_format is None:
      data_format = "NWC"
    channel_index = 1 if data_format.startswith("NC") else 2
    ksize = [1] + _get_sequence(ksize, 1, channel_index, "ksize")
    strides = [1] + _get_sequence(strides, 1, channel_index, "strides")

    expanding_dim = 1 if data_format == "NWC" else 2
    data_format = "NHWC" if data_format == "NWC" else "NCHW"

    input = array_ops.expand_dims_v2(input, expanding_dim)
    result = gen_nn_ops.avg_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)
    return array_ops.squeeze(result, expanding_dim)


@tf_export("nn.avg_pool3d")
def avg_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):  # pylint: disable=redefined-builtin
  """Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    input: A 5-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: An int or list of `ints` that has length `1`, `3` or `5`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `3` or `5`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NDHWC' and 'NCDHW' are supported.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  with ops.name_scope(name, "AvgPool3D", [input]) as name:
    if data_format is None:
      data_format = "NDHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    ksize = _get_sequence(ksize, 3, channel_index, "ksize")
    strides = _get_sequence(strides, 3, channel_index, "strides")

    return gen_nn_ops.avg_pool3d(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)


# pylint: disable=redefined-builtin
@tf_export("nn.max_pool", v1=["nn.max_pool_v2"])
def max_pool_v2(input, ksize, strides, padding, data_format=None, name=None):
  """Performs the max pooling on the input.

  Args:
    input:  Tensor of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]` if `data_format` does not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC". Pooling happens over the spatial dimensions only.
    ksize: An int or list of `ints` that has length `1`, `N` or `N+2`. The size
      of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `N` or `N+2`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. Specifies the channel dimension. For N=1 it can be
      either "NWC" (default) or "NCW", for N=2 it can be either "NHWC" (default)
      or "NCHW" and for N=3 either "NDHWC" (default) or "NCDHW".
    name: Optional name for the operation.

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  if input.shape is not None:
    n = len(input.shape) - 2
  elif data_format is not None:
    n = len(data_format) - 2
  else:
    raise ValueError(
        "The input must have a rank or a data format must be given.")
  if not 1 <= n <= 3:
    raise ValueError(
        "Input tensor must be of rank 3, 4 or 5 but was {}.".format(n + 2))

  if data_format is None:
    channel_index = n + 1
  else:
    channel_index = 1 if data_format.startswith("NC") else n + 1

  ksize = _get_sequence(ksize, n, channel_index, "ksize")
  strides = _get_sequence(strides, n, channel_index, "strides")

  max_pooling_ops = {
      1: max_pool1d,
      2: gen_nn_ops.max_pool,
      3: gen_nn_ops.max_pool3d
  }

  op = max_pooling_ops[n]
  return op(
      input,
      ksize=ksize,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=name)
# pylint: enable=redefined-builtin


@tf_export(v1=["nn.max_pool"])
def max_pool(value,
             ksize,
             strides,
             padding,
             data_format="NHWC",
             name=None,
             input=None):  # pylint: disable=redefined-builtin
  """Performs the max pooling on the input.

  Args:
    value: A 4-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`.
      The size of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`.
      The stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.
    name: Optional name for the operation.
    input: Alias for value.

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  value = deprecation.deprecated_argument_lookup("input", input, "value", value)
  with ops.name_scope(name, "MaxPool", [value]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")
    if ((np.isscalar(ksize) and ksize == 0) or
        (isinstance(ksize,
                    (list, tuple, np.ndarray)) and any(v == 0 for v in ksize))):
      raise ValueError("ksize cannot be zero.")

    return gen_nn_ops.max_pool(
        value,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)


# pylint: disable=redefined-builtin
@tf_export("nn.max_pool1d")
def max_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
  """Performs the max pooling on the input.

  Note internally this op reshapes and uses the underlying 2d operation.

  Args:
    input: A 3-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1` or `3`. The size of the
      window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1` or `3`. The stride of
      the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: An optional string from: "NWC", "NCW". Defaults to "NWC".
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  with ops.name_scope(name, "MaxPool1d", [input]) as name:
    if data_format is None:
      data_format = "NWC"
    channel_index = 1 if data_format.startswith("NC") else 2
    ksize = [1] + _get_sequence(ksize, 1, channel_index, "ksize")
    strides = [1] + _get_sequence(strides, 1, channel_index, "strides")

    expanding_dim = 1 if data_format == "NWC" else 2
    data_format = "NHWC" if data_format == "NWC" else "NCHW"

    input = array_ops.expand_dims_v2(input, expanding_dim)
    result = gen_nn_ops.max_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)
    return array_ops.squeeze(result, expanding_dim)
# pylint: enable=redefined-builtin


# pylint: disable=redefined-builtin
@tf_export("nn.max_pool2d")
def max_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
  """Performs the max pooling on the input.

  Args:
    input: A 4-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.
    name: Optional name for the operation.

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  with ops.name_scope(name, "MaxPool2d", [input]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")

    return gen_nn_ops.max_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)
# pylint: enable=redefined-builtin


# pylint: disable=redefined-builtin
@tf_export("nn.max_pool3d")
def max_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
  """Performs the max pooling on the input.

  Args:
    input: A 5-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1`, `3` or `5`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `3` or `5`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
      The data format of the input and output data. With the default format
      "NDHWC", the data is stored in the order of: [batch, in_depth, in_height,
        in_width, in_channels]. Alternatively, the format could be "NCDHW", the
      data storage order is: [batch, in_channels, in_depth, in_height,
        in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  with ops.name_scope(name, "MaxPool3D", [input]) as name:
    if data_format is None:
      data_format = "NDHWC"
    channel_index = 1 if data_format.startswith("NC") else 4

    ksize = _get_sequence(ksize, 3, channel_index, "ksize")
    strides = _get_sequence(strides, 3, channel_index, "strides")

    return gen_nn_ops.max_pool3d(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)
# pylint: enable=redefined-builtin


@tf_export("nn.max_pool_with_argmax", v1=[])
def max_pool_with_argmax_v2(
    input,  # pylint: disable=redefined-builtin
    ksize,
    strides,
    padding,
    data_format="NHWC",
    output_dtype=dtypes.int64,
    include_batch_in_index=False,
    name=None):
  """Performs max pooling on the input and outputs both max values and indices.

  The indices in `argmax` are flattened, so that a maximum value at position
  `[b, y, x, c]` becomes flattened index: `(y * width + x) * channels + c` if
  `include_batch_in_index` is False;
  `((b * height + y) * width + x) * channels + c`
  if `include_batch_in_index` is True.

  The indices returned are always in `[0, height) x [0, width)` before
  flattening, even if padding is involved and the mathematically correct answer
  is outside (either negative or too large).  This is a bug, but fixing it is
  difficult to do in a safe backwards compatible way, especially due to
  flattening.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`,
      `uint32`, `uint64`.
      4-D with shape `[batch, height, width, channels]`.  Input to pool over.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`.
      The size of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string`, must be set to `"NHWC"`. Defaults to
      `"NHWC"`.
      Specify the data format of the input and output data.
    output_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`.
      Defaults to `tf.int64`.
      The dtype of the returned argmax tensor.
    include_batch_in_index: An optional `boolean`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, argmax).

    output: A `Tensor`. Has the same type as `input`.
    argmax: A `Tensor` of type `output_dtype`.
  """

  if data_format != "NHWC":
    raise ValueError("Data formats other than 'NHWC' are not yet supported")

  ksize = _get_sequence(ksize, 2, 3, "ksize")
  strides = _get_sequence(strides, 2, 3, "strides")

  return gen_nn_ops.max_pool_with_argmax(
      input=input,
      ksize=ksize,
      strides=strides,
      padding=padding,
      Targmax=output_dtype,
      include_batch_in_index=include_batch_in_index,
      name=name)


@tf_export(v1=["nn.max_pool_with_argmax"])
def max_pool_with_argmax_v1(  # pylint: disable=missing-docstring,invalid-name
    input,  # pylint: disable=redefined-builtin
    ksize,
    strides,
    padding,
    data_format="NHWC",
    Targmax=None,
    name=None,
    output_dtype=None,
    include_batch_in_index=False):
  if data_format != "NHWC":
    raise ValueError("Data formats other than 'NHWC' are not yet supported")

  Targmax = deprecated_argument_lookup(
      "output_dtype", output_dtype, "Targmax", Targmax)
  if Targmax is None:
    Targmax = dtypes.int64
  return gen_nn_ops.max_pool_with_argmax(
      input=input,
      ksize=ksize,
      strides=strides,
      padding=padding,
      Targmax=Targmax,
      include_batch_in_index=include_batch_in_index,
      name=name)


max_pool_with_argmax_v1.__doc__ = gen_nn_ops.max_pool_with_argmax.__doc__


@ops.RegisterStatistics("Conv3D", "flops")
def _calc_conv3d_flops(graph, node):
  """Calculates the compute resources needed for Conv3D."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(
      graph, node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_time = int(filter_shape[0])
  filter_height = int(filter_shape[1])
  filter_width = int(filter_shape[2])
  filter_in_depth = int(filter_shape[3])
  output_count = np.prod(output_shape.as_list(), dtype=np.int64)
  return ops.OpStats("flops", (output_count * filter_in_depth * filter_time *
                               filter_height * filter_width * 2))


@ops.RegisterStatistics("Conv2D", "flops")
def _calc_conv_flops(graph, node):
  """Calculates the compute resources needed for Conv2D."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(
      graph, node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_height = int(filter_shape[0])
  filter_width = int(filter_shape[1])
  filter_in_depth = int(filter_shape[2])
  output_count = np.prod(output_shape.as_list(), dtype=np.int64)
  return ops.OpStats(
      "flops",
      (output_count * filter_in_depth * filter_height * filter_width * 2))


@ops.RegisterStatistics("DepthwiseConv2dNative", "flops")
def _calc_depthwise_conv_flops(graph, node):
  """Calculates the compute resources needed for DepthwiseConv2dNative."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(
      graph, node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_height = int(filter_shape[0])
  filter_width = int(filter_shape[1])
  output_count = np.prod(output_shape.as_list(), dtype=np.int64)
  return ops.OpStats("flops", (output_count * filter_height * filter_width * 2))


@ops.RegisterStatistics("BiasAdd", "flops")
def _calc_bias_add_flops(graph, node):
  """Calculates the computing needed for BiasAdd."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  input_count = np.prod(input_shape.as_list())
  return ops.OpStats("flops", input_count)


@tf_export(v1=["nn.xw_plus_b"])
def xw_plus_b(x, weights, biases, name=None):  # pylint: disable=invalid-name
  """Computes matmul(x, weights) + biases.

  Args:
    x: a 2D tensor.  Dimensions typically: batch, in_units
    weights: a 2D tensor.  Dimensions typically: in_units, out_units
    biases: a 1D tensor.  Dimensions: out_units
    name: A name for the operation (optional).  If not specified
      "xw_plus_b" is used.

  Returns:
    A 2-D Tensor computing matmul(x, weights) + biases.
    Dimensions typically: batch, out_units.
  """
  with ops.name_scope(name, "xw_plus_b", [x, weights, biases]) as name:
    x = ops.convert_to_tensor(x, name="x")
    weights = ops.convert_to_tensor(weights, name="weights")
    biases = ops.convert_to_tensor(biases, name="biases")
    mm = math_ops.matmul(x, weights)
    return bias_add(mm, biases, name=name)


def xw_plus_b_v1(x, weights, biases, name=None):
  """Computes matmul(x, weights) + biases.

  This is a deprecated version of that will soon be removed.

  Args:
    x: a 2D tensor.  Dimensions typically: batch, in_units
    weights: a 2D tensor.  Dimensions typically: in_units, out_units
    biases: a 1D tensor.  Dimensions: out_units
    name: A name for the operation (optional).  If not specified
      "xw_plus_b_v1" is used.

  Returns:
    A 2-D Tensor computing matmul(x, weights) + biases.
    Dimensions typically: batch, out_units.
  """
  with ops.name_scope(name, "xw_plus_b_v1", [x, weights, biases]) as name:
    x = ops.convert_to_tensor(x, name="x")
    weights = ops.convert_to_tensor(weights, name="weights")
    biases = ops.convert_to_tensor(biases, name="biases")
    mm = math_ops.matmul(x, weights)
    return bias_add_v1(mm, biases, name=name)


def _get_noise_shape(x, noise_shape):
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape


@tf_export(v1=["nn.dropout"])
@deprecation.deprecated_args(None, "Please use `rate` instead of `keep_prob`. "
                             "Rate should be set to `rate = 1 - keep_prob`.",
                             "keep_prob")
def dropout(x, keep_prob=None, noise_shape=None, seed=None, name=None,
            rate=None):
  """Computes dropout.

  For each element of `x`, with probability `rate`, outputs `0`, and otherwise
  scales up the input by `1 / (1-rate)`. The scaling is such that the expected
  sum is unchanged.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.

  Args:
    x: A floating point tensor.
    keep_prob: (deprecated) A deprecated alias for `(1-rate)`.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
    name: A name for this operation (optional).
    rate: A scalar `Tensor` with the same type as `x`. The probability that each
      element of `x` is discarded.

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating
      point tensor.
  """
  try:
    keep = 1. - keep_prob if keep_prob is not None else None
  except TypeError:
    raise ValueError("keep_prob must be a floating point number or Tensor "
                     "(got %r)" % keep_prob)

  rate = deprecation.deprecated_argument_lookup(
      "rate", rate,
      "keep_prob", keep)

  if rate is None:
    raise ValueError("You must provide a rate to dropout.")

  return dropout_v2(x, rate, noise_shape=noise_shape, seed=seed, name=name)


@tf_export("nn.dropout", v1=[])
def dropout_v2(x, rate, noise_shape=None, seed=None, name=None):
  """Computes dropout: randomly sets elements to zero to prevent overfitting.

  Note: The behavior of dropout has changed between TensorFlow 1.x and 2.x.
  When converting 1.x code, please use named arguments to ensure behavior stays
  consistent.

  See also: `tf.keras.layers.Dropout` for a dropout layer.

  [Dropout](https://arxiv.org/abs/1207.0580) is useful for regularizing DNN
  models. Inputs elements are randomly set to zero (and the other elements are
  rescaled). This encourages each node to be independently useful, as it cannot
  rely on the output of other nodes.

  More precisely: With probability `rate` elements of `x` are set to `0`.
  The remaining elemenst are scaled up by `1.0 / (1 - rate)`, so that the
  expected value is preserved.

  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate = 0.5, seed = 1).numpy()
  array([[2., 0., 0., 2., 2.],
       [2., 2., 2., 2., 2.],
       [2., 0., 2., 0., 2.]], dtype=float32)

  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate = 0.8, seed = 1).numpy()
  array([[0., 0., 0., 5., 5.],
       [0., 5., 0., 5., 0.],
       [5., 0., 5., 0., 5.]], dtype=float32)

  >>> tf.nn.dropout(x, rate = 0.0) == x
  <tf.Tensor: shape=(3, 5), dtype=bool, numpy=
    array([[ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]])>


  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions. This is useful for dropping whole
  channels from an image or sequence. For example:

  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,10])
  >>> tf.nn.dropout(x, rate = 2/3, noise_shape=[1,10], seed=1).numpy()
  array([[0., 0., 0., 3., 3., 0., 3., 3., 3., 0.],
       [0., 0., 0., 3., 3., 0., 3., 3., 3., 0.],
       [0., 0., 0., 3., 3., 0., 3., 3., 3., 0.]], dtype=float32)

  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because theoutput would be all zeros,
      which is likely not what was intended.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    # TODO(b/144930399): Remove this once the compatible window is passed.
    if compat.forward_compatible(2019, 12, 16):
      is_rate_number = isinstance(rate, numbers.Real)
      if is_rate_number and (rate < 0 or rate >= 1):
        raise ValueError("rate must be a scalar tensor or a float in the "
                         "range [0, 1), got %g" % rate)
      x = ops.convert_to_tensor(x, name="x")
      x_dtype = x.dtype
      if not x_dtype.is_floating:
        raise ValueError("x has to be a floating point tensor since it's going "
                         "to be scaled. Got a %s tensor instead." % x_dtype)
      is_executing_eagerly = context.executing_eagerly()
      if not tensor_util.is_tensor(rate):
        if is_rate_number:
          keep_prob = 1 - rate
          scale = 1 / keep_prob
          scale = ops.convert_to_tensor(scale, dtype=x_dtype)
          ret = gen_math_ops.mul(x, scale)
        else:
          raise ValueError("rate is neither scalar nor scalar tensor %r" % rate)
      else:
        rate.get_shape().assert_has_rank(0)
        rate_dtype = rate.dtype
        if rate_dtype != x_dtype:
          if not rate_dtype.is_compatible_with(x_dtype):
            raise ValueError(
                "Tensor dtype %s is incomptaible with Tensor dtype %s: %r" %
                (x_dtype.name, rate_dtype.name, rate))
          rate = gen_math_ops.cast(rate, x_dtype, name="rate")
        one_tensor = constant_op.constant(1, dtype=x_dtype)
        ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))

      noise_shape = _get_noise_shape(x, noise_shape)
      # Sample a uniform distribution on [0.0, 1.0) and select values larger
      # than rate.
      #
      # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
      # and subtract 1.0.
      random_tensor = random_ops.random_uniform(
          noise_shape, seed=seed, dtype=x_dtype)
      # NOTE: if (1.0 + rate) - 1 is equal to rate, then that float is selected,
      # hence a >= comparison is used.
      keep_mask = random_tensor >= rate
      ret = gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
      if not is_executing_eagerly:
        ret.set_shape(x.get_shape())
      return ret
    else:
      x = ops.convert_to_tensor(x, name="x")
      if not x.dtype.is_floating:
        raise ValueError("x has to be a floating point tensor since it will "
                         "be scaled. Got a %s tensor instead." % x.dtype)
      if isinstance(rate, numbers.Real):
        if not (rate >= 0 and rate < 1):
          raise ValueError("rate must be a scalar tensor or a float in the "
                           "range [0, 1), got %g" % rate)
        if rate > 0.5:
          logging.log_first_n(
              logging.WARN, "Large dropout rate: %g (>0.5). In TensorFlow "
              "2.x, dropout() uses dropout rate instead of keep_prob. "
              "Please ensure that this is intended.", 5, rate)

      # Early return if nothing needs to be dropped.
      if isinstance(rate, numbers.Real) and rate == 0:
        return x
      if context.executing_eagerly():
        if isinstance(rate, ops.EagerTensor):
          if rate.numpy() == 0:
            return x
      else:
        rate = ops.convert_to_tensor(rate, dtype=x.dtype, name="rate")
        rate.get_shape().assert_has_rank(0)

        # Do nothing if we know rate == 0
        if tensor_util.constant_value(rate) == 0:
          return x

      noise_shape = _get_noise_shape(x, noise_shape)
      # Sample a uniform distribution on [0.0, 1.0) and select values larger
      # than rate.
      #
      # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
      # and subtract 1.0.
      random_tensor = random_ops.random_uniform(
          noise_shape, seed=seed, dtype=x.dtype)
      keep_prob = 1 - rate
      scale = 1 / keep_prob
      # NOTE: if (1.0 + rate) - 1 is equal to rate, then that
      # float is selected, hence we use a >= comparison.
      keep_mask = random_tensor >= rate
      ret = x * scale * math_ops.cast(keep_mask, x.dtype)
      if not context.executing_eagerly():
        ret.set_shape(x.get_shape())
      return ret


@tf_export("math.top_k", "nn.top_k")
def top_k(input, k=1, sorted=True, name=None):  # pylint: disable=redefined-builtin
  """Finds values and indices of the `k` largest entries for the last dimension.

  If the input is a vector (rank=1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  Args:
    input: 1-D or higher `Tensor` with last dimension at least `k`.
    k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last
      dimension (along each row for matrices).
    sorted: If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: Optional name for the operation.

  Returns:
    values: The `k` largest elements along each last dimensional slice.
    indices: The indices of `values` within the last dimension of `input`.
  """
  return gen_nn_ops.top_kv2(input, k=k, sorted=sorted, name=name)


def nth_element(input, n, reverse=False, name=None):  # pylint: disable=redefined-builtin
  r"""Finds values of the `n`-th smallest value for the last dimension.

  Note that n is zero-indexed.

  If the input is a vector (rank-1), finds the entries which is the nth-smallest
  value in the vector and outputs their values as scalar tensor.

  For matrices (resp. higher rank input), computes the entries which is the
  nth-smallest value in each row (resp. vector along the last dimension). Thus,

      values.shape = input.shape[:-1]

  Args:
    input: 1-D or higher `Tensor` with last dimension at least `n+1`.
    n: A `Tensor` of type `int32`.
      0-D. Position of sorted vector to select along the last dimension (along
      each row for matrices). Valid range of n is `[0, input.shape[:-1])`
    reverse: An optional `bool`. Defaults to `False`.
      When set to True, find the nth-largest value in the vector and vice
      versa.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    The `n`-th order statistic along each last dimensional slice.
  """
  return gen_nn_ops.nth_element(input, n, reverse=reverse, name=name)


@tf_export(v1=["nn.fractional_max_pool"])
@deprecation.deprecated(date=None, instructions="`seed2` and `deterministic` "
                        "args are deprecated.  Use fractional_max_pool_v2.")
def fractional_max_pool(value,
                        pooling_ratio,
                        pseudo_random=False,
                        overlapping=False,
                        deterministic=False,
                        seed=0,
                        seed2=0,
                        name=None):   # pylint: disable=redefined-builtin
  r"""Performs fractional max pooling on the input.

  This is a deprecated version of `fractional_max_pool`.

  Fractional max pooling is slightly different than regular max pooling.  In
  regular max pooling, you downsize an input set by taking the maximum value of
  smaller N x N subsections of the set (often 2x2), and try to reduce the set by
  a factor of N, where N is an integer.  Fractional max pooling, as you might
  expect from the word "fractional", means that the overall reduction ratio N
  does not have to be an integer.

  The sizes of the pooling regions are generated randomly but are fairly
  uniform.  For example, let's look at the height dimension, and the constraints
  on the list of rows that will be pool boundaries.

  First we define the following:

  1.  input_row_length : the number of rows from the input set
  2.  output_row_length : which will be smaller than the input
  3.  alpha = input_row_length / output_row_length : our reduction ratio
  4.  K = floor(alpha)
  5.  row_pooling_sequence : this is the result list of pool boundary rows

  Then, row_pooling_sequence should satisfy:

  1.  a[0] = 0 : the first value of the sequence is 0
  2.  a[end] = input_row_length : the last value of the sequence is the size
  3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
  4.  length(row_pooling_sequence) = output_row_length+1

  Args:
    value: A `Tensor`. 4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length >= 4.  Pooling ratio for
      each dimension of `value`, currently only supports row and col dimension
      and should be >= 1.0. For example, a valid pooling ratio looks like [1.0,
      1.44, 1.73, 1.0]. The first and last elements must be 1.0 because we don't
      allow pooling on batch and channels dimensions.  1.44 and 1.73 are pooling
      ratio on height and width dimensions respectively.
    pseudo_random: An optional `bool`.  Defaults to `False`. When set to `True`,
      generates the pooling sequence in a pseudorandom fashion, otherwise, in a
      random fashion. Check (Graham, 2015) for difference between
      pseudorandom and random.
    overlapping: An optional `bool`.  Defaults to `False`.  When set to `True`,
      it means when pooling, the values at the boundary of adjacent pooling
      cells are used by both cells. For example:
      `index  0  1  2  3  4`
      `value  20 5  16 3  7`
      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used
      twice.  The result would be [20, 16] for fractional max pooling.
    deterministic: An optional `bool`.  Deprecated; use `fractional_max_pool_v2`
      instead.
    seed: An optional `int`.  Defaults to `0`.  If set to be non-zero, the
      random number generator is seeded by the given seed.  Otherwise it is
      seeded by a random seed.
    seed2: An optional `int`.  Deprecated; use `fractional_max_pool_v2` instead.
    name: A name for the operation (optional).

  Returns:
  A tuple of `Tensor` objects (`output`, `row_pooling_sequence`,
  `col_pooling_sequence`).
    output: Output `Tensor` after fractional max pooling.  Has the same type as
      `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.

  References:
    Fractional Max-Pooling:
      [Graham, 2015](https://arxiv.org/abs/1412.6071)
      ([pdf](https://arxiv.org/pdf/1412.6071.pdf))
  """
  return gen_nn_ops.fractional_max_pool(value, pooling_ratio, pseudo_random,
                                        overlapping, deterministic, seed, seed2,
                                        name)


@tf_export("nn.fractional_max_pool", v1=[])
def fractional_max_pool_v2(value,
                           pooling_ratio,
                           pseudo_random=False,
                           overlapping=False,
                           seed=0,
                           name=None):  # pylint: disable=redefined-builtin
  r"""Performs fractional max pooling on the input.

  Fractional max pooling is slightly different than regular max pooling.  In
  regular max pooling, you downsize an input set by taking the maximum value of
  smaller N x N subsections of the set (often 2x2), and try to reduce the set by
  a factor of N, where N is an integer.  Fractional max pooling, as you might
  expect from the word "fractional", means that the overall reduction ratio N
  does not have to be an integer.

  The sizes of the pooling regions are generated randomly but are fairly
  uniform.  For example, let's look at the height dimension, and the constraints
  on the list of rows that will be pool boundaries.

  First we define the following:

  1.  input_row_length : the number of rows from the input set
  2.  output_row_length : which will be smaller than the input
  3.  alpha = input_row_length / output_row_length : our reduction ratio
  4.  K = floor(alpha)
  5.  row_pooling_sequence : this is the result list of pool boundary rows

  Then, row_pooling_sequence should satisfy:

  1.  a[0] = 0 : the first value of the sequence is 0
  2.  a[end] = input_row_length : the last value of the sequence is the size
  3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
  4.  length(row_pooling_sequence) = output_row_length+1

  Args:
    value: A `Tensor`. 4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: An int or list of `ints` that has length `1`, `2` or `4`.
      Pooling ratio for each dimension of `value`, currently only supports row
      and col dimension and should be >= 1.0. For example, a valid pooling ratio
      looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements must be 1.0
      because we don't allow pooling on batch and channels dimensions.  1.44 and
      1.73 are pooling ratio on height and width dimensions respectively.
    pseudo_random: An optional `bool`.  Defaults to `False`. When set to `True`,
      generates the pooling sequence in a pseudorandom fashion, otherwise, in a
      random fashion. Check paper (Graham, 2015) for difference between
      pseudorandom and random.
    overlapping: An optional `bool`.  Defaults to `False`.  When set to `True`,
      it means when pooling, the values at the boundary of adjacent pooling
      cells are used by both cells. For example:
      `index  0  1  2  3  4`
      `value  20 5  16 3  7`
      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used
      twice.  The result would be [20, 16] for fractional max pooling.
    seed: An optional `int`.  Defaults to `0`.  If set to be non-zero, the
      random number generator is seeded by the given seed.  Otherwise it is
      seeded by a random seed.
    name: A name for the operation (optional).

  Returns:
  A tuple of `Tensor` objects (`output`, `row_pooling_sequence`,
  `col_pooling_sequence`).
    output: Output `Tensor` after fractional max pooling.  Has the same type as
      `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.

  References:
    Fractional Max-Pooling:
      [Graham, 2015](https://arxiv.org/abs/1412.6071)
      ([pdf](https://arxiv.org/pdf/1412.6071.pdf))
  """
  pooling_ratio = _get_sequence(pooling_ratio, 2, 3, "pooling_ratio")

  if seed == 0:
    return gen_nn_ops.fractional_max_pool(value, pooling_ratio, pseudo_random,
                                          overlapping, deterministic=False,
                                          seed=0, seed2=0, name=name)
  else:
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_nn_ops.fractional_max_pool(value, pooling_ratio, pseudo_random,
                                          overlapping, deterministic=True,
                                          seed=seed1, seed2=seed2, name=name)


@tf_export(v1=["nn.fractional_avg_pool"])
@deprecation.deprecated(date=None, instructions="`seed2` and `deterministic` "
                        "args are deprecated.  Use fractional_avg_pool_v2.")
def fractional_avg_pool(value,
                        pooling_ratio,
                        pseudo_random=False,
                        overlapping=False,
                        deterministic=False,
                        seed=0,
                        seed2=0,
                        name=None):  # pylint: disable=redefined-builtin
  r"""Performs fractional average pooling on the input.

  This is a deprecated version of `fractional_avg_pool`.

  Fractional average pooling is similar to Fractional max pooling in the pooling
  region generation step. The only difference is that after pooling regions are
  generated, a mean operation is performed instead of a max operation in each
  pooling region.

  Args:
    value: A `Tensor`. 4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length >= 4.  Pooling ratio for
      each dimension of `value`, currently only supports row and col dimension
      and should be >= 1.0. For example, a valid pooling ratio looks like [1.0,
      1.44, 1.73, 1.0]. The first and last elements must be 1.0 because we don't
      allow pooling on batch and channels dimensions.  1.44 and 1.73 are pooling
      ratio on height and width dimensions respectively.
    pseudo_random: An optional `bool`.  Defaults to `False`. When set to `True`,
      generates the pooling sequence in a pseudorandom fashion, otherwise, in a
      random fashion. Check paper (Graham, 2015) for difference between
      pseudorandom and random.
    overlapping: An optional `bool`.  Defaults to `False`.  When set to `True`,
      it means when pooling, the values at the boundary of adjacent pooling
      cells are used by both cells. For example:
      `index  0  1  2  3  4`
      `value  20 5  16 3  7`
      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used
      twice.  The result would be [20, 16] for fractional avg pooling.
    deterministic: An optional `bool`.  Deprecated; use `fractional_avg_pool_v2`
      instead.
    seed: An optional `int`.  Defaults to `0`.  If set to be non-zero, the
      random number generator is seeded by the given seed.  Otherwise it is
      seeded by a random seed.
    seed2: An optional `int`.  Deprecated; use `fractional_avg_pool_v2` instead.
    name: A name for the operation (optional).

  Returns:
  A tuple of `Tensor` objects (`output`, `row_pooling_sequence`,
  `col_pooling_sequence`).
    output: Output `Tensor` after fractional avg pooling.  Has the same type as
      `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.

  References:
    Fractional Max-Pooling:
      [Graham, 2015](https://arxiv.org/abs/1412.6071)
      ([pdf](https://arxiv.org/pdf/1412.6071.pdf))
  """
  return gen_nn_ops.fractional_avg_pool(value, pooling_ratio, pseudo_random,
                                        overlapping, deterministic, seed, seed2,
                                        name=name)


@tf_export("nn.fractional_avg_pool", v1=[])
def fractional_avg_pool_v2(value,
                           pooling_ratio,
                           pseudo_random=False,
                           overlapping=False,
                           seed=0,
                           name=None):  # pylint: disable=redefined-builtin
  r"""Performs fractional average pooling on the input.

  Fractional average pooling is similar to Fractional max pooling in the pooling
  region generation step. The only difference is that after pooling regions are
  generated, a mean operation is performed instead of a max operation in each
  pooling region.

  Args:
    value: A `Tensor`. 4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length >= 4.  Pooling ratio for
      each dimension of `value`, currently only supports row and col dimension
      and should be >= 1.0. For example, a valid pooling ratio looks like [1.0,
      1.44, 1.73, 1.0]. The first and last elements must be 1.0 because we don't
      allow pooling on batch and channels dimensions.  1.44 and 1.73 are pooling
      ratio on height and width dimensions respectively.
    pseudo_random: An optional `bool`.  Defaults to `False`. When set to `True`,
      generates the pooling sequence in a pseudorandom fashion, otherwise, in a
      random fashion. Check paper (Graham, 2015) for difference between
      pseudorandom and random.
    overlapping: An optional `bool`.  Defaults to `False`.  When set to `True`,
      it means when pooling, the values at the boundary of adjacent pooling
      cells are used by both cells. For example:
      `index  0  1  2  3  4`
      `value  20 5  16 3  7`
      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used
      twice.  The result would be [20, 16] for fractional avg pooling.
    seed: An optional `int`.  Defaults to `0`.  If set to be non-zero, the
      random number generator is seeded by the given seed.  Otherwise it is
      seeded by a random seed.
    name: A name for the operation (optional).

  Returns:
  A tuple of `Tensor` objects (`output`, `row_pooling_sequence`,
  `col_pooling_sequence`).
    output: Output `Tensor` after fractional avg pooling.  Has the same type as
      `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.

  References:
    Fractional Max-Pooling:
      [Graham, 2015](https://arxiv.org/abs/1412.6071)
      ([pdf](https://arxiv.org/pdf/1412.6071.pdf))
  """
  if seed == 0:
    return gen_nn_ops.fractional_avg_pool(value, pooling_ratio, pseudo_random,
                                          overlapping, deterministic=False,
                                          seed=0, seed2=0, name=name)
  else:
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_nn_ops.fractional_avg_pool(value, pooling_ratio, pseudo_random,
                                          overlapping, deterministic=True,
                                          seed=seed1, seed2=seed2, name=name)


@ops.RegisterStatistics("Dilation2D", "flops")
def _calc_dilation2d_flops(graph, node):
  """Calculates the compute resources needed for Dilation2D."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(
      graph, node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_height = int(filter_shape[0])
  filter_width = int(filter_shape[1])
  output_count = np.prod(output_shape.as_list(), dtype=np.int64)
  return ops.OpStats("flops", (output_count * filter_height * filter_width * 2))


@tf_export(v1=["nn.erosion2d"])
def erosion2d(value, kernel, strides, rates, padding, name=None):
  """Computes the grayscale erosion of 4-D `value` and 3-D `kernel` tensors.

  The `value` tensor has shape `[batch, in_height, in_width, depth]` and the
  `kernel` tensor has shape `[kernel_height, kernel_width, depth]`, i.e.,
  each input channel is processed independently of the others with its own
  structuring function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the
  output tensor depend on the `padding` algorithm. We currently only support the
  default "NHWC" `data_format`.

  In detail, the grayscale morphological 2-D erosion is given by:

      output[b, y, x, c] =
         min_{dy, dx} value[b,
                            strides[1] * y - rates[1] * dy,
                            strides[2] * x - rates[2] * dx,
                            c] -
                      kernel[dy, dx, c]

  Duality: The erosion of `value` by the `kernel` is equal to the negation of
  the dilation of `-value` by the reflected `kernel`.

  Args:
    value: A `Tensor`. 4-D with shape `[batch, in_height, in_width, depth]`.
    kernel: A `Tensor`. Must have the same type as `value`.
      3-D with shape `[kernel_height, kernel_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional). If not specified "erosion2d"
      is used.

  Returns:
    A `Tensor`. Has the same type as `value`.
    4-D with shape `[batch, out_height, out_width, depth]`.

  Raises:
    ValueError: If the `value` depth does not match `kernel`' shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  with ops.name_scope(name, "erosion2d", [value, kernel]) as name:
    # Reduce erosion to dilation by duality.
    return math_ops.negative(
        gen_nn_ops.dilation2d(
            input=math_ops.negative(value),
            filter=array_ops.reverse_v2(kernel, [0, 1]),
            strides=strides,
            rates=rates,
            padding=padding,
            name=name))


@tf_export("nn.erosion2d", v1=[])
def erosion2d_v2(value,
                 filters,
                 strides,
                 padding,
                 data_format,
                 dilations,
                 name=None):
  """Computes the grayscale erosion of 4-D `value` and 3-D `filters` tensors.

  The `value` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filters` tensor has shape `[filters_height, filters_width, depth]`, i.e.,
  each input channel is processed independently of the others with its own
  structuring function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the
  output tensor depend on the `padding` algorithm. We currently only support the
  default "NHWC" `data_format`.

  In detail, the grayscale morphological 2-D erosion is given by:

      output[b, y, x, c] =
         min_{dy, dx} value[b,
                            strides[1] * y - dilations[1] * dy,
                            strides[2] * x - dilations[2] * dx,
                            c] -
                      filters[dy, dx, c]

  Duality: The erosion of `value` by the `filters` is equal to the negation of
  the dilation of `-value` by the reflected `filters`.

  Args:
    value: A `Tensor`. 4-D with shape `[batch, in_height, in_width, depth]`.
    filters: A `Tensor`. Must have the same type as `value`.
      3-D with shape `[filters_height, filters_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: A `string`, only `"NHWC"` is currently supported.
    dilations: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    name: A name for the operation (optional). If not specified "erosion2d"
      is used.

  Returns:
    A `Tensor`. Has the same type as `value`.
    4-D with shape `[batch, out_height, out_width, depth]`.

  Raises:
    ValueError: If the `value` depth does not match `filters`' shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  if data_format != "NHWC":
    raise ValueError("Data formats other than NHWC are not yet supported")

  with ops.name_scope(name, "erosion2d", [value, filters]) as name:
    # Reduce erosion to dilation by duality.
    return math_ops.negative(
        gen_nn_ops.dilation2d(
            input=math_ops.negative(value),
            filter=array_ops.reverse_v2(filters, [0, 1]),
            strides=strides,
            rates=dilations,
            padding=padding,
            name=name))


@tf_export(v1=["math.in_top_k", "nn.in_top_k"])
def in_top_k(predictions, targets, k, name=None):
  r"""Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is finite (not inf, -inf, or nan) and among
  the top `k` predictions among all predictions for example `i`. Note that the
  behavior of `InTopK` differs from the `TopK` op in its handling of ties; if
  multiple classes have the same prediction value and straddle the top-`k`
  boundary, all of those classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: An `int`. Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. Computed Precision at `k` as a `bool Tensor`.
  """
  with ops.name_scope(name, "in_top_k"):
    return gen_nn_ops.in_top_kv2(predictions, targets, k, name=name)


@tf_export("math.in_top_k", "nn.in_top_k", v1=[])
def in_top_k_v2(targets, predictions, k, name=None):
  return in_top_k(predictions, targets, k, name)


in_top_k_v2.__doc__ = in_top_k.__doc__


tf_export(v1=["nn.quantized_avg_pool"])(gen_nn_ops.quantized_avg_pool)
tf_export(v1=["nn.quantized_conv2d"])(gen_nn_ops.quantized_conv2d)
tf_export(v1=["nn.quantized_relu_x"])(gen_nn_ops.quantized_relu_x)
tf_export(v1=["nn.quantized_max_pool"])(gen_nn_ops.quantized_max_pool)
