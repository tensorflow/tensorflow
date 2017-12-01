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

import numbers

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import


# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn

# pylint: disable=protected-access


def _non_atrous_convolution(input, filter, padding, data_format=None,  # pylint: disable=redefined-builtin
                            strides=None, name=None):
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
    input = ops.convert_to_tensor(input, name="input")
    input_shape = input.get_shape()
    filter = ops.convert_to_tensor(filter, name="filter")
    filter_shape = filter.get_shape()
    op = _NonAtrousConvolution(input_shape,
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

  def __init__(self,
               input_shape,
               filter_shape,  # pylint: disable=redefined-builtin
               padding, data_format=None,
               strides=None, name=None):
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
      raise ValueError("len(strides)=%d, but should be %d" %
                       (len(strides), conv_dims))
    if conv_dims == 1:
      # conv1d uses the 2-d data format names
      if data_format is None or data_format == "NWC":
        data_format_2d = "NHWC"
      elif data_format == "NCW":
        data_format_2d = "NCHW"
      else:
        raise ValueError("data_format must be \"NWC\" or \"NCW\".")
      self.strides = strides[0]
      self.data_format = data_format_2d
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
      self.conv_op = gen_nn_ops.conv2d
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
    return conv1d(value=input, filters=filter, stride=strides, padding=padding,
                  data_format=data_format, name=name)
  # pylint: enable=redefined-builtin

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.conv_op(
        input=inp,
        filter=filter,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        name=self.name)


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
  input = ops.convert_to_tensor(input, name="input")
  input_shape = input.get_shape()

  def build_op(num_spatial_dims, padding):
    return lambda inp, _: op(inp, num_spatial_dims, padding)

  new_op = _WithSpaceToBatch(input_shape,
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
    dilation_rate = ops.convert_to_tensor(dilation_rate,
                                          dtypes.int32,
                                          name="dilation_rate")
    try:
      rate_shape = dilation_rate.get_shape().with_rank(1)
    except ValueError:
      raise ValueError("rate must be rank 1")

    if not dilation_rate.get_shape().is_fully_defined():
      raise ValueError("rate must have known shape")

    num_spatial_dims = rate_shape[0].value

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
          "integers")  # pylint: disable=line-too-long

    if data_format is not None and data_format.startswith("NC"):
      expected_input_rank = spatial_dims[-1]
    else:
      expected_input_rank = spatial_dims[-1] + 1

    try:
      input_shape.with_rank_at_least(expected_input_rank)
    except ValueError:
      ValueError("input tensor must have rank %d at least" %
                 (expected_input_rank))

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
            const_filter_shape,
            num_spatial_dims,
            rate_or_const_rate)
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
          filter_shape,
          self.num_spatial_dims,
          self.rate_or_const_rate)
    paddings, crops = array_ops.required_space_to_batch_paddings(
        input_shape=input_spatial_shape,
        base_paddings=base_paddings,
        block_shape=self.dilation_rate)

    dilation_rate = _with_space_to_batch_adjust(self.dilation_rate, 1,
                                                spatial_dims)
    paddings = _with_space_to_batch_adjust(paddings, 0, spatial_dims)
    crops = _with_space_to_batch_adjust(crops, 0, spatial_dims)
    input_converted = array_ops.space_to_batch_nd(
        input=inp,
        block_shape=dilation_rate,
        paddings=paddings)

    result = self.op(input_converted, filter)

    result_converted = array_ops.batch_to_space_nd(
        input=result, block_shape=dilation_rate, crops=crops)
    return result_converted

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.call(inp, filter)


def _with_space_to_batch_base_paddings(filter_shape, num_spatial_dims,
                                       rate_or_const_rate):
  """Helper function to compute base_paddings."""
  # Spatial dimensions of the filters and the upsampled filters in which we
  # introduce (rate - 1) zeros between consecutive filter values.
  filter_spatial_shape = filter_shape[:num_spatial_dims]
  dilated_filter_spatial_shape = (filter_spatial_shape +
                                  (filter_spatial_shape - 1) *
                                  (rate_or_const_rate - 1))
  pad_extra_shape = dilated_filter_spatial_shape - 1

  # When full_padding_shape is odd, we pad more at end, following the same
  # convention as conv2d.
  pad_extra_start = pad_extra_shape // 2
  pad_extra_end = pad_extra_shape - pad_extra_start
  base_paddings = array_ops.stack([[pad_extra_start[i], pad_extra_end[i]]
                                   for i in range(num_spatial_dims)])
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
    raise ValueError("len(strides)=%d but should be %d" %
                     (len(strides), num_spatial_dims))
  strides = np.array(strides, dtype=np.int32)
  if np.any(strides < 1):
    raise ValueError("all values of strides must be positive")

  if np.any(strides > 1) and np.any(dilation_rate > 1):
    raise ValueError(
        "strides > 1 not supported in conjunction with dilation_rate > 1")
  return strides, dilation_rate


def convolution(input, filter,  # pylint: disable=redefined-builtin
                padding, strides=None, dilation_rate=None,
                name=None, data_format=None):
  # pylint: disable=line-too-long
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
  @{tf.nn.convolution$comment here}.

  In the case that `data_format` does start with `"NC"`, the `input` and output
  (but not the `filter`) are simply transposed as follows:

    convolution(input, data_format, **kwargs) =
      tf.transpose(convolution(tf.transpose(input, [0] + range(2,N+2) + [1]),
                               **kwargs),
                   [0, N+1] + range(1, N+1))

  It is required that 1 <= N <= 3.

  Args:
    input: An N-D `Tensor` of type `T`, of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    filter: An N-D `Tensor` with the same type as `input` and shape
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
  # pylint: enable=line-too-long
  with ops.name_scope(name, "convolution", [input, filter]) as name:
    input = ops.convert_to_tensor(input, name="input")
    input_shape = input.get_shape()
    filter = ops.convert_to_tensor(filter, name="filter")
    filter_shape = filter.get_shape()
    op = Convolution(input_shape,
                     filter_shape,
                     padding,
                     strides=strides,
                     dilation_rate=dilation_rate,
                     name=name, data_format=data_format)
    return op(input, filter)


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
               padding, strides=None, dilation_rate=None,
               name=None, data_format=None):
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
      ValueError("input tensor must have rank %d" % (num_spatial_dims + 2))

    try:
      filter_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("filter tensor must have rank %d" % (num_spatial_dims + 2))

    if data_format is None or not data_format.startswith("NC"):
      input_channels_dim = input_shape[num_spatial_dims + 1]
      spatial_dims = range(1, num_spatial_dims+1)
    else:
      input_channels_dim = input_shape[1]
      spatial_dims = range(2, num_spatial_dims+2)

    if not input_channels_dim.is_compatible_with(filter_shape[
        num_spatial_dims]):
      raise ValueError(
          "number of input channels does not match corresponding dimension of "
          "filter, {} != {}".format(input_channels_dim, filter_shape[
              num_spatial_dims]))

    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.data_format = data_format
    self.strides = strides
    self.name = name
    self.conv_op = _WithSpaceToBatch(
        input_shape,
        dilation_rate=dilation_rate,
        padding=padding,
        build_op=self._build_op,
        filter_shape=filter_shape,
        spatial_dims=spatial_dims)

  def _build_op(self, _, padding):
    return _NonAtrousConvolution(
        self.input_shape,
        filter_shape=self.filter_shape,
        padding=padding,
        data_format=self.data_format,
        strides=self.strides,
        name=self.name)

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.conv_op(inp, filter)


def pool(input,  # pylint: disable=redefined-builtin
         window_shape,
         pooling_type,
         padding,
         dilation_rate=None,
         strides=None,
         name=None,
         data_format=None):
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
  and pad_before is defined based on the value of `padding` as described in the
  @{tf.nn.convolution$comment here}.
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
      See the @{tf.nn.convolution$comment here}
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
  # pylint: enable=line-too-long
  with ops.name_scope(name, "%s_pool" %
                      (pooling_type.lower()), [input]) as scope:
    input = ops.convert_to_tensor(input, name="input")

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

    pooling_ops = {("MAX", 1): max_pool,
                   ("MAX", 2): max_pool,
                   ("MAX", 3): max_pool3d,  # pylint: disable=undefined-variable
                   ("AVG", 1): avg_pool,
                   ("AVG", 2): avg_pool,
                   ("AVG", 3): avg_pool3d,  # pylint: disable=undefined-variable
                  }
    op_key = (pooling_type, num_spatial_dims)
    if op_key not in pooling_ops:
      raise ValueError("%d-D %s pooling is not supported." %
                       (op_key[1], op_key[0]))

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
      result = pooling_ops[op_key](converted_input,
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


def atrous_conv2d(value, filters, rate, padding, name=None):
  """Atrous convolution (a.k.a. convolution with holes or dilated convolution).

  This function is a simpler wrapper around the more general
  @{tf.nn.convolution}, and exists only for backwards compatibility. You can
  use @{tf.nn.convolution} to perform 1-D, 2-D, or 3-D atrous convolution.


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
  feature extraction, please see: [Semantic Image Segmentation with Deep
  Convolutional Nets and Fully Connected CRFs](http://arxiv.org/abs/1412.7062).
  The same operation is investigated further in [Multi-Scale Context Aggregation
  by Dilated Convolutions](http://arxiv.org/abs/1511.07122). Previous works
  that effectively use atrous convolution in different ways are, among others,
  [OverFeat: Integrated Recognition, Localization and Detection using
  Convolutional Networks](http://arxiv.org/abs/1312.6229) and [Fast Image
  Scanning with Deep Max-Pooling Convolutional Neural Networks](http://arxiv.org/abs/1302.1700).
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
    Output shape with `'VALID`` padding is:

        [batch, height - 2 * (filter_width - 1),
         width - 2 * (filter_height - 1), out_channels].

    Output shape with `'SAME'` padding is:

        [batch, height, width, out_channels].

  Raises:
    ValueError: If input/output depth does not match `filters`' shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  return convolution(
      input=value,
      filter=filters,
      padding=padding,
      dilation_rate=np.broadcast_to(rate, (2,)),
      name=name)


def conv2d_transpose(value,
                     filter,  # pylint: disable=redefined-builtin
                     output_shape,
                     strides,
                     padding="SAME",
                     data_format="NHWC",
                     name=None):
  """The transpose of `conv2d`.

  This operation is sometimes called "deconvolution" after [Deconvolutional
  Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
  actually the transpose (gradient) of `conv2d` rather than an actual
  deconvolution.

  Args:
    value: A 4-D `Tensor` of type `float` and shape
      `[batch, height, width, in_channels]` for `NHWC` data format or
      `[batch, in_channels, height, width]` for `NCHW` data format.
    filter: A 4-D `Tensor` with the same type as `value` and shape
      `[height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: A list of ints. The stride of the sliding window for each
      dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the @{tf.nn.convolution$comment here}
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  with ops.name_scope(name, "conv2d_transpose",
                      [value, filter, output_shape]) as name:
    if data_format not in ("NCHW", "NHWC"):
      raise ValueError("data_format has to be either NCHW or NHWC.")
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    axis = 3 if data_format == "NHWC" else 1
    if not value.get_shape()[axis].is_compatible_with(filter.get_shape()[3]):
      raise ValueError("input channels does not match filter's input channels, "
                       "{} != {}".format(value.get_shape()[axis],
                                         filter.get_shape()[3]))

    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(4)):
      raise ValueError("output_shape must have shape (4,), got {}"
                       .format(output_shape_.get_shape()))

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [4] if reached this point.
      if not filter.get_shape()[2].is_compatible_with(output_shape[axis]):
        raise ValueError(
            "output_shape does not match filter's output channels, "
            "{} != {}".format(output_shape[axis], filter.get_shape()[2]))

    if padding != "VALID" and padding != "SAME":
      raise ValueError("padding must be either VALID or SAME:"
                       " {}".format(padding))

    return gen_nn_ops.conv2d_backprop_input(input_sizes=output_shape_,
                                            filter=filter,
                                            out_backprop=value,
                                            strides=strides,
                                            padding=padding,
                                            data_format=data_format,
                                            name=name)


def atrous_conv2d_transpose(value,
                            filters,
                            output_shape,
                            rate,
                            padding,
                            name=None):
  """The transpose of `atrous_conv2d`.

  This operation is sometimes called "deconvolution" after [Deconvolutional
  Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
  actually the transpose (gradient) of `atrous_conv2d` rather than an actual
  deconvolution.

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
  """
  with ops.name_scope(name, "atrous_conv2d_transpose",
                      [value, filters, output_shape]) as name:
    value = ops.convert_to_tensor(value, name="value")
    filters = ops.convert_to_tensor(filters, name="filters")
    if not value.get_shape()[3].is_compatible_with(filters.get_shape()[3]):
      raise ValueError(
          "value's input channels does not match filters' input channels, "
          "{} != {}".format(value.get_shape()[3], filters.get_shape()[3]))
    if rate < 1:
      raise ValueError("rate {} cannot be less than one".format(rate))

    if rate == 1:
      return conv2d_transpose(value,
                              filters,
                              output_shape,
                              strides=[1, 1, 1, 1],
                              padding=padding,
                              data_format="NHWC")

    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(4)):
      raise ValueError("output_shape must have shape (4,), got {}"
                       .format(output_shape_.get_shape()))

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [4] if reached this point.
      if not filters.get_shape()[2].is_compatible_with(output_shape[3]):
        raise ValueError(
            "output_shape does not match filter's output channels, "
            "{} != {}".format(output_shape[3], filters.get_shape()[2]))

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

    value = array_ops.space_to_batch(input=value,
                                     paddings=space_to_batch_pad,
                                     block_size=rate)

    input_sizes = [rate * rate * output_shape[0],
                   (in_height + pad_bottom_extra) // rate,
                   (in_width + pad_right_extra) // rate,
                   output_shape[3]]

    value = gen_nn_ops.conv2d_backprop_input(input_sizes=input_sizes,
                                             filter=filters,
                                             out_backprop=value,
                                             strides=[1, 1, 1, 1],
                                             padding="VALID",
                                             data_format="NHWC")

    # The crops argument to batch_to_space includes both padding components.
    batch_to_space_crop = [[pad_top, pad_bottom + pad_bottom_extra],
                           [pad_left, pad_right + pad_right_extra]]

    return array_ops.batch_to_space(input=value,
                                    crops=batch_to_space_crop,
                                    block_size=rate)


def conv3d_transpose(value,
                     filter,  # pylint: disable=redefined-builtin
                     output_shape,
                     strides,
                     padding="SAME",
                     data_format="NDHWC",
                     name=None):
  """The transpose of `conv3d`.

  This operation is sometimes called "deconvolution" after [Deconvolutional
  Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
  actually the transpose (gradient) of `conv3d` rather than an actual
  deconvolution.

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
      See the @{tf.nn.convolution$comment here}
    data_format: A string, either `'NDHWC'` or `'NCDHW`' specifying the layout
      of the input and output tensors. Defaults to `'NDHWC'`.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  with ops.name_scope(name, "conv3d_transpose",
                      [value, filter, output_shape]) as name:
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    axis = 1 if data_format == "NCDHW" else 4
    if not value.get_shape()[axis].is_compatible_with(filter.get_shape()[4]):
      raise ValueError("input channels does not match filter's input channels, "
                       "{} != {}".format(value.get_shape()[axis],
                                         filter.get_shape()[4]))

    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(5)):
      raise ValueError("output_shape must have shape (5,), got {}"
                       .format(output_shape_.get_shape()))

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [5] if reached this point.
      if not filter.get_shape()[3].is_compatible_with(output_shape[4]):
        raise ValueError(
            "output_shape does not match filter's output channels, "
            "{} != {}".format(output_shape[4], filter.get_shape()[3]))

    if padding != "VALID" and padding != "SAME":
      raise ValueError("padding must be either VALID or SAME:"
                       " {}".format(padding))

    return gen_nn_ops.conv3d_backprop_input_v2(input_sizes=output_shape_,
                                               filter=filter,
                                               out_backprop=value,
                                               strides=strides,
                                               padding=padding,
                                               data_format=data_format,
                                               name=name)


# pylint: disable=protected-access
def bias_add(value, bias, data_format=None, name=None):
  """Adds `bias` to `value`.

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
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `value`.
  """
  with ops.name_scope(name, "BiasAdd", [value, bias]) as name:
    value = ops.convert_to_tensor(value, name="input")
    bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")
    return gen_nn_ops._bias_add(value, bias, data_format=data_format, name=name)


# pylint: disable=protected-access
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
    return gen_nn_ops._bias_add_v1(value, bias, name=name)


def crelu(features, name=None):
  """Computes Concatenated ReLU.

  Concatenates a ReLU which selects only the positive part of the activation
  with a ReLU which selects only the *negative* part of the activation.
  Note that as a result this non-linearity doubles the depth of the activations.
  Source: [Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units. W. Shang, et al.](https://arxiv.org/abs/1603.05201)

  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `features`.
  """
  with ops.name_scope(name, "CRelu", [features]) as name:
    features = ops.convert_to_tensor(features, name="features")
    c = array_ops.concat([features, -features], -1, name=name)
    return gen_nn_ops.relu(c)


def relu6(features, name=None):
  """Computes Rectified Linear 6: `min(max(features, 0), 6)`.
  Source: [Convolutional Deep Belief Networks on CIFAR-10. A. Krizhevsky](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf)

  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `features`.
  """
  with ops.name_scope(name, "Relu6", [features]) as name:
    features = ops.convert_to_tensor(features, name="features")
    return gen_nn_ops._relu6(features, name=name)


def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.

  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

  Args:
    features: A `Tensor` representing preactivation values.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)


def _flatten_outer_dims(logits):
  """Flattens logits' outer dimensions and keep its last dimension."""
  rank = array_ops.rank(logits)
  last_dim_size = array_ops.slice(
      array_ops.shape(logits), [math_ops.subtract(rank, 1)], [1])
  output = array_ops.reshape(logits, array_ops.concat([[-1], last_dim_size], 0))

  # Set output shape if known.
  if context.in_graph_mode():
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
    compute_op: Either gen_nn_ops._softmax or gen_nn_ops._log_softmax
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
    return array_ops.transpose(logits,
                               array_ops.concat([
                                   math_ops.range(dim_index), [last_index],
                                   math_ops.range(dim_index + 1, last_index),
                                   [dim_index]
                               ], 0), name=name)

  logits = ops.convert_to_tensor(logits)

  # We need its original shape for shape inference.
  shape = logits.get_shape()
  is_last_dim = (dim is -1) or (dim == shape.ndims - 1)

  if shape.ndims is 2 and is_last_dim:
    return compute_op(logits, name=name)

  # If dim is the last dimension, simply reshape the logits to a matrix and
  # apply the internal softmax.
  if is_last_dim:
    input_shape = array_ops.shape(logits)
    logits = _flatten_outer_dims(logits)
    output = compute_op(logits)
    output = array_ops.reshape(output, input_shape, name=name)
    return output

  # If dim is not the last dimension, we have to do a reshape and transpose so
  # that we can still perform softmax on its last dimension.

  # Swap logits' dimension of dim and its last dimension.
  input_rank = array_ops.rank(logits)
  logits = _swap_axis(logits, dim, math_ops.subtract(input_rank, 1))
  shape_after_swap = array_ops.shape(logits)

  # Reshape logits into a matrix.
  logits = _flatten_outer_dims(logits)

  # Do the actual softmax on its last dimension.
  output = compute_op(logits)

  # Transform back the output tensor.
  output = array_ops.reshape(output, shape_after_swap)
  output = _swap_axis(output, dim, math_ops.subtract(input_rank, 1), name=name)

  # Make shape inference work since reshape and transpose may erase its static
  # shape.
  output.set_shape(shape)

  return output


def softmax(logits, dim=-1, name=None):
  """Computes softmax activations.

  This function performs the equivalent of

      softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), dim)

  Args:
    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    dim: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type and shape as `logits`.

  Raises:
    InvalidArgumentError: if `logits` is empty or `dim` is beyond the last
      dimension of `logits`.
  """
  return _softmax(logits, gen_nn_ops._softmax, dim, name)


def log_softmax(logits, dim=-1, name=None):
  """Computes log softmax activations.

  For each batch `i` and class `j` we have

      logsoftmax = logits - log(reduce_sum(exp(logits), dim))

  Args:
    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    dim: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.

  Raises:
    InvalidArgumentError: if `logits` is empty or `dim` is beyond the last
      dimension of `logits`.
  """
  return _softmax(logits, gen_nn_ops._log_softmax, dim, name)


def _ensure_xent_args(name, sentinel, labels, logits):
  # Make sure that all arguments were passed as named arguments.
  if sentinel is not None:
    raise ValueError("Only call `%s` with "
                     "named arguments (labels=..., logits=..., ...)" % name)
  if labels is None or logits is None:
    raise ValueError("Both labels and logits must be provided.")


def softmax_cross_entropy_with_logits(_sentinel=None,  # pylint: disable=invalid-name
                                      labels=None, logits=None,
                                      dim=-1, name=None):
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

  `logits` and `labels` must have the same shape, e.g.
  `[batch_size, num_classes]` and the same dtype (either `float16`, `float32`,
  or `float64`).

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: Each row `labels[i]` must be a valid probability distribution.
    logits: Unscaled log probabilities.
    dim: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).

  Returns:
    A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
    softmax cross entropy loss.
  """
  _ensure_xent_args("softmax_cross_entropy_with_logits", _sentinel,
                    labels, logits)

  # TODO(pcmurray) Raise an error when the labels do not sum to 1. Note: This
  # could break users who call this with bad labels, but disregard the bad
  # results.

  logits = ops.convert_to_tensor(logits)
  labels = ops.convert_to_tensor(labels)
  precise_logits = math_ops.cast(logits, dtypes.float32) if (
      logits.dtype == dtypes.float16) else logits
  # labels and logits must be of the same type
  labels = math_ops.cast(labels, precise_logits.dtype)
  input_rank = array_ops.rank(precise_logits)
  # For shape inference.
  shape = logits.get_shape()

  # Move the dim to the end if dim is not the last dimension.
  if dim is not -1:
    def _move_dim_to_end(tensor, dim_index, rank):
      return array_ops.transpose(tensor,
                                 array_ops.concat([
                                     math_ops.range(dim_index),
                                     math_ops.range(dim_index + 1, rank),
                                     [dim_index]
                                 ], 0))

    precise_logits = _move_dim_to_end(precise_logits, dim, input_rank)
    labels = _move_dim_to_end(labels, dim, input_rank)

  input_shape = array_ops.shape(precise_logits)

  # Make precise_logits and labels into matrices.
  precise_logits = _flatten_outer_dims(precise_logits)
  labels = _flatten_outer_dims(labels)

  # Do the actual op computation.
  # The second output tensor contains the gradients.  We use it in
  # _CrossEntropyGrad() in nn_grad but not here.
  cost, unused_backprop = gen_nn_ops._softmax_cross_entropy_with_logits(
      precise_logits, labels, name=name)

  # The output cost shape should be the input minus dim.
  output_shape = array_ops.slice(input_shape, [0],
                                 [math_ops.subtract(input_rank, 1)])
  cost = array_ops.reshape(cost, output_shape)

  # Make shape inference work since reshape and transpose may erase its static
  # shape.
  if context.in_graph_mode() and shape is not None and shape.dims is not None:
    shape = shape.as_list()
    del shape[dim]
    cost.set_shape(shape)

  if logits.dtype == dtypes.float16:
    return math_ops.cast(cost, dtypes.float16)
  else:
    return cost


def sparse_softmax_cross_entropy_with_logits(_sentinel=None,  # pylint: disable=invalid-name
                                             labels=None, logits=None,
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
  `softmax_cross_entropy_with_logits`.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits of shape `[batch_size, num_classes]` and
  labels of shape `[batch_size]`. But higher dimensions are supported.

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Unscaled log probabilities of shape
      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float32` or `float64`.
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
    precise_logits = math_ops.cast(logits, dtypes.float32) if (
        dtypes.as_dtype(logits.dtype) == dtypes.float16) else logits

    # Store label shape for result later.
    labels_static_shape = labels.get_shape()
    labels_shape = array_ops.shape(labels)
    if logits.get_shape().ndims is not None and logits.get_shape().ndims == 0:
      raise ValueError("Logits cannot be scalars - received shape %s." %
                       logits.get_shape())
    if logits.get_shape().ndims is not None and (
        labels_static_shape.ndims is not None and
        labels_static_shape.ndims != logits.get_shape().ndims - 1):
      raise ValueError("Rank mismatch: Rank of labels (received %s) should "
                       "equal rank of logits minus 1 (received %s)." %
                       (labels_static_shape.ndims, logits.get_shape().ndims))
    # Check if no reshapes are required.
    if logits.get_shape().ndims == 2:
      cost, _ = gen_nn_ops._sparse_softmax_cross_entropy_with_logits(
          precise_logits, labels, name=name)
      if logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost

    # Reshape logits to 2 dim, labels to 1 dim.
    num_classes = array_ops.shape(logits)[array_ops.rank(logits) - 1]
    precise_logits = array_ops.reshape(precise_logits, [-1, num_classes])
    labels = array_ops.reshape(labels, [-1])
    # The second output tensor contains the gradients.  We use it in
    # _CrossEntropyGrad() in nn_grad but not here.
    cost, _ = gen_nn_ops._sparse_softmax_cross_entropy_with_logits(
        precise_logits, labels, name=name)
    cost = array_ops.reshape(cost, labels_shape)
    cost.set_shape(labels_static_shape)
    if logits.dtype == dtypes.float16:
      return math_ops.cast(cost, dtypes.float16)
    else:
      return cost


def avg_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
  """Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: A 1-D int Tensor of 4 elements.
      The size of the window for each dimension of the input tensor.
    strides: A 1-D int Tensor of 4 elements
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the @{tf.nn.convolution$comment here}
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  with ops.name_scope(name, "AvgPool", [value]) as name:
    value = ops.convert_to_tensor(value, name="input")
    return gen_nn_ops._avg_pool(value,
                                ksize=ksize,
                                strides=strides,
                                padding=padding,
                                data_format=data_format,
                                name=name)


def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
  """Performs the max pooling on the input.

  Args:
    value: A 4-D `Tensor` of the format specified by `data_format`.
    ksize: A 1-D int Tensor of 4 elements.  The size of the window for
      each dimension of the input tensor.
    strides: A 1-D int Tensor of 4 elements.  The stride of the sliding
      window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the @{tf.nn.convolution$comment here}
    data_format: A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.
    name: Optional name for the operation.

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  with ops.name_scope(name, "MaxPool", [value]) as name:
    value = ops.convert_to_tensor(value, name="input")
    return gen_nn_ops._max_pool(value,
                                ksize=ksize,
                                strides=strides,
                                padding=padding,
                                data_format=data_format,
                                name=name)


@ops.RegisterStatistics("Conv2D", "flops")
def _calc_conv_flops(graph, node):
  """Calculates the compute resources needed for Conv2D."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                            node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_height = int(filter_shape[0])
  filter_width = int(filter_shape[1])
  filter_in_depth = int(filter_shape[2])
  output_count = np.prod(output_shape.as_list())
  return ops.OpStats("flops", (output_count * filter_in_depth * filter_height *
                               filter_width * 2))


@ops.RegisterStatistics("DepthwiseConv2dNative", "flops")
def _calc_depthwise_conv_flops(graph, node):
  """Calculates the compute resources needed for DepthwiseConv2dNative."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                            node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_height = int(filter_shape[0])
  filter_width = int(filter_shape[1])
  output_count = np.prod(output_shape.as_list())
  return ops.OpStats("flops", (output_count * filter_height * filter_width * 2))


@ops.RegisterStatistics("BiasAdd", "flops")
def _calc_bias_add_flops(graph, node):
  """Calculates the computing needed for BiasAdd."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  input_count = np.prod(input_shape.as_list())
  return ops.OpStats("flops", input_count)


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


def xw_plus_b_v1(x, weights, biases, name=None):  # pylint: disable=invalid-name
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


def dropout(x, keep_prob, noise_shape=None, seed=None, name=None):  # pylint: disable=invalid-name
  """Computes dropout.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
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
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
      point tensor.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    # Do nothing if we know keep_prob == 1
    if tensor_util.constant_value(keep_prob) == 1:
      return x

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(noise_shape,
                                               seed=seed,
                                               dtype=x.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    ret = math_ops.div(x, keep_prob) * binary_tensor
    if context.in_graph_mode():
      ret.set_shape(x.get_shape())
    return ret


def top_k(input, k=1, sorted=True, name=None):
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
  return gen_nn_ops._top_kv2(input, k=k, sorted=sorted, name=name)


def nth_element(input, n, reverse=False, name=None):
  r"""Finds values of the `n`-th order statistic for the last dmension.

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


def conv1d(value, filters, stride, padding,
           use_cudnn_on_gpu=None, data_format=None,
           name=None):
  r"""Computes a 1-D convolution given 3-D input and filter tensors.

  Given an input tensor of shape
    [batch, in_width, in_channels]
  if data_format is "NHWC", or
    [batch, in_channels, in_width]
  if data_format is "NCHW",
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
    value: A 3D `Tensor`.  Must be of type `float32` or `float64`.
    filters: A 3D `Tensor`.  Must have the same type as `input`.
    stride: An `integer`.  The number of entries by which
      the filter is moved right at each step.
    padding: 'SAME' or 'VALID'
    use_cudnn_on_gpu: An optional `bool`.  Defaults to `True`.
    data_format: An optional `string` from `"NHWC", "NCHW"`.  Defaults
      to `"NHWC"`, the data is stored in the order of
      [batch, in_width, in_channels].  The `"NCHW"` format stores
      data as [batch, in_channels, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.  Has the same type as input.

  Raises:
    ValueError: if `data_format` is invalid.
  """
  with ops.name_scope(name, "conv1d", [value, filters]) as name:
    # Reshape the input tensor to [batch, 1, in_width, in_channels]
    if data_format is None or data_format == "NHWC":
      data_format = "NHWC"
      spatial_start_dim = 1
      strides = [1, 1, stride, 1]
    elif data_format == "NCHW":
      spatial_start_dim = 2
      strides = [1, 1, 1, stride]
    else:
      raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
    value = array_ops.expand_dims(value, spatial_start_dim)
    filters = array_ops.expand_dims(filters, 0)
    result = gen_nn_ops.conv2d(value, filters, strides, padding,
                               use_cudnn_on_gpu=use_cudnn_on_gpu,
                               data_format=data_format)
    return array_ops.squeeze(result, [spatial_start_dim])


@ops.RegisterStatistics("Dilation2D", "flops")
def _calc_dilation2d_flops(graph, node):
  """Calculates the compute resources needed for Dilation2D."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                            node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_height = int(filter_shape[0])
  filter_width = int(filter_shape[1])
  output_count = np.prod(output_shape.as_list())
  return ops.OpStats("flops", (output_count * filter_height * filter_width * 2))


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
        gen_nn_ops.dilation2d(input=math_ops.negative(value),
                              filter=array_ops.reverse_v2(kernel, [0, 1]),
                              strides=strides,
                              rates=rates,
                              padding=padding,
                              name=name))


def in_top_k(predictions, targets, k, name=None):
  r"""Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

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
  with ops.name_scope(name, 'in_top_k'):
    return gen_nn_ops._in_top_kv2(predictions, targets, k, name=name)
