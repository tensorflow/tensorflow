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
"""Primitive Neural Net (NN) Operations.

## Notes on padding

Several neural network operations, such as `tf.nn.conv2d` and
`tf.nn.max_pool2d`, take a `padding` parameter, which controls how the input is
padded before running the operation. The input is padded by inserting values
(typically zeros) before and after the tensor in each spatial dimension. The
`padding` parameter can either be the string `'VALID'`, which means use no
padding, or `'SAME'` which adds padding according to a formula which is
described below. Certain ops also allow the amount of padding per dimension to
be explicitly specified by passing a list to `padding`.

In the case of convolutions, the input is padded with zeros. In case of pools,
the padded input values are ignored. For example, in a max pool, the sliding
window ignores padded values, which is equivalent to the padded values being
`-infinity`.

### `'VALID'` padding

Passing `padding='VALID'` to an op causes no padding to be used. This causes the
output size to typically be smaller than the input size, even when the stride is
one. In the 2D case, the output size is computed as:

```python
out_height = ceil((in_height - filter_height + 1) / stride_height)
out_width  = ceil((in_width - filter_width + 1) / stride_width)
```

The 1D and 3D cases are similar. Note `filter_height` and `filter_width` refer
to the filter size after dilations (if any) for convolutions, and refer to the
window size for pools.

### `'SAME'` padding

With `'SAME'` padding, padding is applied to each spatial dimension. When the
strides are 1, the input is padded such that the output size is the same as the
input size. In the 2D case, the output size is computed as:

```python
out_height = ceil(in_height / stride_height)
out_width  = ceil(in_width / stride_width)
```

The amount of padding used is the smallest amount that results in the output
size. The formula for the total amount of padding per dimension is:

```python
if (in_height % strides[1] == 0):
  pad_along_height = max(filter_height - stride_height, 0)
else:
  pad_along_height = max(filter_height - (in_height % stride_height), 0)
if (in_width % strides[2] == 0):
  pad_along_width = max(filter_width - stride_width, 0)
else:
  pad_along_width = max(filter_width - (in_width % stride_width), 0)
```

Finally, the padding on the top, bottom, left and right are:

```python
pad_top = pad_along_height // 2
pad_bottom = pad_along_height - pad_top
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left
```

Note that the division by 2 means that there might be cases when the padding on
both sides (top vs bottom, right vs left) are off by one. In this case, the
bottom and right sides always get the one additional padded pixel. For example,
when pad_along_height is 5, we pad 2 pixels at the top and 3 pixels at the
bottom. Note that this is different from existing libraries such as PyTorch and
Caffe, which explicitly specify the number of padded pixels and always pad the
same number of pixels on both sides.

Here is an example of `'SAME'` padding:

>>> in_height = 5
>>> filter_height = 3
>>> stride_height = 2
>>>
>>> in_width = 2
>>> filter_width = 2
>>> stride_width = 1
>>>
>>> inp = tf.ones((2, in_height, in_width, 2))
>>> filter = tf.ones((filter_height, filter_width, 2, 2))
>>> strides = [stride_height, stride_width]
>>> output = tf.nn.conv2d(inp, filter, strides, padding='SAME')
>>> output.shape[1]  # output_height: ceil(5 / 2)
3
>>> output.shape[2] # output_width: ceil(2 / 1)
2

### Explicit padding

Certain ops, like `tf.nn.conv2d`, also allow a list of explicit padding amounts
to be passed to the `padding` parameter. This list is in the same format as what
is passed to `tf.pad`, except the padding must be a nested list, not a tensor.
For example, in the 2D case, the list is in the format `[[0, 0], [pad_top,
pad_bottom], [pad_left, pad_right], [0, 0]]` when `data_format` is its default
value of `'NHWC'`. The two `[0, 0]` pairs  indicate the batch and channel
dimensions have no padding, which is required, as only spatial dimensions can
have padding.

For example:

>>> inp = tf.ones((1, 3, 3, 1))
>>> filter = tf.ones((2, 2, 1, 1))
>>> strides = [1, 1]
>>> padding = [[0, 0], [1, 2], [0, 1], [0, 0]]
>>> output = tf.nn.conv2d(inp, filter, strides, padding=padding)
>>> tuple(output.shape)
(1, 5, 3, 1)
>>> # Equivalently, tf.pad can be used, since convolutions pad with zeros.
>>> inp = tf.pad(inp, padding)
>>> # 'VALID' means to use no padding in conv2d (we already padded inp)
>>> output2 = tf.nn.conv2d(inp, filter, strides, padding='VALID')
>>> tf.debugging.assert_equal(output, output2)

### Difference between convolution and pooling layers
How padding is used in convolution layers and pooling layers is different. For
convolution layers, padding is filled with values of zero, and padding is
multiplied with kernels. For pooling layers, padding is excluded from the
computation. For example when applying average pooling to a 4x4 grid, how much
padding is added will not impact the output. Here is an example that
demonstrates the difference.

>>> x_in = np.array([[
...   [[2], [2]],
...   [[1], [1]],
...   [[1], [1]]]])
>>> kernel_in = np.array([  # simulate the avg_pool with conv2d
...  [ [[0.25]], [[0.25]] ],
...  [ [[0.25]], [[0.25]] ]])
>>> x = tf.constant(x_in, dtype=tf.float32)
>>> kernel = tf.constant(kernel_in, dtype=tf.float32)
>>> conv_out = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
>>> pool_out = tf.nn.avg_pool(x, [2, 2], strides=[1, 1, 1, 1], padding='SAME')
>>> print(conv_out.shape, pool_out.shape)
(1, 3, 2, 1) (1, 3, 2, 1)
>>> tf.reshape(conv_out, [3, 2]).numpy()  # conv2d takes account of padding
array([[1.5 , 0.75],
       [1.  , 0.5 ],
       [0.5 , 0.25]], dtype=float32)
>>> tf.reshape(pool_out, [3, 2]).numpy()  # avg_pool excludes padding
array([[1.5, 1.5],
       [1. , 1. ],
       [1. , 1. ]], dtype=float32)

"""

import functools
import numbers

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import device_context
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

from tensorflow.python.util.tf_export import tf_export

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn

# pylint: disable=protected-access
# pylint: disable=g-classes-have-attributes

# Acceptable channels last formats (robust to H, W, D order).
_CHANNELS_LAST_FORMATS = frozenset({
    "NWC", "NHC", "NHWC", "NWHC", "NDHWC", "NDWHC", "NHDWC", "NHWDC", "NWDHC",
    "NWHDC"
})


def _get_sequence(value, n, channel_index, name):
  """Formats a value input for gen_nn_ops."""
  # Performance is fast-pathed for common cases:
  # `None`, `list`, `tuple` and `int`.
  if value is None:
    return [1] * (n + 2)

  # Always convert `value` to a `list`.
  if isinstance(value, list):
    pass
  elif isinstance(value, tuple):
    value = list(value)
  elif isinstance(value, int):
    value = [value]
  elif not isinstance(value, collections_abc.Sized):
    value = [value]
  else:
    value = list(value)  # Try casting to a list.

  len_value = len(value)

  # Fully specified, including batch and channel dims.
  if len_value == n + 2:
    return value

  # Apply value to spatial dims only.
  if len_value == 1:
    value = value * n  # Broadcast to spatial dimensions.
  elif len_value != n:
    raise ValueError(f"{name} should be of length 1, {n} or {n + 2}. "
                     f"Received: {name}={value} of length {len_value}")

  # Add batch and channel dims (always 1).
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
    input_shape = input.shape
    filter = ops.convert_to_tensor(filter, name="filter")  # pylint: disable=redefined-builtin
    filter_shape = filter.shape
    op = _NonAtrousConvolution(
        input_shape,
        filter_shape=filter_shape,
        padding=padding,
        data_format=data_format,
        strides=strides,
        name=scope)
    return op(input, filter)


class _NonAtrousConvolution:
  """Helper class for _non_atrous_convolution.

  Note that this class assumes that shapes of input and filter passed to
  `__call__` are compatible with `input_shape` and filter_shape passed to the
  constructor.

  Args:
    input_shape: static input shape, i.e. input.shape.
    filter_shape: static filter shape, i.e. filter.shape.
    padding: see _non_atrous_convolution.
    data_format: see _non_atrous_convolution.
    strides: see _non_atrous_convolution.
    name: see _non_atrous_convolution.
    num_batch_dims: (Optional.)  The number of batch dimensions in the input;
     if not provided, the default of `1` is used.
  """

  def __init__(
      self,
      input_shape,
      filter_shape,
      padding,
      data_format=None,
      strides=None,
      name=None,
      num_batch_dims=1):
    # filter shape is always rank num_spatial_dims + 2
    # and num_spatial_dims == input_shape.ndims - num_batch_dims - 1
    if input_shape.ndims is not None:
      filter_shape = filter_shape.with_rank(
          input_shape.ndims - num_batch_dims + 1)
    self.padding = padding
    self.name = name
    # input shape is == num_spatial_dims + num_batch_dims + 1
    # and filter_shape is always rank num_spatial_dims + 2
    if filter_shape.ndims is not None:
      input_shape = input_shape.with_rank(
          filter_shape.ndims + num_batch_dims - 1)
    if input_shape.ndims is None:
      raise ValueError(
          "Rank of convolution must be known. "
          f"Received: input_shape={input_shape} of rank {input_shape.rank}")
    if input_shape.ndims < 3 or input_shape.ndims - num_batch_dims + 1 > 5:
      raise ValueError(
          "`input_shape.rank - num_batch_dims + 1` must be at least 3 and at "
          f"most 5. Received: input_shape.rank={input_shape.rank} and "
          f"num_batch_dims={num_batch_dims}")
    conv_dims = input_shape.ndims - num_batch_dims - 1
    if strides is None:
      strides = [1] * conv_dims
    elif len(strides) != conv_dims:
      raise ValueError(
          f"`len(strides)` should be {conv_dims}. "
          f"Received: strides={strides} of length {len(strides)}")
    if conv_dims == 1:
      # conv1d uses the 2-d data format names
      if data_format is None:
        data_format = "NWC"
      elif data_format not in {"NCW", "NWC", "NCHW", "NHWC"}:
        raise ValueError("`data_format` must be 'NWC' or 'NCW'. "
                         f"Received: data_format={data_format}")
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
        raise ValueError("`data_format` must be 'NHWC' or 'NCHW'. "
                         f"Received: data_format={data_format}")
      self.strides = strides
      self.data_format = data_format
      self.conv_op = conv2d
    elif conv_dims == 3:
      if data_format is None or data_format == "NDHWC":
        strides = [1] + list(strides) + [1]
      elif data_format == "NCDHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("`data_format` must be 'NDHWC' or 'NCDHW'. "
                         f"Received: data_format={data_format}")
      self.strides = strides
      self.data_format = data_format
      self.conv_op = _conv3d_expanded_batch

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


def squeeze_batch_dims(inp, op, inner_rank, name=None):
  """Returns `unsqueeze_batch(op(squeeze_batch(inp)))`.

  Where `squeeze_batch` reshapes `inp` to shape
  `[prod(inp.shape[:-inner_rank])] + inp.shape[-inner_rank:]`
  and `unsqueeze_batch` does the reverse reshape but on the output.

  Args:
    inp: A tensor with dims `batch_shape + inner_shape` where `inner_shape`
      is length `inner_rank`.
    op: A callable that takes a single input tensor and returns a single.
      output tensor.
    inner_rank: A python integer.
    name: A string.

  Returns:
    `unsqueeze_batch_op(squeeze_batch(inp))`.
  """
  with ops.name_scope(name, "squeeze_batch_dims", [inp]):
    inp = ops.convert_to_tensor(inp, name="input")
    shape = inp.shape

    inner_shape = shape[-inner_rank:]
    if not inner_shape.is_fully_defined():
      inner_shape = array_ops.shape(inp)[-inner_rank:]

    batch_shape = shape[:-inner_rank]
    if not batch_shape.is_fully_defined():
      batch_shape = array_ops.shape(inp)[:-inner_rank]

    if isinstance(inner_shape, tensor_shape.TensorShape):
      inp_reshaped = array_ops.reshape(inp, [-1] + inner_shape.as_list())
    else:
      inp_reshaped = array_ops.reshape(
          inp, array_ops.concat(([-1], inner_shape), axis=-1))

    out_reshaped = op(inp_reshaped)

    out_inner_shape = out_reshaped.shape[-inner_rank:]
    if not out_inner_shape.is_fully_defined():
      out_inner_shape = array_ops.shape(out_reshaped)[-inner_rank:]

    out = array_ops.reshape(
        out_reshaped, array_ops.concat((batch_shape, out_inner_shape), axis=-1))

    out.set_shape(inp.shape[:-inner_rank] + out.shape[-inner_rank:])
    return out


@tf_export("nn.dilation2d", v1=[])
@dispatch.add_dispatch_support
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
      The type of padding algorithm to use. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A `string`, only `"NHWC"` is currently supported.
    dilations: A list of `ints` that has length `>= 4`.
      The input stride for atrous morphological dilation. Must be:
      `[1, rate_height, rate_width, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  if data_format != "NHWC":
    raise ValueError("`data_format` values other  than 'NHWC' are not "
                     f"supported. Received: data_format={data_format}")

  return gen_nn_ops.dilation2d(input=input,
                               filter=filters,
                               strides=strides,
                               rates=dilations,
                               padding=padding,
                               name=name)


@tf_export(v1=["nn.dilation2d"])
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
  input_shape = input.shape

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


class _WithSpaceToBatch:
  """Helper class for with_space_to_batch.

  Note that this class assumes that shapes of input and filter passed to
  `__call__` are compatible with `input_shape`, `filter_shape`, and
  `spatial_dims` passed to the constructor.

  Arguments
    input_shape: static shape of input. i.e. input.shape.
    dilation_rate: see `with_space_to_batch`.
    padding: see `with_space_to_batch`.
    build_op: Function that maps (num_spatial_dims, paddings) -> (function that
      maps (input, filter) -> output).
    filter_shape: see `with_space_to_batch`.
    spatial_dims: `see with_space_to_batch`.
    data_format: see `with_space_to_batch`.
    num_batch_dims: (Optional).  Number of batch dims in `input_shape`.
  """

  def __init__(self,
               input_shape,
               dilation_rate,
               padding,
               build_op,
               filter_shape=None,
               spatial_dims=None,
               data_format=None,
               num_batch_dims=1):
    """Helper class for _with_space_to_batch."""
    dilation_rate = ops.convert_to_tensor(
        dilation_rate, dtypes.int32, name="dilation_rate")
    if dilation_rate.shape.ndims not in (None, 1):
      raise ValueError(
          "`dilation_rate.shape.rank` must be 1. Received: "
          f"dilation_rate={dilation_rate} of rank {dilation_rate.shape.rank}")

    if not dilation_rate.shape.is_fully_defined():
      raise ValueError(
          "`dilation_rate.shape` must be fully defined. Received: "
          f"dilation_rate={dilation_rate} with shape "
          f"{dilation_rate.shape}")

    num_spatial_dims = dilation_rate.shape.dims[0].value

    if data_format is not None and data_format.startswith("NC"):
      starting_spatial_dim = num_batch_dims + 1
    else:
      starting_spatial_dim = num_batch_dims

    if spatial_dims is None:
      spatial_dims = range(starting_spatial_dim,
                           num_spatial_dims + starting_spatial_dim)
    orig_spatial_dims = list(spatial_dims)
    spatial_dims = sorted(set(int(x) for x in orig_spatial_dims))
    if spatial_dims != orig_spatial_dims or any(x < 1 for x in spatial_dims):
      raise ValueError(
          "`spatial_dims` must be a monotonically increasing sequence of "
          f"positive integers. Received: spatial_dims={orig_spatial_dims}")

    if data_format is not None and data_format.startswith("NC"):
      expected_input_rank = spatial_dims[-1]
    else:
      expected_input_rank = spatial_dims[-1] + 1

    try:
      input_shape.with_rank_at_least(expected_input_rank)
    except ValueError:
      raise ValueError(
          f"`input.shape.rank` must be at least {expected_input_rank}. "
          f"Received: input.shape={input_shape} with rank {input_shape.rank}")

    const_rate = tensor_util.constant_value(dilation_rate)
    rate_or_const_rate = dilation_rate
    if const_rate is not None:
      rate_or_const_rate = const_rate
      if np.any(const_rate < 1):
        raise ValueError(
            "`dilation_rate` must be positive. "
            f"Received: dilation_rate={const_rate}")
      if np.all(const_rate == 1):
        self.call = build_op(num_spatial_dims, padding)
        return

    padding, explicit_paddings = convert_padding(padding)

    # We have two padding contributions. The first is used for converting "SAME"
    # to "VALID". The second is required so that the height and width of the
    # zero-padded value tensor are multiples of rate.

    # Padding required to reduce to "VALID" convolution
    if padding == "SAME":
      if filter_shape is None:
        raise ValueError(
            "`filter_shape` must be specified for `padding='SAME'`. "
            f"Received: filter_shape={filter_shape} and padding={padding}")
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
    elif padding == "EXPLICIT":
      base_paddings = (np.array(explicit_paddings)
                       .reshape([num_spatial_dims + 2, 2]))
      # Remove batch and channel dimensions
      if data_format is not None and data_format.startswith("NC"):
        self.base_paddings = base_paddings[2:]
      else:
        self.base_paddings = base_paddings[1:-1]
    else:
      raise ValueError("`padding` must be one of 'SAME' or 'VALID'. "
                       f"Received: padding={padding}")

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
      input_spatial_shape = array_ops_stack.stack(
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
  pad_extra_shape = (filter_spatial_shape - 1) * rate_or_const_rate

  # When full_padding_shape is odd, we pad more at end, following the same
  # convention as conv2d.
  pad_extra_start = pad_extra_shape // 2
  pad_extra_end = pad_extra_shape - pad_extra_start
  base_paddings = array_ops_stack.stack(
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
    strides: Optional.  List of N ints >= 1.  Defaults to `[1]*N`.  If any value
      of strides is > 1, then all values of dilation_rate must be 1.
    dilation_rate: Optional.  List of N ints >= 1.  Defaults to `[1]*N`.  If any
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
    raise ValueError(f"`len(dilation_rate)` should be {num_spatial_dims}. "
                     f"Received: dilation_rate={dilation_rate} of length "
                     f"{len(dilation_rate)}")
  dilation_rate = np.array(dilation_rate, dtype=np.int32)
  if np.any(dilation_rate < 1):
    raise ValueError("all values of `dilation_rate` must be positive. "
                     f"Received: dilation_rate={dilation_rate}")

  if strides is None:
    strides = [1] * num_spatial_dims
  elif len(strides) != num_spatial_dims:
    raise ValueError(f"`len(strides)` should be {num_spatial_dims}. "
                     f"Received: strides={strides} of length {len(strides)}")
  strides = np.array(strides, dtype=np.int32)
  if np.any(strides < 1):
    raise ValueError("all values of `strides` must be positive. "
                     f"Received: strides={strides}")

  if np.any(strides > 1) and np.any(dilation_rate > 1):
    raise ValueError(
        "`strides > 1` not supported in conjunction with `dilation_rate > 1`. "
        f"Received: strides={strides} and dilation_rate={dilation_rate}")
  return strides, dilation_rate


@tf_export(v1=["nn.convolution"])
@dispatch.add_dispatch_support
def convolution(
    input,  # pylint: disable=redefined-builtin
    filter,  # pylint: disable=redefined-builtin
    padding,
    strides=None,
    dilation_rate=None,
    name=None,
    data_format=None,
    filters=None,
    dilations=None):  # pylint: disable=g-doc-args
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

  an optional `dilation_rate` tensor of shape N (defaults to `[1]*N`) specifying
  the filter upsampling/input downsampling rate, and an optional list of N
  `strides` (defaults to `[1]*N`), this computes for each N-D spatial output
  position `(x[0], ..., x[N-1])`:

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
  output striding `strides`.

  In the case that `data_format` does start with `"NC"`, the `input` and output
  (but not the `filter`) are simply transposed as follows:

  ```python
  convolution(input, data_format, **kwargs) =
    tf.transpose(convolution(tf.transpose(input, [0] + range(2,N+2) + [1]),
                             **kwargs),
                 [0, N+1] + range(1, N+1))
  ```

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
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input when the strides are 1. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    strides: Optional.  Sequence of N ints >= 1.  Specifies the output stride.
      Defaults to `[1]*N`.  If any value of strides is > 1, then all values of
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
@dispatch.add_dispatch_support
def convolution_v2(  # pylint: disable=missing-docstring
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
    call_from_convolution=True,
    num_spatial_dims=None):
  """Internal function which performs rank agnostic convolution.

  Args:
    input: See `convolution`.
    filters: See `convolution`.
    strides: See `convolution`.
    padding: See `convolution`.
    data_format: See `convolution`.
    dilations: See `convolution`.
    name: See `convolution`.
    call_from_convolution: See `convolution`.
    num_spatial_dims: (Optional.).  It is a integer describing the
      rank of the spatial dimensions.  For `1-D`, `2-D` and `3-D` convolutions,
      the value of `num_spatial_dims` is `1`, `2`, and `3`, respectively.
      This argument is only required to disambiguate the rank of `batch_shape`
      when `filter_shape.ndims is None` and `len(batch_shape) > 1`.  For
      backwards compatibility, if `num_spatial_dims is None` and
     `filter_shape.ndims is None`, then `len(batch_shape)` is assumed to be
     `1` (i.e., the input is expected to be
     `[batch_size, num_channels] + input_spatial_shape`
     or `[batch_size] + input_spatial_shape + [num_channels]`.

  Returns:
    A tensor of shape and dtype matching that of `input`.

  Raises:
    ValueError: If input and filter both have unknown shapes, or if
      `num_spatial_dims` is provided and incompatible with the value
      estimated from `filters.shape`.
  """
  if (not isinstance(filters, variables_lib.Variable) and
      not tensor_util.is_tf_type(filters)):
    with ops.name_scope("convolution_internal", None, [filters, input]):
      filters = ops.convert_to_tensor(filters, name='filters')
  if (not isinstance(input, ops.Tensor) and not tensor_util.is_tf_type(input)):
    with ops.name_scope("convolution_internal", None, [filters, input]):
      input = ops.convert_to_tensor(input, name="input")

  filters_rank = filters.shape.rank
  inputs_rank = input.shape.rank
  if num_spatial_dims is None:
    if filters_rank:
      num_spatial_dims = filters_rank - 2
    elif inputs_rank:
      num_spatial_dims = inputs_rank - 2
    else:
      raise ValueError(
          "When `num_spatial_dims` is not set, one of `input.shape.rank` or "
          "`filters.shape.rank` must be known. "
          f"Received: input.shape={input.shape} of rank {inputs_rank} and "
          f"filters.shape={filters.shape} of rank {filters_rank}")
  elif filters_rank and filters_rank - 2 != num_spatial_dims:
    raise ValueError(
        "`filters.shape.rank - 2` should equal `num_spatial_dims`. Received: "
        f"filters.shape={filters.shape} of rank {filters_rank} and "
        f"num_spatial_dims={num_spatial_dims}")

  if inputs_rank:
    num_batch_dims = inputs_rank - num_spatial_dims - 1  # Channel dimension.
  else:
    num_batch_dims = 1  # By default, assume single batch dimension.

  if num_spatial_dims not in {1, 2, 3}:
    raise ValueError(
        "`num_spatial_dims` must be 1, 2, or 3. "
        f"Received: num_spatial_dims={num_spatial_dims}.")

  if data_format is None or data_format in _CHANNELS_LAST_FORMATS:
    channel_index = num_batch_dims + num_spatial_dims
  else:
    channel_index = num_batch_dims

  if dilations is None:
    dilations = _get_sequence(dilations, num_spatial_dims, channel_index,
                              "dilations")
    is_dilated_conv = False
  else:
    dilations = _get_sequence(dilations, num_spatial_dims, channel_index,
                              "dilations")
    is_dilated_conv = any(i != 1 for i in dilations)

  strides = _get_sequence(strides, num_spatial_dims, channel_index, "strides")
  has_tpu_context = device_context.enclosing_tpu_context() is not None

  if name:
    default_name = None
  elif not has_tpu_context or call_from_convolution:
    default_name = "convolution"
  elif num_spatial_dims == 2:  # Most common case.
    default_name = "Conv2D"
  elif num_spatial_dims == 3:
    default_name = "Conv3D"
  else:
    default_name = "conv1d"

  with ops.name_scope(name, default_name, [input, filters]) as name:
    # Fast path for TPU or if no dilation, as gradient only supported on TPU
    # for dilations.
    if not is_dilated_conv or has_tpu_context:
      if num_spatial_dims == 2:  # Most common case.
        op = _conv2d_expanded_batch
      elif num_spatial_dims == 3:
        op = _conv3d_expanded_batch
      else:
        op = conv1d

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
          data_format=data_format,
          num_spatial_dims=num_spatial_dims)
      return op(input, filters)


class Convolution:
  """Helper class for convolution.

  Note that this class assumes that shapes of input and filter passed to
  `__call__` are compatible with `input_shape`, `filter_shape`, and
  `num_spatial_dims` passed to the constructor.

  Arguments
    input_shape: static shape of input. i.e. input.shape.  Its length is
      `batch_shape + input_spatial_shape + [num_channels]` if `data_format`
      does not start with `NC`, or
      `batch_shape + [num_channels] + input_spatial_shape` if `data_format`
      starts with `NC`.
    filter_shape: static shape of the filter. i.e. filter.shape.
    padding: The padding algorithm, must be "SAME" or "VALID".
    strides: see convolution.
    dilation_rate: see convolution.
    name: see convolution.
    data_format: A string or `None`.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (if `data_format` is `None`
      or does not start with `NC`), or the first post-batch dimension (i.e. if
      `data_format` starts with `NC`).
    num_spatial_dims: (Usually optional.)  Python integer, the rank of the
      spatial and channel dimensions.  For `1-D`, `2-D` and `3-D` convolutions,
      the value of `num_spatial_dims` is `1`, `2`, and `3`, respectively.
      This argument is only required to disambiguate the rank of `batch_shape`
      when `filter_shape.ndims is None` and `len(batch_shape) > 1`.  For
      backwards compatibility, if `num_spatial_dims is None` and
      `filter_shape.ndims is None`, then `len(batch_shape)` is assumed to be
      `1` (i.e., the input is expected to be
      `[batch_size, num_channels] + input_spatial_shape`
      or `[batch_size] + input_spatial_shape + [num_channels]`.
  """

  def __init__(self,
               input_shape,
               filter_shape,
               padding,
               strides=None,
               dilation_rate=None,
               name=None,
               data_format=None,
               num_spatial_dims=None):
    """Helper function for convolution."""
    num_batch_dims = None
    filter_shape = tensor_shape.as_shape(filter_shape)
    input_shape = tensor_shape.as_shape(input_shape)

    if filter_shape.ndims is not None:
      if (num_spatial_dims is not None and
          filter_shape.ndims != num_spatial_dims + 2):
        raise ValueError(
            "`filters.shape.rank` must be `num_spatial_dims + 2`. Received: "
            f"filters.shape={filter_shape} of rank {filter_shape.rank} and "
            f"num_spatial_dims={num_spatial_dims}")
      else:
        num_spatial_dims = filter_shape.ndims - 2

    if input_shape.ndims is not None and num_spatial_dims is not None:
      num_batch_dims = input_shape.ndims - num_spatial_dims - 1

    if num_spatial_dims is None:
      num_spatial_dims = input_shape.ndims - 2
    else:
      if input_shape.ndims is not None:
        if input_shape.ndims < num_spatial_dims + 2:
          raise ValueError(
              "`input.shape.rank` must be >= than `num_spatial_dims + 2`. "
              f"Received: input.shape={input_shape} of rank {input_shape.rank} "
              f"and num_spatial_dims={num_spatial_dims}")
        else:
          if num_batch_dims is None:
            num_batch_dims = input_shape.ndims - num_spatial_dims - 1

    if num_spatial_dims is None:
      raise ValueError(
          "When `num_spatial_dims` is not set, one of `input.shape.rank` or "
          "`filters.shape.rank` must be known. "
          f"Received: input.shape={input_shape} of rank {input_shape.rank} and "
          f"`filters.shape={filter_shape}` of rank {filter_shape.rank}")

    if num_batch_dims is None:
      num_batch_dims = 1

    if num_batch_dims < 1:
      raise ValueError(
          f"Batch dims should be >= 1, but found {num_batch_dims}. "
          "Batch dims was estimated as "
          "`input.shape.rank - num_spatial_dims - 1` and `num_spatial_dims` "
          "was either provided or estimated as `filters.shape.rank - 2`. "
          f"Received: input.shape={input_shape} of rank {input_shape.rank}, "
          f"filters.shape={filter_shape} of rank {filter_shape.rank}, and "
          f"num_spatial_dims={num_spatial_dims}")

    if data_format is None or not data_format.startswith("NC"):
      input_channels_dim = tensor_shape.dimension_at_index(
          input_shape, num_spatial_dims + num_batch_dims)
      spatial_dims = range(num_batch_dims, num_spatial_dims + num_batch_dims)
    else:
      input_channels_dim = tensor_shape.dimension_at_index(
          input_shape, num_batch_dims)
      spatial_dims = range(
          num_batch_dims + 1, num_spatial_dims + num_batch_dims + 1)

    filter_dim = tensor_shape.dimension_at_index(filter_shape, num_spatial_dims)
    if not (input_channels_dim % filter_dim).is_compatible_with(0):
      raise ValueError(
          "The number of input channels is not divisible by the corresponding "
          f"number of output filters. Received: input.shape={input_shape} with "
          f"{input_channels_dim} channels and filters.shape={filter_shape} "
          f"with {filter_dim} output filters.")

    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.data_format = data_format
    self.strides = strides
    self.padding = padding
    self.name = name
    self.dilation_rate = dilation_rate
    self.num_batch_dims = num_batch_dims
    self.num_spatial_dims = num_spatial_dims
    self.conv_op = _WithSpaceToBatch(
        input_shape,
        dilation_rate=dilation_rate,
        padding=padding,
        build_op=self._build_op,
        filter_shape=filter_shape,
        spatial_dims=spatial_dims,
        data_format=data_format,
        num_batch_dims=num_batch_dims)

  def _build_op(self, _, padding):
    return _NonAtrousConvolution(
        self.input_shape,
        filter_shape=self.filter_shape,
        padding=padding,
        data_format=self.data_format,
        strides=self.strides,
        name=self.name,
        num_batch_dims=self.num_batch_dims)

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
          call_from_convolution=False,
          num_spatial_dims=self.num_spatial_dims)
    else:
      return self.conv_op(inp, filter)


@tf_export(v1=["nn.pool"])
@dispatch.add_dispatch_support
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

  ```python
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
      Defaults to `[1]*N`.  If any value of dilation_rate is > 1, then all
      values of strides must be 1.
    strides: Optional.  Sequence of N ints >= 1.  Defaults to `[1]*N`.
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
      raise ValueError("`len(window_shape)` must be 1, 2, or 3. Received: "
                       f"window_shape={window_shape} of length "
                       f"{len(window_shape)}")

    input.get_shape().with_rank(num_spatial_dims + 2)

    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    if padding == "SAME" and np.any(dilation_rate > 1):
      raise ValueError(
          "pooling with 'SAME' padding is not implemented for "
          f"`dilation_rate` > 1. Received: padding={padding} and "
          f"dilation_rate={dilation_rate}")

    if np.any(strides > window_shape):
      raise ValueError(
          "`strides` > `window_shape` not supported due to inconsistency "
          f"between CPU and GPU implementations. Received: strides={strides} "
          f"and window_shape={window_shape}")

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
      raise ValueError(
          f"{num_spatial_dims}-D {pooling_type} pooling is not supported.")

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
        raise ValueError("data_format must be either 'NWC' or 'NCW'. "
                         f"Received: data_format={data_format}")
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
@dispatch.add_dispatch_support
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

  ```python
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
    strides: Optional. Sequence of N ints >= 1.  Defaults to `[1]*N`. If any value of
      strides is > 1, then all values of dilation_rate must be 1.
    padding: The padding algorithm, must be "SAME" or "VALID". Defaults to "SAME".
      See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW". For
      N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations: Optional.  Dilation rate.  List of N ints >= 1. Defaults to
      `[1]*N`.  If any value of dilation_rate is > 1, then all values of strides
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
@dispatch.add_dispatch_support
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
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.
    Output shape with `'VALID'` padding is:

        [batch, height - rate * (filter_width - 1),
         width - rate * (filter_height - 1), out_channels].

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


def convert_padding(padding, expected_length=4):
  """Converts Python padding to C++ padding for ops which take EXPLICIT padding.

  Args:
    padding: the `padding` argument for a Python op which supports EXPLICIT
      padding.
    expected_length: Expected number of entries in the padding list when
      explicit padding is used.

  Returns:
    (padding, explicit_paddings) pair, which should be passed as attributes to a
    C++ op.

  Raises:
    ValueError: If padding is invalid.
  """
  explicit_paddings = []
  if padding == "EXPLICIT":
    raise ValueError("'EXPLICIT' is not a valid value for `padding`. To use "
                     "explicit padding, `padding` must be a list.")
  if isinstance(padding, (list, tuple)):
    for i, dim_paddings in enumerate(padding):
      if not isinstance(dim_paddings, (list, tuple)):
        raise ValueError("When `padding` is a list, each element of `padding` "
                         "must be a list/tuple of size 2. Received: "
                         f"padding={padding} with element at index {i} of type "
                         f"{type(dim_paddings)}")
      if len(dim_paddings) != 2:
        raise ValueError("When `padding` is a list, each element of `padding` "
                         "must be a list/tuple of size 2. Received: "
                         f"padding={padding} with element at index {i} of size "
                         f"{len(dim_paddings)}")
      explicit_paddings.extend(dim_paddings)
    if len(padding) != expected_length:
      raise ValueError(
          f"When padding is a list, it must be of size {expected_length}. "
          f"Received: padding={padding} of size {len(padding)}")
    padding = "EXPLICIT"
  return padding, explicit_paddings


@tf_export(v1=["nn.conv1d"])
@dispatch.add_dispatch_support
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
  r"""Computes a 1-D convolution of input with rank `>=3` and a `3-D` filter.

  Given an input tensor of shape
    `batch_shape + [in_width, in_channels]`
  if `data_format` is `"NWC"`, or
    `batch_shape + [in_channels, in_width]`
  if `data_format` is `"NCW"`,
  and a filter / kernel tensor of shape
  `[filter_width, in_channels, out_channels]`, this op reshapes
  the arguments to pass them to `conv2d` to perform the equivalent
  convolution operation.

  Internally, this op reshapes the input tensors and invokes `tf.nn.conv2d`.
  For example, if `data_format` does not start with "NC", a tensor of shape
    `batch_shape + [in_width, in_channels]`
  is reshaped to
    `batch_shape + [1, in_width, in_channels]`,
  and the filter is reshaped to
    `[1, filter_width, in_channels, out_channels]`.
  The result is then reshaped back to
    `batch_shape + [out_width, out_channels]`
  \(where out_width is a function of the stride and padding as in conv2d\) and
  returned to the caller.

  Args:
    value: A Tensor of rank at least 3. Must be of type `float16`, `float32`, or
      `float64`.
    filters: A Tensor of rank at least 3.  Must have the same type as `value`.
    stride: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: 'SAME' or 'VALID'
    use_cudnn_on_gpu: An optional `bool`.  Defaults to `True`.
    data_format: An optional `string` from `"NWC", "NCW"`.  Defaults to `"NWC"`,
      the data is stored in the order of `batch_shape + [in_width,
      in_channels]`.  The `"NCW"` format stores data as `batch_shape +
      [in_channels, in_width]`.
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
    # Reshape the input tensor to batch_shape + [1, in_width, in_channels]
    if data_format is None or data_format == "NHWC" or data_format == "NWC":
      data_format = "NHWC"
      spatial_start_dim = -3
      channel_index = 2
    elif data_format == "NCHW" or data_format == "NCW":
      data_format = "NCHW"
      spatial_start_dim = -2
      channel_index = 1
    else:
      raise ValueError("`data_format` must be 'NWC' or 'NCW'. "
                       f"Received: data_format={data_format}")
    strides = [1] + _get_sequence(stride, 1, channel_index, "stride")
    dilations = [1] + _get_sequence(dilations, 1, channel_index, "dilations")

    value = array_ops.expand_dims(value, spatial_start_dim)
    filters = array_ops.expand_dims(filters, 0)
    if value.shape.ndims in (4, 3, 2, 1, 0, None):
      result = gen_nn_ops.conv2d(
          value,
          filters,
          strides,
          padding,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          data_format=data_format,
          dilations=dilations,
          name=name)
    else:
      result = squeeze_batch_dims(
          value,
          functools.partial(
              gen_nn_ops.conv2d,
              filter=filters,
              strides=strides,
              padding=padding,
              use_cudnn_on_gpu=use_cudnn_on_gpu,
              data_format=data_format,
              dilations=dilations,
          ),
          inner_rank=3,
          name=name)
    return array_ops.squeeze(result, [spatial_start_dim])


@tf_export("nn.conv1d", v1=[])
@dispatch.add_dispatch_support
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
    `batch_shape + [in_width, in_channels]`
  if `data_format` is `"NWC"`, or
    `batch_shape + [in_channels, in_width]`
  if `data_format` is `"NCW"`,
  and a filter / kernel tensor of shape
  `[filter_width, in_channels, out_channels]`, this op reshapes
  the arguments to pass them to `conv2d` to perform the equivalent
  convolution operation.

  Internally, this op reshapes the input tensors and invokes `tf.nn.conv2d`.
  For example, if `data_format` does not start with `"NC"`, a tensor of shape
    `batch_shape + [in_width, in_channels]`
  is reshaped to
    `batch_shape + [1, in_width, in_channels]`,
  and the filter is reshaped to
    `[1, filter_width, in_channels, out_channels]`.
  The result is then reshaped back to
    `batch_shape + [out_width, out_channels]`
  \(where out_width is a function of the stride and padding as in conv2d\) and
  returned to the caller.

  Args:
    input: A Tensor of rank at least 3. Must be of type `float16`, `float32`, or
      `float64`.
    filters: A Tensor of rank at least 3.  Must have the same type as `input`.
    stride: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: 'SAME' or 'VALID'. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: An optional `string` from `"NWC", "NCW"`.  Defaults to `"NWC"`,
      the data is stored in the order of
      `batch_shape + [in_width, in_channels]`.  The `"NCW"` format stores data
      as `batch_shape + [in_channels, in_width]`.
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
@dispatch.add_dispatch_support
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
    filters: A 3-D `Tensor` with the same type as `input` and shape
      `[filter_width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `input`.
    output_shape: A 1-D `Tensor`, containing three elements, representing the
      output shape of the deconvolution op.
    strides: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A string. `'NWC'` and `'NCW'` are supported.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `input`.

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
      raise ValueError("`data_format` must be 'NWC' or 'NCW'. "
                       f"Received: data_format={data_format}")

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
@dispatch.add_dispatch_support
def conv2d_v2(input,  # pylint: disable=redefined-builtin
              filters,
              strides,
              padding,
              data_format="NHWC",
              dilations=None,
              name=None):
  # pylint: disable=line-too-long
  r"""Computes a 2-D convolution given `input` and 4-D `filters` tensors.

  The `input` tensor may have rank `4` or higher, where shape dimensions `[:-3]`
  are considered batch dimensions (`batch_shape`).

  Given an input tensor of shape
  `batch_shape + [in_height, in_width, in_channels]` and a filter / kernel
  tensor of shape `[filter_height, filter_width, in_channels, out_channels]`,
  this op performs the following:

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
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

  Usage Example:

  >>> x_in = np.array([[
  ...   [[2], [1], [2], [0], [1]],
  ...   [[1], [3], [2], [2], [3]],
  ...   [[1], [1], [3], [3], [0]],
  ...   [[2], [2], [0], [1], [1]],
  ...   [[0], [0], [3], [1], [2]], ]])
  >>> kernel_in = np.array([
  ...  [ [[2, 0.1]], [[3, 0.2]] ],
  ...  [ [[0, 0.3]], [[1, 0.4]] ], ])
  >>> x = tf.constant(x_in, dtype=tf.float32)
  >>> kernel = tf.constant(kernel_in, dtype=tf.float32)
  >>> tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
  <tf.Tensor: shape=(1, 4, 4, 2), dtype=float32, numpy=..., dtype=float32)>

  Args:
    input: A `Tensor`. Must be one of the following types:
      `half`, `bfloat16`, `float32`, `float64`.
      A Tensor of rank at least 4. The dimension order is interpreted according
      to the value of `data_format`; with the all-but-inner-3 dimensions acting
      as batch dimensions. See below for details.
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
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`.
      Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          `batch_shape + [height, width, channels]`.
      Alternatively, the format could be "NCHW", the data storage order of:
          `batch_shape + [channels, height, width]`.
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
    A `Tensor`. Has the same type as `input` and the same outer batch shape.
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
@dispatch.add_dispatch_support
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
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

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
  padding, explicit_paddings = convert_padding(padding)
  if data_format is None:
    data_format = "NHWC"
  channel_index = 1 if data_format.startswith("NC") else 3

  strides = _get_sequence(strides, 2, channel_index, "strides")
  dilations = _get_sequence(dilations, 2, channel_index, "dilations")

  shape = input.shape
  # shape object may lack ndims, e.g., if input is an np.ndarray.  In that case,
  # we fall back to len(shape).
  ndims = getattr(shape, "ndims", -1)
  if ndims == -1:
    ndims = len(shape)
  if ndims in (4, 3, 2, 1, 0, None):
    # We avoid calling squeeze_batch_dims to reduce extra python function
    # call slowdown in eager mode.  This branch doesn't require reshapes.
    return gen_nn_ops.conv2d(
        input,
        filter=filter,
        strides=strides,
        padding=padding,
        use_cudnn_on_gpu=use_cudnn_on_gpu,
        explicit_paddings=explicit_paddings,
        data_format=data_format,
        dilations=dilations,
        name=name)
  return squeeze_batch_dims(
      input,
      functools.partial(
          gen_nn_ops.conv2d,
          filter=filter,
          strides=strides,
          padding=padding,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          explicit_paddings=explicit_paddings,
          data_format=data_format,
          dilations=dilations),
      inner_rank=3,
      name=name)


@tf_export(v1=["nn.conv2d_backprop_filter"])
@dispatch.add_dispatch_support
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
  padding, explicit_paddings = convert_padding(padding)
  return gen_nn_ops.conv2d_backprop_filter(
      input, filter_sizes, out_backprop, strides, padding, use_cudnn_on_gpu,
      explicit_paddings, data_format, dilations, name)


@tf_export(v1=["nn.conv2d_backprop_input"])
@dispatch.add_dispatch_support
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
  padding, explicit_paddings = convert_padding(padding)
  return gen_nn_ops.conv2d_backprop_input(
      input_sizes, filter, out_backprop, strides, padding, use_cudnn_on_gpu,
      explicit_paddings, data_format, dilations, name)


@tf_export(v1=["nn.conv2d_transpose"])
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.  When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
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
    padding, explicit_paddings = convert_padding(padding)

    return gen_nn_ops.conv2d_backprop_input(
        input_sizes=output_shape,
        filter=filters,
        out_backprop=input,
        strides=strides,
        padding=padding,
        explicit_paddings=explicit_paddings,
        data_format=data_format,
        dilations=dilations,
        name=name)


def _conv2d_expanded_batch(
    input,  # pylint: disable=redefined-builtin
    filters,
    strides,
    padding,
    data_format,
    dilations,
    name):
  """Helper function for `convolution_internal`; handles expanded batches."""
  # Try really hard to avoid modifying the legacy name scopes - return early.
  input_rank = input.shape.rank
  if input_rank is None or input_rank < 5:
    # We avoid calling squeeze_batch_dims to reduce extra python function
    # call slowdown in eager mode.  This branch doesn't require reshapes.
    return gen_nn_ops.conv2d(
        input,
        filter=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)
  return squeeze_batch_dims(
      input,
      functools.partial(
          gen_nn_ops.conv2d,
          filter=filters,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations),
      inner_rank=3,
      name=name)


@tf_export("nn.atrous_conv2d_transpose")
@dispatch.add_dispatch_support
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
      deconvolution op, of form `[batch, out_height, out_width, out_channels]`.
    rate: A positive int32. The stride with which we sample input values across
      the `height` and `width` dimensions. Equivalently, the rate by which we
      upsample the filter values by inserting zeros across the `height` and
      `width` dimensions. In the literature, the same parameter is sometimes
      called `input stride` or `dilation`.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
          "`value` channel count must be compatible with `filters` input "
          f"channel count. Received: value.shape={value.get_shape()} with "
          f"channel count {value.get_shape()[3]} and "
          f"filters.shape={filters.get_shape()} with input channel count "
          f"{filters.get_shape()[3]}.")
    if rate < 1:
      raise ValueError(f"`rate` cannot be less than one. Received: rate={rate}")

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
      raise ValueError("`output_shape` must have shape (4,). "
                       f"Received: output_shape={output_shape_.get_shape()}")

    if isinstance(output_shape, tuple):
      output_shape = list(output_shape)

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [4] if reached this point.
      if not filters.get_shape().dims[2].is_compatible_with(output_shape[3]):
        raise ValueError(
            "`output_shape` channel count must be compatible with `filters` "
            f"output channel count. Received: output_shape={output_shape} with "
            f"channel count {output_shape[3]} and "
            f"filters.shape={filters.get_shape()} with output channel count "
            f"{filters.get_shape()[3]}.")

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
      raise ValueError("`padding` must be either 'VALID' or 'SAME'. "
                       f"Received: padding={padding}")

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


@tf_export(v1=["nn.depthwise_conv2d_native"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("nn.depthwise_conv2d_native")
def depthwise_conv2d_native(  # pylint: disable=redefined-builtin,dangerous-default-value
    input,
    filter,
    strides,
    padding,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name=None):
  r"""Computes a 2-D depthwise convolution.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
  `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
  a different filter to each input channel (expanding from 1 channel to
  `channel_multiplier` channels for each), then concatenates the results
  together. Thus, the output has `in_channels * channel_multiplier` channels.

  ```
  for k in 0..in_channels-1
    for q in 0..channel_multiplier-1
      output[b, i, j, k * channel_multiplier + q] =
        sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                          filter[di, dj, k, q]
  ```

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`,
      `float32`, `float64`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`. 1-D of length 4.  The stride of the sliding
      window for each dimension of `input`.
    padding: Controls how to pad the image before applying the convolution. Can
      be the string `"SAME"` or `"VALID"` indicating the type of padding
      algorithm to use, or a list indicating the explicit paddings at the start
      and end of each dimension. When explicit padding is used and data_format
      is `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to
      `"NHWC"`. Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of: [batch, height,
        width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`. 1-D
      tensor of length 4.  The dilation factor for each dimension of `input`. If
      set to k > 1, there will be k-1 skipped cells between each filter element
      on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  padding, explicit_paddings = convert_padding(padding)
  return gen_nn_ops.depthwise_conv2d_native(
      input,
      filter,
      strides,
      padding,
      explicit_paddings=explicit_paddings,
      data_format=data_format,
      dilations=dilations,
      name=name)


@tf_export(
    "nn.depthwise_conv2d_backprop_input",
    v1=[
        "nn.depthwise_conv2d_native_backprop_input",
        "nn.depthwise_conv2d_backprop_input"
    ])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("nn.depthwise_conv2d_native_backprop_input")
def depthwise_conv2d_native_backprop_input(  # pylint: disable=redefined-builtin,dangerous-default-value
    input_sizes,
    filter,
    out_backprop,
    strides,
    padding,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name=None):
  r"""Computes the gradients of depthwise convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`. An integer vector representing the
      shape of `input`, based on `data_format`.  For example, if `data_format`
      is 'NHWC' then `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`,
      `float32`, `float64`. 4-D with shape `[filter_height, filter_width,
      in_channels, depthwise_multiplier]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`. 4-D with
      shape  based on `data_format`. For example, if `data_format` is 'NHWC'
      then out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`. The stride of the sliding window for each
      dimension of the input of the convolution.
    padding: Controls how to pad the image before applying the convolution. Can
      be the string `"SAME"` or `"VALID"` indicating the type of padding
      algorithm to use, or a list indicating the explicit paddings at the start
      and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to
      `"NHWC"`. Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of: [batch, height,
        width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`. 1-D
      tensor of length 4.  The dilation factor for each dimension of `input`. If
      set to k > 1, there will be k-1 skipped cells between each filter element
      on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  padding, explicit_paddings = convert_padding(padding)
  return gen_nn_ops.depthwise_conv2d_native_backprop_input(
      input_sizes,
      filter,
      out_backprop,
      strides,
      padding,
      explicit_paddings=explicit_paddings,
      data_format=data_format,
      dilations=dilations,
      name=name)


@tf_export(
    "nn.depthwise_conv2d_backprop_filter",
    v1=[
        "nn.depthwise_conv2d_native_backprop_filter",
        "nn.depthwise_conv2d_backprop_filter"
    ])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("nn.depthwise_conv2d_native_backprop_filter")
def depthwise_conv2d_native_backprop_filter(  # pylint: disable=redefined-builtin,dangerous-default-value
    input,
    filter_sizes,
    out_backprop,
    strides,
    padding,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name=None):
  r"""Computes the gradients of depthwise convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`,
      `float32`, `float64`. 4-D with shape based on `data_format`.  For example,
      if `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
      in_width, in_channels]` tensor.
    filter_sizes: A `Tensor` of type `int32`. An integer vector representing the
      tensor shape of `filter`, where `filter` is a 4-D `[filter_height,
      filter_width, in_channels, depthwise_multiplier]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`. 4-D with shape
      based on `data_format`. For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`. The stride of the sliding window for each
      dimension of the input of the convolution.
    padding: Controls how to pad the image before applying the convolution. Can
      be the string `"SAME"` or `"VALID"` indicating the type of padding
      algorithm to use, or a list indicating the explicit paddings at the start
      and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to
      `"NHWC"`. Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of: [batch, height,
        width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`. 1-D
      tensor of length 4.  The dilation factor for each dimension of `input`. If
      set to k > 1, there will be k-1 skipped cells between each filter element
      on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  padding, explicit_paddings = convert_padding(padding)
  return gen_nn_ops.depthwise_conv2d_native_backprop_filter(
      input,
      filter_sizes,
      out_backprop,
      strides,
      padding,
      explicit_paddings=explicit_paddings,
      data_format=data_format,
      dilations=dilations,
      name=name)


def _conv3d_expanded_batch(
    input,  # pylint: disable=redefined-builtin
    filter,  # pylint: disable=redefined-builtin
    strides,
    padding,
    data_format,
    dilations=None,
    name=None):
  """Helper function for `conv3d`; handles expanded batches."""
  shape = input.shape
  # shape object may lack ndims, e.g., if input is an np.ndarray.  In that case,
  # we fall back to len(shape).
  ndims = getattr(shape, "ndims", -1)
  if ndims == -1:
    ndims = len(shape)
  if ndims in (5, 4, 3, 2, 1, 0, None):
    # We avoid calling squeeze_batch_dims to reduce extra python function
    # call slowdown in eager mode.  This branch doesn't require reshapes.
    return gen_nn_ops.conv3d(
        input,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
        name=name)
  else:
    return squeeze_batch_dims(
        input,
        functools.partial(
            gen_nn_ops.conv3d,
            filter=filter,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilations=dilations),
        inner_rank=4,
        name=name)


@tf_export("nn.conv3d", v1=[])
@dispatch.add_dispatch_support
def conv3d_v2(input,  # pylint: disable=redefined-builtin,missing-docstring
              filters,
              strides,
              padding,
              data_format="NDHWC",
              dilations=None,
              name=None):
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  return _conv3d_expanded_batch(input, filters, strides, padding, data_format,
                                dilations, name)


@tf_export(v1=["nn.conv3d"])
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
    input: A 5-D `Tensor` of type `float` and shape `[batch, depth, height,
      width, in_channels]` for `NDHWC` data format or `[batch, in_channels,
      depth, height, width]` for `NCDHW` data format.
    filters: A 5-D `Tensor` with the same type as `input` and shape `[depth,
      height, width, output_channels, in_channels]`.  `filter`'s `in_channels`
      dimension must match that of `input`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: An int or list of `ints` that has length `1`, `3` or `5`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `D`, `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 0. The dimension order is
      determined by the value of `data_format`, see below for details.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
    A `Tensor` with the same type as `input`.

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
@dispatch.add_dispatch_support
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
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
    if tensor_util.is_tf_type(output_shape):
      n = output_shape.shape[0] - 2
    elif isinstance(output_shape, collections_abc.Sized):
      n = len(output_shape) - 2
    else:
      raise ValueError("`output_shape` must be a tensor or sized collection. "
                       f"Received: output_shape={output_shape}")

    if not 1 <= n <= 3:
      raise ValueError(
          f"`output_shape` must be of length 3, 4 or 5. "
          f"Received: output_shape={output_shape} of length {n + 2}.")

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


@tf_export("nn.bias_add")
@dispatch.add_dispatch_support
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
        raise ValueError("`data_format` must be of the form `N...C` or "
                         f"`NC...`. Received: data_format={data_format}")

    if not context.executing_eagerly():
      value = ops.convert_to_tensor(value, name="input")
      bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")

    return gen_nn_ops.bias_add(value, bias, data_format=data_format, name=name)


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
@dispatch.add_dispatch_support
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
    c = array_ops.concat([features, -features], axis, name=name)  # pylint: disable=invalid-unary-operand-type
    return gen_nn_ops.relu(c)


@tf_export("nn.crelu", v1=[])
@dispatch.add_dispatch_support
def crelu_v2(features, axis=-1, name=None):
  return crelu(features, name=name, axis=axis)
crelu_v2.__doc__ = crelu.__doc__


@tf_export("nn.relu6")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def relu6(features, name=None):
  """Computes Rectified Linear 6: `min(max(features, 0), 6)`.

  In comparison with `tf.nn.relu`, relu6 activation functions have shown to
  empirically perform better under low-precision conditions (e.g. fixed point
  inference) by encouraging the model to learn sparse features earlier.
  Source: [Convolutional Deep Belief Networks on CIFAR-10: Krizhevsky et al.,
  2010](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf).

  For example:

  >>> x = tf.constant([-3.0, -1.0, 0.0, 6.0, 10.0], dtype=tf.float32)
  >>> y = tf.nn.relu6(x)
  >>> y.numpy()
  array([0., 0., 0., 6., 6.], dtype=float32)

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
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
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


@tf_export("nn.gelu", v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def gelu(features, approximate=False, name=None):
  """Compute the Gaussian Error Linear Unit (GELU) activation function.

  Gaussian error linear unit (GELU) computes
  `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
  The (GELU) nonlinearity weights inputs by their value, rather than gates
  inputs by their sign as in ReLU.

  For example:

  >>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
  >>> y = tf.nn.gelu(x)
  >>> y.numpy()
  array([-0.00404951, -0.15865529,  0.        ,  0.8413447 ,  2.9959507 ],
      dtype=float32)
  >>> y = tf.nn.gelu(x, approximate=True)
  >>> y.numpy()
  array([-0.00363752, -0.15880796,  0.        ,  0.841192  ,  2.9963627 ],
      dtype=float32)

  Args:
    features: A `float Tensor` representing preactivation values.
    approximate: An optional `bool`. Defaults to `False`. Whether to enable
      approximation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `features`.

  Raises:
    ValueError: if `features` is not a floating point `Tensor`.

  References:
    [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
  """
  with ops.name_scope(name, "Gelu", [features]):
    features = ops.convert_to_tensor(features, name="features")
    if not features.dtype.is_floating:
      raise ValueError(
          "`features.dtype` must be a floating point tensor."
          f"Received:features.dtype={features.dtype}")
    if approximate:
      coeff = math_ops.cast(0.044715, features.dtype)
      return 0.5 * features * (
          1.0 + math_ops.tanh(0.7978845608028654 *
                              (features + coeff * math_ops.pow(features, 3))))
    else:
      return 0.5 * features * (1.0 + math_ops.erf(
          features / math_ops.cast(1.4142135623730951, features.dtype)))


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


def _wrap_2d_function(inputs, compute_op, dim=-1, name=None):
  """Helper function for ops that accept and return 2d inputs of same shape.

  It reshapes and transposes the inputs into a 2-D Tensor and then invokes
  the given function. The output would be transposed and reshaped back.
  If the given function returns a tuple of tensors, each of them will be
  transposed and reshaped.

  Args:
    inputs: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    compute_op: The function to wrap. Must accept the input tensor as its first
      arugment, and a second keyword argument `name`.
    dim: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same shape as inputs. If compute_op returns multiple
      tensors, each of them have the same shape as the input.
  Raises:
    InvalidArgumentError: if `inputs` is empty or `dim` is beyond the last
      dimension of `inputs`.
  """

  def _swap_axis(input_tensor, dim_index, last_index, name=None):
    """Swaps logits's dim_index and last_index."""
    return array_ops.transpose(
        input_tensor,
        array_ops.concat([
            math_ops.range(dim_index), [last_index],
            math_ops.range(dim_index + 1, last_index), [dim_index]
        ], 0),
        name=name)

  inputs = ops.convert_to_tensor(inputs)

  # We need its original shape for shape inference.
  shape = inputs.get_shape()
  is_last_dim = (dim == -1) or (dim == shape.ndims - 1)

  if is_last_dim:
    return compute_op(inputs, name=name)

  dim_val = dim
  if isinstance(dim, ops.Tensor):
    dim_val = tensor_util.constant_value(dim)
  if dim_val is not None and not -shape.ndims <= dim_val < shape.ndims:
    raise errors_impl.InvalidArgumentError(
        None, None,
        f"`dim` must be in the range [{-shape.ndims}, {shape.ndims}) where "
        f"{shape.ndims} is the number of dimensions in the input. "
        f"Received: dim={dim_val}")

  # If dim is not the last dimension, we have to do a transpose so that we can
  # still perform the op on its last dimension.

  # In case dim is negative (and is not last dimension -1), add shape.ndims
  ndims = array_ops.rank(inputs)
  if not isinstance(dim, ops.Tensor):
    if dim < 0:
      dim += ndims
  else:
    dim = array_ops.where(math_ops.less(dim, 0), dim + ndims, dim)

  # Swap logits' dimension of dim and its last dimension.
  input_rank = array_ops.rank(inputs)
  dim_axis = dim % shape.ndims
  inputs = _swap_axis(inputs, dim_axis, math_ops.subtract(input_rank, 1))

  # Do the actual call on its last dimension.
  def fix_output(output):
    output = _swap_axis(
        output, dim_axis, math_ops.subtract(input_rank, 1), name=name)

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output

  outputs = compute_op(inputs)
  if isinstance(outputs, tuple):
    return tuple(fix_output(output) for output in outputs)
  else:
    return fix_output(outputs)


@tf_export("nn.softmax", "math.softmax", v1=[])
@dispatch.add_dispatch_support
def softmax_v2(logits, axis=None, name=None):
  """Computes softmax activations.

  Used for multi-class predictions. The sum of all outputs generated by softmax
  is 1.

  This function performs the equivalent of

  ```python
  softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)
  ```
  Example usage:

  >>> softmax = tf.nn.softmax([-1, 0., 1.])
  >>> softmax
  <tf.Tensor: shape=(3,), dtype=float32,
  numpy=array([0.09003057, 0.24472848, 0.66524094], dtype=float32)>
  >>> sum(softmax)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

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
  return _wrap_2d_function(logits, gen_nn_ops.softmax, axis, name)


@tf_export(v1=["nn.softmax", "math.softmax"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def softmax(logits, axis=None, name=None, dim=None):
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dim", dim)
  if axis is None:
    axis = -1
  return _wrap_2d_function(logits, gen_nn_ops.softmax, axis, name)


softmax.__doc__ = softmax_v2.__doc__


@tf_export(v1=["nn.log_softmax", "math.log_softmax"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
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
  return _wrap_2d_function(logits, gen_nn_ops.log_softmax, axis, name)


@tf_export("nn.log_softmax", "math.log_softmax", v1=[])
@dispatch.add_dispatch_support
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
  return _wrap_2d_function(logits, gen_nn_ops.log_softmax, axis, name)


def _ensure_xent_args(name, labels, logits):
  if labels is None or logits is None:
    raise ValueError(f"Both `labels` and `logits` must be provided for {name}"
                     f"Received: labels={labels} and logits={logits}")


@tf_export("nn.softmax_cross_entropy_with_logits", v1=[])
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
    if config.is_op_determinism_enabled():
      log_probs = log_softmax_v2(precise_logits)
      cost = -math_ops.reduce_sum(labels * log_probs, axis=1)
    else:
      # The second output tensor contains the gradients.  We use it in
      # CrossEntropyGrad() in nn_grad but not here.
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
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions=_XENT_DEPRECATION)
def softmax_cross_entropy_with_logits(
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
  _ensure_xent_args("softmax_cross_entropy_with_logits", labels, logits)

  with ops.name_scope(name, "softmax_cross_entropy_with_logits_sg",
                      [logits, labels]) as name:
    labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")

  return softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits, axis=dim, name=name)


def _sparse_softmax_cross_entropy_with_rank_2_logits(logits, labels, name):
  if config.is_op_determinism_enabled():
    # TODO(duncanriach): Implement a GPU-deterministic version of this op at
    #     the C++/CUDA level.

    # The actual op functionality
    log_probs = log_softmax_v2(logits)
    cost = math_ops.negative(array_ops.gather(log_probs, labels, batch_dims=1))

    # Force the output to be NaN when the corresponding label is invalid.
    # Without the selective gradient gating provided by the following code,
    # backprop into the actual op functionality above, when there are invalid
    # labels, leads to corruption of the gradients associated with valid labels.
    # TODO(duncanriach): Uncover the source of the aforementioned corruption.
    nan_tensor = constant_op.constant(float("Nan"), dtype=logits.dtype)
    cost_all_nans = array_ops.broadcast_to(nan_tensor, array_ops.shape(cost))
    class_count = math_ops.cast(array_ops.shape(logits)[-1], labels.dtype)
    cost = array_ops.where(
        math_ops.logical_or(
            math_ops.less(labels, 0),
            math_ops.greater_equal(labels, class_count)), cost_all_nans, cost)
  else:
    # The second output tensor contains the gradients. We use it in
    # _CrossEntropyGrad() in nn_grad but not here.
    cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name=name)
  return cost


@tf_export(v1=["nn.sparse_softmax_cross_entropy_with_logits"])
@dispatch.add_dispatch_support
def sparse_softmax_cross_entropy_with_logits(
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
  _ensure_xent_args("sparse_softmax_cross_entropy_with_logits", labels, logits)

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
          f"`logits` cannot be a scalar. Received logits={logits}`")
    if logits.get_shape().ndims is not None and (
        labels_static_shape.ndims is not None and
        labels_static_shape.ndims != logits.get_shape().ndims - 1):
      raise ValueError(
          "`labels.shape.rank` must equal `logits.shape.rank - 1`. "
          f"Received: labels.shape={labels_static_shape} of rank "
          f"{labels_static_shape.rank} and logits.shape={logits.get_shape()} "
          f"of rank {logits.get_shape().rank}")
    if (static_shapes_fully_defined and
        labels_static_shape != logits.get_shape()[:-1]):
      raise ValueError(
          "`labels.shape` must equal `logits.shape` except for "
          f"the last dimension. Received: labels.shape={labels_static_shape} "
          f"and logits.shape={logits.get_shape()}")
    # Check if no reshapes are required.
    if logits.get_shape().ndims == 2:
      cost = _sparse_softmax_cross_entropy_with_rank_2_logits(
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
      cost = _sparse_softmax_cross_entropy_with_rank_2_logits(
          precise_logits, labels, name=name)
      cost = array_ops.reshape(cost, labels_shape)
      cost.set_shape(labels_static_shape)
      if logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost


@tf_export("nn.sparse_softmax_cross_entropy_with_logits", v1=[])
@dispatch.add_dispatch_support
def sparse_softmax_cross_entropy_with_logits_v2(labels, logits, name=None):
  """Computes sparse softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  Note:  For this operation, the probability of a given label is considered
  exclusive.  That is, soft classes are not allowed, and the `labels` vector
  must provide a single specific index for the true class for each row of
  `logits` (each minibatch entry).  For soft softmax classification with
  a probability distribution for each entry, see
  `softmax_cross_entropy_with_logits_v2`.

  Warning: This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits of shape
  `[batch_size, num_classes]` and have labels of shape
  `[batch_size]`, but higher dimensions are supported, in which
  case the `dim`-th dimension is assumed to be of size `num_classes`.
  `logits` must have the dtype of `float16`, `float32`, or `float64`, and
  `labels` must have the dtype of `int32` or `int64`.

  >>> logits = tf.constant([[2., -5., .5, -.1],
  ...                       [0., 0., 1.9, 1.4],
  ...                       [-100., 100., -100., -100.]])
  >>> labels = tf.constant([0, 3, 1])
  >>> tf.nn.sparse_softmax_cross_entropy_with_logits(
  ...     labels=labels, logits=logits).numpy()
  array([0.29750752, 1.1448325 , 0.        ], dtype=float32)

  To avoid confusion, passing only named arguments to this function is
  recommended.

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
@dispatch.add_dispatch_support
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
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
        "`input` must have a static shape or `data_format` must be given. "
        f"Received: input.shape={input.shape} and "
        f"data_format={data_format}")
  if not 1 <= n <= 3:
    raise ValueError(
        f"`input.shape.rank` must be 3, 4 or 5. Received: "
        f"input.shape={input.shape} of rank {n + 2}.")

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
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
@dispatch.add_dispatch_support
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
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
@dispatch.add_dispatch_support
def avg_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):  # pylint: disable=redefined-builtin
  """Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    input: A 5-D `Tensor` of shape `[batch, depth, height, width, channels]`
      and type `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: An int or list of `ints` that has length `1`, `3` or `5`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `3` or `5`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
@dispatch.add_dispatch_support
def max_pool_v2(input, ksize, strides, padding, data_format=None, name=None):
  """Performs max pooling on the input.

  For a given window of `ksize`, takes the maximum value within that window.
  Used for reducing computation and preventing overfitting.

  Consider an example of pooling with 2x2, non-overlapping windows:

  >>> matrix = tf.constant([
  ...     [0, 0, 1, 7],
  ...     [0, 2, 0, 0],
  ...     [5, 2, 0, 0],
  ...     [0, 0, 9, 8],
  ... ])
  >>> reshaped = tf.reshape(matrix, (1, 4, 4, 1))
  >>> tf.nn.max_pool(reshaped, ksize=2, strides=2, padding="SAME")
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=int32, numpy=
  array([[[[2],
           [7]],
          [[5],
           [9]]]], dtype=int32)>

  We can adjust the window size using the `ksize` parameter. For example, if we
  were to expand the window to 3:

  >>> tf.nn.max_pool(reshaped, ksize=3, strides=2, padding="SAME")
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=int32, numpy=
  array([[[[5],
           [7]],
          [[9],
           [9]]]], dtype=int32)>

  We've now picked up two additional large numbers (5 and 9) in two of the
  pooled spots.

  Note that our windows are now overlapping, since we're still moving by 2 units
  on each iteration. This is causing us to see the same 9 repeated twice, since
  it is part of two overlapping windows.

  We can adjust how far we move our window with each iteration using the
  `strides` parameter. Updating this to the same value as our window size
  eliminates the overlap:

  >>> tf.nn.max_pool(reshaped, ksize=3, strides=3, padding="SAME")
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=int32, numpy=
  array([[[[2],
           [7]],
          [[5],
           [9]]]], dtype=int32)>

  Because the window does not neatly fit into our input, padding is added around
  the edges, giving us the same result as when we used a 2x2 window. We can skip
  padding altogether and simply drop the windows that do not fully fit into our
  input by instead passing `"VALID"` to the `padding` argument:

  >>> tf.nn.max_pool(reshaped, ksize=3, strides=3, padding="VALID")
  <tf.Tensor: shape=(1, 1, 1, 1), dtype=int32, numpy=array([[[[5]]]],
   dtype=int32)>

  Now we've grabbed the largest value in the 3x3 window starting from the upper-
  left corner. Since no other windows fit in our input, they are dropped.

  Args:
    input:  Tensor of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]` if `data_format` does not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC". Pooling happens over the spatial dimensions only.
    ksize: An int or list of `ints` that has length `1`, `N` or `N+2`. The size
      of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `N` or `N+2`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`. When using explicit
      padding, the size of the paddings cannot be greater than the sliding
      window size.
    data_format: A string. Specifies the channel dimension. For N=1 it can be
      either "NWC" (default) or "NCW", for N=2 it can be either "NHWC" (default)
      or "NCHW" and for N=3 either "NDHWC" (default) or "NCDHW".
    name: Optional name for the operation.

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.

  Raises:
    ValueError: If
      - explicit padding is used with an input tensor of rank 5.
      - explicit padding is used with data_format='NCHW_VECT_C'.
  """
  if input.shape is not None:
    n = len(input.shape) - 2
  elif data_format is not None:
    n = len(data_format) - 2
  else:
    raise ValueError(
        "`input` must have a static shape or a data format must be given. "
        f"Received: input.shape={input.shape} and "
        f"data_format={data_format}")
  if not 1 <= n <= 3:
    raise ValueError(
        f"`input.shape.rank` must be 3, 4 or 5. Received: "
        f"input.shape={input.shape} of rank {n + 2}.")
  if data_format is None:
    channel_index = n + 1
  else:
    channel_index = 1 if data_format.startswith("NC") else n + 1

  if isinstance(padding, (list, tuple)) and data_format == "NCHW_VECT_C":
    raise ValueError("`data_format='NCHW_VECT_C'` is not supported with "
                     f"explicit padding. Received: padding={padding}")

  ksize = _get_sequence(ksize, n, channel_index, "ksize")
  strides = _get_sequence(strides, n, channel_index, "strides")

  if (isinstance(padding, (list, tuple)) and n == 3):
    raise ValueError("Explicit padding is not supported with an input "
                     f"tensor of rank 5. Received: padding={padding}")

  max_pooling_ops = {
      1: max_pool1d,
      2: max_pool2d,
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
@dispatch.add_dispatch_support
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
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. When explicit padding is used and
      data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
      pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
      and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`. When using explicit
      padding, the size of the paddings cannot be greater than the sliding
      window size.
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
    if isinstance(padding, (list, tuple)) and data_format == "NCHW_VECT_C":
      raise ValueError("`data_format='NCHW_VECT_C'` is not supported with "
                       f"explicit padding. Received: padding={padding}")
    padding, explicit_paddings = convert_padding(padding)
    if ((np.isscalar(ksize) and ksize == 0) or
        (isinstance(ksize,
                    (list, tuple, np.ndarray)) and any(v == 0 for v in ksize))):
      raise ValueError(f"`ksize` cannot be zero. Received: ksize={ksize}")

    return gen_nn_ops.max_pool(
        value,
        ksize=ksize,
        strides=strides,
        padding=padding,
        explicit_paddings=explicit_paddings,
        data_format=data_format,
        name=name)


# pylint: disable=redefined-builtin
@tf_export("nn.max_pool1d")
@dispatch.add_dispatch_support
def max_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
  """Performs the max pooling on the input.

  Note internally this op reshapes and uses the underlying 2d operation.

  Args:
    input: A 3-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1` or `3`. The size of the
      window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1` or `3`. The stride of
      the sliding window for each dimension of the input tensor.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NWC"`, this should be in the form `[[0, 0], [pad_left, pad_right], [0,
      0]]`. When explicit padding used and data_format is `"NCW"`, this should
      be in the form `[[0, 0], [0, 0], [pad_left, pad_right]]`. When using
      explicit padding, the size of the paddings cannot be greater than the
      sliding window size.
    data_format: An optional string from: "NWC", "NCW". Defaults to "NWC".
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  with ops.name_scope(name, "MaxPool1d", [input]) as name:
    if isinstance(padding, (list, tuple)) and data_format == "NCHW_VECT_C":
      raise ValueError("`data_format='NCHW_VECT_C'` is not supported with "
                       f"explicit padding. Received: padding={padding}")
    if data_format is None:
      data_format = "NWC"
    channel_index = 1 if data_format.startswith("NC") else 2
    ksize = [1] + _get_sequence(ksize, 1, channel_index, "ksize")
    strides = [1] + _get_sequence(strides, 1, channel_index, "strides")
    padding, explicit_paddings = convert_padding(padding, 3)
    if padding == "EXPLICIT":
      explicit_paddings = [0, 0] + explicit_paddings

    expanding_dim = 1 if data_format == "NWC" else 2
    data_format = "NHWC" if data_format == "NWC" else "NCHW"

    input = array_ops.expand_dims_v2(input, expanding_dim)
    result = gen_nn_ops.max_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        explicit_paddings=explicit_paddings,
        data_format=data_format,
        name=name)
    return array_ops.squeeze(result, expanding_dim)
# pylint: enable=redefined-builtin


# pylint: disable=redefined-builtin
@tf_export("nn.max_pool2d")
@dispatch.add_dispatch_support
def max_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
  """Performs max pooling on 2D spatial data such as images.

  This is a more specific version of `tf.nn.max_pool` where the input tensor
  is 4D, representing 2D spatial data such as images. Using these APIs are
  equivalent

  Downsamples the input images along theirs spatial dimensions (height and
  width) by taking its maximum over an input window defined by `ksize`.
  The window is shifted by `strides` along each dimension.

  For example, for `strides=(2, 2)` and `padding=VALID` windows that extend
  outside of the input are not included in the output:

  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> # Add the `batch` and `channels` dimensions.
  >>> x = x[tf.newaxis, :, :, tf.newaxis]
  >>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
  ...                           padding="VALID")
  >>> result[0, :, :, 0]
  <tf.Tensor: shape=(1, 2), dtype=float32, numpy=
  array([[6., 8.]], dtype=float32)>

  With `padding=SAME`, we get:

  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> x = x[tf.newaxis, :, :, tf.newaxis]
  >>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
  ...                           padding='SAME')
  >>> result[0, :, :, 0]
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[ 6., 8.],
         [10.,12.]], dtype=float32)>

  We can also specify padding explicitly. The following example adds width-1
  padding on all sides (top, bottom, left, right):

  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> x = x[tf.newaxis, :, :, tf.newaxis]
  >>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
  ...                           padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
  >>> result[0, :, :, 0]
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[ 1., 3., 4.],
         [ 9., 11., 12.]], dtype=float32)>

  For more examples and detail, see `tf.nn.max_pool`.

  Args:
    input: A 4-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`. The size of
      the window for each dimension of the input tensor. If only one integer is
      specified, then we apply the same window for all 4 dims. If two are
      provided then we use those for H, W dimensions and keep N, C dimension
      window size = 1.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
      stride of the sliding window for each dimension of the input tensor. If
      only one integer is specified, we apply the same stride to all 4 dims. If
      two are provided we use those for the H, W dimensions and keep N, C of
      stride = 1.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
        for more information. When explicit padding is used and data_format is
        `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
        [pad_left, pad_right], [0, 0]]`. When explicit padding used and
        data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
        [pad_top, pad_bottom], [pad_left, pad_right]]`. When using explicit
        padding, the size of the paddings cannot be greater than the sliding
        window size.
    data_format: A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.
    name: Optional name for the operation.

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.

  Raises:
    ValueError: If explicit padding is used with data_format='NCHW_VECT_C'.
  """
  with ops.name_scope(name, "MaxPool2d", [input]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")
    if isinstance(padding, (list, tuple)) and data_format == "NCHW_VECT_C":
      raise ValueError("`data_format='NCHW_VECT_C'` is not supported with "
                       f"explicit padding. Received: padding={padding}")
    padding, explicit_paddings = convert_padding(padding)

    return gen_nn_ops.max_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        explicit_paddings=explicit_paddings,
        data_format=data_format,
        name=name)
# pylint: enable=redefined-builtin


# pylint: disable=redefined-builtin
@tf_export("nn.max_pool3d")
@dispatch.add_dispatch_support
def max_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
  """Performs the max pooling on the input.

  Args:
    input: A 5-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1`, `3` or `5`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `3` or `5`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
@dispatch.add_dispatch_support
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
      The type of padding algorithm to use. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
    raise ValueError("`data_format` values other  than 'NHWC' are not "
                     f"supported. Received: data_format={data_format}")

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
@dispatch.add_dispatch_support
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
    raise ValueError("`data_format` values other  than 'NHWC' are not "
                     f"supported. Received: data_format={data_format}")

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
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
    noise_shape: A 1-D integer `Tensor`, representing the
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
    rate_from_keep_prob = 1. - keep_prob if keep_prob is not None else None
  except TypeError:
    raise ValueError("`keep_prob` must be a floating point number or Tensor. "
                     f"Received: keep_prob={keep_prob}")

  rate = deprecation.deprecated_argument_lookup(
      "rate", rate,
      "keep_prob", rate_from_keep_prob)

  if rate is None:
    raise ValueError(f"`rate` must be provided. Received: rate={rate}")

  return dropout_v2(x, rate, noise_shape=noise_shape, seed=seed, name=name)


@tf_export("nn.dropout", v1=[])
@dispatch.add_dispatch_support
def dropout_v2(x, rate, noise_shape=None, seed=None, name=None):
  """Computes dropout: randomly sets elements to zero to prevent overfitting.

  Warning: You should consider using
  `tf.nn.experimental.stateless_dropout` instead of this function. The
  difference between `tf.nn.experimental.stateless_dropout` and this
  function is analogous to the difference between
  `tf.random.stateless_uniform` and `tf.random.uniform`. Please see
  [Random number
  generation](https://www.tensorflow.org/guide/random_numbers) guide
  for a detailed description of the various RNG systems in TF. As the
  guide states, legacy stateful RNG ops like `tf.random.uniform` and
  `tf.nn.dropout` are not deprecated yet but highly discouraged,
  because their states are hard to control.

  Note: The behavior of dropout has changed between TensorFlow 1.x and 2.x.
  When converting 1.x code, please use named arguments to ensure behavior stays
  consistent.

  See also: `tf.keras.layers.Dropout` for a dropout layer.

  [Dropout](https://arxiv.org/abs/1207.0580) is useful for regularizing DNN
  models. Inputs elements are randomly set to zero (and the other elements are
  rescaled). This encourages each node to be independently useful, as it cannot
  rely on the output of other nodes.

  More precisely: With probability `rate` elements of `x` are set to `0`.
  The remaining elements are scaled up by `1.0 / (1 - rate)`, so that the
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
    noise_shape: A 1-D integer `Tensor`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  uniform_sampler = functools.partial(random_ops.random_uniform, seed=seed)
  def dummy_rng_step():
    random_seed.get_seed(seed)
  return _dropout(x=x, rate=rate, noise_shape=noise_shape,
                  uniform_sampler=uniform_sampler,
                  dummy_rng_step=dummy_rng_step, name=name,
                  default_name="dropout")


@tf_export("nn.experimental.stateless_dropout")
@dispatch.add_dispatch_support
def stateless_dropout(x, rate, seed, rng_alg=None, noise_shape=None, name=None):
  """Computes dropout: randomly sets elements to zero to prevent overfitting.

  [Dropout](https://arxiv.org/abs/1207.0580) is useful for regularizing DNN
  models. Inputs elements are randomly set to zero (and the other elements are
  rescaled). This encourages each node to be independently useful, as it cannot
  rely on the output of other nodes.

  More precisely: With probability `rate` elements of `x` are set to `0`.
  The remaining elements are scaled up by `1.0 / (1 - rate)`, so that the
  expected value is preserved.

  >>> x = tf.ones([3,5])
  >>> tf.nn.experimental.stateless_dropout(x, rate=0.5, seed=[1, 0])
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[2., 0., 2., 0., 0.],
         [0., 0., 2., 0., 2.],
         [0., 0., 0., 0., 2.]], dtype=float32)>

  >>> x = tf.ones([3,5])
  >>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[1, 0])
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[5., 0., 0., 0., 0.],
         [0., 0., 0., 0., 5.],
         [0., 0., 0., 0., 5.]], dtype=float32)>

  >>> tf.nn.experimental.stateless_dropout(x, rate=0.0, seed=[1, 0]) == x
  <tf.Tensor: shape=(3, 5), dtype=bool, numpy=
  array([[ True,  True,  True,  True,  True],
         [ True,  True,  True,  True,  True],
         [ True,  True,  True,  True,  True]])>


  This function is a stateless version of `tf.nn.dropout`, in the
  sense that no matter how many times you call this function, the same
  `seed` will lead to the same results, and different `seed` will lead
  to different results.

  >>> x = tf.ones([3,5])
  >>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[1, 0])
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[5., 0., 0., 0., 0.],
         [0., 0., 0., 0., 5.],
         [0., 0., 0., 0., 5.]], dtype=float32)>
  >>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[1, 0])
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[5., 0., 0., 0., 0.],
         [0., 0., 0., 0., 5.],
         [0., 0., 0., 0., 5.]], dtype=float32)>
  >>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[2, 0])
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[5., 0., 0., 0., 0.],
         [0., 0., 0., 5., 0.],
         [0., 0., 0., 0., 0.]], dtype=float32)>
  >>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[2, 0])
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[5., 0., 0., 0., 0.],
         [0., 0., 0., 5., 0.],
         [0., 0., 0., 0., 0.]], dtype=float32)>

  Compare the above results to those of `tf.nn.dropout` below. The
  second time `tf.nn.dropout` is called with the same seed, it will
  give a different output.

  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate=0.8, seed=1)
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[0., 0., 0., 5., 5.],
         [0., 5., 0., 5., 0.],
         [5., 0., 5., 0., 5.]], dtype=float32)>
  >>> tf.nn.dropout(x, rate=0.8, seed=1)
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[0., 0., 0., 0., 0.],
         [0., 0., 0., 5., 0.],
         [0., 0., 0., 0., 0.]], dtype=float32)>
  >>> tf.nn.dropout(x, rate=0.8, seed=2)
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[0., 0., 0., 0., 0.],
         [0., 5., 0., 5., 0.],
         [0., 0., 0., 0., 0.]], dtype=float32)>
  >>> tf.nn.dropout(x, rate=0.8, seed=2)
  <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
  array([[0., 0., 0., 0., 0.],
         [5., 0., 5., 0., 5.],
         [0., 5., 0., 0., 5.]], dtype=float32)>

  The difference between this function and `tf.nn.dropout` is
  analogous to the difference between `tf.random.stateless_uniform`
  and `tf.random.uniform`. Please see [Random number
  generation](https://www.tensorflow.org/guide/random_numbers) guide
  for a detailed description of the various RNG systems in TF. As the
  guide states, legacy stateful RNG ops like `tf.random.uniform` and
  `tf.nn.dropout` are not deprecated yet but highly discouraged,
  because their states are hard to control.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions. This is useful for dropping whole
  channels from an image or sequence. For example:

  >>> x = tf.ones([3,10])
  >>> tf.nn.experimental.stateless_dropout(x, rate=2/3, noise_shape=[1,10],
  ...                                      seed=[1, 0])
  <tf.Tensor: shape=(3, 10), dtype=float32, numpy=
  array([[3., 0., 0., 0., 0., 0., 0., 3., 0., 3.],
         [3., 0., 0., 0., 0., 0., 0., 3., 0., 3.],
         [3., 0., 0., 0., 0., 0., 0., 3., 0., 3.]], dtype=float32)>

  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    seed: An integer tensor of shape `[2]`. The seed of the random numbers.
    rng_alg: The algorithm used to generate the random numbers
      (default to `"auto_select"`). See the `alg` argument of
      `tf.random.stateless_uniform` for the supported values.
    noise_shape: A 1-D integer `Tensor`, representing the
      shape for randomly generated keep/drop flags.
    name: A name for this operation.

  Returns:
    A Tensor of the same shape and dtype of `x`.

  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  uniform_sampler = functools.partial(
      stateless_random_ops.stateless_random_uniform, seed=seed, alg=rng_alg)
  def dummy_rng_step():
    pass
  return _dropout(x=x, rate=rate, noise_shape=noise_shape,
                  uniform_sampler=uniform_sampler,
                  dummy_rng_step=dummy_rng_step, name=name,
                  default_name="stateless_dropout")


@tf_export("nn.experimental.general_dropout")
@dispatch.add_dispatch_support
def general_dropout(x, rate, uniform_sampler, noise_shape=None, name=None):
  """Computes dropout: randomly sets elements to zero to prevent overfitting.

  Please see `tf.nn.experimental.stateless_dropout` for an overview
  of dropout.

  Unlike `tf.nn.experimental.stateless_dropout`, here you can supply a
  custom sampler function `uniform_sampler` that (given a shape and a
  dtype) generates a random, `Uniform[0, 1)`-distributed tensor (of
  that shape and dtype).  `uniform_sampler` can be
  e.g. `tf.random.stateless_random_uniform` or
  `tf.random.Generator.uniform`.

  For example, if you are using `tf.random.Generator` to generate
  random numbers, you can use this code to do dropouts:

  >>> g = tf.random.Generator.from_seed(7)
  >>> sampler = g.uniform
  >>> x = tf.constant([1.1, 2.2, 3.3, 4.4, 5.5])
  >>> rate = 0.5
  >>> tf.nn.experimental.general_dropout(x, rate, sampler)
  <tf.Tensor: shape=(5,), ..., numpy=array([ 0. ,  4.4,  6.6,  8.8, 11. ], ...)>
  >>> tf.nn.experimental.general_dropout(x, rate, sampler)
  <tf.Tensor: shape=(5,), ..., numpy=array([2.2, 0. , 0. , 8.8, 0. ], ...)>

  It has better performance than using
  `tf.nn.experimental.stateless_dropout` and
  `tf.random.Generator.make_seeds`:

  >>> g = tf.random.Generator.from_seed(7)
  >>> x = tf.constant([1.1, 2.2, 3.3, 4.4, 5.5])
  >>> rate = 0.5
  >>> tf.nn.experimental.stateless_dropout(x, rate, g.make_seeds(1)[:, 0])
  <tf.Tensor: shape=(5,), ..., numpy=array([ 2.2,  4.4,  6.6,  0. , 11. ], ...)>
  >>> tf.nn.experimental.stateless_dropout(x, rate, g.make_seeds(1)[:, 0])
  <tf.Tensor: shape=(5,), ..., numpy=array([2.2, 0. , 6.6, 8.8, 0. ], ...>

  because generating and consuming seeds cost extra
  computation. `tf.nn.experimental.general_dropout` can let you avoid
  them.

  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    uniform_sampler: a callable of signature `(shape, dtype) ->
      Tensor[shape, dtype]`, used to generate a tensor of uniformly-distributed
      random numbers in the range `[0, 1)`, of the given shape and dtype.
    noise_shape: A 1-D integer `Tensor`, representing the
      shape for randomly generated keep/drop flags.
    name: A name for this operation.

  Returns:
    A Tensor of the same shape and dtype of `x`.

  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  def dummy_rng_step():
    pass
  return _dropout(x=x, rate=rate, noise_shape=noise_shape,
                  uniform_sampler=uniform_sampler,
                  dummy_rng_step=dummy_rng_step, name=name,
                  default_name="general_dropout")


def _dropout(x, rate, noise_shape, uniform_sampler, dummy_rng_step, name,
             default_name):
  """Shared implementation of the various dropout functions.

  Args:
    x: same as the namesake in `dropout_v2`.
    rate: same as the namesake in `dropout_v2`.
    noise_shape: same as the namesake in `dropout_v2`.
    uniform_sampler: a callable of signature `(shape, dtype) ->
      Tensor`, used to generate a tensor of uniformly-distributed
      random numbers in the range `[0, 1)`, of the given shape and dtype.
    dummy_rng_step: a callable of signature `() -> None`, to make a
      dummy RNG call in the fast path. In the fast path where rate is
      0, we don't need to generate random numbers, but some samplers
      still require you to make an RNG call, to make sure that RNG
      states won't depend on whether the fast path is taken.
    name: same as the namesake in `dropout_v2`.
    default_name: a default name in case `name` is `None`.

  Returns:
    A Tensor of the same shape and dtype of `x`.
  """
  with ops.name_scope(name, default_name, [x]) as name:
    is_rate_number = isinstance(rate, numbers.Real)
    if is_rate_number and (rate < 0 or rate >= 1):
      raise ValueError("`rate` must be a scalar tensor or a float in the "
                       f"range [0, 1). Received: rate={rate}")
    x = ops.convert_to_tensor(x, name="x")
    x_dtype = x.dtype
    if not x_dtype.is_floating:
      raise ValueError(
          "`x.dtype` must be a floating point tensor as `x` will be "
          f"scaled. Received: x_dtype={x_dtype}")
    if is_rate_number and rate == 0:
      # Fast-path: Return the input immediately if rate is non-tensor & is `0`.
      # We trigger this after all error checking
      # and after `x` has been converted to a tensor, to prevent inconsistent
      # tensor conversions/error raising if rate is changed to/from 0.
      #
      # We also explicitly call `dummy_rng_step` to make sure
      # we don't change the random number generation behavior of
      # stateful random ops by entering a fastpath,
      # despite not generating a random tensor in the fastpath
      dummy_rng_step()
      return x

    is_executing_eagerly = context.executing_eagerly()
    if not tensor_util.is_tf_type(rate):
      if is_rate_number:
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
        ret = gen_math_ops.mul(x, scale)
      else:
        raise ValueError(
            f"`rate` must be a scalar or scalar tensor. Received: rate={rate}")
    else:
      rate.get_shape().assert_has_rank(0)
      rate_dtype = rate.dtype
      if rate_dtype != x_dtype:
        if not rate_dtype.is_compatible_with(x_dtype):
          raise ValueError(
              "`x.dtype` must be compatible with `rate.dtype`. "
              f"Received: x.dtype={x_dtype} and rate.dtype={rate_dtype}")
        rate = gen_math_ops.cast(rate, x_dtype, name="rate")
      one_tensor = constant_op.constant(1, dtype=x_dtype)
      ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))

    noise_shape = _get_noise_shape(x, noise_shape)
    # Sample a uniform distribution on [0.0, 1.0) and select values larger
    # than or equal to `rate`.
    random_tensor = uniform_sampler(shape=noise_shape, dtype=x_dtype)
    keep_mask = random_tensor >= rate
    zero_tensor = constant_op.constant(0, dtype=x_dtype)
    ret = array_ops.where_v2(keep_mask, ret, zero_tensor)
    if not is_executing_eagerly:
      ret.set_shape(x.get_shape())
    return ret


@tf_export("math.top_k", "nn.top_k")
@dispatch.add_dispatch_support
def top_k(input, k=1, sorted=True, name=None):  # pylint: disable=redefined-builtin
  """Finds values and indices of the `k` largest entries for the last dimension.

  If the input is a vector (rank=1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  >>> result = tf.math.top_k([1, 2, 98, 1, 1, 99, 3, 1, 3, 96, 4, 1],
  ...                         k=3)
  >>> result.values.numpy()
  array([99, 98, 96], dtype=int32)
  >>> result.indices.numpy()
  array([5, 2, 9], dtype=int32)

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

  >>> input = tf.random.normal(shape=(3,4,5,6))
  >>> k = 2
  >>> values, indices  = tf.math.top_k(input, k=k)
  >>> values.shape.as_list()
  [3, 4, 5, 2]
  >>>
  >>> values.shape == indices.shape == input.shape[:-1] + [k]
  True

  The indices can be used to `gather` from a tensor who's shape matches `input`.

  >>> gathered_values = tf.gather(input, indices, batch_dims=-1)
  >>> assert tf.reduce_all(gathered_values == values)

  If two elements are equal, the lower-index element appears first.

  >>> result = tf.math.top_k([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
  ...                        k=3)
  >>> result.indices.numpy()
  array([0, 1, 3], dtype=int32)

  Args:
    input: 1-D or higher `Tensor` with last dimension at least `k`.
    k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last
      dimension (along each row for matrices).
    sorted: If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: Optional name for the operation.

  Returns:
    A tuple with two named fields:
    values: The `k` largest elements along each last dimensional slice.
    indices: The indices of `values` within the last dimension of `input`.
  """
  return gen_nn_ops.top_kv2(input, k=k, sorted=sorted, name=name)


@tf_export("math.approx_max_k", "nn.approx_max_k")
@dispatch.add_dispatch_support
def approx_max_k(operand,
                 k,
                 reduction_dimension=-1,
                 recall_target=0.95,
                 reduction_input_size_override=-1,
                 aggregate_to_topk=True,
                 name=None):
  """Returns max `k` values and their indices of the input `operand` in an approximate manner.

  See https://arxiv.org/abs/2206.14286 for the algorithm details. This op is
  only optimized on TPU currently.

  Args:
    operand : Array to search for max-k. Must be a floating number type.
    k : Specifies the number of max-k.
    reduction_dimension : Integer dimension along which to search. Default: -1.
    recall_target : Recall target for the approximation.
    reduction_input_size_override : When set to a positive value, it overrides
      the size determined by `operand[reduction_dim]` for evaluating the recall.
      This option is useful when the given `operand` is only a subset of the
      overall computation in SPMD or distributed pipelines, where the true input
      size cannot be deferred by the `operand` shape.
    aggregate_to_topk : When true, aggregates approximate results to top-k. When
      false, returns the approximate results. The number of the approximate
      results is implementation defined and is greater equals to the specified
      `k`.
    name: Optional name for the operation.

  Returns:
    Tuple of two arrays. The arrays are the max `k` values and the
    corresponding indices along the `reduction_dimension` of the input
    `operand`. The arrays' dimensions are the same as the input `operand`
    except for the `reduction_dimension`: when `aggregate_to_topk` is true,
    the reduction dimension is `k`; otherwise, it is greater equals to `k`
    where the size is implementation-defined.

  We encourage users to wrap `approx_max_k` with jit. See the following
  example for maximal inner production search (MIPS):

  >>> import tensorflow as tf
  >>> @tf.function(jit_compile=True)
  ... def mips(qy, db, k=10, recall_target=0.95):
  ...   dists = tf.einsum('ik,jk->ij', qy, db)
  ...   # returns (f32[qy_size, k], i32[qy_size, k])
  ...   return tf.nn.approx_max_k(dists, k=k, recall_target=recall_target)
  >>>
  >>> qy = tf.random.uniform((256,128))
  >>> db = tf.random.uniform((2048,128))
  >>> dot_products, neighbors = mips(qy, db, k=20)
  """
  return gen_nn_ops.approx_top_k(
      operand,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=True,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk,
      name=name)


@tf_export("math.approx_min_k", "nn.approx_min_k")
@dispatch.add_dispatch_support
def approx_min_k(operand,
                 k,
                 reduction_dimension=-1,
                 recall_target=0.95,
                 reduction_input_size_override=-1,
                 aggregate_to_topk=True,
                 name=None):
  """Returns min `k` values and their indices of the input `operand` in an approximate manner.

  See https://arxiv.org/abs/2206.14286 for the algorithm details. This op is
  only optimized on TPU currently.

  Args:
    operand : Array to search for min-k. Must be a floating number type.
    k : Specifies the number of min-k.
    reduction_dimension: Integer dimension along which to search. Default: -1.
    recall_target: Recall target for the approximation.
    reduction_input_size_override : When set to a positive value, it overrides
      the size determined by `operand[reduction_dim]` for evaluating the recall.
      This option is useful when the given `operand` is only a subset of the
      overall computation in SPMD or distributed pipelines, where the true input
      size cannot be deferred by the `operand` shape.
    aggregate_to_topk: When true, aggregates approximate results to top-k. When
      false, returns the approximate results. The number of the approximate
      results is implementation defined and is greater equals to the specified
      `k`.
    name: Optional name for the operation.

  Returns:
    Tuple of two arrays. The arrays are the least `k` values and the
    corresponding indices along the `reduction_dimension` of the input
    `operand`.  The arrays' dimensions are the same as the input `operand`
    except for the `reduction_dimension`: when `aggregate_to_topk` is true,
    the reduction dimension is `k`; otherwise, it is greater equals to `k`
    where the size is implementation-defined.

  We encourage users to wrap `approx_min_k` with jit. See the following example
  for nearest neighbor search over the squared l2 distance:

  >>> import tensorflow as tf
  >>> @tf.function(jit_compile=True)
  ... def l2_ann(qy, db, half_db_norms, k=10, recall_target=0.95):
  ...   dists = half_db_norms - tf.einsum('ik,jk->ij', qy, db)
  ...   return tf.nn.approx_min_k(dists, k=k, recall_target=recall_target)
  >>>
  >>> qy = tf.random.uniform((256,128))
  >>> db = tf.random.uniform((2048,128))
  >>> half_db_norms = tf.norm(db, axis=1) / 2
  >>> dists, neighbors = l2_ann(qy, db, half_db_norms)

  In the example above, we compute `db_norms/2 - dot(qy, db^T)` instead of
  `qy^2 - 2 dot(qy, db^T) + db^2` for performance reason. The former uses less
  arithmetics and produces the same set of neighbors.
  """
  return gen_nn_ops.approx_top_k(
      operand,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=False,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk,
      name=name)


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
@dispatch.add_dispatch_support
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

  Raises:
    ValueError: If op determinism is enabled and either the seeds are not set or
      the "deterministic" argument is False.

  References:
    Fractional Max-Pooling:
      [Graham, 2015](https://arxiv.org/abs/1412.6071)
      ([pdf](https://arxiv.org/pdf/1412.6071.pdf))
  """
  if config.is_op_determinism_enabled() and (not seed or not seed2 or
                                             not deterministic):
    raise ValueError(
        f'tf.compat.v1.nn.fractional_max_pool requires "seed" and '
        f'"seed2" to be non-zero and "deterministic" to be true when op '
        f"determinism is enabled. Please pass in such values, e.g. by passing"
        f'"seed=1, seed2=1, deterministic=True". Got: seed={seed}, '
        f'seed2={seed2}, deterministic={deterministic}')
  return gen_nn_ops.fractional_max_pool(value, pooling_ratio, pseudo_random,
                                        overlapping, deterministic, seed, seed2,
                                        name)


@tf_export("nn.fractional_max_pool", v1=[])
@dispatch.add_dispatch_support
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

  Raises:
    ValueError: If no seed is specified and op determinism is enabled.

  References:
    Fractional Max-Pooling:
      [Graham, 2015](https://arxiv.org/abs/1412.6071)
      ([pdf](https://arxiv.org/pdf/1412.6071.pdf))
  """
  if (isinstance(pooling_ratio, (list, tuple))):
    if (pooling_ratio[0] != 1.0 or pooling_ratio[-1] != 1.0):
      raise ValueError(
          "`pooling_ratio` should have first and last elements with value 1.0. "
          f"Received: pooling_ratio={pooling_ratio}")
    for element in pooling_ratio:
      if element < 1.0:
        raise ValueError(
            f"`pooling_ratio` elements should be >= 1.0. "
            f"Received: pooling_ratio={pooling_ratio}")
  elif (isinstance(pooling_ratio, (int, float))):
    if pooling_ratio < 1.0:
      raise ValueError(
          "`pooling_ratio` should be >= 1.0. "
          f"Received: pooling_ratio={pooling_ratio}")
  else:
    raise ValueError(
        "`pooling_ratio` should be an int or a list of ints. "
        f"Received: pooling_ratio={pooling_ratio}")

  pooling_ratio = _get_sequence(pooling_ratio, 2, 3, "pooling_ratio")

  if seed == 0:
    if config.is_op_determinism_enabled():
      raise ValueError(
          f"tf.nn.fractional_max_pool requires a non-zero seed to be passed in "
          f"when determinism is enabled, but got seed={seed}. Please pass in a "
          f'non-zero seed, e.g. by passing "seed=1".')
    return gen_nn_ops.fractional_max_pool(value, pooling_ratio, pseudo_random,
                                          overlapping, deterministic=False,
                                          seed=0, seed2=0, name=name)
  else:
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_nn_ops.fractional_max_pool(value, pooling_ratio, pseudo_random,
                                          overlapping, deterministic=True,
                                          seed=seed1, seed2=seed2, name=name)


@tf_export(v1=["nn.fractional_avg_pool"])
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
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
      The type of padding algorithm to use. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
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
    raise ValueError("`data_format` values other  than 'NHWC' are not "
                     f"supported. Received: data_format={data_format}")

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
@dispatch.add_dispatch_support
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
@dispatch.add_dispatch_support
def in_top_k_v2(targets, predictions, k, name=None):
  """Outputs whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is finite (not inf, -inf, or nan) and among
  the top `k` predictions among all predictions for example `i`.
  `predictions` does not have to be normalized.

  Note that the behavior of `InTopK` differs from the `TopK` op in its handling
  of ties; if multiple classes have the same prediction value and straddle the
  top-`k` boundary, all of those classes are considered to be in the top `k`.

  >>> target = tf.constant([0, 1, 3])
  >>> pred = tf.constant([
  ...  [1.2, -0.3, 2.8, 5.2],
  ...  [0.1, 0.0, 0.0, 0.0],
  ...  [0.0, 0.5, 0.3, 0.3]],
  ...  dtype=tf.float32)
  >>> print(tf.math.in_top_k(target, pred, 2))
  tf.Tensor([False  True  True], shape=(3,), dtype=bool)

  Args:
    targets: A `batch_size` vector of class ids. Must be `int32` or `int64`.
    predictions: A `batch_size` x `classes` tensor of type `float32`.
    k: An `int`. The parameter to specify search space.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same shape of `targets` with type of `bool`. Each
      element specifies if the target falls into top-k predictions.
  """
  return in_top_k(predictions, targets, k, name)


tf_export(v1=["nn.quantized_avg_pool"])(
    dispatch.add_dispatch_support(gen_nn_ops.quantized_avg_pool))
tf_export(v1=["nn.quantized_conv2d"])(
    dispatch.add_dispatch_support(gen_nn_ops.quantized_conv2d))
tf_export(v1=["nn.quantized_relu_x"])(
    dispatch.add_dispatch_support(gen_nn_ops.quantized_relu_x))
tf_export(v1=["nn.quantized_max_pool"])(
    dispatch.add_dispatch_support(gen_nn_ops.quantized_max_pool))


@tf_export("nn.isotonic_regression", v1=[])
@dispatch.add_dispatch_support
def isotonic_regression(inputs, decreasing=True, axis=-1):
  r"""Solves isotonic regression problems along the given axis.

  For each vector x, the problem solved is

  $$\argmin_{y_1 >= y_2 >= ... >= y_n} \sum_i (x_i - y_i)^2.$$

  As the solution is component-wise constant, a second tensor is returned that
  encodes the segments. The problems are solved over the given axis.

  Consider the following example, where we solve a batch of two problems. The
  first input is [3, 1, 2], while the second [1, 3, 4] (as the axis is 1).
  >>> x = tf.constant([[3, 1, 2], [1, 3, 4]], dtype=tf.float32)
  >>> y, segments = tf.nn.isotonic_regression(x, axis=1)
  >>> y  # The solution.
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[3.       , 1.5      , 1.5      ],
         [2.6666667, 2.6666667, 2.6666667]], dtype=float32)>

  Note that the first solution has two blocks [2] and [1.5, 1.5]. The second
  solution is constant, and thus has a single segment. These segments are
  exactly what the second returned tensor encodes:

  >>> segments
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[0, 1, 1],
         [0, 0, 0]], dtype=int32)>


  Args:
    inputs: A tensor holding the inputs.
    decreasing: If set to False, the inequalities in the optimizing constrained
      are flipped.
    axis: The axis along which the problems should be solved.

  Returns:
    output: The solutions, same shape as type as the input.
    segments: An int32 tensor, same shape as the input indicating the segments
      that have the same value. Specifically, those positions that have the same
      value correspond to the same segment. These values start at zero, and are
      monotonously increasing for each solution.
  """
  type_promotions = {
      # Float types get mapped to themselves, int8/16 to float32, rest to double
      dtypes.float32:
          dtypes.float32,
      dtypes.half:
          dtypes.half,
      dtypes.bfloat16:
          dtypes.bfloat16,
      dtypes.int8:
          dtypes.float32,
      dtypes.int16:
          dtypes.float32,
  }
  inputs = ops.convert_to_tensor(inputs)
  try:
    output_dtype = type_promotions[inputs.dtype]
  except KeyError:
    output_dtype = dtypes.float64

  def compute_on_matrix(matrix, name=None):
    iso_fn = functools.partial(
        gen_nn_ops.isotonic_regression, output_dtype=output_dtype, name=name)
    if decreasing:
      return iso_fn(matrix)
    else:
      output, segments = iso_fn(-matrix)
      return -output, segments

  return _wrap_2d_function(inputs, compute_on_matrix, axis)


# Register elementwise ops that don't have Python wrappers.
# Unary elementwise ops.
dispatch.register_unary_elementwise_api(gen_nn_ops.elu)
dispatch.register_unary_elementwise_api(gen_nn_ops.relu)
dispatch.register_unary_elementwise_api(gen_nn_ops.selu)
dispatch.register_unary_elementwise_api(gen_nn_ops.softsign)
