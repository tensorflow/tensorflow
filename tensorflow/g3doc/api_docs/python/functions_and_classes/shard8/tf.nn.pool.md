### `tf.nn.pool(input, window_shape, pooling_type, padding, dilation_rate=None, strides=None, name=None, data_format=None)` {#pool}

Performs an N-D pooling operation.

In the case that `data_format` does not start with "NC", computes for
    0 <= b < batch_size,
    0 <= x[i] < output_spatial_shape[i],
    0 <= c < num_channels:

  output[b, x[0], ..., x[N-1], c] =
    REDUCE_{z[0], ..., z[N-1]}
      input[b,
            x[0] * strides[0] - pad_before[0] + dilation_rate[0]*z[0],
            ...
            x[N-1]*strides[N-1] - pad_before[N-1] + dilation_rate[N-1]*z[N-1],
            c],

where the reduction function REDUCE depends on the value of `pooling_type`,
and pad_before is defined based on the value of `padding` as described in the
[comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution).
The reduction never includes out-of-bounds positions.

In the case that `data_format` starts with `"NC"`, the `input` and output are
simply transposed as follows:

  pool(input, data_format, **kwargs) =
    tf.transpose(pool(tf.transpose(input, [0] + range(2,N+2) + [1]),
                      **kwargs),
                 [0, N+1] + range(1, N+1))

##### Args:


*  <b>`input`</b>: Tensor of rank N+2, of shape
    `[batch_size] + input_spatial_shape + [num_channels]` if data_format does
    not start with "NC" (default), or
    `[batch_size, num_channels] + input_spatial_shape` if data_format starts
    with "NC".  Pooling happens over the spatial dimensions only.
*  <b>`window_shape`</b>: Sequence of N ints >= 1.
*  <b>`pooling_type`</b>: Specifies pooling operation, must be "AVG" or "MAX".
*  <b>`padding`</b>: The padding algorithm, must be "SAME" or "VALID".
    See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
*  <b>`dilation_rate`</b>: Optional.  Dilation rate.  List of N ints >= 1.
    Defaults to [1]*N.  If any value of dilation_rate is > 1, then all values
    of strides must be 1.
*  <b>`strides`</b>: Optional.  Sequence of N ints >= 1.  Defaults to [1]*N.
    If any value of strides is > 1, then all values of dilation_rate must be
    1.
*  <b>`name`</b>: Optional. Name of the op.
*  <b>`data_format`</b>: A string or None.  Specifies whether the channel dimension of
    the `input` and output is the last dimension (default, or if `data_format`
    does not start with "NC"), or the second dimension (if `data_format`
    starts with "NC").  For N=1, the valid values are "NWC" (default) and
    "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".  For
    N=3, the valid value is "NDHWC".

##### Returns:

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

##### Raises:


*  <b>`ValueError`</b>: if arguments are invalid.

