### `tf.nn.pool(input, window_shape, pooling_type, padding, dilation_rate=None, strides=None, name=None)` {#pool}

Performs an N-D pooling operation.

Computes for
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

##### Args:


*  <b>`input`</b>: Tensor of rank N+2, of shape
    [batch_size] + input_spatial_shape + [num_channels].
    Pooling happens over the spatial dimensions only.
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

##### Returns:

  Tensor of rank N+2, of shape
    [batch_size] + output_spatial_shape + [num_channels],
  where `output_spatial_shape` depends on the value of padding:

  If padding = "SAME":
    output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
  If padding = "VALID":
    output_spatial_shape[i] =
      ceil((input_spatial_shape[i] - (window_shape[i] - 1) * dilation_rate[i])
           / strides[i]).

##### Raises:


*  <b>`ValueError`</b>: if arguments are invalid.

