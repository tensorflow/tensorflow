### `tf.nn.dilation2d(input, filter, strides, rates, padding, name=None)` {#dilation2d}

Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
`filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
input channel is processed independently of the others with its own structuring
function. The `output` tensor has shape
`[batch, out_height, out_width, depth]`. The spatial dimensions of the output
tensor depend on the `padding` algorithm. We currently only support the default
"NHWC" `data_format`.

In detail, the grayscale morphological 2-D dilation is the max-sum correlation
(for consistency with `conv2d`, we use unmirrored filters):

    output[b, y, x, c] =
       max_{dy, dx} input[b,
                          strides[1] * y + rates[1] * dy,
                          strides[2] * x + rates[2] * dx,
                          c] +
                    filter[dy, dx, c]

Max-pooling is a special case when the filter has size equal to the pooling
kernel size and contains all zeros.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`filter`</b>: A `Tensor`. Must have the same type as `input`.
*  <b>`strides`</b>: A list of `ints` that has length `>= 4`.
*  <b>`rates`</b>: A list of `ints` that has length `>= 4`.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.

