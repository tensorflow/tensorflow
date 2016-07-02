### `tf.nn.erosion2d(value, kernel, strides, rates, padding, name=None)` {#erosion2d}

Computes the grayscale erosion of 4-D `value` and 3-D `kernel` tensors.

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

##### Args:


*  <b>`value`</b>: A `Tensor`. 4-D with shape `[batch, in_height, in_width, depth]`.
*  <b>`kernel`</b>: A `Tensor`. Must have the same type as `value`.
    3-D with shape `[kernel_height, kernel_width, depth]`.
*  <b>`strides`</b>: A list of `ints` that has length `>= 4`.
    1-D of length 4. The stride of the sliding window for each dimension of
    the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
*  <b>`rates`</b>: A list of `ints` that has length `>= 4`.
    1-D of length 4. The input stride for atrous morphological dilation.
    Must be: `[1, rate_height, rate_width, 1]`.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`name`</b>: A name for the operation (optional). If not specified "erosion2d"
    is used.

##### Returns:

  A `Tensor`. Has the same type as `value`.
  4-D with shape `[batch, out_height, out_width, depth]`.

##### Raises:


*  <b>`ValueError`</b>: If the `value` depth does not match `kernel`' shape, or if
    padding is other than `'VALID'` or `'SAME'`.

