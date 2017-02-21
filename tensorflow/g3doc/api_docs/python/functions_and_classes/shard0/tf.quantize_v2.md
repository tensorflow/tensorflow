### `tf.quantize_v2(input, min_range, max_range, T, mode=None, name=None)` {#quantize_v2}

Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
if T == qint8, out[i] -= (range(T) + 1) / 2.0
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN_COMBINED Mode Example*

Assume the input is type float and has a possible range of [0.0, 6.0] and the
output type is quint8 ([0, 255]). The min_range and max_range values should be
specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
value of the input by 255/6 and cast to quint8.

If the output type was qint8 ([-128, 127]), the operation will additionally
subtract each value by 128 prior to casting, so that the range of values aligns
with the range of qint8.

If the mode is 'MIN_FIRST', then this approach is used:

```
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = number_of_steps / range
quantized = round(input * range_scale) - round(range_min * range_scale) +
  numeric_limits<T>::min()
quantized = max(quantized, numeric_limits<T>::min())
quantized = min(quantized, numeric_limits<T>::max())
```

The biggest difference between this and MIN_COMBINED is that the minimum range
is rounded first, before it's subtracted from the rounded value. With
MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
and dequantizing will introduce a larger and larger error.

One thing to watch out for is that the operator may choose to adjust the
requested minimum and maximum values slightly during the quantization process,
so you should always use the output ports as the range for further calculations.
For example, if the requested minimum and maximum values are close to equal,
they will be separated by a small epsilon value to prevent ill-formed quantized
buffers from being created. Otherwise, you can end up with buffers where all the
quantized values map to the same float value, which causes problems for
operations that have to perform further calculations on them.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `float32`.
*  <b>`min_range`</b>: A `Tensor` of type `float32`.
    The minimum scalar value possibly produced for the input.
*  <b>`max_range`</b>: A `Tensor` of type `float32`.
    The maximum scalar value possibly produced for the input.
*  <b>`T`</b>: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
*  <b>`mode`</b>: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST"`. Defaults to `"MIN_COMBINED"`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (output, output_min, output_max).

*  <b>`output`</b>: A `Tensor` of type `T`. The quantized data produced from the float input.
*  <b>`output_min`</b>: A `Tensor` of type `float32`. The actual minimum scalar value used for the output.
*  <b>`output_max`</b>: A `Tensor` of type `float32`. The actual maximum scalar value used for the output.

