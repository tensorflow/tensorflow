### `tf.dequantize(input, min_range, max_range, mode=None, name=None)` {#dequantize}

Dequantize the 'input' tensor into a float Tensor.

[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
if T == qint8, in[i] += (range(T) + 1)/ 2.0
out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN_COMBINED Mode Example*

If the input comes from a QuantizedRelu6, the output type is
quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
Dequantize on quint8 will take each value, cast to float, and multiply
by 6 / 255.
Note that if quantizedtype is qint8, the operation will additionally add
each value by 128 prior to casting.

If the mode is 'MIN_FIRST', then this approach is used:

```
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = range / number_of_steps
const double offset_input = static_cast<double>(input) - lowest_quantized;
result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
```

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
*  <b>`min_range`</b>: A `Tensor` of type `float32`.
    The minimum scalar value possibly produced for the input.
*  <b>`max_range`</b>: A `Tensor` of type `float32`.
    The maximum scalar value possibly produced for the input.
*  <b>`mode`</b>: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST"`. Defaults to `"MIN_COMBINED"`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.

