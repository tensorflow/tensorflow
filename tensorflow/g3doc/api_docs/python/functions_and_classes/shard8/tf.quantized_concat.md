### `tf.quantized_concat(concat_dim, values, input_mins, input_maxes, name=None)` {#quantized_concat}

Concatenates quantized tensors along one dimension.

##### Args:


*  <b>`concat_dim`</b>: An `Output` of type `int32`.
    0-D.  The dimension along which to concatenate.  Must be in the
    range [0, rank(values)).
*  <b>`values`</b>: A list of at least 2 `Output` objects of the same type.
    The `N` Tensors to concatenate. Their ranks and types must match,
    and their sizes must match in all dimensions except `concat_dim`.
*  <b>`input_mins`</b>: A list with the same number of `Output` objects as `values` of `Output` objects of type `float32`.
    The minimum scalar values for each of the input tensors.
*  <b>`input_maxes`</b>: A list with the same number of `Output` objects as `values` of `Output` objects of type `float32`.
    The maximum scalar values for each of the input tensors.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Output` objects (output, output_min, output_max).

*  <b>`output`</b>: A `Output`. Has the same type as `values`. An `Output` with the concatenation of values stacked along the
    `concat_dim` dimension.  This tensor's shape matches that of `values` except
    in `concat_dim` where it has the sum of the sizes.
*  <b>`output_min`</b>: An `Output` of type `float32`. The float value that the minimum quantized output value represents.
*  <b>`output_max`</b>: An `Output` of type `float32`. The float value that the maximum quantized output value represents.

