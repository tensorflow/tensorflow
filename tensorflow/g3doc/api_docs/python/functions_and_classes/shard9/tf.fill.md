### `tf.fill(dims, value, name=None)` {#fill}

Creates a tensor filled with a scalar value.

This operation creates a tensor of shape `dims` and fills it with `value`.

For example:

```prettyprint
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```

##### Args:


*  <b>`dims`</b>: A `Tensor` of type `int32`.
    1-D. Represents the shape of the output tensor.
*  <b>`value`</b>: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.

    @compatibility(numpy)
    Equivalent to np.full
    @end_compatibility

*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `value`.

