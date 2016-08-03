### `tf.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)` {#cumsum}

Compute the cumulative sum of the tensor `x` along `axis`.

By default, this op performs an inclusive cumsum, which means that the first
element of the input is identical to the first element of the output:
```prettyprint
tf.cumsum([a, b, c]) ==> [a, a + b, a + b + c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumsum is performed
instead:
```prettyprint
tf.cumsum([a, b, c], exclusive=True) ==> [0, a, a + b]
```

By setting the `reverse` kwarg to `True`, the cumsum is performed in the
opposite direction:
```prettyprint
tf.cumsum([a, b, c], reverse=True) ==> [a + b + c, b + c, c]
```
This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:
```prettyprint
tf.cumsum([a, b, c], exclusive=True, reverse=True) ==> [b + c, c, 0]
```

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`,
     `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
     `complex128`, `qint8`, `quint8`, `qint32`, `half`.
*  <b>`axis`</b>: A `Tensor` of type `int32` (default: 0).
*  <b>`reverse`</b>: A `bool` (default: False).
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.

