### `tf.nn.zero_fraction(value, name=None)` {#zero_fraction}

Returns the fraction of zeros in `value`.

If `value` is empty, the result is `nan`.

This is useful in summaries to measure and report sparsity.  For example,

    z = tf.Relu(...)
    summ = tf.scalar_summary('sparsity', tf.nn.zero_fraction(z))

##### Args:


*  <b>`value`</b>: A tensor of numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The fraction of zeros in `value`, with type `float32`.

