### `tf.nn.log_softmax(logits, dim=-1, name=None)` {#log_softmax}

Computes log softmax activations.

For each batch `i` and class `j` we have

    logsoftmax = logits - reduce_sum(exp(logits), dim)

##### Args:


*  <b>`logits`</b>: A non-empty `Tensor`. Must be one of the following types: `half`,
    `float32`, `float64`.
*  <b>`dim`</b>: The dimension softmax would be performed on. The default is -1 which
    indicates the last dimension.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `logits`. Same shape as `logits`.

##### Raises:


*  <b>`InvalidArgumentError`</b>: if `logits` is empty or `dim` is beyond the last
    dimension of `logits`.

