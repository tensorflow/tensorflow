### `tf.nn.softmax(logits, name=None)` {#softmax}

Computes softmax activations.

For each batch `i` and class `j` we have

    softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))

##### Args:


*  <b>`logits`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    2-D with shape `[batch_size, num_classes]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `logits`. Same shape as `logits`.

