### `tf.nn.log_softmax(logits, name=None)` {#log_softmax}

Computes log softmax activations.

For each batch `i` and class `j` we have

    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

##### Args:


*  <b>`logits`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    2-D with shape `[batch_size, num_classes]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `logits`. Same shape as `logits`.

