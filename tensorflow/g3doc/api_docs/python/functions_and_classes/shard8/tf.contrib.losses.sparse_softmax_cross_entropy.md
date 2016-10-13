### `tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels, weight=1.0, scope=None)` {#sparse_softmax_cross_entropy}

Cross-entropy loss using tf.nn.sparse_softmax_cross_entropy_with_logits.

`weight` acts as a coefficient for the loss. If a scalar is provided,
then the loss is simply scaled by the given value. If `weight` is a
tensor of size [`batch_size`], then the loss weights apply to each
corresponding sample.

##### Args:


*  <b>`logits`</b>: [batch_size, num_classes] logits outputs of the network .
*  <b>`labels`</b>: [batch_size, 1] or [batch_size] target labels of dtype `int32` or
    `int64` in the range `[0, num_classes)`.
*  <b>`weight`</b>: Coefficients for the loss. The tensor must be a scalar or a tensor
    of shape [batch_size] or [batch_size, 1].
*  <b>`scope`</b>: the scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shapes of logits, labels, and weight are incompatible, or
    if `weight` is None.

