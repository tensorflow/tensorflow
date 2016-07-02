### `tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels, weight=1.0, label_smoothing=0, scope=None)` {#softmax_cross_entropy}

Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits.

It can scale the loss by weight factor, and smooth the labels.

##### Args:


*  <b>`logits`</b>: [batch_size, num_classes] logits outputs of the network .
*  <b>`onehot_labels`</b>: [batch_size, num_classes] target one_hot_encoded labels.
*  <b>`weight`</b>: Coefficients for the loss. The tensor must be a scalar or a tensor
    of shape [batch_size].
*  <b>`label_smoothing`</b>: If greater than 0 then smooth the labels.
*  <b>`scope`</b>: the scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

