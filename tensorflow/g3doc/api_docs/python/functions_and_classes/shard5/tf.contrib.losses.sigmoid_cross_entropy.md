### `tf.contrib.losses.sigmoid_cross_entropy(logits, multi_class_labels, weight=1.0, label_smoothing=0, scope=None)` {#sigmoid_cross_entropy}

Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

##### Args:


*  <b>`logits`</b>: [batch_size, num_classes] logits outputs of the network .
*  <b>`multi_class_labels`</b>: [batch_size, num_classes] target labels in (0, 1).
*  <b>`weight`</b>: Coefficients for the loss. The tensor must be a scalar, a tensor of
    shape [batch_size] or shape [batch_size, num_classes].
*  <b>`label_smoothing`</b>: If greater than 0 then smooth the labels.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

