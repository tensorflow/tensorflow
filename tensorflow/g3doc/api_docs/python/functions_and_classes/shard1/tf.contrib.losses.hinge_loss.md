### `tf.contrib.losses.hinge_loss(logits, target, scope=None)` {#hinge_loss}

Method that returns the loss tensor for hinge loss.

##### Args:


*  <b>`logits`</b>: The logits, a float tensor.
*  <b>`target`</b>: The ground truth output tensor. Its shape should match the shape of
    logits. The values of the tensor are expected to be 0.0 or 1.0.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A `Tensor` of same shape as logits and target representing the loss values
    across the batch.

##### Raises:


*  <b>`ValueError`</b>: If the shapes of `logits` and `target` don't match.

