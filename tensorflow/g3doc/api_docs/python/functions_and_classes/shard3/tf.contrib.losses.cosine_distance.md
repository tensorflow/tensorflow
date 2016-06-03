### `tf.contrib.losses.cosine_distance(predictions, targets, dim, weight=1.0, scope=None)` {#cosine_distance}

Adds a cosine-distance loss to the training procedure.

Note that the function assumes that the predictions and targets are already
unit-normalized.

##### Args:


*  <b>`predictions`</b>: An arbitrary matrix.
*  <b>`targets`</b>: A `Tensor` whose shape matches 'predictions'
*  <b>`dim`</b>: The dimension along which the cosine distance is computed.
*  <b>`weight`</b>: Coefficients for the loss a scalar, a tensor of shape
    [batch_size] or a tensor whose shape matches `predictions`.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If predictions.shape doesn't match targets.shape, if the ignore
              mask is provided and its shape doesn't match targets.shape or if
              the ignore mask is not boolean valued.

