<!-- This file is machine generated: DO NOT EDIT! -->

# Losses (contrib)
[TOC]

Ops for building neural network losses.

## Other Functions and Classes
- - -

### `tf.contrib.losses.absolute_difference(predictions, targets, weight=1.0, scope=None)` {#absolute_difference}

Adds an Absolute Difference loss to the training procedure.

`weight` acts as a coefficient for the loss. If a scalar is provided, then the
loss is simply scaled by the given value. If `weight` is a tensor of size
[batch_size], then the total loss for each sample of the batch is rescaled
by the corresponding element in the `weight` vector. If the shape of
`weight` matches the shape of `predictions`, then the loss of each
measurable element of `predictions` is scaled by the corresponding value of
`weight`.

##### Args:


*  <b>`predictions`</b>: The predicted outputs.
*  <b>`targets`</b>: The ground truth output tensor, same dimensions as 'predictions'.
*  <b>`weight`</b>: Coefficients for the loss a scalar, a tensor of shape
    [batch_size] or a tensor whose shape matches `predictions`.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` doesn't match that of `targets` or
    if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.add_loss(*args, **kwargs)` {#add_loss}

Adds a externally defined loss to the collection of losses.

##### Args:


*  <b>`loss`</b>: A loss `Tensor`.
*  <b>`loss_collection`</b>: Optional collection to add the loss to.


- - -

### `tf.contrib.losses.compute_weighted_loss(losses, weight=1.0)` {#compute_weighted_loss}

Computes the weighted loss.

##### Args:


*  <b>`losses`</b>: A tensor of size [batch_size, d1, ... dN].
*  <b>`weight`</b>: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.

##### Returns:

  A scalar `Tensor` that returns the weighted loss.

##### Raises:


*  <b>`ValueError`</b>: If the weight is None or the shape is not compatible with the
    losses shape or if the number of dimensions (rank) of either losses or
    weight is missing.


- - -

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


- - -

### `tf.contrib.losses.get_losses(scope=None, loss_collection='losses')` {#get_losses}

Gets the list of losses from the loss_collection.

##### Args:


*  <b>`scope`</b>: an optional scope for filtering the losses to return.
*  <b>`loss_collection`</b>: Optional losses collection.

##### Returns:

  a list of loss tensors.


- - -

### `tf.contrib.losses.get_regularization_losses(scope=None)` {#get_regularization_losses}

Gets the regularization losses.

##### Args:


*  <b>`scope`</b>: an optional scope for filtering the losses to return.

##### Returns:

  A list of loss variables.


- - -

### `tf.contrib.losses.get_total_loss(add_regularization_losses=True, name='total_loss')` {#get_total_loss}

Returns a tensor whose value represents the total loss.

Notice that the function adds the given losses to the regularization losses.

##### Args:


*  <b>`add_regularization_losses`</b>: A boolean indicating whether or not to use the
    regularization losses in the sum.
*  <b>`name`</b>: The name of the returned tensor.

##### Returns:

  A `Tensor` whose value represents the total loss.

##### Raises:


*  <b>`ValueError`</b>: if `losses` is not iterable.


- - -

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


- - -

### `tf.contrib.losses.log_loss(predictions, targets, weight=1.0, epsilon=1e-07, scope=None)` {#log_loss}

Adds a Log Loss term to the training procedure.

`weight` acts as a coefficient for the loss. If a scalar is provided, then the
loss is simply scaled by the given value. If `weight` is a tensor of size
[batch_size], then the total loss for each sample of the batch is rescaled
by the corresponding element in the `weight` vector. If the shape of
`weight` matches the shape of `predictions`, then the loss of each
measurable element of `predictions` is scaled by the corresponding value of
`weight`.

##### Args:


*  <b>`predictions`</b>: The predicted outputs.
*  <b>`targets`</b>: The ground truth output tensor, same dimensions as 'predictions'.
*  <b>`weight`</b>: Coefficients for the loss a scalar, a tensor of shape
    [batch_size] or a tensor whose shape matches `predictions`.
*  <b>`epsilon`</b>: A small increment to add to avoid taking a log of zero.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` doesn't match that of `targets` or
    if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.mean_pairwise_squared_error(*args, **kwargs)` {#mean_pairwise_squared_error}

Adds a pairwise-errors-squared loss to the training procedure. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-01.
Instructions for updating:
Use mean_pairwise_squared_error.

  Unlike the sum_of_squares loss, which is a measure of the differences between
  corresponding elements of `predictions` and `targets`, sum_of_pairwise_squares
  is a measure of the differences between pairs of corresponding elements of
  `predictions` and `targets`.

  For example, if `targets`=[a, b, c] and `predictions`=[x, y, z], there are
  three pairs of differences are summed to compute the loss:
    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

  Note that since the inputs are of size [batch_size, d0, ... dN], the
  corresponding pairs are computed within each batch sample but not across
  samples within a batch. For example, if `predictions` represents a batch of
  16 grayscale images of dimension [batch_size, 100, 200], then the set of pairs
  is drawn from each image, but not across images.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector.

  Args:
    predictions: The predicted outputs, a tensor of size [batch_size, d0, .. dN]
      where N+1 is the total number of dimensions in `predictions`.
    targets: The ground truth output tensor, whose shape must match the shape of
      the `predictions` tensor.
    weight: Coefficients for the loss a scalar, a tensor of shape [batch_size]
      or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.mean_squared_error(*args, **kwargs)` {#mean_squared_error}

Adds a Sum-of-Squares loss to the training procedure. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-01.
Instructions for updating:
Use mean_squared_error.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    targets: The ground truth output tensor, same dimensions as 'predictions'.
    weight: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.sigmoid_cross_entropy(logits, multi_class_labels, weight=1.0, label_smoothing=0, scope=None)` {#sigmoid_cross_entropy}

Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

`weight` acts as a coefficient for the loss. If a scalar is provided,
then the loss is simply scaled by the given value. If `weight` is a
tensor of size [`batch_size`], then the loss weights apply to each
corresponding sample.

If `label_smoothing` is nonzero, smooth the labels towards 1/2:

    new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                            + 0.5 * label_smoothing

##### Args:


*  <b>`logits`</b>: [batch_size, num_classes] logits outputs of the network .
*  <b>`multi_class_labels`</b>: [batch_size, num_classes] target labels in (0, 1).
*  <b>`weight`</b>: Coefficients for the loss. The tensor must be a scalar, a tensor of
    shape [batch_size] or shape [batch_size, num_classes].
*  <b>`label_smoothing`</b>: If greater than 0 then smooth the labels.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` doesn't match that of `targets` or
    if the shape of `weight` is invalid or if `weight` is None.


- - -

### `tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels, weight=1.0, label_smoothing=0, scope=None)` {#softmax_cross_entropy}

Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits.

`weight` acts as a coefficient for the loss. If a scalar is provided,
then the loss is simply scaled by the given value. If `weight` is a
tensor of size [`batch_size`], then the loss weights apply to each
corresponding sample.

If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
    new_onehot_labels = onehot_labels * (1 - label_smoothing)
                        + label_smoothing / num_classes

##### Args:


*  <b>`logits`</b>: [batch_size, num_classes] logits outputs of the network .
*  <b>`onehot_labels`</b>: [batch_size, num_classes] target one_hot_encoded labels.
*  <b>`weight`</b>: Coefficients for the loss. The tensor must be a scalar or a tensor
    of shape [batch_size].
*  <b>`label_smoothing`</b>: If greater than 0 then smooth the labels.
*  <b>`scope`</b>: the scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `logits` doesn't match that of `onehot_labels`
    or if the shape of `weight` is invalid or if `weight` is None.


- - -

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


- - -

### `tf.contrib.losses.sum_of_pairwise_squares(*args, **kwargs)` {#sum_of_pairwise_squares}

Adds a pairwise-errors-squared loss to the training procedure. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-01.
Instructions for updating:
Use mean_pairwise_squared_error.

  Unlike the sum_of_squares loss, which is a measure of the differences between
  corresponding elements of `predictions` and `targets`, sum_of_pairwise_squares
  is a measure of the differences between pairs of corresponding elements of
  `predictions` and `targets`.

  For example, if `targets`=[a, b, c] and `predictions`=[x, y, z], there are
  three pairs of differences are summed to compute the loss:
    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

  Note that since the inputs are of size [batch_size, d0, ... dN], the
  corresponding pairs are computed within each batch sample but not across
  samples within a batch. For example, if `predictions` represents a batch of
  16 grayscale images of dimension [batch_size, 100, 200], then the set of pairs
  is drawn from each image, but not across images.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector.

  Args:
    predictions: The predicted outputs, a tensor of size [batch_size, d0, .. dN]
      where N+1 is the total number of dimensions in `predictions`.
    targets: The ground truth output tensor, whose shape must match the shape of
      the `predictions` tensor.
    weight: Coefficients for the loss a scalar, a tensor of shape [batch_size]
      or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.sum_of_squares(*args, **kwargs)` {#sum_of_squares}

Adds a Sum-of-Squares loss to the training procedure. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-01.
Instructions for updating:
Use mean_squared_error.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    targets: The ground truth output tensor, same dimensions as 'predictions'.
    weight: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.


