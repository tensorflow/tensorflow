<!-- This file is machine generated: DO NOT EDIT! -->

# Losses (contrib)
[TOC]

Ops for building neural network losses.

## Other Functions and Classes
- - -

### `tf.contrib.losses.absolute_difference(*args, **kwargs)` {#absolute_difference}

Adds an Absolute Difference loss to the training procedure. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`targets` is being deprecated, use `labels`. `weight` is being deprecated, use `weights`.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    targets: Deprecated alias for `labels`.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.add_loss(*args, **kwargs)` {#add_loss}

Adds a externally defined loss to the collection of losses.

##### Args:


*  <b>`loss`</b>: A loss `Tensor`.
*  <b>`loss_collection`</b>: Optional collection to add the loss to.


- - -

### `tf.contrib.losses.compute_weighted_loss(*args, **kwargs)` {#compute_weighted_loss}

Computes the weighted loss. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`weight` is being deprecated, use `weights`.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weights: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.


- - -

### `tf.contrib.losses.cosine_distance(*args, **kwargs)` {#cosine_distance}

Adds a cosine-distance loss to the training procedure. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`targets` is being deprecated, use `labels`. `weight` is being deprecated, use `weights`.

  Note that the function assumes that `predictions` and `labels` are already
  unit-normalized.

  Args:
    predictions: An arbitrary matrix.
    labels: A `Tensor` whose shape matches 'predictions'
    dim: The dimension along which the cosine distance is computed.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    targets: Deprecated alias for `labels`.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If `predictions` shape doesn't match `labels` shape, or
      `weights` is `None`.


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

### `tf.contrib.losses.hinge_loss(*args, **kwargs)` {#hinge_loss}

Method that returns the loss tensor for hinge loss. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`target` is being deprecated, use `labels`.

  Args:
    logits: The logits, a float tensor.
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0.
    scope: The scope for the operations performed in computing the loss.
    target: Deprecated alias for `labels`.

  Returns:
    A `Tensor` of same shape as logits and target representing the loss values
      across the batch.

  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match.


- - -

### `tf.contrib.losses.log_loss(*args, **kwargs)` {#log_loss}

Adds a Log Loss term to the training procedure. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`targets` is being deprecated, use `labels`. `weight` is being deprecated, use `weights`.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.
    targets: Deprecated alias for `labels`.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.mean_pairwise_squared_error(*args, **kwargs)` {#mean_pairwise_squared_error}

Adds a pairwise-errors-squared loss to the training procedure. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`targets` is being deprecated, use `labels`. `weight` is being deprecated, use `weights`.

  Unlike `mean_squared_error`, which is a measure of the differences between
  corresponding elements of `predictions` and `labels`,
  `mean_pairwise_squared_error` is a measure of the differences between pairs of
  corresponding elements of `predictions` and `labels`.

  For example, if `labels`=[a, b, c] and `predictions`=[x, y, z], there are
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
    labels: The ground truth output tensor, whose shape must match the shape of
      the `predictions` tensor.
    weights: Coefficients for the loss a scalar, a tensor of shape [batch_size]
      or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    targets: Deprecated alias for `labels`.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.mean_squared_error(*args, **kwargs)` {#mean_squared_error}

Adds a Sum-of-Squares loss to the training procedure. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`targets` is being deprecated, use `labels`. `weight` is being deprecated, use `weights`.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    targets: Deprecated alias for `labels`.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weight` is invalid.


- - -

### `tf.contrib.losses.sigmoid_cross_entropy(*args, **kwargs)` {#sigmoid_cross_entropy}

Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`weight` is being deprecated, use `weights`

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/2:

      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    multi_class_labels: [batch_size, num_classes] target labels in (0, 1).
    weights: Coefficients for the loss. The tensor must be a scalar, a tensor of
      shape [batch_size] or shape [batch_size, num_classes].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: The scope for the operations performed in computing the loss.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `multi_class_labels` or if the shape of `weight` is invalid, or if
      `weight` is None.


- - -

### `tf.contrib.losses.softmax_cross_entropy(*args, **kwargs)` {#softmax_cross_entropy}

Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`weight` is being deprecated, use `weights`

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    onehot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    weights: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weight` is invalid or if `weight` is None.


- - -

### `tf.contrib.losses.sparse_softmax_cross_entropy(*args, **kwargs)` {#sparse_softmax_cross_entropy}

Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`weight` is being deprecated, use `weights`

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    labels: [batch_size, 1] or [batch_size] target labels of dtype `int32` or
      `int64` in the range `[0, num_classes)`.
    weights: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size] or [batch_size, 1].
    scope: the scope for the operations performed in computing the loss.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shapes of logits, labels, and weight are incompatible, or
      if `weight` is None.


