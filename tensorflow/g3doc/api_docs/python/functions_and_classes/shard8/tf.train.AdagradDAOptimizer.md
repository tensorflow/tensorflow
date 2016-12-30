Adagrad Dual Averaging algorithm for sparse linear models.

See this [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).

This optimizer takes care of regularization of unseen features in a mini batch
by updating them when they are seen with a closed form update rule that is
equivalent to having updated them on every mini-batch.

AdagradDA is typically used when there is a need for large sparsity in the
trained model. This optimizer only guarantees sparsity for linear models. Be
careful when using AdagradDA for deep networks as it will require careful
initialization of the gradient accumulators for it to train.

- - -

#### `tf.train.AdagradDAOptimizer.__init__(learning_rate, global_step, initial_gradient_squared_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='AdagradDA')` {#AdagradDAOptimizer.__init__}

Construct a new AdagradDA optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value.  The learning rate.
*  <b>`global_step`</b>: A `Tensor` containing the current training step number.
*  <b>`initial_gradient_squared_accumulator_value`</b>: A floating point value.
    Starting value for the accumulators, must be positive.
*  <b>`l1_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`l2_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "AdagradDA".

##### Raises:


*  <b>`ValueError`</b>: If the `initial_gradient_squared_accumulator_value` is
  invalid.


