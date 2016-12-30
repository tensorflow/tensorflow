Optimizer that implements the Proximal Adagrad algorithm.

See this [paper](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf).

- - -

#### `tf.train.ProximalAdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='ProximalAdagrad')` {#ProximalAdagradOptimizer.__init__}

Construct a new ProximalAdagrad optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value.  The learning rate.
*  <b>`initial_accumulator_value`</b>: A floating point value.
    Starting value for the accumulators, must be positive.
*  <b>`l1_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`l2_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Adagrad".

##### Raises:


*  <b>`ValueError`</b>: If the `initial_accumulator_value` is invalid.


