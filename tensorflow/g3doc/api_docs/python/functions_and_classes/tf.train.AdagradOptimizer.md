Optimizer that implements the Adagrad algorithm.

See this [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).

- - -

#### `tf.train.AdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')` {#AdagradOptimizer.__init__}

Construct a new Adagrad optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value.  The learning rate.
*  <b>`initial_accumulator_value`</b>: A floating point value.
    Starting value for the accumulators, must be positive.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Adagrad".

##### Raises:


*  <b>`ValueError`</b>: If the `initial_accumulator_value` is invalid.


