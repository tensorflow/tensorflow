Optimizer that implements the FTRL algorithm.

See this [paper](
https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).

- - -

#### `tf.train.FtrlOptimizer.__init__(learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='Ftrl')` {#FtrlOptimizer.__init__}

Construct a new FTRL optimizer.

##### Args:


*  <b>`learning_rate`</b>: A float value or a constant float `Tensor`.
*  <b>`learning_rate_power`</b>: A float value, must be less or equal to zero.
*  <b>`initial_accumulator_value`</b>: The starting value for accumulators.
    Only positive values are allowed.
*  <b>`l1_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`l2_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Ftrl".

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


