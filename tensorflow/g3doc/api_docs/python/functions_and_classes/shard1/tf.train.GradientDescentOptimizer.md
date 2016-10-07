Optimizer that implements the gradient descent algorithm.

- - -

#### `tf.train.GradientDescentOptimizer.__init__(learning_rate, use_locking=False, name='GradientDescent')` {#GradientDescentOptimizer.__init__}

Construct a new gradient descent optimizer.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning
    rate to use.
*  <b>`use_locking`</b>: If True use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients. Defaults to "GradientDescent".


