Optimizer that implements the proximal gradient descent algorithm.

See this [paper](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf).

- - -

#### `tf.train.ProximalGradientDescentOptimizer.__init__(learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='ProximalGradientDescent')` {#ProximalGradientDescentOptimizer.__init__}

Construct a new proximal gradient descent optimizer.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning
    rate to use.
*  <b>`l1_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`l2_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`use_locking`</b>: If True use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients. Defaults to "GradientDescent".


