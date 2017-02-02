Optimizer that implements the Adadelta algorithm.

See [M. D. Zeiler](http://arxiv.org/abs/1212.5701)
([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))

- - -

#### `tf.train.AdadeltaOptimizer.__init__(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')` {#AdadeltaOptimizer.__init__}

Construct a new Adadelta optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value. The learning rate.
*  <b>`rho`</b>: A `Tensor` or a floating point value. The decay rate.
*  <b>`epsilon`</b>: A `Tensor` or a floating point value.  A constant epsilon used
           to better conditioning the grad update.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Adadelta".


