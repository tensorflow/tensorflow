Optimizer that implements the Momentum algorithm.

- - -

#### `tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False)` {#MomentumOptimizer.__init__}

Construct a new Momentum optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value.  The learning rate.
*  <b>`momentum`</b>: A `Tensor` or a floating point value.  The momentum.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Momentum".
*  <b>`use_nesterov`</b>: If `True` use Nesterov Momentum.
    See [Sutskever et. al., 2013](
*  <b>`http`</b>: //jmlr.org/proceedings/papers/v28/sutskever13.pdf)


