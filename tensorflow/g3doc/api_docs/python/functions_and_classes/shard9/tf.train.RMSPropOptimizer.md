Optimizer that implements the RMSProp algorithm.

See the [paper]
(http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

- - -

#### `tf.train.RMSPropOptimizer.__init__(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')` {#RMSPropOptimizer.__init__}

Construct a new RMSProp optimizer.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning rate.
*  <b>`decay`</b>: Discounting factor for the history/coming gradient
*  <b>`momentum`</b>: A scalar tensor.
*  <b>`epsilon`</b>: Small value to avoid zero denominator.
*  <b>`use_locking`</b>: If True use locks for update operation.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients. Defaults to "RMSProp".


