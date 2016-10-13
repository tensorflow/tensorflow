Optimizer that implements the RMSProp algorithm.

See the [paper](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

- - -

#### `tf.train.RMSPropOptimizer.__init__(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp')` {#RMSPropOptimizer.__init__}

Construct a new RMSProp optimizer.

Note that in dense implement of this algorithm, m_t and v_t will
update even if g is zero, but in sparse implement, m_t and v_t
will not update in iterations g is zero.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning rate.
*  <b>`decay`</b>: Discounting factor for the history/coming gradient
*  <b>`momentum`</b>: A scalar tensor.
*  <b>`epsilon`</b>: Small value to avoid zero denominator.
*  <b>`use_locking`</b>: If True use locks for update operation.
*  <b>`centered`</b>: If True, gradients are normalized by the estimated variance of
    the gradient; if False, by the uncentered second moment. Setting this to
    True may help with training, but is slightly more expensive in terms of
    computation and memory. Defaults to False.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients. Defaults to "RMSProp".


