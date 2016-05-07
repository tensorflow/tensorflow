Optimizer that implements the Adam algorithm.

See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

- - -

#### `tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')` {#AdamOptimizer.__init__}

Construct a new Adam optimizer.

Initialization:

```
m_0 <- 0 (Initialize initial 1st moment vector)
v_0 <- 0 (Initialize initial 2nd moment vector)
t <- 0 (Initialize timestep)
```

The update rule for `variable` with gradient `g` uses an optimization
described at the end of section2 of the paper:

```
t <- t + 1
lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

m_t <- beta1 * m_{t-1} + (1 - beta1) * g
v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
```

The default value of 1e-8 for epsilon might not be a good default in
general. For example, when training an Inception network on ImageNet a
current good choice is 1.0 or 0.1.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning rate.
*  <b>`beta1`</b>: A float value or a constant float tensor.
    The exponential decay rate for the 1st moment estimates.
*  <b>`beta2`</b>: A float value or a constant float tensor.
    The exponential decay rate for the 2nd moment estimates.
*  <b>`epsilon`</b>: A small constant for numerical stability.
*  <b>`use_locking`</b>: If True use locks for update operations.
*  <b>`name`</b>: Optional name for the operations created when applying gradients.
    Defaults to "Adam".


