### `tf.train.inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)` {#inverse_time_decay}

Applies inverse time decay to the initial learning rate.

When training a model, it is often recommended to lower the learning rate as
the training progresses.  This function applies an inverse decay function
to a provided initial learning rate.  It requires an `global_step` value to
compute the decayed learning rate.  You can just pass a TensorFlow variable
that you increment at each training step.

The function returns the decayed learning rate.  It is computed as:

```python
decayed_learning_rate = learning_rate / (1 + decay_rate * t)
```

Example: decay 1/t with a rate of 0.5:

```python
...
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1
k = 0.5
learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, k)

# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)
```

##### Args:


*  <b>`learning_rate`</b>: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The initial learning rate.
*  <b>`global_step`</b>: A Python number.
    Global step to use for the decay computation.  Must not be negative.
*  <b>`decay_steps`</b>: How often to apply decay.
*  <b>`decay_rate`</b>: A Python number.  The decay rate.
*  <b>`staircase`</b>: Whether to apply decay in a discrete staircase, as opposed to
    continuous, fashion.
*  <b>`name`</b>: String.  Optional name of the operation.  Defaults to
    'InverseTimeDecay'.

##### Returns:

  A scalar `Tensor` of the same type as `learning_rate`.  The decayed
  learning rate.

##### Raises:


*  <b>`ValueError`</b>: if `global_step` is not supplied.

