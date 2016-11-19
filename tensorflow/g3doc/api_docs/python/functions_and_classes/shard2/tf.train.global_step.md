### `tf.train.global_step(sess, global_step_tensor)` {#global_step}

Small helper to get the global step.

```python
# Creates a variable to hold the global_step.
global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
# Creates a session.
sess = tf.Session()
# Initializes the variable.
print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))

global_step: 10
```

##### Args:


*  <b>`sess`</b>: A TensorFlow `Session` object.
*  <b>`global_step_tensor`</b>: `Tensor` or the `name` of the operation that contains
    the global step.

##### Returns:

  The global step value.

