"""Various learning rate decay functions."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
                      staircase=False, name=None):
  """Applies exponential decay to the learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `global_step /decay_steps` is an
  integer division and the decayed learning rate follows a staircase function.

  Example: decay every 100000 steps with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.exponential_decay(starter_learning_rate, global_step,
                                       100000, 0.96, staircase=True)
  optimizer = tf.GradientDescent(learning_rate)
  # Passing global_step to minimize() will increment it at each step.
  optimizer.minimize(...my loss..., global_step=global_step)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate.
    staircase: Boolean.  It `True` decay the learning rate at discrete intervals.
    name: string.  Optional name of the operation.  Defaults to 'ExponentialDecay'

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  """
  with ops.op_scope([learning_rate, global_step, decay_steps, decay_rate],
                   name, "ExponentialDecay") as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    decay_rate = math_ops.cast(decay_rate, dtype)
    p = global_step / decay_steps
    if staircase:
      p = math_ops.floor(p)
    return math_ops.mul(learning_rate, math_ops.pow(decay_rate, p), name=name)
