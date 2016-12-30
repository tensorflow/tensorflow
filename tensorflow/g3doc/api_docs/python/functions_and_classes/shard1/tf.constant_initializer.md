### `tf.constant_initializer(value=0, dtype=tf.float32)` {#constant_initializer}

Returns an initializer that generates tensors with constant values.

The resulting tensor is populated with values of type `dtype`, as
specified by arguments `value` following the desired `shape` of the
new tensor (see examples below).

The argument `value` can be a constant value, or a list of values of type
`dtype`. If `value` is a list, then the length of the list must be less
than or equal to the number of elements implied by the desired shape of the
tensor. In the case where the total number of elements in `value` is less
than the number of elements required by the tensor shape, the last element
in `value` will be used to fill the remaining entries. If the total number of
elements in `value` is greater than the number of elements required by the
tensor shape, the initializer will raise a `ValueError`.

##### Args:


*  <b>`value`</b>: A Python scalar, list of values, or a N-dimensional numpy array. All
    elements of the initialized variable will be set to the corresponding
    value in the `value` argument.
*  <b>`dtype`</b>: The data type.

##### Returns:

  An initializer that generates tensors with constant values.

##### Examples:

  The following example can be rewritten using a numpy.ndarray instead
  of the `value` list, even reshaped, as shown in the two commented lines
  below the `value` list initialization.

```python
  >>> import numpy as np
  >>> import tensorflow as tf

  >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
  >>> # value = np.array(value)
  >>> # value = value.reshape([2, 4])
  >>> init = tf.constant_initializer(value)

  >>> print('fitting shape:')
  >>> tf.reset_default_graph()
  >>> with tf.Session():
  >>>   x = tf.get_variable('x', shape=[2, 4], initializer=init)
  >>>   x.initializer.run()
  >>>   print(x.eval())

  fitting shape:
  [[ 0.  1.  2.  3.]
   [ 4.  5.  6.  7.]]

  >>> print('larger shape:')
  >>> tf.reset_default_graph()
  >>> with tf.Session():
  >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init)
  >>>   x.initializer.run()
  >>>   print(x.eval())

  larger shape:
  [[ 0.  1.  2.  3.]
   [ 4.  5.  6.  7.]
   [ 7.  7.  7.  7.]]

  >>> print('smaller shape:')
  >>> tf.reset_default_graph()
  >>> with tf.Session():
  >>>   x = tf.get_variable('x', shape=[2, 3], initializer=init)


*  <b>`ValueError`</b>: Too many elements provided. Needed at most 6, but received 8
```

