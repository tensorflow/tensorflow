### `tf.convert_to_tensor(value, dtype=None, name=None, as_ref=False)` {#convert_to_tensor}

Converts the given `value` to a `Tensor`.

This function converts Python objects of various types to `Tensor`
objects. It accepts `Tensor` objects, numpy arrays, Python lists,
and Python scalars. For example:

```python
import numpy as np

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
```

This function can be useful when composing a new operation in Python
(such as `my_func` in the example above). All standard Python op
constructors apply this function to each of their Tensor-valued
inputs, which allows those ops to accept numpy arrays, Python lists,
and scalars in addition to `Tensor` objects.

##### Args:


*  <b>`value`</b>: An object whose type has a registered `Tensor` conversion function.
*  <b>`dtype`</b>: Optional element type for the returned tensor. If missing, the
    type is inferred from the type of `value`.
*  <b>`name`</b>: Optional name to use if a new `Tensor` is created.
*  <b>`as_ref`</b>: True if we want the result as a ref tensor. Only used if a new
    `Tensor` is created.

##### Returns:

  A `Tensor` based on `value`.

##### Raises:


*  <b>`TypeError`</b>: If no conversion function is registered for `value`.
*  <b>`RuntimeError`</b>: If a registered conversion function returns an invalid value.

