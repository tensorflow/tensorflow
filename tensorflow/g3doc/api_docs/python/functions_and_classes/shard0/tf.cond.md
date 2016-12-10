### `tf.cond(pred, fn1, fn2, name=None)` {#cond}

Return either fn1() or fn2() based on the boolean predicate `pred`.

`fn1` and `fn2` both return lists of output tensors. `fn1` and `fn2` must have
the same non-zero number and type of outputs.

Note that the conditional execution applies only to the operations defined in
fn1 and fn2. Consider the following simple program:

```python
z = tf.mul(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
```

If x < y, the `tf.add` operation will be executed and tf.square
operation will not be executed. Since z is needed for at least one
branch of the cond, the tf.mul operation is always executed, unconditionally.
Although this behavior is consistent with the dataflow model of TensorFlow,
it has occasionally surprised some users who expected a lazier semantics.

##### Args:


*  <b>`pred`</b>: A scalar determining whether to return the result of `fn1` or `fn2`.
*  <b>`fn1`</b>: The callable to be performed if pred is true.
*  <b>`fn2`</b>: The callable to be performed if pref is false.
*  <b>`name`</b>: Optional name prefix for the returned tensors.

##### Returns:

  Tensors returned by the call to either `fn1` or `fn2`. If the callables
  return a singleton list, the element is extracted from the list.

##### Raises:


*  <b>`TypeError`</b>: if `fn1` or `fn2` is not callable.
*  <b>`ValueError`</b>: if `fn1` and `fn2` do not return the same number of tensors, or
              return tensors of different types.


*  <b>`Example`</b>: 

```python
  x = tf.constant(2)
  y = tf.constant(5)
  def f1(): return tf.mul(x, 17)
  def f2(): return tf.add(y, 23)
  r = tf.cond(tf.less(x, y), f1, f2)
  # r is set to f1().
  # Operations in f2 (e.g., tf.add) are not executed.
```

