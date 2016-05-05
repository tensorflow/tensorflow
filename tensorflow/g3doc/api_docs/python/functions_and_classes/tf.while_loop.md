### `tf.while_loop(cond, body, loop_vars, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)` {#while_loop}

Repeat `body` while the condition `cond` is true.

`cond` is a callable taking a list of tensors and returning a boolean scalar
tensor. `body` is a callable taking a list of tensors and returning a list of
tensors of the same length and with the same types as the input. `loop_vars`
is a list of tensors that is passed to both `cond` and `body`.

In addition to regular Tensors or IndexedSlices, the body may accept and
return TensorArray objects.  The flows of the TensorArray objects will
be appropriately forwarded between loops and during gradient calculations.

While `cond` evaluates to true, `body` is executed.

##### Args:


*  <b>`cond`</b>: The termination condition of the loop.
*  <b>`body`</b>: A callable that represents the loop body.
*  <b>`loop_vars`</b>: The list of variable input tensors.
*  <b>`parallel_iterations`</b>: The number of iterations allowed to run in parallel.
*  <b>`back_prop`</b>: Whether backprop is enabled for this while loop.
*  <b>`swap_memory`</b>: Whether GPU-CPU memory swap is enabled for this loop.
*  <b>`name`</b>: Optional name prefix for the returned tensors.

##### Returns:

  The output tensors for the loop variables after the loop.

##### Raises:


*  <b>`TypeError`</b>: if `cond` or `body` is not callable.
*  <b>`ValueError`</b>: if `loop_var` is empty.


*  <b>`Example`</b>: 

  ```python
  i = tf.constant(0)
  c = lambda i: tf.less(i, 10)
  b = lambda i: tf.add(i, 1)
  r = tf.while_loop(c, b, [i])
  ```

