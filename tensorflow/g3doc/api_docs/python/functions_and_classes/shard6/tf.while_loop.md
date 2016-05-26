### `tf.while_loop(cond, body, loop_vars, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)` {#while_loop}

Repeat `body` while the condition `cond` is true.

`cond` is a callable returning a boolean scalar tensor. `body` is a callable
returning a list of tensors of the same length and with the same types as
`loop_vars`. `loop_vars` is a list of tensors that is passed to both `cond`
and `body`. `cond` and `body` both take as many arguments as there are
`loop_vars`.

In addition to regular Tensors or IndexedSlices, the body may accept and
return TensorArray objects.  The flows of the TensorArray objects will
be appropriately forwarded between loops and during gradient calculations.

While `cond` evaluates to true, `body` is executed.

`while_loop` implements non-strict semantics, enabling multiple iterations
to run in parallel. The maximum number of parallel iterations can be
controlled by `parallel_iterations`, which gives users some control over
memory consumption and execution order. For correct programs, `while_loop`
should return the same result for any parallel_iterations > 0.

For training, TensorFlow remembers the tensors that are produced in the
forward inference but needed in back propagation. These tensors can be a
main source of memory consumption and often cause OOM problems when training
on GPUs.  When the flag swap_memory is true, we swap out these tensors from
GPU to CPU.  This for example allows us to train RNN models with very long
sequences and large batches.

##### Args:


*  <b>`cond`</b>: A callable that represents the termination condition of the loop.
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

