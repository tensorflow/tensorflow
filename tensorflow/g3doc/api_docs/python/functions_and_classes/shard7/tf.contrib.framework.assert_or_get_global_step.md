### `tf.contrib.framework.assert_or_get_global_step(graph=None, global_step_tensor=None)` {#assert_or_get_global_step}

Verifies that a global step tensor is valid or gets one if None is given.

If `global_step_tensor` is not None, check that it is a valid global step
tensor (using `assert_global_step`). Otherwise find a global step tensor using
`get_global_step` and return it.

##### Args:


*  <b>`graph`</b>: The graph to find the global step tensor for.
*  <b>`global_step_tensor`</b>: The tensor to check for suitability as a global step.
    If None is given (the default), find a global step tensor.

##### Returns:

  A tensor suitable as a global step, or `None` if none was provided and none
  was found.

