### `tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)` {#assign}

Update 'ref' by assigning 'value' to it.

This operation outputs "ref" after the assignment is done.
This makes it easier to chain operations that need to use the reset value.

##### Args:


*  <b>`ref`</b>: A mutable `Tensor`.
    Should be from a `Variable` node. May be uninitialized.
*  <b>`value`</b>: A `Tensor`. Must have the same type as `ref`.
    The value to be assigned to the variable.
*  <b>`validate_shape`</b>: An optional `bool`. Defaults to `True`.
    If true, the operation will validate that the shape
    of 'value' matches the shape of the Tensor being assigned to.  If false,
    'ref' will take on the shape of 'value'.
*  <b>`use_locking`</b>: An optional `bool`. Defaults to `True`.
    If True, the assignment will be protected by a lock;
    otherwise the behavior is undefined, but may exhibit less contention.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same as "ref".  Returned as a convenience for operations that want
  to use the new value after the variable has been reset.

