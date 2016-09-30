### `tf.contrib.framework.variable(*args, **kwargs)` {#variable}

Gets an existing variable with these parameters or creates a new one.

##### Args:


*  <b>`name`</b>: the name of the new or existing variable.
*  <b>`shape`</b>: shape of the new or existing variable.
*  <b>`dtype`</b>: type of the new or existing variable (defaults to `DT_FLOAT`).
*  <b>`initializer`</b>: initializer for the variable if one is created.
*  <b>`regularizer`</b>: a (Tensor -> Tensor or None) function; the result of
      applying it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
*  <b>`trainable`</b>: If `True` also add the variable to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
*  <b>`collections`</b>: A list of collection names to which the Variable will be added.
    If None it would default to `tf.GraphKeys.VARIABLES`.
*  <b>`caching_device`</b>: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's
      device.
*  <b>`device`</b>: Optional device to place the variable. It can be an string or a
    function that is called to get the device for the variable.

##### Returns:

  The created or existing variable.

