### `tf.contrib.graph_editor.swap_ts(ts0, ts1, can_modify=None, cannot_modify=None)` {#swap_ts}

For each tensor's pair, swap the end of (t0,t1).

B0 B1     B0 B1
|  |    =>  X
A0 A1     A0 A1

##### Args:


*  <b>`ts0`</b>: an object convertible to a list of tf.Tensor.
*  <b>`ts1`</b>: an object convertible to a list of tf.Tensor.
*  <b>`can_modify`</b>: iterable of operations which can be modified. Any operation
    outside within_ops will be left untouched by this function.
*  <b>`cannot_modify`</b>: iterable of operations which cannot be modified.
    Any operation within cannot_modify will be left untouched by this
    function.

##### Returns:

  The number of individual modifications made by the function.

##### Raises:


*  <b>`TypeError`</b>: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
*  <b>`TypeError`</b>: if can_modify or cannot_modify is not None and cannot be
    converted to a list of tf.Operation.

