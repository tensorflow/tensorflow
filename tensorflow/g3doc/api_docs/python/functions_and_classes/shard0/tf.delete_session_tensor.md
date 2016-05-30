### `tf.delete_session_tensor(name=None)` {#delete_session_tensor}

Delete the tensor by feeding a tensor handle.

This is EXPERIMENTAL and subject to change.

Delete the tensor of a given tensor handle. The tensor is produced
in a previous run() and stored in the state of the session.

##### Args:


*  <b>`name`</b>: Optional name prefix for the return tensor.

##### Returns:

  A pair of graph elements. The first is a placeholder for feeding a
  tensor handle and the second is a deletion operation.

