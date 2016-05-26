### `tf.train.slice_input_producer(tensor_list, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None)` {#slice_input_producer}

Produces a slice of each `Tensor` in `tensor_list`.

Implemented using a Queue -- a `QueueRunner` for the Queue
is added to the current `Graph`'s `QUEUE_RUNNER` collection.

##### Args:


*  <b>`tensor_list`</b>: A list of `Tensor` objects. Every `Tensor` in
    `tensor_list` must have the same size in the first dimension.
*  <b>`num_epochs`</b>: An integer (optional). If specified, `slice_input_producer`
    produces each slice `num_epochs` times before generating
    an `OutOfRange` error. If not specified, `slice_input_producer` can cycle
    through the slices an unlimited number of times.
*  <b>`shuffle`</b>: Boolean. If true, the integers are randomly shuffled within each
    epoch.
*  <b>`seed`</b>: An integer (optional). Seed used if shuffle == True.
*  <b>`capacity`</b>: An integer. Sets the queue capacity.
*  <b>`shared_name`</b>: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: A name for the operations (optional).

##### Returns:

  A list of tensors, one for each element of `tensor_list`.  If the tensor
  in `tensor_list` has shape `[N, a, b, .., z]`, then the corresponding output
  tensor will have shape `[a, b, ..., z]`.

##### Raises:


*  <b>`ValueError`</b>: if `slice_input_producer` produces nothing from `tensor_list`.

