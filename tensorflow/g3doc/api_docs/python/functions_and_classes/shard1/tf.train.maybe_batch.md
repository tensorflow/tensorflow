### `tf.train.maybe_batch(tensors, keep_input, batch_size, num_threads=1, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)` {#maybe_batch}

Conditionally creates batches of tensors based on `keep_input`.

See docstring in `batch` for more details.

##### Args:


*  <b>`tensors`</b>: The list or dictionary of tensors to enqueue.
*  <b>`keep_input`</b>: A `bool` scalar Tensor.  This tensor controls whether the input
    is added to the queue or not.  If it evaluates `True`, then `tensors` are
    added to the queue; otherwise they are dropped.  This tensor essentially
    acts as a filtering mechanism.
*  <b>`batch_size`</b>: The new batch size pulled from the queue.
*  <b>`num_threads`</b>: The number of threads enqueuing `tensors`.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensors` is a single example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same types as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.

