### `tf.train.maybe_shuffle_batch(tensors, batch_size, capacity, min_after_dequeue, keep_input, num_threads=1, seed=None, enqueue_many=False, shapes=None, allow_smaller_final_batch=False, shared_name=None, name=None)` {#maybe_shuffle_batch}

Creates batches by randomly shuffling conditionally-enqueued tensors.

See docstring in `shuffle_batch` for more details.

##### Args:


*  <b>`tensors`</b>: The list or dictionary of tensors to enqueue.
*  <b>`batch_size`</b>: The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`min_after_dequeue`</b>: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
*  <b>`keep_input`</b>: A `bool` scalar Tensor.  This tensor controls whether the input
    is added to the queue or not.  If it evaluates `True`, then `tensors` are
    added to the queue; otherwise they are dropped.  This tensor essentially
    acts as a filtering mechanism.
*  <b>`num_threads`</b>: The number of threads enqueuing `tensor_list`.
*  <b>`seed`</b>: Seed for the random shuffling within the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list` is a single example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list`.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the types as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.

