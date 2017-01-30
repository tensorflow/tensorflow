### `tf.train.maybe_batch_join(tensors_list, keep_input, batch_size, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)` {#maybe_batch_join}

Runs a list of tensors to conditionally fill a queue to create batches.

See docstring in `batch_join` for more details.

##### Args:


*  <b>`tensors_list`</b>: A list of tuples or dictionaries of tensors to enqueue.
*  <b>`keep_input`</b>: A `bool` Tensor.  This tensor controls whether the input is
    added to the queue or not.  If it is a scalar and evaluates `True`, then
    `tensors` are all added to the queue. If it is a vector and `enqueue_many`
    is `True`, then each example is added to the queue only if the
    corresonding value in `keep_input` is `True`. This tensor essentially acts
    as a filtering mechanism.
*  <b>`batch_size`</b>: An integer. The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list_list` is a single
    example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list_list[i]`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensor_list_list`.

