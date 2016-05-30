### `tf.train.batch_join(tensors_list, batch_size, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, shared_name=None, name=None)` {#batch_join}

Runs a list of tensors to fill a queue to create batches of examples.

The `tensors_list` argument is a list of tuples of tensors, or a list of
dictionaries of tensors.  Each element in the list is treated similarily
to the `tensors` argument of `tf.train.batch()`.

Enqueues a different list of tensors in different threads.
Implemented using a queue -- a `QueueRunner` for the queue
is added to the current `Graph`'s `QUEUE_RUNNER` collection.

`len(tensors_list)` threads will be started,
with thread `i` enqueuing the tensors from
`tensors_list[i]`. `tensors_list[i1][j]` must match
`tensors_list[i2][j]` in type and shape, except in the first
dimension if `enqueue_many` is true.

If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
to represent a single example. An input tensor `x` will be output as a
tensor with shape `[batch_size] + x.shape`.

If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
represent a batch of examples, where the first dimension is indexed
by example, and all members of `tensors_list[i]` should have the
same size in the first dimension.  The slices of any input tensor
`x` are treated as examples, and the output tensors will have shape
`[batch_size] + x.shape[1:]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

*N.B.:* If `dynamic_pad` is `False`, you must ensure that either
(i) the `shapes` argument is passed, or (ii) all of the tensors in
`tensors_list` must have fully-defined shapes. `ValueError` will be
raised if neither of these conditions holds.

If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
tensors is known, but individual dimensions may have value `None`.
In this case, for each enqueue the dimensions with value `None`
may have a variable length; upon dequeue, the output tensors will be padded
on the right to the maximum shape of the tensors in the current minibatch.
For numbers, this padding takes value 0.  For strings, this padding is
the empty string.  See `PaddingFIFOQueue` for more info.

##### Args:


*  <b>`tensors_list`</b>: A list of tuples or dictionaries of tensors to enqueue.
*  <b>`batch_size`</b>: An integer. The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list_list` is a single
    example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list_list[i]`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`shared_name`</b>: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensor_list_list`.

