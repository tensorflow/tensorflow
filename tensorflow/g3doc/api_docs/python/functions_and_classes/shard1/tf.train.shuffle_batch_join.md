### `tf.train.shuffle_batch_join(tensors_list, batch_size, capacity, min_after_dequeue, seed=None, enqueue_many=False, shapes=None, shared_name=None, name=None)` {#shuffle_batch_join}

Create batches by randomly shuffling tensors.

The `tensors_list` argument is a list of tuples of tensors, or a list of
dictionaries of tensors.  Each element in the list is treated similarily
to the `tensors` argument of `tf.train.shuffle_batch()`.

This version enqueues a different list of tensors in different threads.
It adds the following to the current `Graph`:

* A shuffling queue into which tensors from `tensors_list` are enqueued.
* A `dequeue_many` operation to create batches from the queue.
* A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
  from `tensors_list`.

`len(tensors_list)` threads will be started, with thread `i` enqueuing
the tensors from `tensors_list[i]`. `tensors_list[i1][j]` must match
`tensors_list[i2][j]` in type and shape, except in the first dimension if
`enqueue_many` is true.

If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
to represent a single example.  An input tensor with shape `[x, y, z]`
will be output as a tensor with shape `[batch_size, x, y, z]`.

If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
represent a batch of examples, where the first dimension is indexed
by example, and all members of `tensors_list[i]` should have the
same size in the first dimension.  If an input tensor has shape `[*, x,
y, z]`, the output will have shape `[batch_size, x, y, z]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

##### Args:


*  <b>`tensors_list`</b>: A list of tuples or dictionaries of tensors to enqueue.
*  <b>`batch_size`</b>: An integer. The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`min_after_dequeue`</b>: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
*  <b>`seed`</b>: Seed for the random shuffling within the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list_list` is a single
    example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors_list[i]`.
*  <b>`shared_name`</b>: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors_list`.

