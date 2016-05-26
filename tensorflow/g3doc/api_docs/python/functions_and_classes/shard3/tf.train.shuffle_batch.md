### `tf.train.shuffle_batch(tensors, batch_size, capacity, min_after_dequeue, num_threads=1, seed=None, enqueue_many=False, shapes=None, shared_name=None, name=None)` {#shuffle_batch}

Creates batches by randomly shuffling tensors.

This function adds the following to the current `Graph`:

* A shuffling queue into which tensors from `tensors` are enqueued.
* A `dequeue_many` operation to create batches from the queue.
* A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
  from `tensors`.

If `enqueue_many` is `False`, `tensors` is assumed to represent a
single example.  An input tensor with shape `[x, y, z]` will be output
as a tensor with shape `[batch_size, x, y, z]`.

If `enqueue_many` is `True`, `tensors` is assumed to represent a
batch of examples, where the first dimension is indexed by example,
and all members of `tensors` should have the same size in the
first dimension.  If an input tensor has shape `[*, x, y, z]`, the
output will have shape `[batch_size, x, y, z]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

For example:

```python
# Creates batches of 32 images and 32 labels.
image_batch, label_batch = tf.train.shuffle_batch(
      [single_image, single_label],
      batch_size=32,
      num_threads=4,
      capacity=50000,
      min_after_dequeue=10000)
```

*N.B.:* You must ensure that either (i) the `shapes` argument is
passed, or (ii) all of the tensors in `tensors` must have
fully-defined shapes. `ValueError` will be raised if neither of
these conditions holds.

##### Args:


*  <b>`tensors`</b>: The list or dictionary of tensors to enqueue.
*  <b>`batch_size`</b>: The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`min_after_dequeue`</b>: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
*  <b>`num_threads`</b>: The number of threads enqueuing `tensor_list`.
*  <b>`seed`</b>: Seed for the random shuffling within the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list` is a single example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list`.
*  <b>`shared_name`</b>: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the types as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.

