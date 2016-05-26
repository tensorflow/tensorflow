A queue implementation that dequeues elements in a random order.

See [`tf.QueueBase`](#QueueBase) for a description of the methods on
this class.

- - -

#### `tf.RandomShuffleQueue.__init__(capacity, min_after_dequeue, dtypes, shapes=None, names=None, seed=None, shared_name=None, name='random_shuffle_queue')` {#RandomShuffleQueue.__init__}

Create a queue that dequeues elements in a random order.

A `RandomShuffleQueue` has bounded capacity; supports multiple
concurrent producers and consumers; and provides exactly-once
delivery.

A `RandomShuffleQueue` holds a list of up to `capacity`
elements. Each element is a fixed-length tuple of tensors whose
dtypes are described by `dtypes`, and whose shapes are optionally
described by the `shapes` argument.

If the `shapes` argument is specified, each component of a queue
element must have the respective fixed shape. If it is
unspecified, different queue elements may have different shapes,
but the use of `dequeue_many` is disallowed.

The `min_after_dequeue` argument allows the caller to specify a
minimum number of elements that will remain in the queue after a
`dequeue` or `dequeue_many` operation completes, to ensure a
minimum level of mixing of elements. This invariant is maintained
by blocking those operations until sufficient elements have been
enqueued. The `min_after_dequeue` argument is ignored after the
queue has been closed.

##### Args:


*  <b>`capacity`</b>: An integer. The upper bound on the number of elements
    that may be stored in this queue.
*  <b>`min_after_dequeue`</b>: An integer (described above).
*  <b>`dtypes`</b>: A list of `DType` objects. The length of `dtypes` must equal
    the number of tensors in each queue element.
*  <b>`shapes`</b>: (Optional.) A list of fully-defined `TensorShape` objects
    with the same length as `dtypes`, or `None`.
*  <b>`names`</b>: (Optional.) A list of string naming the components in the queue
    with the same length as `dtypes`, or `None`.  If specified the dequeue
    methods return a dictionary with the names as keys.
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`shared_name`</b>: (Optional.) If non-empty, this queue will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the queue operation.


