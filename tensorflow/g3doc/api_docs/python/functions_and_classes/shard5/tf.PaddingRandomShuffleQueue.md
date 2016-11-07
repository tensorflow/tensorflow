A RandomQueue that supports batching variable-sized tensors by padding.

A `PaddingRandomShuffleQueue` may contain components with dynamic shape, while also
supporting `dequeue_many`.  See the constructor for more details.

See [`tf.QueueBase`](#QueueBase) for a description of the methods on
this class.

- - -

#### `tf.PaddingRandomShuffleQueue.__init__(capacity, min_after_dequeue, dtypes, shapes, names=None, seed=None, shared_name=None, name='random_shuffle_queue')` {#PaddingRandomShuffleQueue.__init__}

Create a queue that dequeues elements in a random order.

A `PaddingRandomShuffleQueue` has bounded capacity; supports multiple
concurrent producers and consumers; and provides exactly-once
delivery.

A `PaddingRandomShuffleQueue` holds a list of up to `capacity`
elements. Each element is a fixed-length tuple of tensors whose
dtypes are described by `dtypes`, and whose shapes are
described by the `shapes` argument.

The `shapes` argument must be specified; each component of a queue
element must have the respective shape.  Shapes of fixed
rank but variable size are allowed by setting any shape dimension to None.
In this case, the inputs' shape may vary along the given dimension, and
`dequeue_many` will pad the given dimension with zeros up to the maximum
shape of all elements in the given batch.

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
*  <b>`shapes`</b>: A list of `TensorShape` objects, with the same length as
    `dtypes`.  Any dimension in the `TensorShape` containing value
    `None` is dynamic and allows values to be enqueued with
     variable size in that dimension.
*  <b>`names`</b>: (Optional.) A list of string naming the components in the queue
    with the same length as `dtypes`, or `None`.  If specified the dequeue
    methods return a dictionary with the names as keys.
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`shared_name`</b>: (Optional.) If non-empty, this queue will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the queue operation.

##### Raises:

*  <b>`ValueError`</b>: If shapes is not a list of shapes, or the lengths of dtypes and shapes do not match, or if names is specified and the lengths of dtypes and names do not match.
