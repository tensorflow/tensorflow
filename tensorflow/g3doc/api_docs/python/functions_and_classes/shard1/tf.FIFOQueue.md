A queue implementation that dequeues elements in first-in first-out order.

See [`tf.QueueBase`](#QueueBase) for a description of the methods on
this class.

- - -

#### `tf.FIFOQueue.__init__(capacity, dtypes, shapes=None, names=None, shared_name=None, name='fifo_queue')` {#FIFOQueue.__init__}

Creates a queue that dequeues elements in a first-in first-out order.

A `FIFOQueue` has bounded capacity; supports multiple concurrent
producers and consumers; and provides exactly-once delivery.

A `FIFOQueue` holds a list of up to `capacity` elements. Each
element is a fixed-length tuple of tensors whose dtypes are
described by `dtypes`, and whose shapes are optionally described
by the `shapes` argument.

If the `shapes` argument is specified, each component of a queue
element must have the respective fixed shape. If it is
unspecified, different queue elements may have different shapes,
but the use of `dequeue_many` is disallowed.

##### Args:


*  <b>`capacity`</b>: An integer. The upper bound on the number of elements
    that may be stored in this queue.
*  <b>`dtypes`</b>: A list of `DType` objects. The length of `dtypes` must equal
    the number of tensors in each queue element.
*  <b>`shapes`</b>: (Optional.) A list of fully-defined `TensorShape` objects
    with the same length as `dtypes`, or `None`.
*  <b>`names`</b>: (Optional.) A list of string naming the components in the queue
    with the same length as `dtypes`, or `None`.  If specified the dequeue
    methods return a dictionary with the names as keys.
*  <b>`shared_name`</b>: (Optional.) If non-empty, this queue will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the queue operation.


