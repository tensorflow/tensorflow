A queue implementation that dequeues elements in prioritized order.

See [`tf.QueueBase`](#QueueBase) for a description of the methods on
this class.

- - -

#### `tf.PriorityQueue.__init__(capacity, types, shapes=None, names=None, shared_name=None, name='priority_queue')` {#PriorityQueue.__init__}

Creates a queue that dequeues elements in a first-in first-out order.

A `PriorityQueue` has bounded capacity; supports multiple concurrent
producers and consumers; and provides exactly-once delivery.

A `PriorityQueue` holds a list of up to `capacity` elements. Each
element is a fixed-length tuple of tensors whose dtypes are
described by `types`, and whose shapes are optionally described
by the `shapes` argument.

If the `shapes` argument is specified, each component of a queue
element must have the respective fixed shape. If it is
unspecified, different queue elements may have different shapes,
but the use of `dequeue_many` is disallowed.

Enqueues and Dequeues to the `PriorityQueue` must include an additional
tuple entry at the beginning: the `priority`.  The priority must be
an int64 scalar (for `enqueue`) or an int64 vector (for `enqueue_many`).

##### Args:


*  <b>`capacity`</b>: An integer. The upper bound on the number of elements
    that may be stored in this queue.
*  <b>`types`</b>: A list of `DType` objects. The length of `types` must equal
    the number of tensors in each queue element, except the first priority
    element.  The first tensor in each element is the priority,
    which must be type int64.
*  <b>`shapes`</b>: (Optional.) A list of fully-defined `TensorShape` objects,
    with the same length as `types`, or `None`.
*  <b>`names`</b>: (Optional.) A list of strings naming the components in the queue
    with the same length as `dtypes`, or `None`.  If specified, the dequeue
    methods return a dictionary with the names as keys.
*  <b>`shared_name`</b>: (Optional.) If non-empty, this queue will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the queue operation.


