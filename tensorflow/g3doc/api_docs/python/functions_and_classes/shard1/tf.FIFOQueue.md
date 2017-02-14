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


- - -

#### `tf.FIFOQueue.close(cancel_pending_enqueues=False, name=None)` {#FIFOQueue.close}

Closes this queue.

This operation signals that no more elements will be enqueued in
the given queue. Subsequent `enqueue` and `enqueue_many`
operations will fail. Subsequent `dequeue` and `dequeue_many`
operations will continue to succeed if sufficient elements remain
in the queue. Subsequent `dequeue` and `dequeue_many` operations
that would block will fail immediately.

If `cancel_pending_enqueues` is `True`, all pending requests will also
be cancelled.

##### Args:


*  <b>`cancel_pending_enqueues`</b>: (Optional.) A boolean, defaulting to
    `False` (described above).
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that closes the queue.


- - -

#### `tf.FIFOQueue.dequeue(name=None)` {#FIFOQueue.dequeue}

Dequeues one element from this queue.

If the queue is empty when this operation executes, it will block
until there is an element to dequeue.

At runtime, this operation may raise an error if the queue is
[closed](#QueueBase.close) before or during its execution. If the
queue is closed, the queue is empty, and there are no pending
enqueue operations that can fulfill this request,
`tf.errors.OutOfRangeError` will be raised. If the session is
[closed](../../api_docs/python/client.md#Session.close),
`tf.errors.CancelledError` will be raised.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of tensors that was dequeued.


- - -

#### `tf.FIFOQueue.dequeue_many(n, name=None)` {#FIFOQueue.dequeue_many}

Dequeues and concatenates `n` elements from this queue.

This operation concatenates queue-element component tensors along
the 0th dimension to make a single component tensor.  All of the
components in the dequeued tuple will have size `n` in the 0th dimension.

If the queue is closed and there are less than `n` elements left, then an
`OutOfRange` exception is raised.

At runtime, this operation may raise an error if the queue is
[closed](#QueueBase.close) before or during its execution. If the
queue is closed, the queue contains fewer than `n` elements, and
there are no pending enqueue operations that can fulfill this
request, `tf.errors.OutOfRangeError` will be raised. If the
session is [closed](../../api_docs/python/client.md#Session.close),
`tf.errors.CancelledError` will be raised.

##### Args:


*  <b>`n`</b>: A scalar `Tensor` containing the number of elements to dequeue.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of concatenated tensors that was dequeued.


- - -

#### `tf.FIFOQueue.dequeue_up_to(n, name=None)` {#FIFOQueue.dequeue_up_to}

Dequeues and concatenates `n` elements from this queue.

**Note** This operation is not supported by all queues.  If a queue does not
support DequeueUpTo, then a `tf.errors.UnimplementedError` is raised.

This operation concatenates queue-element component tensors along
the 0th dimension to make a single component tensor. If the queue
has not been closed, all of the components in the dequeued tuple
will have size `n` in the 0th dimension.

If the queue is closed and there are more than `0` but fewer than
`n` elements remaining, then instead of raising a
`tf.errors.OutOfRangeError` like [`dequeue_many`](#QueueBase.dequeue_many),
less than `n` elements are returned immediately.  If the queue is
closed and there are `0` elements left in the queue, then a
`tf.errors.OutOfRangeError` is raised just like in `dequeue_many`.
Otherwise the behavior is identical to `dequeue_many`.

##### Args:


*  <b>`n`</b>: A scalar `Tensor` containing the number of elements to dequeue.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of concatenated tensors that was dequeued.


- - -

#### `tf.FIFOQueue.dtypes` {#FIFOQueue.dtypes}

The list of dtypes for each component of a queue element.


- - -

#### `tf.FIFOQueue.enqueue(vals, name=None)` {#FIFOQueue.enqueue}

Enqueues one element to this queue.

If the queue is full when this operation executes, it will block
until the element has been enqueued.

At runtime, this operation may raise an error if the queue is
[closed](#QueueBase.close) before or during its execution. If the
queue is closed before this operation runs,
`tf.errors.CancelledError` will be raised. If this operation is
blocked, and either (i) the queue is closed by a close operation
with `cancel_pending_enqueues=True`, or (ii) the session is
[closed](../../api_docs/python/client.md#Session.close),
`tf.errors.CancelledError` will be raised.

##### Args:


*  <b>`vals`</b>: A tensor, a list or tuple of tensors, or a dictionary containing
    the values to enqueue.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that enqueues a new tuple of tensors to the queue.


- - -

#### `tf.FIFOQueue.enqueue_many(vals, name=None)` {#FIFOQueue.enqueue_many}

Enqueues zero or more elements to this queue.

This operation slices each component tensor along the 0th dimension to
make multiple queue elements. All of the tensors in `vals` must have the
same size in the 0th dimension.

If the queue is full when this operation executes, it will block
until all of the elements have been enqueued.

At runtime, this operation may raise an error if the queue is
[closed](#QueueBase.close) before or during its execution. If the
queue is closed before this operation runs,
`tf.errors.CancelledError` will be raised. If this operation is
blocked, and either (i) the queue is closed by a close operation
with `cancel_pending_enqueues=True`, or (ii) the session is
[closed](../../api_docs/python/client.md#Session.close),
`tf.errors.CancelledError` will be raised.

##### Args:


*  <b>`vals`</b>: A tensor, a list or tuple of tensors, or a dictionary
    from which the queue elements are taken.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that enqueues a batch of tuples of tensors to the queue.


- - -

#### `tf.FIFOQueue.from_list(index, queues)` {#FIFOQueue.from_list}

Create a queue using the queue reference from `queues[index]`.

##### Args:


*  <b>`index`</b>: An integer scalar tensor that determines the input that gets
    selected.
*  <b>`queues`</b>: A list of `QueueBase` objects.

##### Returns:

  A `QueueBase` object.

##### Raises:


*  <b>`TypeError`</b>: When `queues` is not a list of `QueueBase` objects,
    or when the data types of `queues` are not all the same.


- - -

#### `tf.FIFOQueue.name` {#FIFOQueue.name}

The name of the underlying queue.


- - -

#### `tf.FIFOQueue.names` {#FIFOQueue.names}

The list of names for each component of a queue element.


- - -

#### `tf.FIFOQueue.queue_ref` {#FIFOQueue.queue_ref}

The underlying queue reference.


- - -

#### `tf.FIFOQueue.shapes` {#FIFOQueue.shapes}

The list of shapes for each component of a queue element.


- - -

#### `tf.FIFOQueue.size(name=None)` {#FIFOQueue.size}

Compute the number of elements in this queue.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar tensor containing the number of elements in this queue.


