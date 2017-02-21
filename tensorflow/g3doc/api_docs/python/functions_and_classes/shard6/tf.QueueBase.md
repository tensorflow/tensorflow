Base class for queue implementations.

A queue is a TensorFlow data structure that stores tensors across
multiple steps, and exposes operations that enqueue and dequeue
tensors.

Each queue element is a tuple of one or more tensors, where each
tuple component has a static dtype, and may have a static shape. The
queue implementations support versions of enqueue and dequeue that
handle single elements, versions that support enqueuing and
dequeuing a batch of elements at once.

See [`tf.FIFOQueue`](#FIFOQueue) and
[`tf.RandomShuffleQueue`](#RandomShuffleQueue) for concrete
implementations of this class, and instructions on how to create
them.
- - -

#### `tf.QueueBase.__init__(dtypes, shapes, names, queue_ref)` {#QueueBase.__init__}

Constructs a queue object from a queue reference.

The two optional lists, `shapes` and `names`, must be of the same length
as `dtypes` if provided.  The values at a given index `i` indicate the
shape and name to use for the corresponding queue component in `dtypes`.

##### Args:


*  <b>`dtypes`</b>: A list of types.  The length of dtypes must equal the number
    of tensors in each element.
*  <b>`shapes`</b>: Constraints on the shapes of tensors in an element:
    A list of shape tuples or None. This list is the same length
    as dtypes.  If the shape of any tensors in the element are constrained,
    all must be; shapes can be None if the shapes should not be constrained.
*  <b>`names`</b>: Optional list of names.  If provided, the `enqueue()` and
    `dequeue()` methods will use dictionaries with these names as keys.
    Must be None or a list or tuple of the same length as `dtypes`.
*  <b>`queue_ref`</b>: The queue reference, i.e. the output of the queue op.

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


- - -

#### `tf.QueueBase.close(cancel_pending_enqueues=False, name=None)` {#QueueBase.close}

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

#### `tf.QueueBase.dequeue(name=None)` {#QueueBase.dequeue}

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

#### `tf.QueueBase.dequeue_many(n, name=None)` {#QueueBase.dequeue_many}

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

#### `tf.QueueBase.dequeue_up_to(n, name=None)` {#QueueBase.dequeue_up_to}

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

#### `tf.QueueBase.dtypes` {#QueueBase.dtypes}

The list of dtypes for each component of a queue element.


- - -

#### `tf.QueueBase.enqueue(vals, name=None)` {#QueueBase.enqueue}

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

#### `tf.QueueBase.enqueue_many(vals, name=None)` {#QueueBase.enqueue_many}

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

#### `tf.QueueBase.from_list(index, queues)` {#QueueBase.from_list}

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

#### `tf.QueueBase.name` {#QueueBase.name}

The name of the underlying queue.


- - -

#### `tf.QueueBase.names` {#QueueBase.names}

The list of names for each component of a queue element.


- - -

#### `tf.QueueBase.queue_ref` {#QueueBase.queue_ref}

The underlying queue reference.


- - -

#### `tf.QueueBase.shapes` {#QueueBase.shapes}

The list of shapes for each component of a queue element.


- - -

#### `tf.QueueBase.size(name=None)` {#QueueBase.size}

Compute the number of elements in this queue.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar tensor containing the number of elements in this queue.


