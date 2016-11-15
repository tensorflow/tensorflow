### `tf.train.input_producer(input_tensor, element_shape=None, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, summary_name=None, name=None, cancel_op=None)` {#input_producer}

Output the rows of `input_tensor` to a queue for an input pipeline.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variable_initializer()` to initialize local variables.

##### Args:


*  <b>`input_tensor`</b>: A tensor with the rows to produce. Must be at least
    one-dimensional. Must either have a fully-defined shape, or
    `element_shape` must be defined.
*  <b>`element_shape`</b>: (Optional.) A `TensorShape` representing the shape of a
    row of `input_tensor`, if it cannot be inferred.
*  <b>`num_epochs`</b>: (Optional.) An integer. If specified `input_producer` produces
    each row of `input_tensor` `num_epochs` times before generating an
    `OutOfRange` error. If not specified, `input_producer` can cycle through
    the rows of `input_tensor` an unlimited number of times.
*  <b>`shuffle`</b>: (Optional.) A boolean. If true, the rows are randomly shuffled
    within each epoch.
*  <b>`seed`</b>: (Optional.) An integer. The seed to use if `shuffle` is true.
*  <b>`capacity`</b>: (Optional.) The capacity of the queue to be used for buffering
    the input.
*  <b>`shared_name`</b>: (Optional.) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`summary_name`</b>: (Optional.) If set, a scalar summary for the current queue
    size will be generated, using this name as part of the tag.
*  <b>`name`</b>: (Optional.) A name for queue.
*  <b>`cancel_op`</b>: (Optional.) Cancel op for the queue

##### Returns:

  A queue with the output rows.  A `QueueRunner` for the queue is
  added to the current `QUEUE_RUNNER` collection of the current
  graph.

##### Raises:


*  <b>`ValueError`</b>: If the shape of the input cannot be inferred from the arguments.

