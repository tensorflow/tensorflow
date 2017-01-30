### `tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None, cancel_op=None)` {#string_input_producer}

Output strings (e.g. filenames) to a queue for an input pipeline.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variables_initializer()` to initialize local variables.

##### Args:


*  <b>`string_tensor`</b>: A 1-D string tensor with the strings to produce.
*  <b>`num_epochs`</b>: An integer (optional). If specified, `string_input_producer`
    produces each string from `string_tensor` `num_epochs` times before
    generating an `OutOfRange` error. If not specified,
    `string_input_producer` can cycle through the strings in `string_tensor`
    an unlimited number of times.
*  <b>`shuffle`</b>: Boolean. If true, the strings are randomly shuffled within each
    epoch.
*  <b>`seed`</b>: An integer (optional). Seed used if shuffle == True.
*  <b>`capacity`</b>: An integer. Sets the queue capacity.
*  <b>`shared_name`</b>: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: A name for the operations (optional).
*  <b>`cancel_op`</b>: Cancel op for the queue (optional).

##### Returns:

  A queue with the output strings.  A `QueueRunner` for the Queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

##### Raises:


*  <b>`ValueError`</b>: If the string_tensor is a null Python list.  At runtime,
  will fail with an assertion if string_tensor becomes a null tensor.

