#### `tf.PaddingFIFOQueue.from_list(index, queues)` {#PaddingFIFOQueue.from_list}

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

