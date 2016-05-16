### `tf.contrib.learn.read_batch_record_features(file_pattern, batch_size, features, randomize_input=True, queue_capacity=10000, num_threads=1, name='dequeue_record_examples')` {#read_batch_record_features}

Reads TFRecord, queues, batches and parses `Example` proto.

See more detailed description in `read_examples`.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`num_threads`</b>: The number of threads enqueuing examples.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.

