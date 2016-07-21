### `tf.contrib.learn.read_batch_features(file_pattern, batch_size, features, reader, randomize_input=True, num_epochs=None, queue_capacity=10000, reader_num_threads=1, parser_num_threads=1, name=None)` {#read_batch_features}

Adds operations to read, queue, batch and parse `Example` protos.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
*  <b>`reader`</b>: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`num_epochs`</b>: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.initialize_local_variables() as shown in the tests.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`reader_num_threads`</b>: The number of threads to read examples.
*  <b>`parser_num_threads`</b>: The number of threads to parse examples.
    records to read at once
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  A dict of `Tensor` or `SparseTensor` objects for each in `features`.
  If `keep_keys` is `True`, returns tuple of string `Tensor` and above dict.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.

