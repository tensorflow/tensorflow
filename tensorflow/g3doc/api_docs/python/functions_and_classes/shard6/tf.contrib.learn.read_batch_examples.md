### `tf.contrib.learn.read_batch_examples(file_pattern, batch_size, reader, randomize_input=True, num_epochs=None, queue_capacity=10000, num_threads=1, read_batch_size=1, parse_fn=None, name=None)` {#read_batch_examples}

Adds operations to read, queue, batch `Example` protos.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size`.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Use `parse_fn` if you need to do parsing / processing on single examples.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`reader`</b>: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`num_epochs`</b>: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.initialize_all_variables()` as shown in the tests.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`num_threads`</b>: The number of threads enqueuing examples.
*  <b>`read_batch_size`</b>: An int or scalar `Tensor` specifying the number of
    records to read at once
*  <b>`parse_fn`</b>: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  String `Tensor` of batched `Example` proto. If `keep_keys` is True, then
  returns tuple of string `Tensor`s, where first value is the key.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.

