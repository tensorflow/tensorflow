### `tf.contrib.learn.run_feeds(output_dict, feed_dicts, restore_checkpoint_path=None)` {#run_feeds}

Run `output_dict` tensors with each input in `feed_dicts`.

If `checkpoint_path` is supplied, restore from checkpoint. Otherwise, init all
variables.

##### Args:


*  <b>`output_dict`</b>: A `dict` mapping string names to `Tensor` objects to run.
    Tensors must all be from the same graph.
*  <b>`feed_dicts`</b>: Iterable of `dict` objects of input values to feed.
*  <b>`restore_checkpoint_path`</b>: A string containing the path to a checkpoint to
    restore.

##### Returns:

  A list of dicts of values read from `output_dict` tensors, one item in the
  list for each item in `feed_dicts`. Keys are the same as `output_dict`,
  values are the results read from the corresponding `Tensor` in
  `output_dict`.

##### Raises:


*  <b>`ValueError`</b>: if `output_dict` or `feed_dicts` is None or empty.

