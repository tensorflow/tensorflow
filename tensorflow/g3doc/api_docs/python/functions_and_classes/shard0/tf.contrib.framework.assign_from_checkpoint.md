### `tf.contrib.framework.assign_from_checkpoint(model_path, var_list)` {#assign_from_checkpoint}

Creates an operation to assign specific variables from a checkpoint.

##### Args:


*  <b>`model_path`</b>: The full path to the model checkpoint. To get latest checkpoint
      use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
*  <b>`var_list`</b>: A list of `Variable` objects or a dictionary mapping names in the
      checkpoint to the correspoing variables to initialize. If empty or None,
      it would return  no_op(), None.

##### Returns:

  the restore_op and the feed_dict that need to be run to restore var_list.

##### Raises:


*  <b>`ValueError`</b>: If the checkpoint specified at `model_path` is missing one of
    the variables in `var_list`.

