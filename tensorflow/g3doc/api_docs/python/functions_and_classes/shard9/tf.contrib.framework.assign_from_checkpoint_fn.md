### `tf.contrib.framework.assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False, reshape_variables=False)` {#assign_from_checkpoint_fn}

Returns a function that assigns specific variables from a checkpoint.

##### Args:


*  <b>`model_path`</b>: The full path to the model checkpoint. To get latest checkpoint
      use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
*  <b>`var_list`</b>: A list of `Variable` objects or a dictionary mapping names in the
      checkpoint to the correspoing variables to initialize. If empty or None,
      it would return  no_op(), None.
*  <b>`ignore_missing_vars`</b>: Boolean, if True it would ignore variables missing in
      the checkpoint with a warning instead of failing.
*  <b>`reshape_variables`</b>: Boolean, if True it would automatically reshape variables
      which are of different shape then the ones stored in the checkpoint but
      which have the same number of elements.

##### Returns:

  A function that takes a single argument, a `tf.Session`, that applies the
  assignment operation.

##### Raises:


*  <b>`ValueError`</b>: If the checkpoint specified at `model_path` is missing one of
    the variables in `var_list`.

