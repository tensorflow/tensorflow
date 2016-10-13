### `tf.report_uninitialized_variables(var_list=None, name='report_uninitialized_variables')` {#report_uninitialized_variables}

Adds ops to list the names of uninitialized variables.

When run, it returns a 1-D tensor containing the names of uninitialized
variables if there are any, or an empty array if there are none.

##### Args:


*  <b>`var_list`</b>: List of `Variable` objects to check. Defaults to the
    value of `all_variables() + local_variables()`
*  <b>`name`</b>: Optional name of the `Operation`.

##### Returns:

  A 1-D tensor containing names of the uninitialized variables, or an empty
  1-D tensor if there are no variables or no uninitialized variables.

