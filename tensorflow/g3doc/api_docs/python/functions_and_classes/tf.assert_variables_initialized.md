### `tf.assert_variables_initialized(var_list=None)` {#assert_variables_initialized}

Returns an Op to check if variables are initialized.

NOTE: This function is obsolete and will be removed in 6 months.  Please
change your implementation to use `report_uninitialized_variables()`.

When run, the returned Op will raise the exception `FailedPreconditionError`
if any of the variables has not yet been initialized.

Note: This function is implemented by trying to fetch the values of the
variables. If one of the variables is not initialized a message may be
logged by the C++ runtime. This is expected.

##### Args:


*  <b>`var_list`</b>: List of `Variable` objects to check. Defaults to the
    value of `all_variables().`

##### Returns:

  An Op, or None if there are no variables.

