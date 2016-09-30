### `tf.contrib.framework.get_variables(scope=None, suffix=None, collection='variables')` {#get_variables}

Gets the list of variables, filtered by scope and/or suffix.

##### Args:


*  <b>`scope`</b>: an optional scope for filtering the variables to return.
*  <b>`suffix`</b>: an optional suffix for filtering the variables to return.
*  <b>`collection`</b>: in which collection search for. Defaults to
    `GraphKeys.VARIABLES`.

##### Returns:

  a list of variables in collection with scope and suffix.

