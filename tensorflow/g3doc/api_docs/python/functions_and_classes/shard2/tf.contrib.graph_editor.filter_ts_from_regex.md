### `tf.contrib.graph_editor.filter_ts_from_regex(ops, regex)` {#filter_ts_from_regex}

Get all the tensors linked to ops that match the given regex.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of tf.Operation.
*  <b>`regex`</b>: a regular expression matching the tensors' name.
    For example, "^foo(/.*)?:\d+$" will match all the tensors in the "foo"
    scope.

##### Returns:

  A list of tf.Tensor.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of tf.Operation.

