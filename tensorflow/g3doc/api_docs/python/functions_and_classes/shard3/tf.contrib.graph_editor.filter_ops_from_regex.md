### `tf.contrib.graph_editor.filter_ops_from_regex(ops, regex)` {#filter_ops_from_regex}

Get all the operations that match the given regex.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of `tf.Operation`.
*  <b>`regex`</b>: a regular expression matching the operation's name.
    For example, `"^foo(/.*)?$"` will match all the operations in the "foo"
    scope.

##### Returns:

  A list of `tf.Operation`.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of `tf.Operation`.

