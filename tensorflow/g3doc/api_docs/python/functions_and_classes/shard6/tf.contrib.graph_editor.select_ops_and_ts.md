### `tf.contrib.graph_editor.select_ops_and_ts(*args, **kwargs)` {#select_ops_and_ts}

Helper to select operations and tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    tf.Operation 3) (array of) tf.Tensor. Regular expressions matching tensors
    must start with the comment "(?#ts)", for instance: "(?#ts)^foo/.*".
*  <b>`**kwargs`</b>: 'graph': tf.Graph in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if positive_filter(elem) is
      True. This is optional.

##### Returns:

  A tuple `(ops, ts)` where:
    `ops` is a list of tf.Operation
    `ts` is a list of tf.Tensor

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a tf.Graph
    or if an argument in args is not an (array of) tf.Tensor
    or an (array of) tf.Operation or a string or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.

