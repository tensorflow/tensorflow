### `tf.contrib.graph_editor.select_ops(*args, **kwargs)` {#select_ops}

Helper to select operations.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    tf.Operation. tf.Tensor instances are silently ignored.
*  <b>`**kwargs`</b>: 'graph': tf.Graph in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if positive_filter(elem) is
      True. This is optional.
    'restrict_ops_regex': a regular expression is ignored if it doesn't start
      with the substring "(?#ops)".

##### Returns:

  list of tf.Operation

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a tf.Graph
    or if an argument in args is not an (array of) tf.Operation
    or an (array of) tf.Tensor (silently ignored) or a string
    or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.

