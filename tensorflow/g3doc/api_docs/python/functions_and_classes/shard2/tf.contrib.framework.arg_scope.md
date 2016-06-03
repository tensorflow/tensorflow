### `tf.contrib.framework.arg_scope(list_ops_or_scope, **kwargs)` {#arg_scope}

Stores the default arguments for the given set of list_ops.

For usage, please see examples at top of the file.

##### Args:


*  <b>`list_ops_or_scope`</b>: List or tuple of operations to set argument scope for or
    a dictionary containg the current scope. When list_ops_or_scope is a dict,
    kwargs must be empty. When list_ops_or_scope is a list or tuple, then
    every op in it need to be decorated with @add_arg_scope to work.
*  <b>`**kwargs`</b>: keyword=value that will define the defaults for each op in
            list_ops. All the ops need to accept the given set of arguments.

##### Yields:

  the current_scope, which is a dictionary of {op: {arg: value}}

##### Raises:


*  <b>`TypeError`</b>: if list_ops is not a list or a tuple.
*  <b>`ValueError`</b>: if any op in list_ops has not be decorated with @add_arg_scope.

