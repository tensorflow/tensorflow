### `tf.contrib.framework.init_from_checkpoint(checkpoint_dir, assignment_map)` {#init_from_checkpoint}

Using assingment map initializes current variables with loaded tensors.

Note: This overrides default initialization ops of specified variables and
redefines dtype.

##### Assignment map supports following syntax:

  `'checkpoint_scope_name/': 'scope_name/'` - will load all variables in
    current `scope_name` from `checkpoint_scope_name` with matching variable
    names.
  `'checkpoint_scope_name/some_other_variable': 'scope_name/variable_name'` -
    will initalize `scope_name/variable_name` variable
    from `checkpoint_scope_name/some_other_variable`.
  `'scope_variable_name': variable` - will initialize given `tf.Variable`
    object with variable from the checkpoint.
  `'scope_variable_name': list(variable)` - will initialize list of
    partitioned variables with variable from the checkpoint.
  `'/': 'scope_name/'` - will load all variables in current `scope_name` from
    checkpoint's root (e.g. no scope).

Supports loading into partitioned variables, which are represented as
'<variable>/part_<part #>'.


*  <b>`Example`</b>: 
```python
  # Create variables.
  with tf.variable_scope('test'):
    m = tf.get_variable('my_var')
  with tf.variable_scope('test2'):
    var2 = tf.get_variable('my_var')
  var3 = tf.get_variable(name="my1", shape=[100, 100],
                         partitioner=lambda shape, dtype: [5, 1])
  ...
  # Specify which variables to intialize from checkpoint.
  init_from_checkpoint(checkpoint_dir, {
    'some_var': 'test/my_var',
    'some_scope/': 'test2/'})
  ...
  # Or use `Variable` objects to identify what to initialize.
  init_from_checkpoint(checkpoint_dir, {
    'some_scope/var2': var2,
  })
  # Initialize partitioned variables
  init_from_checkpoint(checkpoint_dir, {
    'some_var_from_ckpt': 'part_var',
  })
  # Or specifying the list of `Variable` objects.
  init_from_checkpoint(checkpoint_dir, {
    'some_var_from_ckpt': var3._get_variable_list(),
  })
  ...
  # Initialize variables as usual.
  session.run(tf.get_all_variables())
```

##### Args:


*  <b>`checkpoint_dir`</b>: Directory with checkpoints file or path to checkpoint.
*  <b>`assignment_map`</b>: Dict, where keys are names of the variables in the
    checkpoint and values are current variables or names of current variables
    (in default graph).

##### Raises:

  tf.errors.OpError: If missing checkpoints or tensors in checkpoints.

*  <b>`ValueError`</b>: If missing variables in current graph.

