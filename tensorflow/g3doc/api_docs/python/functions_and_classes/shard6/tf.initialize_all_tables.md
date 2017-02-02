### `tf.initialize_all_tables(*args, **kwargs)` {#initialize_all_tables}

Returns an Op that initializes all tables of the default graph. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-02.
Instructions for updating:
Use `tf.tables_initializer` instead.

##### Args:


*  <b>`name`</b>: Optional name for the initialization op.

##### Returns:

  An Op that initializes all tables.  Note that if there are
  not tables the returned Op is a NoOp.

