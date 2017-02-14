A class to write records to a TFRecords file.

This class implements `__enter__` and `__exit__`, and can be used
in `with` blocks like a normal file.
- - -

#### `tf.python_io.TFRecordWriter.__enter__()` {#TFRecordWriter.__enter__}

Enter a `with` block.


- - -

#### `tf.python_io.TFRecordWriter.__exit__(unused_type, unused_value, unused_traceback)` {#TFRecordWriter.__exit__}

Exit a `with` block, closing the file.


- - -

#### `tf.python_io.TFRecordWriter.__init__(path, options=None)` {#TFRecordWriter.__init__}

Opens file `path` and creates a `TFRecordWriter` writing to it.

##### Args:


*  <b>`path`</b>: The path to the TFRecords file.
*  <b>`options`</b>: (optional) A TFRecordOptions object.

##### Raises:


*  <b>`IOError`</b>: If `path` cannot be opened for writing.


- - -

#### `tf.python_io.TFRecordWriter.close()` {#TFRecordWriter.close}

Close the file.


- - -

#### `tf.python_io.TFRecordWriter.write(record)` {#TFRecordWriter.write}

Write a string record to the file.

##### Args:


*  <b>`record`</b>: str


