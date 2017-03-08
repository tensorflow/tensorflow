### `tf.python_io.tf_record_iterator(path, options=None)` {#tf_record_iterator}

An iterator that read the records from a TFRecords file.

##### Args:


*  <b>`path`</b>: The path to the TFRecords file.
*  <b>`options`</b>: (optional) A TFRecordOptions object.

##### Yields:

  Strings.

##### Raises:


*  <b>`IOError`</b>: If `path` cannot be opened for reading.

