<!-- This file is machine generated: DO NOT EDIT! -->

# Data IO (Python functions)
[TOC]

Python functions for directly manipulating TFRecord-formatted files.

See the @{$python/python_io} guide.

- - -

### `class tf.python_io.TFRecordWriter` {#TFRecordWriter}

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



- - -

### `tf.python_io.tf_record_iterator(path, options=None)` {#tf_record_iterator}

An iterator that read the records from a TFRecords file.

##### Args:


*  <b>`path`</b>: The path to the TFRecords file.
*  <b>`options`</b>: (optional) A TFRecordOptions object.

##### Yields:

  Strings.

##### Raises:


*  <b>`IOError`</b>: If `path` cannot be opened for reading.


- - -

### `class tf.python_io.TFRecordCompressionType` {#TFRecordCompressionType}

The type of compression for the record.

- - -

### `class tf.python_io.TFRecordOptions` {#TFRecordOptions}

Options used for manipulating TFRecord files.
- - -

#### `tf.python_io.TFRecordOptions.__init__(compression_type)` {#TFRecordOptions.__init__}




- - -

#### `tf.python_io.TFRecordOptions.get_compression_type_string(cls, options)` {#TFRecordOptions.get_compression_type_string}





