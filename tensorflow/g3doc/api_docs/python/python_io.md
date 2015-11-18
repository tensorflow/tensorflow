<!-- This file is machine generated: DO NOT EDIT! -->

# Data IO (Python functions)
[TOC]

## Data IO (Python Functions)

A TFRecords file represents a sequence of (binary) strings.  The format is not
random access, so it is suitable for streaming large amounts of data but not
suitable if fast sharding or other non-sequential access is desired.

- - -

### `class tf.python_io.TFRecordWriter` {#TFRecordWriter}

A class to write records to a TFRecords file.

This class implements `__enter__` and `__exit__`, and can be used
in `with` blocks like a normal file.

- - -

#### `tf.python_io.TFRecordWriter.__init__(path)` {#TFRecordWriter.__init__}

Opens file `path` and creates a `TFRecordWriter` writing to it.

##### Args:


*  <b>`path`</b>: The path to the TFRecords file.

##### Raises:


*  <b>`IOError`</b>: If `path` cannot be opened for writing.


- - -

#### `tf.python_io.TFRecordWriter.write(record)` {#TFRecordWriter.write}

Write a string record to the file.

##### Args:


*  <b>`record`</b>: str


- - -

#### `tf.python_io.TFRecordWriter.close()` {#TFRecordWriter.close}

Close the file.



- - -

### `tf.python_io.tf_record_iterator(path)` {#tf_record_iterator}

An iterator that read the records from a TFRecords file.

##### Args:


*  <b>`path`</b>: The path to the TFRecords file.

##### Yields:

  Strings.

##### Raises:


*  <b>`IOError`</b>: If `path` cannot be opened for reading.



- - -

### TFRecords Format Details

A TFRecords file contains a sequence of strings with CRC hashes.  Each record
has the format

    uint64 length
    uint32 masked_crc32_of_length
    byte   data[length]
    uint32 masked_crc32_of_data

and the records are concatenated together to produce the file.  The CRC32s
are [described here](https://en.wikipedia.org/wiki/Cyclic_redundancy_check),
and the mask of a CRC is

    masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul
