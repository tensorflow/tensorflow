A Reader that outputs fixed-length records from a file.

See ReaderBase for supported methods.
- - -

#### `tf.FixedLengthRecordReader.__init__(record_bytes, header_bytes=None, footer_bytes=None, name=None)` {#FixedLengthRecordReader.__init__}

Create a FixedLengthRecordReader.

##### Args:


*  <b>`record_bytes`</b>: An int.
*  <b>`header_bytes`</b>: An optional int. Defaults to 0.
*  <b>`footer_bytes`</b>: An optional int. Defaults to 0.
*  <b>`name`</b>: A name for the operation (optional).


- - -

#### `tf.FixedLengthRecordReader.num_records_produced(name=None)` {#FixedLengthRecordReader.num_records_produced}

Returns the number of records this reader has produced.

This is the same as the number of Read executions that have
succeeded.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.FixedLengthRecordReader.num_work_units_completed(name=None)` {#FixedLengthRecordReader.num_work_units_completed}

Returns the number of work units this reader has finished processing.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.FixedLengthRecordReader.read(queue, name=None)` {#FixedLengthRecordReader.read}

Returns the next record (key, value pair) produced by a reader.

Will dequeue a work unit from queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has
finished with the previous file).

##### Args:


*  <b>`queue`</b>: A Queue or a mutable string Tensor representing a handle
    to a Queue, with string work items.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of Tensors (key, value).

*  <b>`key`</b>: A string scalar Tensor.
*  <b>`value`</b>: A string scalar Tensor.


- - -

#### `tf.FixedLengthRecordReader.read_up_to(queue, num_records, name=None)` {#FixedLengthRecordReader.read_up_to}

Returns up to num_records (key, value pairs) produced by a reader.

Will dequeue a work unit from queue if necessary (e.g., when the
Reader needs to start reading from a new file since it has
finished with the previous file).

##### Args:


*  <b>`queue`</b>: A Queue or a mutable string Tensor representing a handle
    to a Queue, with string work items.
*  <b>`num_records`</b>: Number of records to read.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of Tensors (keys, values).

*  <b>`keys`</b>: A 1-D string Tensor.
*  <b>`values`</b>: A 1-D string Tensor.


- - -

#### `tf.FixedLengthRecordReader.reader_ref` {#FixedLengthRecordReader.reader_ref}

Op that implements the reader.


- - -

#### `tf.FixedLengthRecordReader.reset(name=None)` {#FixedLengthRecordReader.reset}

Restore a reader to its initial clean state.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

#### `tf.FixedLengthRecordReader.restore_state(state, name=None)` {#FixedLengthRecordReader.restore_state}

Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
Unimplemented error.

##### Args:


*  <b>`state`</b>: A string Tensor.
    Result of a SerializeState of a Reader with matching type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

#### `tf.FixedLengthRecordReader.serialize_state(name=None)` {#FixedLengthRecordReader.serialize_state}

Produce a string tensor that encodes the state of a reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A string Tensor.


- - -

#### `tf.FixedLengthRecordReader.supports_serialize` {#FixedLengthRecordReader.supports_serialize}

Whether the Reader implementation can serialize its state.


