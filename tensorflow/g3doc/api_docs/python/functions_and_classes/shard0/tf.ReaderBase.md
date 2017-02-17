Base class for different Reader types, that produce a record every step.

Conceptually, Readers convert string 'work units' into records (key,
value pairs).  Typically the 'work units' are filenames and the
records are extracted from the contents of those files.  We want a
single record produced per step, but a work unit can correspond to
many records.

Therefore we introduce some decoupling using a queue.  The queue
contains the work units and the Reader dequeues from the queue when
it is asked to produce a record (via Read()) but it has finished the
last work unit.
- - -

#### `tf.ReaderBase.__init__(reader_ref, supports_serialize=False)` {#ReaderBase.__init__}

Creates a new ReaderBase.

##### Args:


*  <b>`reader_ref`</b>: The operation that implements the reader.
*  <b>`supports_serialize`</b>: True if the reader implementation can
    serialize its state.


- - -

#### `tf.ReaderBase.num_records_produced(name=None)` {#ReaderBase.num_records_produced}

Returns the number of records this reader has produced.

This is the same as the number of Read executions that have
succeeded.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.ReaderBase.num_work_units_completed(name=None)` {#ReaderBase.num_work_units_completed}

Returns the number of work units this reader has finished processing.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.ReaderBase.read(queue, name=None)` {#ReaderBase.read}

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

#### `tf.ReaderBase.read_up_to(queue, num_records, name=None)` {#ReaderBase.read_up_to}

Returns up to num_records (key, value pairs) produced by a reader.

Will dequeue a work unit from queue if necessary (e.g., when the
Reader needs to start reading from a new file since it has
finished with the previous file).
It may return less than num_records even before the last batch.

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

#### `tf.ReaderBase.reader_ref` {#ReaderBase.reader_ref}

Op that implements the reader.


- - -

#### `tf.ReaderBase.reset(name=None)` {#ReaderBase.reset}

Restore a reader to its initial clean state.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

#### `tf.ReaderBase.restore_state(state, name=None)` {#ReaderBase.restore_state}

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

#### `tf.ReaderBase.serialize_state(name=None)` {#ReaderBase.serialize_state}

Produce a string tensor that encodes the state of a reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A string Tensor.


- - -

#### `tf.ReaderBase.supports_serialize` {#ReaderBase.supports_serialize}

Whether the Reader implementation can serialize its state.


