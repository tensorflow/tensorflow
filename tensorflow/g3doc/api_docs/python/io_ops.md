<!-- This file is machine generated: DO NOT EDIT! -->

# Inputs and Readers

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Placeholders

TensorFlow provides a placeholder operation that must be fed with data
on execution.  For more info, see the section on [Feeding
data](../../how_tos/reading_data/index.md#feeding).

- - -

### `tf.placeholder(dtype, shape=None, name=None)` {#placeholder}

Inserts a placeholder for a tensor that will be always fed.

**Important**: This tensor will produce an error if evaluated. Its value must
be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`.

For example:

```python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

##### Args:


*  <b>`dtype`</b>: The type of elements in the tensor to be fed.
*  <b>`shape`</b>: The shape of the tensor to be fed (optional). If the shape is not
    specified, you can feed a tensor of any shape.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` that may be used as a handle for feeding a value, but not
  evaluated directly.


- - -

### `tf.placeholder_with_default(input, shape, name=None)` {#placeholder_with_default}

A placeholder op that passes through `input` when its output is not fed.

##### Args:


*  <b>`input`</b>: A `Tensor`. The default value to produce when `output` is not fed.
*  <b>`shape`</b>: A `tf.TensorShape` or list of `ints`.
    The (possibly partial) shape of the tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  A placeholder tensor that defaults to `input` if it is not fed.



For feeding `SparseTensor`s which are composite type,
there is a convenience function:

- - -

### `tf.sparse_placeholder(dtype, shape=None, name=None)` {#sparse_placeholder}

Inserts a placeholder for a sparse tensor that will be always fed.

**Important**: This sparse tensor will produce an error if evaluated.
Its value must be fed using the `feed_dict` optional argument to
`Session.run()`, `Tensor.eval()`, or `Operation.run()`.

For example:

```python
x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
  values = np.array([1.0, 2.0], dtype=np.float32)
  shape = np.array([7, 9, 2], dtype=np.int64)
  print(sess.run(y, feed_dict={
    x: tf.SparseTensorValue(indices, values, shape)}))  # Will succeed.
  print(sess.run(y, feed_dict={
    x: (indices, values, shape)}))  # Will succeed.

  sp = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
  sp_value = sp.eval(session)
  print(sess.run(y, feed_dict={x: sp_value}))  # Will succeed.
```

##### Args:


*  <b>`dtype`</b>: The type of `values` elements in the tensor to be fed.
*  <b>`shape`</b>: The shape of the tensor to be fed (optional). If the shape is not
    specified, you can feed a sparse tensor of any shape.
*  <b>`name`</b>: A name for prefixing the operations (optional).

##### Returns:

  A `SparseTensor` that may be used as a handle for feeding a value, but not
  evaluated directly.



## Readers

TensorFlow provides a set of Reader classes for reading data formats.
For more information on inputs and readers, see [Reading
data](../../how_tos/reading_data/index.md).

- - -

### `class tf.ReaderBase` {#ReaderBase}

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



- - -

### `class tf.TextLineReader` {#TextLineReader}

A Reader that outputs the lines of a file delimited by newlines.

Newlines are stripped from the output.
See ReaderBase for supported methods.
- - -

#### `tf.TextLineReader.__init__(skip_header_lines=None, name=None)` {#TextLineReader.__init__}

Create a TextLineReader.

##### Args:


*  <b>`skip_header_lines`</b>: An optional int. Defaults to 0.  Number of lines
    to skip from the beginning of every file.
*  <b>`name`</b>: A name for the operation (optional).


- - -

#### `tf.TextLineReader.num_records_produced(name=None)` {#TextLineReader.num_records_produced}

Returns the number of records this reader has produced.

This is the same as the number of Read executions that have
succeeded.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.TextLineReader.num_work_units_completed(name=None)` {#TextLineReader.num_work_units_completed}

Returns the number of work units this reader has finished processing.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.TextLineReader.read(queue, name=None)` {#TextLineReader.read}

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

#### `tf.TextLineReader.read_up_to(queue, num_records, name=None)` {#TextLineReader.read_up_to}

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

#### `tf.TextLineReader.reader_ref` {#TextLineReader.reader_ref}

Op that implements the reader.


- - -

#### `tf.TextLineReader.reset(name=None)` {#TextLineReader.reset}

Restore a reader to its initial clean state.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

#### `tf.TextLineReader.restore_state(state, name=None)` {#TextLineReader.restore_state}

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

#### `tf.TextLineReader.serialize_state(name=None)` {#TextLineReader.serialize_state}

Produce a string tensor that encodes the state of a reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A string Tensor.


- - -

#### `tf.TextLineReader.supports_serialize` {#TextLineReader.supports_serialize}

Whether the Reader implementation can serialize its state.



- - -

### `class tf.WholeFileReader` {#WholeFileReader}

A Reader that outputs the entire contents of a file as a value.

To use, enqueue filenames in a Queue.  The output of Read will
be a filename (key) and the contents of that file (value).

See ReaderBase for supported methods.
- - -

#### `tf.WholeFileReader.__init__(name=None)` {#WholeFileReader.__init__}

Create a WholeFileReader.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).


- - -

#### `tf.WholeFileReader.num_records_produced(name=None)` {#WholeFileReader.num_records_produced}

Returns the number of records this reader has produced.

This is the same as the number of Read executions that have
succeeded.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.WholeFileReader.num_work_units_completed(name=None)` {#WholeFileReader.num_work_units_completed}

Returns the number of work units this reader has finished processing.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.WholeFileReader.read(queue, name=None)` {#WholeFileReader.read}

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

#### `tf.WholeFileReader.read_up_to(queue, num_records, name=None)` {#WholeFileReader.read_up_to}

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

#### `tf.WholeFileReader.reader_ref` {#WholeFileReader.reader_ref}

Op that implements the reader.


- - -

#### `tf.WholeFileReader.reset(name=None)` {#WholeFileReader.reset}

Restore a reader to its initial clean state.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

#### `tf.WholeFileReader.restore_state(state, name=None)` {#WholeFileReader.restore_state}

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

#### `tf.WholeFileReader.serialize_state(name=None)` {#WholeFileReader.serialize_state}

Produce a string tensor that encodes the state of a reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A string Tensor.


- - -

#### `tf.WholeFileReader.supports_serialize` {#WholeFileReader.supports_serialize}

Whether the Reader implementation can serialize its state.



- - -

### `class tf.IdentityReader` {#IdentityReader}

A Reader that outputs the queued work as both the key and value.

To use, enqueue strings in a Queue.  Read will take the front
work string and output (work, work).

See ReaderBase for supported methods.
- - -

#### `tf.IdentityReader.__init__(name=None)` {#IdentityReader.__init__}

Create a IdentityReader.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).


- - -

#### `tf.IdentityReader.num_records_produced(name=None)` {#IdentityReader.num_records_produced}

Returns the number of records this reader has produced.

This is the same as the number of Read executions that have
succeeded.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.IdentityReader.num_work_units_completed(name=None)` {#IdentityReader.num_work_units_completed}

Returns the number of work units this reader has finished processing.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.IdentityReader.read(queue, name=None)` {#IdentityReader.read}

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

#### `tf.IdentityReader.read_up_to(queue, num_records, name=None)` {#IdentityReader.read_up_to}

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

#### `tf.IdentityReader.reader_ref` {#IdentityReader.reader_ref}

Op that implements the reader.


- - -

#### `tf.IdentityReader.reset(name=None)` {#IdentityReader.reset}

Restore a reader to its initial clean state.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

#### `tf.IdentityReader.restore_state(state, name=None)` {#IdentityReader.restore_state}

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

#### `tf.IdentityReader.serialize_state(name=None)` {#IdentityReader.serialize_state}

Produce a string tensor that encodes the state of a reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A string Tensor.


- - -

#### `tf.IdentityReader.supports_serialize` {#IdentityReader.supports_serialize}

Whether the Reader implementation can serialize its state.



- - -

### `class tf.TFRecordReader` {#TFRecordReader}

A Reader that outputs the records from a TFRecords file.

See ReaderBase for supported methods.
- - -

#### `tf.TFRecordReader.__init__(name=None, options=None)` {#TFRecordReader.__init__}

Create a TFRecordReader.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).
*  <b>`options`</b>: A TFRecordOptions object (optional).


- - -

#### `tf.TFRecordReader.num_records_produced(name=None)` {#TFRecordReader.num_records_produced}

Returns the number of records this reader has produced.

This is the same as the number of Read executions that have
succeeded.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.TFRecordReader.num_work_units_completed(name=None)` {#TFRecordReader.num_work_units_completed}

Returns the number of work units this reader has finished processing.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An int64 Tensor.


- - -

#### `tf.TFRecordReader.read(queue, name=None)` {#TFRecordReader.read}

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

#### `tf.TFRecordReader.read_up_to(queue, num_records, name=None)` {#TFRecordReader.read_up_to}

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

#### `tf.TFRecordReader.reader_ref` {#TFRecordReader.reader_ref}

Op that implements the reader.


- - -

#### `tf.TFRecordReader.reset(name=None)` {#TFRecordReader.reset}

Restore a reader to its initial clean state.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

#### `tf.TFRecordReader.restore_state(state, name=None)` {#TFRecordReader.restore_state}

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

#### `tf.TFRecordReader.serialize_state(name=None)` {#TFRecordReader.serialize_state}

Produce a string tensor that encodes the state of a reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A string Tensor.


- - -

#### `tf.TFRecordReader.supports_serialize` {#TFRecordReader.supports_serialize}

Whether the Reader implementation can serialize its state.



- - -

### `class tf.FixedLengthRecordReader` {#FixedLengthRecordReader}

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




## Converting

TensorFlow provides several operations that you can use to convert various data
formats into tensors.

- - -

### `tf.decode_csv(records, record_defaults, field_delim=None, name=None)` {#decode_csv}

Convert CSV records to tensors. Each column maps to one tensor.

RFC 4180 format is expected for the CSV records.
(https://tools.ietf.org/html/rfc4180)
Note that we allow leading and trailing spaces with int or float field.

##### Args:


*  <b>`records`</b>: A `Tensor` of type `string`.
    Each string is a record/row in the csv and all records should have
    the same format.
*  <b>`record_defaults`</b>: A list of `Tensor` objects with types from: `float32`, `int32`, `int64`, `string`.
    One tensor per column of the input record, with either a
    scalar default value for that column or empty if the column is required.
*  <b>`field_delim`</b>: An optional `string`. Defaults to `","`.
    delimiter to separate fields in a record.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list of `Tensor` objects. Has the same type as `record_defaults`.
  Each tensor will have the same shape as records.


- - -

### `tf.decode_raw(bytes, out_type, little_endian=None, name=None)` {#decode_raw}

Reinterpret the bytes of a string as a vector of numbers.

##### Args:


*  <b>`bytes`</b>: A `Tensor` of type `string`.
    All the elements must have the same length.
*  <b>`out_type`</b>: A `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64`.
*  <b>`little_endian`</b>: An optional `bool`. Defaults to `True`.
    Whether the input `bytes` are in little-endian order.
    Ignored for `out_type` values that are stored in a single byte like
    `uint8`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `out_type`.
  A Tensor with one more dimension than the input `bytes`.  The
  added dimension will have size equal to the length of the elements
  of `bytes` divided by the number of bytes to represent `out_type`.



- - -

### Example protocol buffer

TensorFlow's [recommended format for training
examples](../../how_tos/reading_data/index.md#standard-tensorflow-format)
is serialized `Example` protocol buffers, [described
here](https://www.tensorflow.org/code/tensorflow/core/example/example.proto).
They contain `Features`, [described
here](https://www.tensorflow.org/code/tensorflow/core/example/feature.proto).

- - -

### `class tf.VarLenFeature` {#VarLenFeature}

Configuration for parsing a variable-length input feature.

Fields:
  dtype: Data type of input.
- - -

#### `tf.VarLenFeature.__getnewargs__()` {#VarLenFeature.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.VarLenFeature.__getstate__()` {#VarLenFeature.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.VarLenFeature.__new__(_cls, dtype)` {#VarLenFeature.__new__}

Create new instance of VarLenFeature(dtype,)


- - -

#### `tf.VarLenFeature.__repr__()` {#VarLenFeature.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.VarLenFeature.dtype` {#VarLenFeature.dtype}

Alias for field number 0



- - -

### `class tf.FixedLenFeature` {#FixedLenFeature}

Configuration for parsing a fixed-length input feature.

To treat sparse input as dense, provide a `default_value`; otherwise,
the parse functions will fail on any examples missing this feature.

Fields:
  shape: Shape of input data.
  dtype: Data type of input.
  default_value: Value to be used if an example is missing this feature. It
      must be compatible with `dtype`.
- - -

#### `tf.FixedLenFeature.__getnewargs__()` {#FixedLenFeature.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.FixedLenFeature.__getstate__()` {#FixedLenFeature.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.FixedLenFeature.__new__(_cls, shape, dtype, default_value=None)` {#FixedLenFeature.__new__}

Create new instance of FixedLenFeature(shape, dtype, default_value)


- - -

#### `tf.FixedLenFeature.__repr__()` {#FixedLenFeature.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.FixedLenFeature.default_value` {#FixedLenFeature.default_value}

Alias for field number 2


- - -

#### `tf.FixedLenFeature.dtype` {#FixedLenFeature.dtype}

Alias for field number 1


- - -

#### `tf.FixedLenFeature.shape` {#FixedLenFeature.shape}

Alias for field number 0



- - -

### `class tf.FixedLenSequenceFeature` {#FixedLenSequenceFeature}

Configuration for a dense input feature in a sequence item.

To treat a sparse input as dense, provide `allow_missing=True`; otherwise,
the parse functions will fail on any examples missing this feature.

Fields:
  shape: Shape of input data.
  dtype: Data type of input.
  allow_missing: Whether to allow this feature to be missing from a feature
    list item.
- - -

#### `tf.FixedLenSequenceFeature.__getnewargs__()` {#FixedLenSequenceFeature.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.FixedLenSequenceFeature.__getstate__()` {#FixedLenSequenceFeature.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.FixedLenSequenceFeature.__new__(_cls, shape, dtype, allow_missing=False)` {#FixedLenSequenceFeature.__new__}

Create new instance of FixedLenSequenceFeature(shape, dtype, allow_missing)


- - -

#### `tf.FixedLenSequenceFeature.__repr__()` {#FixedLenSequenceFeature.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.FixedLenSequenceFeature.allow_missing` {#FixedLenSequenceFeature.allow_missing}

Alias for field number 2


- - -

#### `tf.FixedLenSequenceFeature.dtype` {#FixedLenSequenceFeature.dtype}

Alias for field number 1


- - -

#### `tf.FixedLenSequenceFeature.shape` {#FixedLenSequenceFeature.shape}

Alias for field number 0



- - -

### `class tf.SparseFeature` {#SparseFeature}

Configuration for parsing a sparse input feature.

Fields:
  index_key: Name of index feature.  The underlying feature's type must
    be `int64` and its length must always match that of the `value_key`
    feature.
  value_key: Name of value feature.  The underlying feature's type must
    be `dtype` and its length must always match that of the `index_key`
    feature.
  dtype: Data type of the `value_key` feature.
  size: Each value in the `index_key` feature must be in `[0, size)`.
  already_sorted: A boolean to specify whether the values in `index_key` are
    already sorted. If so skip sorting, False by default (optional).
- - -

#### `tf.SparseFeature.__getnewargs__()` {#SparseFeature.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.SparseFeature.__getstate__()` {#SparseFeature.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.SparseFeature.__new__(_cls, index_key, value_key, dtype, size, already_sorted=False)` {#SparseFeature.__new__}

Create new instance of SparseFeature(index_key, value_key, dtype, size, already_sorted)


- - -

#### `tf.SparseFeature.__repr__()` {#SparseFeature.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.SparseFeature.already_sorted` {#SparseFeature.already_sorted}

Alias for field number 4


- - -

#### `tf.SparseFeature.dtype` {#SparseFeature.dtype}

Alias for field number 2


- - -

#### `tf.SparseFeature.index_key` {#SparseFeature.index_key}

Alias for field number 0


- - -

#### `tf.SparseFeature.size` {#SparseFeature.size}

Alias for field number 3


- - -

#### `tf.SparseFeature.value_key` {#SparseFeature.value_key}

Alias for field number 1



- - -

### `tf.parse_example(serialized, features, name=None, example_names=None)` {#parse_example}

Parses `Example` protos into a `dict` of tensors.

Parses a number of serialized [`Example`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
protos given in `serialized`.

`example_names` may contain descriptive names for the corresponding serialized
protos. These may be useful for debugging purposes, but they have no effect on
the output. If not `None`, `example_names` must be the same length as
`serialized`.

This op parses serialized examples into a dictionary mapping keys to `Tensor`
and `SparseTensor` objects. `features` is a dict from keys to `VarLenFeature`,
`SparseFeature`, and `FixedLenFeature` objects. Each `VarLenFeature`
and `SparseFeature` is mapped to a `SparseTensor`, and each
`FixedLenFeature` is mapped to a `Tensor`.

Each `VarLenFeature` maps to a `SparseTensor` of the specified type
representing a ragged matrix. Its indices are `[batch, index]` where `batch`
is the batch entry the value is from in `serialized`, and `index` is the
value's index in the list of values associated with that feature and example.

Each `SparseFeature` maps to a `SparseTensor` of the specified type
representing a sparse matrix of shape
`(serialized.size(), SparseFeature.size)`. Its indices are `[batch, index]`
where `batch` is the batch entry the value is from in `serialized`, and
`index` is the value's index is given by the values in the
`SparseFeature.index_key` feature column.

Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
`tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.

`FixedLenFeature` entries with a `default_value` are optional. With no default
value, we will fail if that `Feature` is missing from any example in
`serialized`.

Examples:

For example, if one expects a `tf.float32` sparse feature `ft` and three
serialized `Example`s are provided:

```
serialized = [
  features
    { feature { key: "ft" value { float_list { value: [1.0, 2.0] } } } },
  features
    { feature []},
  features
    { feature { key: "ft" value { float_list { value: [3.0] } } }
]
```

then the output will look like:

```
{"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                    values=[1.0, 2.0, 3.0],
                    dense_shape=(3, 2)) }
```

Given two `Example` input protos in `serialized`:

```
[
  features {
    feature { key: "kw" value { bytes_list { value: [ "knit", "big" ] } } }
    feature { key: "gps" value { float_list { value: [] } } }
  },
  features {
    feature { key: "kw" value { bytes_list { value: [ "emmy" ] } } }
    feature { key: "dank" value { int64_list { value: [ 42 ] } } }
    feature { key: "gps" value { } }
  }
]
```

And arguments

```
example_names: ["input0", "input1"],
features: {
    "kw": VarLenFeature(tf.string),
    "dank": VarLenFeature(tf.int64),
    "gps": VarLenFeature(tf.float32),
}
```

Then the output is a dictionary:

```python
{
  "kw": SparseTensor(
      indices=[[0, 0], [0, 1], [1, 0]],
      values=["knit", "big", "emmy"]
      dense_shape=[2, 2]),
  "dank": SparseTensor(
      indices=[[1, 0]],
      values=[42],
      dense_shape=[2, 1]),
  "gps": SparseTensor(
      indices=[],
      values=[],
      dense_shape=[2, 0]),
}
```

For dense results in two serialized `Example`s:

```
[
  features {
    feature { key: "age" value { int64_list { value: [ 0 ] } } }
    feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
   },
   features {
    feature { key: "age" value { int64_list { value: [] } } }
    feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
  }
]
```

We can use arguments:

```
example_names: ["input0", "input1"],
features: {
    "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
    "gender": FixedLenFeature([], dtype=tf.string),
}
```

And the expected output is:

```python
{
  "age": [[0], [-1]],
  "gender": [["f"], ["f"]],
}
```

Given two `Example` input protos in `serialized`:

```
[
  features {
    feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }
    feature { key: "ix" value { int64_list { value: [ 3, 20 ] } } }
  },
  features {
    feature { key: "val" value { float_list { value: [ 0.0 ] } } }
    feature { key: "ix" value { int64_list { value: [ 42 ] } } }
  }
]
```

And arguments

```
example_names: ["input0", "input1"],
features: {
    "sparse": SparseFeature("ix", "val", tf.float32, 100),
}
```

Then the output is a dictionary:

```python
{
  "sparse": SparseTensor(
      indices=[[0, 3], [0, 20], [1, 42]],
      values=[0.5, -1.0, 0.0]
      dense_shape=[2, 100]),
}
```

##### Args:


*  <b>`serialized`</b>: A vector (1-D Tensor) of strings, a batch of binary
    serialized `Example` protos.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature`,
    `VarLenFeature`, and `SparseFeature` values.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`example_names`</b>: A vector (1-D Tensor) of strings (optional), the names of
    the serialized protos in the batch.

##### Returns:

  A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

##### Raises:


*  <b>`ValueError`</b>: if any feature is invalid.


- - -

### `tf.parse_single_example(serialized, features, name=None, example_names=None)` {#parse_single_example}

Parses a single `Example` proto.

Similar to `parse_example`, except:

For dense tensors, the returned `Tensor` is identical to the output of
`parse_example`, except there is no batch dimension, the output shape is the
same as the shape given in `dense_shape`.

For `SparseTensor`s, the first (batch) column of the indices matrix is removed
(the indices matrix is a column vector), the values vector is unchanged, and
the first (`batch_size`) entry of the shape vector is removed (it is now a
single element vector).

One might see performance advantages by batching `Example` protos with
`parse_example` instead of using this function directly.

##### Args:


*  <b>`serialized`</b>: A scalar string Tensor, a single serialized Example.
    See `_parse_single_example_raw` documentation for more details.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`example_names`</b>: (Optional) A scalar string Tensor, the associated name.
    See `_parse_single_example_raw` documentation for more details.

##### Returns:

  A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

##### Raises:


*  <b>`ValueError`</b>: if any feature is invalid.


- - -

### `tf.parse_tensor(serialized, out_type, name=None)` {#parse_tensor}

Transforms a serialized tensorflow.TensorProto proto into a Tensor.

##### Args:


*  <b>`serialized`</b>: A `Tensor` of type `string`.
    A scalar string containing a serialized TensorProto proto.
*  <b>`out_type`</b>: A `tf.DType`.
    The type of the serialized tensor.  The provided type must match the
    type of the serialized tensor and no implicit conversion will take place.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `out_type`. A Tensor of type `out_type`.


- - -

### `tf.decode_json_example(json_examples, name=None)` {#decode_json_example}

Convert JSON-encoded Example records to binary protocol buffer strings.

This op translates a tensor containing Example records, encoded using
the [standard JSON
mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
into a tensor containing the same records encoded as binary protocol
buffers. The resulting tensor can then be fed to any of the other
Example-parsing ops.

##### Args:


*  <b>`json_examples`</b>: A `Tensor` of type `string`.
    Each string is a JSON object serialized according to the JSON
    mapping of the Example proto.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`.
  Each string is a binary Example protocol buffer corresponding
  to the respective element of `json_examples`.



## Queues

TensorFlow provides several implementations of 'Queues', which are
structures within the TensorFlow computation graph to stage pipelines
of tensors together. The following describe the basic Queue interface
and some implementations.  To see an example use, see [Threading and
Queues](../../how_tos/threading_and_queues/index.md).

- - -

### `class tf.QueueBase` {#QueueBase}

Base class for queue implementations.

A queue is a TensorFlow data structure that stores tensors across
multiple steps, and exposes operations that enqueue and dequeue
tensors.

Each queue element is a tuple of one or more tensors, where each
tuple component has a static dtype, and may have a static shape. The
queue implementations support versions of enqueue and dequeue that
handle single elements, versions that support enqueuing and
dequeuing a batch of elements at once.

See [`tf.FIFOQueue`](#FIFOQueue) and
[`tf.RandomShuffleQueue`](#RandomShuffleQueue) for concrete
implementations of this class, and instructions on how to create
them.

- - -

#### `tf.QueueBase.enqueue(vals, name=None)` {#QueueBase.enqueue}

Enqueues one element to this queue.

If the queue is full when this operation executes, it will block
until the element has been enqueued.

At runtime, this operation may raise an error if the queue is
[closed](#QueueBase.close) before or during its execution. If the
queue is closed before this operation runs,
`tf.errors.CancelledError` will be raised. If this operation is
blocked, and either (i) the queue is closed by a close operation
with `cancel_pending_enqueues=True`, or (ii) the session is
[closed](../../api_docs/python/client.md#Session.close),
`tf.errors.CancelledError` will be raised.

##### Args:


*  <b>`vals`</b>: A tensor, a list or tuple of tensors, or a dictionary containing
    the values to enqueue.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that enqueues a new tuple of tensors to the queue.


- - -

#### `tf.QueueBase.enqueue_many(vals, name=None)` {#QueueBase.enqueue_many}

Enqueues zero or more elements to this queue.

This operation slices each component tensor along the 0th dimension to
make multiple queue elements. All of the tensors in `vals` must have the
same size in the 0th dimension.

If the queue is full when this operation executes, it will block
until all of the elements have been enqueued.

At runtime, this operation may raise an error if the queue is
[closed](#QueueBase.close) before or during its execution. If the
queue is closed before this operation runs,
`tf.errors.CancelledError` will be raised. If this operation is
blocked, and either (i) the queue is closed by a close operation
with `cancel_pending_enqueues=True`, or (ii) the session is
[closed](../../api_docs/python/client.md#Session.close),
`tf.errors.CancelledError` will be raised.

##### Args:


*  <b>`vals`</b>: A tensor, a list or tuple of tensors, or a dictionary
    from which the queue elements are taken.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that enqueues a batch of tuples of tensors to the queue.



- - -

#### `tf.QueueBase.dequeue(name=None)` {#QueueBase.dequeue}

Dequeues one element from this queue.

If the queue is empty when this operation executes, it will block
until there is an element to dequeue.

At runtime, this operation may raise an error if the queue is
[closed](#QueueBase.close) before or during its execution. If the
queue is closed, the queue is empty, and there are no pending
enqueue operations that can fulfill this request,
`tf.errors.OutOfRangeError` will be raised. If the session is
[closed](../../api_docs/python/client.md#Session.close),
`tf.errors.CancelledError` will be raised.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of tensors that was dequeued.


- - -

#### `tf.QueueBase.dequeue_many(n, name=None)` {#QueueBase.dequeue_many}

Dequeues and concatenates `n` elements from this queue.

This operation concatenates queue-element component tensors along
the 0th dimension to make a single component tensor.  All of the
components in the dequeued tuple will have size `n` in the 0th dimension.

If the queue is closed and there are less than `n` elements left, then an
`OutOfRange` exception is raised.

At runtime, this operation may raise an error if the queue is
[closed](#QueueBase.close) before or during its execution. If the
queue is closed, the queue contains fewer than `n` elements, and
there are no pending enqueue operations that can fulfill this
request, `tf.errors.OutOfRangeError` will be raised. If the
session is [closed](../../api_docs/python/client.md#Session.close),
`tf.errors.CancelledError` will be raised.

##### Args:


*  <b>`n`</b>: A scalar `Tensor` containing the number of elements to dequeue.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of concatenated tensors that was dequeued.



- - -

#### `tf.QueueBase.size(name=None)` {#QueueBase.size}

Compute the number of elements in this queue.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar tensor containing the number of elements in this queue.



- - -

#### `tf.QueueBase.close(cancel_pending_enqueues=False, name=None)` {#QueueBase.close}

Closes this queue.

This operation signals that no more elements will be enqueued in
the given queue. Subsequent `enqueue` and `enqueue_many`
operations will fail. Subsequent `dequeue` and `dequeue_many`
operations will continue to succeed if sufficient elements remain
in the queue. Subsequent `dequeue` and `dequeue_many` operations
that would block will fail immediately.

If `cancel_pending_enqueues` is `True`, all pending requests will also
be cancelled.

##### Args:


*  <b>`cancel_pending_enqueues`</b>: (Optional.) A boolean, defaulting to
    `False` (described above).
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The operation that closes the queue.



#### Other Methods
- - -

#### `tf.QueueBase.__init__(dtypes, shapes, names, queue_ref)` {#QueueBase.__init__}

Constructs a queue object from a queue reference.

The two optional lists, `shapes` and `names`, must be of the same length
as `dtypes` if provided.  The values at a given index `i` indicate the
shape and name to use for the corresponding queue component in `dtypes`.

##### Args:


*  <b>`dtypes`</b>: A list of types.  The length of dtypes must equal the number
    of tensors in each element.
*  <b>`shapes`</b>: Constraints on the shapes of tensors in an element:
    A list of shape tuples or None. This list is the same length
    as dtypes.  If the shape of any tensors in the element are constrained,
    all must be; shapes can be None if the shapes should not be constrained.
*  <b>`names`</b>: Optional list of names.  If provided, the `enqueue()` and
    `dequeue()` methods will use dictionaries with these names as keys.
    Must be None or a list or tuple of the same length as `dtypes`.
*  <b>`queue_ref`</b>: The queue reference, i.e. the output of the queue op.

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


- - -

#### `tf.QueueBase.dequeue_up_to(n, name=None)` {#QueueBase.dequeue_up_to}

Dequeues and concatenates `n` elements from this queue.

**Note** This operation is not supported by all queues.  If a queue does not
support DequeueUpTo, then a `tf.errors.UnimplementedError` is raised.

This operation concatenates queue-element component tensors along
the 0th dimension to make a single component tensor. If the queue
has not been closed, all of the components in the dequeued tuple
will have size `n` in the 0th dimension.

If the queue is closed and there are more than `0` but fewer than
`n` elements remaining, then instead of raising a
`tf.errors.OutOfRangeError` like [`dequeue_many`](#QueueBase.dequeue_many),
less than `n` elements are returned immediately.  If the queue is
closed and there are `0` elements left in the queue, then a
`tf.errors.OutOfRangeError` is raised just like in `dequeue_many`.
Otherwise the behavior is identical to `dequeue_many`.

##### Args:


*  <b>`n`</b>: A scalar `Tensor` containing the number of elements to dequeue.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The tuple of concatenated tensors that was dequeued.


- - -

#### `tf.QueueBase.dtypes` {#QueueBase.dtypes}

The list of dtypes for each component of a queue element.


- - -

#### `tf.QueueBase.from_list(index, queues)` {#QueueBase.from_list}

Create a queue using the queue reference from `queues[index]`.

##### Args:


*  <b>`index`</b>: An integer scalar tensor that determines the input that gets
    selected.
*  <b>`queues`</b>: A list of `QueueBase` objects.

##### Returns:

  A `QueueBase` object.

##### Raises:


*  <b>`TypeError`</b>: When `queues` is not a list of `QueueBase` objects,
    or when the data types of `queues` are not all the same.


- - -

#### `tf.QueueBase.name` {#QueueBase.name}

The name of the underlying queue.


- - -

#### `tf.QueueBase.names` {#QueueBase.names}

The list of names for each component of a queue element.


- - -

#### `tf.QueueBase.queue_ref` {#QueueBase.queue_ref}

The underlying queue reference.


- - -

#### `tf.QueueBase.shapes` {#QueueBase.shapes}

The list of shapes for each component of a queue element.



- - -

### `class tf.FIFOQueue` {#FIFOQueue}

A queue implementation that dequeues elements in first-in first-out order.

See [`tf.QueueBase`](#QueueBase) for a description of the methods on
this class.

- - -

#### `tf.FIFOQueue.__init__(capacity, dtypes, shapes=None, names=None, shared_name=None, name='fifo_queue')` {#FIFOQueue.__init__}

Creates a queue that dequeues elements in a first-in first-out order.

A `FIFOQueue` has bounded capacity; supports multiple concurrent
producers and consumers; and provides exactly-once delivery.

A `FIFOQueue` holds a list of up to `capacity` elements. Each
element is a fixed-length tuple of tensors whose dtypes are
described by `dtypes`, and whose shapes are optionally described
by the `shapes` argument.

If the `shapes` argument is specified, each component of a queue
element must have the respective fixed shape. If it is
unspecified, different queue elements may have different shapes,
but the use of `dequeue_many` is disallowed.

##### Args:


*  <b>`capacity`</b>: An integer. The upper bound on the number of elements
    that may be stored in this queue.
*  <b>`dtypes`</b>: A list of `DType` objects. The length of `dtypes` must equal
    the number of tensors in each queue element.
*  <b>`shapes`</b>: (Optional.) A list of fully-defined `TensorShape` objects
    with the same length as `dtypes`, or `None`.
*  <b>`names`</b>: (Optional.) A list of string naming the components in the queue
    with the same length as `dtypes`, or `None`.  If specified the dequeue
    methods return a dictionary with the names as keys.
*  <b>`shared_name`</b>: (Optional.) If non-empty, this queue will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the queue operation.



- - -

### `class tf.PaddingFIFOQueue` {#PaddingFIFOQueue}

A FIFOQueue that supports batching variable-sized tensors by padding.

A `PaddingFIFOQueue` may contain components with dynamic shape, while also
supporting `dequeue_many`.  See the constructor for more details.

See [`tf.QueueBase`](#QueueBase) for a description of the methods on
this class.

- - -

#### `tf.PaddingFIFOQueue.__init__(capacity, dtypes, shapes, names=None, shared_name=None, name='padding_fifo_queue')` {#PaddingFIFOQueue.__init__}

Creates a queue that dequeues elements in a first-in first-out order.

A `PaddingFIFOQueue` has bounded capacity; supports multiple concurrent
producers and consumers; and provides exactly-once delivery.

A `PaddingFIFOQueue` holds a list of up to `capacity` elements. Each
element is a fixed-length tuple of tensors whose dtypes are
described by `dtypes`, and whose shapes are described by the `shapes`
argument.

The `shapes` argument must be specified; each component of a queue
element must have the respective shape.  Shapes of fixed
rank but variable size are allowed by setting any shape dimension to None.
In this case, the inputs' shape may vary along the given dimension, and
`dequeue_many` will pad the given dimension with zeros up to the maximum
shape of all elements in the given batch.

##### Args:


*  <b>`capacity`</b>: An integer. The upper bound on the number of elements
    that may be stored in this queue.
*  <b>`dtypes`</b>: A list of `DType` objects. The length of `dtypes` must equal
    the number of tensors in each queue element.
*  <b>`shapes`</b>: A list of `TensorShape` objects, with the same length as
    `dtypes`.  Any dimension in the `TensorShape` containing value
    `None` is dynamic and allows values to be enqueued with
     variable size in that dimension.
*  <b>`names`</b>: (Optional.) A list of string naming the components in the queue
    with the same length as `dtypes`, or `None`.  If specified the dequeue
    methods return a dictionary with the names as keys.
*  <b>`shared_name`</b>: (Optional.) If non-empty, this queue will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the queue operation.

##### Raises:


*  <b>`ValueError`</b>: If shapes is not a list of shapes, or the lengths of dtypes
    and shapes do not match, or if names is specified and the lengths of
    dtypes and names do not match.



- - -

### `class tf.RandomShuffleQueue` {#RandomShuffleQueue}

A queue implementation that dequeues elements in a random order.

See [`tf.QueueBase`](#QueueBase) for a description of the methods on
this class.

- - -

#### `tf.RandomShuffleQueue.__init__(capacity, min_after_dequeue, dtypes, shapes=None, names=None, seed=None, shared_name=None, name='random_shuffle_queue')` {#RandomShuffleQueue.__init__}

Create a queue that dequeues elements in a random order.

A `RandomShuffleQueue` has bounded capacity; supports multiple
concurrent producers and consumers; and provides exactly-once
delivery.

A `RandomShuffleQueue` holds a list of up to `capacity`
elements. Each element is a fixed-length tuple of tensors whose
dtypes are described by `dtypes`, and whose shapes are optionally
described by the `shapes` argument.

If the `shapes` argument is specified, each component of a queue
element must have the respective fixed shape. If it is
unspecified, different queue elements may have different shapes,
but the use of `dequeue_many` is disallowed.

The `min_after_dequeue` argument allows the caller to specify a
minimum number of elements that will remain in the queue after a
`dequeue` or `dequeue_many` operation completes, to ensure a
minimum level of mixing of elements. This invariant is maintained
by blocking those operations until sufficient elements have been
enqueued. The `min_after_dequeue` argument is ignored after the
queue has been closed.

##### Args:


*  <b>`capacity`</b>: An integer. The upper bound on the number of elements
    that may be stored in this queue.
*  <b>`min_after_dequeue`</b>: An integer (described above).
*  <b>`dtypes`</b>: A list of `DType` objects. The length of `dtypes` must equal
    the number of tensors in each queue element.
*  <b>`shapes`</b>: (Optional.) A list of fully-defined `TensorShape` objects
    with the same length as `dtypes`, or `None`.
*  <b>`names`</b>: (Optional.) A list of string naming the components in the queue
    with the same length as `dtypes`, or `None`.  If specified the dequeue
    methods return a dictionary with the names as keys.
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`shared_name`</b>: (Optional.) If non-empty, this queue will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the queue operation.



- - -

### `class tf.PriorityQueue` {#PriorityQueue}

A queue implementation that dequeues elements in prioritized order.

See [`tf.QueueBase`](#QueueBase) for a description of the methods on
this class.

- - -

#### `tf.PriorityQueue.__init__(capacity, types, shapes=None, names=None, shared_name=None, name='priority_queue')` {#PriorityQueue.__init__}

Creates a queue that dequeues elements in a first-in first-out order.

A `PriorityQueue` has bounded capacity; supports multiple concurrent
producers and consumers; and provides exactly-once delivery.

A `PriorityQueue` holds a list of up to `capacity` elements. Each
element is a fixed-length tuple of tensors whose dtypes are
described by `types`, and whose shapes are optionally described
by the `shapes` argument.

If the `shapes` argument is specified, each component of a queue
element must have the respective fixed shape. If it is
unspecified, different queue elements may have different shapes,
but the use of `dequeue_many` is disallowed.

Enqueues and Dequeues to the `PriorityQueue` must include an additional
tuple entry at the beginning: the `priority`.  The priority must be
an int64 scalar (for `enqueue`) or an int64 vector (for `enqueue_many`).

##### Args:


*  <b>`capacity`</b>: An integer. The upper bound on the number of elements
    that may be stored in this queue.
*  <b>`types`</b>: A list of `DType` objects. The length of `types` must equal
    the number of tensors in each queue element, except the first priority
    element.  The first tensor in each element is the priority,
    which must be type int64.
*  <b>`shapes`</b>: (Optional.) A list of fully-defined `TensorShape` objects,
    with the same length as `types`, or `None`.
*  <b>`names`</b>: (Optional.) A list of strings naming the components in the queue
    with the same length as `dtypes`, or `None`.  If specified, the dequeue
    methods return a dictionary with the names as keys.
*  <b>`shared_name`</b>: (Optional.) If non-empty, this queue will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the queue operation.




## Conditional Accumulators

- - -

### `class tf.ConditionalAccumulatorBase` {#ConditionalAccumulatorBase}

A conditional accumulator for aggregating gradients.

Up-to-date gradients (i.e., time step at which gradient was computed is
equal to the accumulator's time step) are added to the accumulator.

Extraction of the average gradient is blocked until the required number of
gradients has been accumulated.
- - -

#### `tf.ConditionalAccumulatorBase.__init__(dtype, shape, accumulator_ref)` {#ConditionalAccumulatorBase.__init__}

Creates a new ConditionalAccumulator.

##### Args:


*  <b>`dtype`</b>: Datatype of the accumulated gradients.
*  <b>`shape`</b>: Shape of the accumulated gradients.
*  <b>`accumulator_ref`</b>: A handle to the conditional accumulator, created by sub-
    classes


- - -

#### `tf.ConditionalAccumulatorBase.accumulator_ref` {#ConditionalAccumulatorBase.accumulator_ref}

The underlying accumulator reference.


- - -

#### `tf.ConditionalAccumulatorBase.dtype` {#ConditionalAccumulatorBase.dtype}

The datatype of the gradients accumulated by this accumulator.


- - -

#### `tf.ConditionalAccumulatorBase.name` {#ConditionalAccumulatorBase.name}

The name of the underlying accumulator.


- - -

#### `tf.ConditionalAccumulatorBase.num_accumulated(name=None)` {#ConditionalAccumulatorBase.num_accumulated}

Number of gradients that have currently been aggregated in accumulator.

##### Args:


*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Number of accumulated gradients currently in accumulator.


- - -

#### `tf.ConditionalAccumulatorBase.set_global_step(new_global_step, name=None)` {#ConditionalAccumulatorBase.set_global_step}

Sets the global time step of the accumulator.

The operation logs a warning if we attempt to set to a time step that is
lower than the accumulator's own time step.

##### Args:


*  <b>`new_global_step`</b>: Value of new time step. Can be a variable or a constant
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Operation that sets the accumulator's time step.



- - -

### `class tf.ConditionalAccumulator` {#ConditionalAccumulator}

A conditional accumulator for aggregating gradients.

Up-to-date gradients (i.e., time step at which gradient was computed is
equal to the accumulator's time step) are added to the accumulator.

Extraction of the average gradient is blocked until the required number of
gradients has been accumulated.
- - -

#### `tf.ConditionalAccumulator.__init__(dtype, shape=None, shared_name=None, name='conditional_accumulator')` {#ConditionalAccumulator.__init__}

Creates a new ConditionalAccumulator.

##### Args:


*  <b>`dtype`</b>: Datatype of the accumulated gradients.
*  <b>`shape`</b>: Shape of the accumulated gradients.
*  <b>`shared_name`</b>: Optional. If non-empty, this accumulator will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the accumulator.


- - -

#### `tf.ConditionalAccumulator.accumulator_ref` {#ConditionalAccumulator.accumulator_ref}

The underlying accumulator reference.


- - -

#### `tf.ConditionalAccumulator.apply_grad(grad, local_step=0, name=None)` {#ConditionalAccumulator.apply_grad}

Attempts to apply a gradient to the accumulator.

The attempt is silently dropped if the gradient is stale, i.e., local_step
is less than the accumulator's global time step.

##### Args:


*  <b>`grad`</b>: The gradient tensor to be applied.
*  <b>`local_step`</b>: Time step at which the gradient was computed.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  The operation that (conditionally) applies a gradient to the accumulator.

##### Raises:


*  <b>`ValueError`</b>: If grad is of the wrong shape


- - -

#### `tf.ConditionalAccumulator.dtype` {#ConditionalAccumulator.dtype}

The datatype of the gradients accumulated by this accumulator.


- - -

#### `tf.ConditionalAccumulator.name` {#ConditionalAccumulator.name}

The name of the underlying accumulator.


- - -

#### `tf.ConditionalAccumulator.num_accumulated(name=None)` {#ConditionalAccumulator.num_accumulated}

Number of gradients that have currently been aggregated in accumulator.

##### Args:


*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Number of accumulated gradients currently in accumulator.


- - -

#### `tf.ConditionalAccumulator.set_global_step(new_global_step, name=None)` {#ConditionalAccumulator.set_global_step}

Sets the global time step of the accumulator.

The operation logs a warning if we attempt to set to a time step that is
lower than the accumulator's own time step.

##### Args:


*  <b>`new_global_step`</b>: Value of new time step. Can be a variable or a constant
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Operation that sets the accumulator's time step.


- - -

#### `tf.ConditionalAccumulator.take_grad(num_required, name=None)` {#ConditionalAccumulator.take_grad}

Attempts to extract the average gradient from the accumulator.

The operation blocks until sufficient number of gradients have been
successfully applied to the accumulator.

Once successful, the following actions are also triggered:
- Counter of accumulated gradients is reset to 0.
- Aggregated gradient is reset to 0 tensor.
- Accumulator's internal time step is incremented by 1.

##### Args:


*  <b>`num_required`</b>: Number of gradients that needs to have been aggregated
*  <b>`name`</b>: Optional name for the operation

##### Returns:

  A tensor holding the value of the average gradient.

##### Raises:


*  <b>`InvalidArgumentError`</b>: If num_required < 1



- - -

### `class tf.SparseConditionalAccumulator` {#SparseConditionalAccumulator}

A conditional accumulator for aggregating sparse gradients.

Sparse gradients are represented by IndexedSlices.

Up-to-date gradients (i.e., time step at which gradient was computed is
equal to the accumulator's time step) are added to the accumulator.

Extraction of the average gradient is blocked until the required number of
gradients has been accumulated.

Args:
  dtype: Datatype of the accumulated gradients.
  shape: Shape of the accumulated gradients.
  shared_name: Optional. If non-empty, this accumulator will be shared under
    the given name across multiple sessions.
  name: Optional name for the accumulator.
- - -

#### `tf.SparseConditionalAccumulator.__init__(dtype, shape=None, shared_name=None, name='sparse_conditional_accumulator')` {#SparseConditionalAccumulator.__init__}




- - -

#### `tf.SparseConditionalAccumulator.accumulator_ref` {#SparseConditionalAccumulator.accumulator_ref}

The underlying accumulator reference.


- - -

#### `tf.SparseConditionalAccumulator.apply_grad(grad_indices, grad_values, grad_shape=None, local_step=0, name=None)` {#SparseConditionalAccumulator.apply_grad}

Attempts to apply a sparse gradient to the accumulator.

The attempt is silently dropped if the gradient is stale, i.e., local_step
is less than the accumulator's global time step.

A sparse gradient is represented by its indices, values and possibly empty
or None shape. Indices must be a vector representing the locations of
non-zero entries in the tensor. Values are the non-zero slices of the
gradient, and must have the same first dimension as indices, i.e., the nnz
represented by indices and values must be consistent. Shape, if not empty or
None, must be consistent with the accumulator's shape (if also provided).

##### Example:

  A tensor [[0, 0], [0. 1], [2, 3]] can be represented

*  <b>`indices`</b>: [1,2]
*  <b>`values`</b>: [[0,1],[2,3]]
*  <b>`shape`</b>: [3, 2]

##### Args:


*  <b>`grad_indices`</b>: Indices of the sparse gradient to be applied.
*  <b>`grad_values`</b>: Values of the sparse gradient to be applied.
*  <b>`grad_shape`</b>: Shape of the sparse gradient to be applied.
*  <b>`local_step`</b>: Time step at which the gradient was computed.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  The operation that (conditionally) applies a gradient to the accumulator.

##### Raises:


*  <b>`InvalidArgumentError`</b>: If grad is of the wrong shape


- - -

#### `tf.SparseConditionalAccumulator.apply_indexed_slices_grad(grad, local_step=0, name=None)` {#SparseConditionalAccumulator.apply_indexed_slices_grad}

Attempts to apply a gradient to the accumulator.

The attempt is silently dropped if the gradient is stale, i.e., local_step
is less than the accumulator's global time step.

##### Args:


*  <b>`grad`</b>: The gradient IndexedSlices to be applied.
*  <b>`local_step`</b>: Time step at which the gradient was computed.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  The operation that (conditionally) applies a gradient to the accumulator.

##### Raises:


*  <b>`InvalidArgumentError`</b>: If grad is of the wrong shape


- - -

#### `tf.SparseConditionalAccumulator.dtype` {#SparseConditionalAccumulator.dtype}

The datatype of the gradients accumulated by this accumulator.


- - -

#### `tf.SparseConditionalAccumulator.name` {#SparseConditionalAccumulator.name}

The name of the underlying accumulator.


- - -

#### `tf.SparseConditionalAccumulator.num_accumulated(name=None)` {#SparseConditionalAccumulator.num_accumulated}

Number of gradients that have currently been aggregated in accumulator.

##### Args:


*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Number of accumulated gradients currently in accumulator.


- - -

#### `tf.SparseConditionalAccumulator.set_global_step(new_global_step, name=None)` {#SparseConditionalAccumulator.set_global_step}

Sets the global time step of the accumulator.

The operation logs a warning if we attempt to set to a time step that is
lower than the accumulator's own time step.

##### Args:


*  <b>`new_global_step`</b>: Value of new time step. Can be a variable or a constant
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Operation that sets the accumulator's time step.


- - -

#### `tf.SparseConditionalAccumulator.take_grad(num_required, name=None)` {#SparseConditionalAccumulator.take_grad}

Attempts to extract the average gradient from the accumulator.

The operation blocks until sufficient number of gradients have been
successfully applied to the accumulator.

Once successful, the following actions are also triggered:
- Counter of accumulated gradients is reset to 0.
- Aggregated gradient is reset to 0 tensor.
- Accumulator's internal time step is incremented by 1.

##### Args:


*  <b>`num_required`</b>: Number of gradients that needs to have been aggregated
*  <b>`name`</b>: Optional name for the operation

##### Returns:

  A tuple of indices, values, and shape representing the average gradient.

##### Raises:


*  <b>`InvalidArgumentError`</b>: If num_required < 1


- - -

#### `tf.SparseConditionalAccumulator.take_indexed_slices_grad(num_required, name=None)` {#SparseConditionalAccumulator.take_indexed_slices_grad}

Attempts to extract the average gradient from the accumulator.

The operation blocks until sufficient number of gradients have been
successfully applied to the accumulator.

Once successful, the following actions are also triggered:
- Counter of accumulated gradients is reset to 0.
- Aggregated gradient is reset to 0 tensor.
- Accumulator's internal time step is incremented by 1.

##### Args:


*  <b>`num_required`</b>: Number of gradients that needs to have been aggregated
*  <b>`name`</b>: Optional name for the operation

##### Returns:

  An IndexedSlices holding the value of the average gradient.

##### Raises:


*  <b>`InvalidArgumentError`</b>: If num_required < 1




## Dealing with the filesystem

- - -

### `tf.matching_files(pattern, name=None)` {#matching_files}

Returns the set of files matching a pattern.

Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion.

##### Args:


*  <b>`pattern`</b>: A `Tensor` of type `string`. A (scalar) shell wildcard pattern.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. A vector of matching filenames.


- - -

### `tf.read_file(filename, name=None)` {#read_file}

Reads and outputs the entire contents of the input filename.

##### Args:


*  <b>`filename`</b>: A `Tensor` of type `string`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`.


- - -

### `tf.write_file(filename, contents, name=None)` {#write_file}

Writes contents to the file at input filename. Creates file if not existing.

##### Args:


*  <b>`filename`</b>: A `Tensor` of type `string`.
    scalar. The name of the file to which we write the contents.
*  <b>`contents`</b>: A `Tensor` of type `string`.
    scalar. The content to be written to the output file.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.



## Input pipeline

TensorFlow functions for setting up an input-prefetching pipeline.
Please see the [reading data how-to](../../how_tos/reading_data/index.md)
for context.

### Beginning of an input pipeline

The "producer" functions add a queue to the graph and a corresponding
`QueueRunner` for running the subgraph that fills that queue.

- - -

### `tf.train.match_filenames_once(pattern, name=None)` {#match_filenames_once}

Save the list of files matching pattern, so it is only computed once.

##### Args:


*  <b>`pattern`</b>: A file pattern (glob).
*  <b>`name`</b>: A name for the operations (optional).

##### Returns:

  A variable that is initialized to the list of files matching pattern.


- - -

### `tf.train.limit_epochs(tensor, num_epochs=None, name=None)` {#limit_epochs}

Returns tensor `num_epochs` times and then raises an `OutOfRange` error.

Note: creates local counter `epochs`. Use `local_variable_initializer()` to
initialize local variables.

##### Args:


*  <b>`tensor`</b>: Any `Tensor`.
*  <b>`num_epochs`</b>: A positive integer (optional).  If specified, limits the number
    of steps the output tensor may be evaluated.
*  <b>`name`</b>: A name for the operations (optional).

##### Returns:

  tensor or `OutOfRange`.

##### Raises:


*  <b>`ValueError`</b>: if `num_epochs` is invalid.


- - -

### `tf.train.input_producer(input_tensor, element_shape=None, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, summary_name=None, name=None, cancel_op=None)` {#input_producer}

Output the rows of `input_tensor` to a queue for an input pipeline.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variable_initializer()` to initialize local variables.

##### Args:


*  <b>`input_tensor`</b>: A tensor with the rows to produce. Must be at least
    one-dimensional. Must either have a fully-defined shape, or
    `element_shape` must be defined.
*  <b>`element_shape`</b>: (Optional.) A `TensorShape` representing the shape of a
    row of `input_tensor`, if it cannot be inferred.
*  <b>`num_epochs`</b>: (Optional.) An integer. If specified `input_producer` produces
    each row of `input_tensor` `num_epochs` times before generating an
    `OutOfRange` error. If not specified, `input_producer` can cycle through
    the rows of `input_tensor` an unlimited number of times.
*  <b>`shuffle`</b>: (Optional.) A boolean. If true, the rows are randomly shuffled
    within each epoch.
*  <b>`seed`</b>: (Optional.) An integer. The seed to use if `shuffle` is true.
*  <b>`capacity`</b>: (Optional.) The capacity of the queue to be used for buffering
    the input.
*  <b>`shared_name`</b>: (Optional.) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`summary_name`</b>: (Optional.) If set, a scalar summary for the current queue
    size will be generated, using this name as part of the tag.
*  <b>`name`</b>: (Optional.) A name for queue.
*  <b>`cancel_op`</b>: (Optional.) Cancel op for the queue

##### Returns:

  A queue with the output rows.  A `QueueRunner` for the queue is
  added to the current `QUEUE_RUNNER` collection of the current
  graph.

##### Raises:


*  <b>`ValueError`</b>: If the shape of the input cannot be inferred from the arguments.


- - -

### `tf.train.range_input_producer(limit, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None)` {#range_input_producer}

Produces the integers from 0 to limit-1 in a queue.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variable_initializer()` to initialize local variables.

##### Args:


*  <b>`limit`</b>: An int32 scalar tensor.
*  <b>`num_epochs`</b>: An integer (optional). If specified, `range_input_producer`
    produces each integer `num_epochs` times before generating an
    OutOfRange error. If not specified, `range_input_producer` can cycle
    through the integers an unlimited number of times.
*  <b>`shuffle`</b>: Boolean. If true, the integers are randomly shuffled within each
    epoch.
*  <b>`seed`</b>: An integer (optional). Seed used if shuffle == True.
*  <b>`capacity`</b>: An integer. Sets the queue capacity.
*  <b>`shared_name`</b>: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: A name for the operations (optional).

##### Returns:

  A Queue with the output integers.  A `QueueRunner` for the Queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.


- - -

### `tf.train.slice_input_producer(tensor_list, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None)` {#slice_input_producer}

Produces a slice of each `Tensor` in `tensor_list`.

Implemented using a Queue -- a `QueueRunner` for the Queue
is added to the current `Graph`'s `QUEUE_RUNNER` collection.

##### Args:


*  <b>`tensor_list`</b>: A list of `Tensor` objects. Every `Tensor` in
    `tensor_list` must have the same size in the first dimension.
*  <b>`num_epochs`</b>: An integer (optional). If specified, `slice_input_producer`
    produces each slice `num_epochs` times before generating
    an `OutOfRange` error. If not specified, `slice_input_producer` can cycle
    through the slices an unlimited number of times.
*  <b>`shuffle`</b>: Boolean. If true, the integers are randomly shuffled within each
    epoch.
*  <b>`seed`</b>: An integer (optional). Seed used if shuffle == True.
*  <b>`capacity`</b>: An integer. Sets the queue capacity.
*  <b>`shared_name`</b>: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: A name for the operations (optional).

##### Returns:

  A list of tensors, one for each element of `tensor_list`.  If the tensor
  in `tensor_list` has shape `[N, a, b, .., z]`, then the corresponding output
  tensor will have shape `[a, b, ..., z]`.

##### Raises:


*  <b>`ValueError`</b>: if `slice_input_producer` produces nothing from `tensor_list`.


- - -

### `tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None, cancel_op=None)` {#string_input_producer}

Output strings (e.g. filenames) to a queue for an input pipeline.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variable_initializer()` to initialize local variables.

##### Args:


*  <b>`string_tensor`</b>: A 1-D string tensor with the strings to produce.
*  <b>`num_epochs`</b>: An integer (optional). If specified, `string_input_producer`
    produces each string from `string_tensor` `num_epochs` times before
    generating an `OutOfRange` error. If not specified,
    `string_input_producer` can cycle through the strings in `string_tensor`
    an unlimited number of times.
*  <b>`shuffle`</b>: Boolean. If true, the strings are randomly shuffled within each
    epoch.
*  <b>`seed`</b>: An integer (optional). Seed used if shuffle == True.
*  <b>`capacity`</b>: An integer. Sets the queue capacity.
*  <b>`shared_name`</b>: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: A name for the operations (optional).
*  <b>`cancel_op`</b>: Cancel op for the queue (optional).

##### Returns:

  A queue with the output strings.  A `QueueRunner` for the Queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

##### Raises:


*  <b>`ValueError`</b>: If the string_tensor is a null Python list.  At runtime,
  will fail with an assertion if string_tensor becomes a null tensor.



### Batching at the end of an input pipeline

These functions add a queue to the graph to assemble a batch of
examples, with possible shuffling.  They also add a `QueueRunner` for
running the subgraph that fills that queue.

Use [`batch`](#batch) or [`batch_join`](#batch_join) for batching
examples that have already been well shuffled.  Use
[`shuffle_batch`](#shuffle_batch) or
[`shuffle_batch_join`](#shuffle_batch_join) for examples that would
benefit from additional shuffling.

Use [`batch`](#batch) or [`shuffle_batch`](#shuffle_batch) if you want a
single thread producing examples to batch, or if you have a
single subgraph producing examples but you want to run it in *N* threads
(where you increase *N* until it can keep the queue full).  Use
[`batch_join`](#batch_join) or [`shuffle_batch_join`](#shuffle_batch_join)
if you have *N* different subgraphs producing examples to batch and you
want them run by *N* threads. Use `maybe_*` to enqueue conditionally.

- - -

### `tf.train.batch(tensors, batch_size, num_threads=1, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)` {#batch}

Creates batches of tensors in `tensors`.

The argument `tensors` can be a list or a dictionary of tensors.
The value returned by the function will be of the same type
as `tensors`.

This function is implemented using a queue. A `QueueRunner` for the
queue is added to the current `Graph`'s `QUEUE_RUNNER` collection.

If `enqueue_many` is `False`, `tensors` is assumed to represent a single
example.  An input tensor with shape `[x, y, z]` will be output as a tensor
with shape `[batch_size, x, y, z]`.

If `enqueue_many` is `True`, `tensors` is assumed to represent a batch of
examples, where the first dimension is indexed by example, and all members of
`tensors` should have the same size in the first dimension.  If an input
tensor has shape `[*, x, y, z]`, the output will have shape `[batch_size, x,
y, z]`.  The `capacity` argument controls the how long the prefetching is
allowed to grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

*N.B.:* If `dynamic_pad` is `False`, you must ensure that either
(i) the `shapes` argument is passed, or (ii) all of the tensors in
`tensors` must have fully-defined shapes. `ValueError` will be
raised if neither of these conditions holds.

If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
tensors is known, but individual dimensions may have shape `None`.
In this case, for each enqueue the dimensions with value `None`
may have a variable length; upon dequeue, the output tensors will be padded
on the right to the maximum shape of the tensors in the current minibatch.
For numbers, this padding takes value 0.  For strings, this padding is
the empty string.  See `PaddingFIFOQueue` for more info.

If `allow_smaller_final_batch` is `True`, a smaller batch value than
`batch_size` is returned when the queue is closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`get_shape` method will have a first `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variable_initializer()` to initialize local variables.

##### Args:


*  <b>`tensors`</b>: The list or dictionary of tensors to enqueue.
*  <b>`batch_size`</b>: The new batch size pulled from the queue.
*  <b>`num_threads`</b>: The number of threads enqueuing `tensors`.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensors` is a single example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same types as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.


- - -

### `tf.train.maybe_batch(tensors, keep_input, batch_size, num_threads=1, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)` {#maybe_batch}

Conditionally creates batches of tensors based on `keep_input`.

See docstring in `batch` for more details.

##### Args:


*  <b>`tensors`</b>: The list or dictionary of tensors to enqueue.
*  <b>`keep_input`</b>: A `bool` scalar Tensor.  This tensor controls whether the input
    is added to the queue or not.  If it evaluates `True`, then `tensors` are
    added to the queue; otherwise they are dropped.  This tensor essentially
    acts as a filtering mechanism.
*  <b>`batch_size`</b>: The new batch size pulled from the queue.
*  <b>`num_threads`</b>: The number of threads enqueuing `tensors`.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensors` is a single example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same types as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.


- - -

### `tf.train.batch_join(tensors_list, batch_size, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)` {#batch_join}

Runs a list of tensors to fill a queue to create batches of examples.

The `tensors_list` argument is a list of tuples of tensors, or a list of
dictionaries of tensors.  Each element in the list is treated similarly
to the `tensors` argument of `tf.train.batch()`.

Enqueues a different list of tensors in different threads.
Implemented using a queue -- a `QueueRunner` for the queue
is added to the current `Graph`'s `QUEUE_RUNNER` collection.

`len(tensors_list)` threads will be started,
with thread `i` enqueuing the tensors from
`tensors_list[i]`. `tensors_list[i1][j]` must match
`tensors_list[i2][j]` in type and shape, except in the first
dimension if `enqueue_many` is true.

If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
to represent a single example. An input tensor `x` will be output as a
tensor with shape `[batch_size] + x.shape`.

If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
represent a batch of examples, where the first dimension is indexed
by example, and all members of `tensors_list[i]` should have the
same size in the first dimension.  The slices of any input tensor
`x` are treated as examples, and the output tensors will have shape
`[batch_size] + x.shape[1:]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

*N.B.:* If `dynamic_pad` is `False`, you must ensure that either
(i) the `shapes` argument is passed, or (ii) all of the tensors in
`tensors_list` must have fully-defined shapes. `ValueError` will be
raised if neither of these conditions holds.

If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
tensors is known, but individual dimensions may have value `None`.
In this case, for each enqueue the dimensions with value `None`
may have a variable length; upon dequeue, the output tensors will be padded
on the right to the maximum shape of the tensors in the current minibatch.
For numbers, this padding takes value 0.  For strings, this padding is
the empty string.  See `PaddingFIFOQueue` for more info.

If `allow_smaller_final_batch` is `True`, a smaller batch value than
`batch_size` is returned when the queue is closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`get_shape` method will have a first `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

##### Args:


*  <b>`tensors_list`</b>: A list of tuples or dictionaries of tensors to enqueue.
*  <b>`batch_size`</b>: An integer. The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list_list` is a single
    example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list_list[i]`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensor_list_list`.


- - -

### `tf.train.maybe_batch_join(tensors_list, keep_input, batch_size, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)` {#maybe_batch_join}

Runs a list of tensors to conditionally fill a queue to create batches.

See docstring in `batch_join` for more details.

##### Args:


*  <b>`tensors_list`</b>: A list of tuples or dictionaries of tensors to enqueue.
*  <b>`keep_input`</b>: A `bool` scalar Tensor.  This tensor controls whether the input
    is added to the queue or not.  If it evaluates `True`, then `tensors` are
    added to the queue; otherwise they are dropped.  This tensor essentially
    acts as a filtering mechanism.
*  <b>`batch_size`</b>: An integer. The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list_list` is a single
    example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list_list[i]`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensor_list_list`.


- - -

### `tf.train.shuffle_batch(tensors, batch_size, capacity, min_after_dequeue, num_threads=1, seed=None, enqueue_many=False, shapes=None, allow_smaller_final_batch=False, shared_name=None, name=None)` {#shuffle_batch}

Creates batches by randomly shuffling tensors.

This function adds the following to the current `Graph`:

* A shuffling queue into which tensors from `tensors` are enqueued.
* A `dequeue_many` operation to create batches from the queue.
* A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
  from `tensors`.

If `enqueue_many` is `False`, `tensors` is assumed to represent a
single example.  An input tensor with shape `[x, y, z]` will be output
as a tensor with shape `[batch_size, x, y, z]`.

If `enqueue_many` is `True`, `tensors` is assumed to represent a
batch of examples, where the first dimension is indexed by example,
and all members of `tensors` should have the same size in the
first dimension.  If an input tensor has shape `[*, x, y, z]`, the
output will have shape `[batch_size, x, y, z]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

For example:

```python
# Creates batches of 32 images and 32 labels.
image_batch, label_batch = tf.train.shuffle_batch(
      [single_image, single_label],
      batch_size=32,
      num_threads=4,
      capacity=50000,
      min_after_dequeue=10000)
```

*N.B.:* You must ensure that either (i) the `shapes` argument is
passed, or (ii) all of the tensors in `tensors` must have
fully-defined shapes. `ValueError` will be raised if neither of
these conditions holds.

If `allow_smaller_final_batch` is `True`, a smaller batch value than
`batch_size` is returned when the queue is closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`get_shape` method will have a first `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variable_initializer()` to initialize local variables.

##### Args:


*  <b>`tensors`</b>: The list or dictionary of tensors to enqueue.
*  <b>`batch_size`</b>: The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`min_after_dequeue`</b>: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
*  <b>`num_threads`</b>: The number of threads enqueuing `tensor_list`.
*  <b>`seed`</b>: Seed for the random shuffling within the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list` is a single example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list`.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the types as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.


- - -

### `tf.train.maybe_shuffle_batch(tensors, batch_size, capacity, min_after_dequeue, keep_input, num_threads=1, seed=None, enqueue_many=False, shapes=None, allow_smaller_final_batch=False, shared_name=None, name=None)` {#maybe_shuffle_batch}

Creates batches by randomly shuffling conditionally-enqueued tensors.

See docstring in `shuffle_batch` for more details.

##### Args:


*  <b>`tensors`</b>: The list or dictionary of tensors to enqueue.
*  <b>`batch_size`</b>: The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`min_after_dequeue`</b>: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
*  <b>`keep_input`</b>: A `bool` scalar Tensor.  This tensor controls whether the input
    is added to the queue or not.  If it evaluates `True`, then `tensors` are
    added to the queue; otherwise they are dropped.  This tensor essentially
    acts as a filtering mechanism.
*  <b>`num_threads`</b>: The number of threads enqueuing `tensor_list`.
*  <b>`seed`</b>: Seed for the random shuffling within the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list` is a single example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list`.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the types as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.


- - -

### `tf.train.shuffle_batch_join(tensors_list, batch_size, capacity, min_after_dequeue, seed=None, enqueue_many=False, shapes=None, allow_smaller_final_batch=False, shared_name=None, name=None)` {#shuffle_batch_join}

Create batches by randomly shuffling tensors.

The `tensors_list` argument is a list of tuples of tensors, or a list of
dictionaries of tensors.  Each element in the list is treated similarly
to the `tensors` argument of `tf.train.shuffle_batch()`.

This version enqueues a different list of tensors in different threads.
It adds the following to the current `Graph`:

* A shuffling queue into which tensors from `tensors_list` are enqueued.
* A `dequeue_many` operation to create batches from the queue.
* A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
  from `tensors_list`.

`len(tensors_list)` threads will be started, with thread `i` enqueuing
the tensors from `tensors_list[i]`. `tensors_list[i1][j]` must match
`tensors_list[i2][j]` in type and shape, except in the first dimension if
`enqueue_many` is true.

If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
to represent a single example.  An input tensor with shape `[x, y, z]`
will be output as a tensor with shape `[batch_size, x, y, z]`.

If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
represent a batch of examples, where the first dimension is indexed
by example, and all members of `tensors_list[i]` should have the
same size in the first dimension.  If an input tensor has shape `[*, x,
y, z]`, the output will have shape `[batch_size, x, y, z]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

If `allow_smaller_final_batch` is `True`, a smaller batch value than
`batch_size` is returned when the queue is closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`get_shape` method will have a first `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

##### Args:


*  <b>`tensors_list`</b>: A list of tuples or dictionaries of tensors to enqueue.
*  <b>`batch_size`</b>: An integer. The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`min_after_dequeue`</b>: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
*  <b>`seed`</b>: Seed for the random shuffling within the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list_list` is a single
    example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors_list[i]`.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors_list`.


- - -

### `tf.train.maybe_shuffle_batch_join(tensors_list, batch_size, capacity, min_after_dequeue, keep_input, seed=None, enqueue_many=False, shapes=None, allow_smaller_final_batch=False, shared_name=None, name=None)` {#maybe_shuffle_batch_join}

Create batches by randomly shuffling conditionally-enqueued tensors.

See docstring in `shuffle_batch_join` for more details.

##### Args:


*  <b>`tensors_list`</b>: A list of tuples or dictionaries of tensors to enqueue.
*  <b>`batch_size`</b>: An integer. The new batch size pulled from the queue.
*  <b>`capacity`</b>: An integer. The maximum number of elements in the queue.
*  <b>`min_after_dequeue`</b>: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
*  <b>`keep_input`</b>: A `bool` scalar Tensor.  If provided, this tensor controls
    whether the input is added to the queue or not.  If it evaluates `True`,
    then `tensors_list` are added to the queue; otherwise they are dropped.
    This tensor essentially acts as a filtering mechanism.
*  <b>`seed`</b>: Seed for the random shuffling within the queue.
*  <b>`enqueue_many`</b>: Whether each tensor in `tensor_list_list` is a single
    example.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors_list[i]`.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
*  <b>`shared_name`</b>: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors_list`.


