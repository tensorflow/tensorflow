<!-- This file is machine generated: DO NOT EDIT! -->

# Summary Operations
[TOC]

## Generation of summaries.

### Class for writing Summaries
- - -

### `class tf.summary.FileWriter` {#FileWriter}

Writes `Summary` protocol buffers to event files.

The `FileWriter` class provides a mechanism to create an event file in a
given directory and add summaries and events to it. The class updates the
file contents asynchronously. This allows a training program to call methods
to add data to the file directly from the training loop, without slowing down
training.

- - -

#### `tf.summary.FileWriter.__init__(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)` {#FileWriter.__init__}

Creates a `FileWriter` and an event file.

On construction the summary writer creates a new event file in `logdir`.
This event file will contain `Event` protocol buffers constructed when you
call one of the following functions: `add_summary()`, `add_session_log()`,
`add_event()`, or `add_graph()`.

If you pass a `Graph` to the constructor it is added to
the event file. (This is equivalent to calling `add_graph()` later).

TensorBoard will pick the graph from the file and display it graphically so
you can interactively explore the graph you built. You will usually pass
the graph from the session in which you launched it:

```python
...create a graph...
# Launch the graph in a session.
sess = tf.Session()
# Create a summary writer, add the 'graph' to the event file.
writer = tf.train.SummaryWriter(<some-directory>, sess.graph)
```

The other arguments to the constructor control the asynchronous writes to
the event file:

*  `flush_secs`: How often, in seconds, to flush the added summaries
   and events to disk.
*  `max_queue`: Maximum number of summaries or events pending to be
   written to disk before one of the 'add' calls block.

##### Args:


*  <b>`logdir`</b>: A string. Directory where event file will be written.
*  <b>`graph`</b>: A `Graph` object, such as `sess.graph`.
*  <b>`max_queue`</b>: Integer. Size of the queue for pending events and summaries.
*  <b>`flush_secs`</b>: Number. How often, in seconds, to flush the
    pending events and summaries to disk.
*  <b>`graph_def`</b>: DEPRECATED: Use the `graph` argument instead.



- - -

#### `tf.summary.FileWriter.add_summary(summary, global_step=None)` {#FileWriter.add_summary}

Adds a `Summary` protocol buffer to the event file.

This method wraps the provided summary in an `Event` protocol buffer
and adds it to the event file.

You can pass the result of evaluating any summary op, using
[`Session.run()`](client.md#Session.run) or
[`Tensor.eval()`](framework.md#Tensor.eval), to this
function. Alternatively, you can pass a `tf.Summary` protocol
buffer that you populate with your own data. The latter is
commonly done to report evaluation results in event files.

##### Args:


*  <b>`summary`</b>: A `Summary` protocol buffer, optionally serialized as a string.
*  <b>`global_step`</b>: Number. Optional global step value to record with the
    summary.


- - -

#### `tf.summary.FileWriter.add_session_log(session_log, global_step=None)` {#FileWriter.add_session_log}

Adds a `SessionLog` protocol buffer to the event file.

This method wraps the provided session in an `Event` protocol buffer
and adds it to the event file.

##### Args:


*  <b>`session_log`</b>: A `SessionLog` protocol buffer.
*  <b>`global_step`</b>: Number. Optional global step value to record with the
    summary.


- - -

#### `tf.summary.FileWriter.add_event(event)` {#FileWriter.add_event}

Adds an event to the event file.

##### Args:


*  <b>`event`</b>: An `Event` protocol buffer.


- - -

#### `tf.summary.FileWriter.add_graph(graph, global_step=None, graph_def=None)` {#FileWriter.add_graph}

Adds a `Graph` to the event file.

The graph described by the protocol buffer will be displayed by
TensorBoard. Most users pass a graph in the constructor instead.

##### Args:


*  <b>`graph`</b>: A `Graph` object, such as `sess.graph`.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    graph.
*  <b>`graph_def`</b>: DEPRECATED. Use the `graph` parameter instead.

##### Raises:


*  <b>`ValueError`</b>: If both graph and graph_def are passed to the method.


- - -

#### `tf.summary.FileWriter.add_run_metadata(run_metadata, tag, global_step=None)` {#FileWriter.add_run_metadata}

Adds a metadata information for a single session.run() call.

##### Args:


*  <b>`run_metadata`</b>: A `RunMetadata` protobuf object.
*  <b>`tag`</b>: The tag name for this metadata.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    StepStats.

##### Raises:


*  <b>`ValueError`</b>: If the provided tag was already used for this type of event.


- - -

#### `tf.summary.FileWriter.get_logdir()` {#FileWriter.get_logdir}

Returns the directory where event file will be written.



- - -

#### `tf.summary.FileWriter.flush()` {#FileWriter.flush}

Flushes the event file to disk.

Call this method to make sure that all pending events have been written to
disk.


- - -

#### `tf.summary.FileWriter.close()` {#FileWriter.close}

Flushes the event file to disk and close the file.

Call this method when you do not need the summary writer anymore.



#### Other Methods
- - -

#### `tf.summary.FileWriter.reopen()` {#FileWriter.reopen}

Reopens the EventFileWriter.

Can be called after `close()` to add more events in the same directory.
The events will go into a new events file.

Does nothing if the EventFileWriter was not closed.




### Summary Ops
- - -

### `tf.summary.tensor_summary(name, tensor, summary_description=None, collections=None)` {#tensor_summary}

Outputs a `Summary` protocol buffer with a serialized tensor.proto.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing the input tensor.

##### Args:


*  <b>`name`</b>: A name for the generated node. Will also serve as the series name in
    TensorBoard.
*  <b>`tensor`</b>: A tensor of any type and shape to serialize.
*  <b>`summary_description`</b>: Optional summary_pb2.SummaryDescription()
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


- - -

### `tf.summary.scalar(name, tensor, collections=None)` {#scalar}

Outputs a `Summary` protocol buffer containing a single scalar value.

The generated Summary has a Tensor.proto containing the input Tensor.

##### Args:


*  <b>`name`</b>: A name for the generated node. Will also serve as the series name in
    TensorBoard.
*  <b>`tensor`</b>: A real numeric Tensor containing a single value.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

##### Returns:

  A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.

##### Raises:


*  <b>`ValueError`</b>: If tensor has the wrong shape or type.


- - -

### `tf.summary.histogram(name, values, collections=None)` {#histogram}

Outputs a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

##### Args:


*  <b>`name`</b>: A name for the generated node. Will also serve as a series name in
    TensorBoard.
*  <b>`values`</b>: A real numeric `Tensor`. Any shape. Values to use to
    build the histogram.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


- - -

### `tf.summary.audio(name, tensor, sample_rate, max_outputs=3, collections=None)` {#audio}

Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
`sample_rate`.

The `tag` in the outputted Summary.Value protobufs is generated based on the
name, with a suffix depending on the max_outputs setting:

*  If `max_outputs` is 1, the summary value tag is '*name*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*name*/audio/0', '*name*/audio/1', etc

##### Args:


*  <b>`name`</b>: A name for the generated node. Will also serve as a series name in
    TensorBoard.
*  <b>`tensor`</b>: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
    or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
*  <b>`sample_rate`</b>: A Scalar `float32` `Tensor` indicating the sample rate of the
    signal in hertz.
*  <b>`max_outputs`</b>: Max number of batch elements to generate audio for.
*  <b>`collections`</b>: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


- - -

### `tf.summary.image(name, tensor, max_outputs=3, collections=None)` {#image}

Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:

*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. For float
input, the values are normalized one image at a time to fit in the range
`[0, 255]`.  `uint8` values are unchanged.  The op uses two different
normalization algorithms:

*  If the input values are all positive, they are rescaled so the largest one
   is 255.

*  If any input value is negative, the values are shifted so input value 0.0
   is at 127.  They are then rescaled so that either the smallest value is 0,
   or the largest one is 255.

The `tag` in the outputted Summary.Value protobufs is generated based on the
name, with a suffix depending on the max_outputs setting:

*  If `max_outputs` is 1, the summary value tag is '*name*/image'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*name*/image/0', '*name*/image/1', etc.

##### Args:


*  <b>`name`</b>: A name for the generated node. Will also serve as a series name in
    TensorBoard.
*  <b>`tensor`</b>: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
    width, channels]` where `channels` is 1, 3, or 4.
*  <b>`max_outputs`</b>: Max number of batch elements to generate images for.
*  <b>`collections`</b>: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


- - -

### `tf.summary.merge(inputs, collections=None, name=None)` {#merge}

Merges summaries.

This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.

##### Args:


*  <b>`inputs`</b>: A list of `string` `Tensor` objects containing serialized `Summary`
    protocol buffers.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer resulting from the merging.


- - -

### `tf.summary.merge_all(key='summaries')` {#merge_all}

Merges all summaries collected in the default graph.

##### Args:


*  <b>`key`</b>: `GraphKey` used to collect the summaries.  Defaults to
    `GraphKeys.SUMMARIES`.

##### Returns:

  If no summaries were collected, returns None.  Otherwise returns a scalar
  `Tensor` of type `string` containing the serialized `Summary` protocol
  buffer resulting from the merging.



## Utilities
- - -

### `tf.summary.get_summary_description(node_def)` {#get_summary_description}

Given a TensorSummary node_def, retrieve its SummaryDescription.

When a Summary op is instantiated, a SummaryDescription of associated
metadata is stored in its NodeDef. This method retrieves the description.

##### Args:


*  <b>`node_def`</b>: the node_def_pb2.NodeDef of a TensorSummary op

##### Returns:

  a summary_pb2.SummaryDescription

##### Raises:


*  <b>`ValueError`</b>: if the node is not a summary op.



## Other Functions and Classes
- - -

### `class tf.summary.SummaryDescription` {#SummaryDescription}


- - -

#### `tf.summary.SummaryDescription.ByteSize()` {#SummaryDescription.ByteSize}




- - -

#### `tf.summary.SummaryDescription.Clear()` {#SummaryDescription.Clear}




- - -

#### `tf.summary.SummaryDescription.ClearExtension(extension_handle)` {#SummaryDescription.ClearExtension}




- - -

#### `tf.summary.SummaryDescription.ClearField(field_name)` {#SummaryDescription.ClearField}




- - -

#### `tf.summary.SummaryDescription.CopyFrom(other_msg)` {#SummaryDescription.CopyFrom}

Copies the content of the specified message into the current message.

The method clears the current message and then merges the specified
message using MergeFrom.

##### Args:


*  <b>`other_msg`</b>: Message to copy into the current one.


- - -

#### `tf.summary.SummaryDescription.DiscardUnknownFields()` {#SummaryDescription.DiscardUnknownFields}




- - -

#### `tf.summary.SummaryDescription.FindInitializationErrors()` {#SummaryDescription.FindInitializationErrors}

Finds required fields which are not initialized.

##### Returns:

  A list of strings.  Each string is a path to an uninitialized field from
  the top-level message, e.g. "foo.bar[5].baz".


- - -

#### `tf.summary.SummaryDescription.FromString(s)` {#SummaryDescription.FromString}




- - -

#### `tf.summary.SummaryDescription.HasExtension(extension_handle)` {#SummaryDescription.HasExtension}




- - -

#### `tf.summary.SummaryDescription.HasField(field_name)` {#SummaryDescription.HasField}




- - -

#### `tf.summary.SummaryDescription.IsInitialized(errors=None)` {#SummaryDescription.IsInitialized}

Checks if all required fields of a message are set.

##### Args:


*  <b>`errors`</b>: A list which, if provided, will be populated with the field
           paths of all missing required fields.

##### Returns:

  True iff the specified message has all required fields set.


- - -

#### `tf.summary.SummaryDescription.ListFields()` {#SummaryDescription.ListFields}




- - -

#### `tf.summary.SummaryDescription.MergeFrom(msg)` {#SummaryDescription.MergeFrom}




- - -

#### `tf.summary.SummaryDescription.MergeFromString(serialized)` {#SummaryDescription.MergeFromString}




- - -

#### `tf.summary.SummaryDescription.ParseFromString(serialized)` {#SummaryDescription.ParseFromString}

Parse serialized protocol buffer data into this message.

Like MergeFromString(), except we clear the object first and
do not return the value that MergeFromString returns.


- - -

#### `tf.summary.SummaryDescription.RegisterExtension(extension_handle)` {#SummaryDescription.RegisterExtension}




- - -

#### `tf.summary.SummaryDescription.SerializePartialToString()` {#SummaryDescription.SerializePartialToString}




- - -

#### `tf.summary.SummaryDescription.SerializeToString()` {#SummaryDescription.SerializeToString}




- - -

#### `tf.summary.SummaryDescription.SetInParent()` {#SummaryDescription.SetInParent}

Sets the _cached_byte_size_dirty bit to true,
and propagates this to our listener iff this was a state change.


- - -

#### `tf.summary.SummaryDescription.WhichOneof(oneof_name)` {#SummaryDescription.WhichOneof}

Returns the name of the currently set field inside a oneof, or None.


- - -

#### `tf.summary.SummaryDescription.__deepcopy__(memo=None)` {#SummaryDescription.__deepcopy__}




- - -

#### `tf.summary.SummaryDescription.__eq__(other)` {#SummaryDescription.__eq__}




- - -

#### `tf.summary.SummaryDescription.__getstate__()` {#SummaryDescription.__getstate__}

Support the pickle protocol.


- - -

#### `tf.summary.SummaryDescription.__hash__()` {#SummaryDescription.__hash__}




- - -

#### `tf.summary.SummaryDescription.__init__(**kwargs)` {#SummaryDescription.__init__}




- - -

#### `tf.summary.SummaryDescription.__ne__(other_msg)` {#SummaryDescription.__ne__}




- - -

#### `tf.summary.SummaryDescription.__repr__()` {#SummaryDescription.__repr__}




- - -

#### `tf.summary.SummaryDescription.__setstate__(state)` {#SummaryDescription.__setstate__}

Support the pickle protocol.


- - -

#### `tf.summary.SummaryDescription.__str__()` {#SummaryDescription.__str__}




- - -

#### `tf.summary.SummaryDescription.__unicode__()` {#SummaryDescription.__unicode__}




- - -

#### `tf.summary.SummaryDescription.type_hint` {#SummaryDescription.type_hint}

Magic attribute generated for "type_hint" proto field.



- - -

### `class tf.summary.TaggedRunMetadata` {#TaggedRunMetadata}


- - -

#### `tf.summary.TaggedRunMetadata.ByteSize()` {#TaggedRunMetadata.ByteSize}




- - -

#### `tf.summary.TaggedRunMetadata.Clear()` {#TaggedRunMetadata.Clear}




- - -

#### `tf.summary.TaggedRunMetadata.ClearExtension(extension_handle)` {#TaggedRunMetadata.ClearExtension}




- - -

#### `tf.summary.TaggedRunMetadata.ClearField(field_name)` {#TaggedRunMetadata.ClearField}




- - -

#### `tf.summary.TaggedRunMetadata.CopyFrom(other_msg)` {#TaggedRunMetadata.CopyFrom}

Copies the content of the specified message into the current message.

The method clears the current message and then merges the specified
message using MergeFrom.

##### Args:


*  <b>`other_msg`</b>: Message to copy into the current one.


- - -

#### `tf.summary.TaggedRunMetadata.DiscardUnknownFields()` {#TaggedRunMetadata.DiscardUnknownFields}




- - -

#### `tf.summary.TaggedRunMetadata.FindInitializationErrors()` {#TaggedRunMetadata.FindInitializationErrors}

Finds required fields which are not initialized.

##### Returns:

  A list of strings.  Each string is a path to an uninitialized field from
  the top-level message, e.g. "foo.bar[5].baz".


- - -

#### `tf.summary.TaggedRunMetadata.FromString(s)` {#TaggedRunMetadata.FromString}




- - -

#### `tf.summary.TaggedRunMetadata.HasExtension(extension_handle)` {#TaggedRunMetadata.HasExtension}




- - -

#### `tf.summary.TaggedRunMetadata.HasField(field_name)` {#TaggedRunMetadata.HasField}




- - -

#### `tf.summary.TaggedRunMetadata.IsInitialized(errors=None)` {#TaggedRunMetadata.IsInitialized}

Checks if all required fields of a message are set.

##### Args:


*  <b>`errors`</b>: A list which, if provided, will be populated with the field
           paths of all missing required fields.

##### Returns:

  True iff the specified message has all required fields set.


- - -

#### `tf.summary.TaggedRunMetadata.ListFields()` {#TaggedRunMetadata.ListFields}




- - -

#### `tf.summary.TaggedRunMetadata.MergeFrom(msg)` {#TaggedRunMetadata.MergeFrom}




- - -

#### `tf.summary.TaggedRunMetadata.MergeFromString(serialized)` {#TaggedRunMetadata.MergeFromString}




- - -

#### `tf.summary.TaggedRunMetadata.ParseFromString(serialized)` {#TaggedRunMetadata.ParseFromString}

Parse serialized protocol buffer data into this message.

Like MergeFromString(), except we clear the object first and
do not return the value that MergeFromString returns.


- - -

#### `tf.summary.TaggedRunMetadata.RegisterExtension(extension_handle)` {#TaggedRunMetadata.RegisterExtension}




- - -

#### `tf.summary.TaggedRunMetadata.SerializePartialToString()` {#TaggedRunMetadata.SerializePartialToString}




- - -

#### `tf.summary.TaggedRunMetadata.SerializeToString()` {#TaggedRunMetadata.SerializeToString}




- - -

#### `tf.summary.TaggedRunMetadata.SetInParent()` {#TaggedRunMetadata.SetInParent}

Sets the _cached_byte_size_dirty bit to true,
and propagates this to our listener iff this was a state change.


- - -

#### `tf.summary.TaggedRunMetadata.WhichOneof(oneof_name)` {#TaggedRunMetadata.WhichOneof}

Returns the name of the currently set field inside a oneof, or None.


- - -

#### `tf.summary.TaggedRunMetadata.__deepcopy__(memo=None)` {#TaggedRunMetadata.__deepcopy__}




- - -

#### `tf.summary.TaggedRunMetadata.__eq__(other)` {#TaggedRunMetadata.__eq__}




- - -

#### `tf.summary.TaggedRunMetadata.__getstate__()` {#TaggedRunMetadata.__getstate__}

Support the pickle protocol.


- - -

#### `tf.summary.TaggedRunMetadata.__hash__()` {#TaggedRunMetadata.__hash__}




- - -

#### `tf.summary.TaggedRunMetadata.__init__(**kwargs)` {#TaggedRunMetadata.__init__}




- - -

#### `tf.summary.TaggedRunMetadata.__ne__(other_msg)` {#TaggedRunMetadata.__ne__}




- - -

#### `tf.summary.TaggedRunMetadata.__repr__()` {#TaggedRunMetadata.__repr__}




- - -

#### `tf.summary.TaggedRunMetadata.__setstate__(state)` {#TaggedRunMetadata.__setstate__}

Support the pickle protocol.


- - -

#### `tf.summary.TaggedRunMetadata.__str__()` {#TaggedRunMetadata.__str__}




- - -

#### `tf.summary.TaggedRunMetadata.__unicode__()` {#TaggedRunMetadata.__unicode__}




- - -

#### `tf.summary.TaggedRunMetadata.run_metadata` {#TaggedRunMetadata.run_metadata}

Magic attribute generated for "run_metadata" proto field.


- - -

#### `tf.summary.TaggedRunMetadata.tag` {#TaggedRunMetadata.tag}

Magic attribute generated for "tag" proto field.



