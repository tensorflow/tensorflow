<!-- This file is machine generated: DO NOT EDIT! -->

# Summary Operations
[TOC]

## Generation of summaries.

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


