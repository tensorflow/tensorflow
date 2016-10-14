<!-- This file is machine generated: DO NOT EDIT! -->

# Summary Operations
[TOC]

This module contains ops for generating summaries.

## Summary Ops
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

### `tf.summary.scalar(name, tensor, summary_description=None, collections=None)` {#scalar}

Outputs a `Summary` protocol buffer containing a single scalar value.

The generated Summary has a Tensor.proto containing the input Tensor.

##### Args:


*  <b>`name`</b>: A name for the generated node. Will also serve as the series name in
    TensorBoard.
*  <b>`tensor`</b>: A tensor containing a single floating point or integer value.
*  <b>`summary_description`</b>: Optional summary_description_pb2.SummaryDescription
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

##### Returns:

  A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.

##### Raises:


*  <b>`ValueError`</b>: If tensor has the wrong shape or type.


