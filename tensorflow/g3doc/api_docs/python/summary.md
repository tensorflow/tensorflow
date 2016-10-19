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


