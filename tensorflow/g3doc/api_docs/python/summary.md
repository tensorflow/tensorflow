<!-- This file is machine generated: DO NOT EDIT! -->

# Summary Operations
[TOC]

This module contains ops for generating summaries.

## Summary Ops
- - -

### `tf.summary.tensor_summary(display_name, tensor, description='', labels=None, collections=None, name=None)` {#tensor_summary}

Outputs a `Summary` protocol buffer with a serialized tensor.proto.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing input_tensor.

##### Args:


*  <b>`display_name`</b>: A name to associate with the data series. Will be used to
    organize output data and as a name in visualizers.
*  <b>`tensor`</b>: A tensor of any type and shape to serialize.
*  <b>`description`</b>: An optional long description of the data being output.
*  <b>`labels`</b>: a list of strings used to specify how the data can be interpreted,
    for example:
    * `'encoding:image/jpg'` for a string tensor containing jpg images
    * `'encoding:proto/X/Y/foo.proto'` for a string tensor containing Foos
    * `'group:$groupName/$roleInGroup'` for a tensor that is related to
       other tensors that are all in a group. (e.g. bounding boxes and images)
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


