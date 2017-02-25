<!-- This file is machine generated: DO NOT EDIT! -->

# Utilities (contrib)
[TOC]

Utilities for dealing with Tensors. See @{$python/contrib.util} guide.

- - -

### `tf.contrib.util.constant_value(tensor)` {#constant_value}

Returns the constant value of the given tensor, if efficiently calculable.

This function attempts to partially evaluate the given tensor, and
returns its value as a numpy ndarray if this succeeds.

TODO(mrry): Consider whether this function should use a registration
mechanism like gradients and ShapeFunctions, so that it is easily
extensible.

NOTE: If `constant_value(tensor)` returns a non-`None` result, it will no
longer be possible to feed a different value for `tensor`. This allows the
result of this function to influence the graph that is constructed, and
permits static shape optimizations.

##### Args:


*  <b>`tensor`</b>: The Tensor to be evaluated.

##### Returns:

  A numpy ndarray containing the constant value of the given `tensor`,
  or None if it cannot be calculated.

##### Raises:


*  <b>`TypeError`</b>: if tensor is not an ops.Tensor.


- - -

### `tf.contrib.util.make_tensor_proto(values, dtype=None, shape=None, verify_shape=False)` {#make_tensor_proto}

Create a TensorProto.

##### Args:


*  <b>`values`</b>: Values to put in the TensorProto.
*  <b>`dtype`</b>: Optional tensor_pb2 DataType value.
*  <b>`shape`</b>: List of integers representing the dimensions of tensor.
*  <b>`verify_shape`</b>: Boolean that enables verification of a shape of values.

##### Returns:

  A TensorProto. Depending on the type, it may contain data in the
  "tensor_content" attribute, which is not directly useful to Python programs.
  To access the values you should convert the proto back to a numpy ndarray
  with tensor_util.MakeNdarray(proto).

##### Raises:


*  <b>`TypeError`</b>: if unsupported types are provided.
*  <b>`ValueError`</b>: if arguments have inappropriate values or if verify_shape is
   True and shape of values is not equals to a shape from the argument.

make_tensor_proto accepts "values" of a python scalar, a python list, a
numpy ndarray, or a numpy scalar.

If "values" is a python scalar or a python list, make_tensor_proto
first convert it to numpy ndarray. If dtype is None, the
conversion tries its best to infer the right numpy data
type. Otherwise, the resulting numpy array has a compatible data
type with the given dtype.

In either case above, the numpy ndarray (either the caller provided
or the auto converted) must have the compatible type with dtype.

make_tensor_proto then converts the numpy array to a tensor proto.

If "shape" is None, the resulting tensor proto represents the numpy
array precisely.

Otherwise, "shape" specifies the tensor's shape and the numpy array
can not have more elements than what "shape" specifies.


- - -

### `tf.contrib.util.make_ndarray(tensor)` {#make_ndarray}

Create a numpy ndarray from a tensor.

Create a numpy ndarray with the same shape and data as the tensor.

##### Args:


*  <b>`tensor`</b>: A TensorProto.

##### Returns:

  A numpy array with the tensor contents.

##### Raises:


*  <b>`TypeError`</b>: if tensor has unsupported type.


- - -

### `tf.contrib.util.ops_used_by_graph_def(graph_def)` {#ops_used_by_graph_def}

Collect the list of ops used by a graph.

Does not validate that the ops are all registered.

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` proto, as from `graph.as_graph_def()`.

##### Returns:

  A list of strings, each naming an op used by the graph.


- - -

### `tf.contrib.util.stripped_op_list_for_graph(graph_def)` {#stripped_op_list_for_graph}

Collect the stripped OpDefs for ops used by a graph.

This function computes the `stripped_op_list` field of `MetaGraphDef` and
similar protos.  The result can be communicated from the producer to the
consumer, which can then use the C++ function
`RemoveNewDefaultAttrsFromGraphDef` to improve forwards compatibility.

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` proto, as from `graph.as_graph_def()`.

##### Returns:

  An `OpList` of ops used by the graph.

##### Raises:


*  <b>`ValueError`</b>: If an unregistered op is used.


