Represents a graph node that performs computation on tensors.

An `Operation` is a node in a TensorFlow `Graph` that takes zero or
more `Tensor` objects as input, and produces zero or more `Tensor`
objects as output. Objects of type `Operation` are created by
calling a Python op constructor (such as
[`tf.matmul()`](../../api_docs/python/math_ops.md#matmul))
or [`Graph.create_op()`](../../api_docs/python/framework.md#Graph.create_op).

For example `c = tf.matmul(a, b)` creates an `Operation` of type
"MatMul" that takes tensors `a` and `b` as input, and produces `c`
as output.

After the graph has been launched in a session, an `Operation` can
be executed by passing it to
[`Session.run()`](../../api_docs/python/client.md#Session.run).
`op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.

- - -

#### `tf.Operation.name` {#Operation.name}

The full name of this operation.


- - -

#### `tf.Operation.type` {#Operation.type}

The type of the op (e.g. `"MatMul"`).


- - -

#### `tf.Operation.inputs` {#Operation.inputs}

The list of `Tensor` objects representing the data inputs of this op.


- - -

#### `tf.Operation.control_inputs` {#Operation.control_inputs}

The `Operation` objects on which this op has a control dependency.

Before this op is executed, TensorFlow will ensure that the
operations in `self.control_inputs` have finished executing. This
mechanism can be used to run ops sequentially for performance
reasons, or to ensure that the side effects of an op are observed
in the correct order.

##### Returns:

  A list of `Operation` objects.


- - -

#### `tf.Operation.outputs` {#Operation.outputs}

The list of `Tensor` objects representing the outputs of this op.


- - -

#### `tf.Operation.device` {#Operation.device}

The name of the device to which this op has been assigned, if any.

##### Returns:

  The string name of the device to which this op has been
  assigned, or an empty string if it has not been assigned to a
  device.


- - -

#### `tf.Operation.graph` {#Operation.graph}

The `Graph` that contains this operation.



- - -

#### `tf.Operation.run(feed_dict=None, session=None)` {#Operation.run}

Runs this operation in a `Session`.

Calling this method will execute all preceding operations that
produce the inputs needed for this operation.

*N.B.* Before invoking `Operation.run()`, its graph must have been
launched in a session, and either a default session must be
available, or `session` must be specified explicitly.

##### Args:


*  <b>`feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    See [`Session.run()`](../../api_docs/python/client.md#Session.run)
    for a description of the valid feed values.
*  <b>`session`</b>: (Optional.) The `Session` to be used to run to this operation. If
    none, the default session will be used.



- - -

#### `tf.Operation.get_attr(name)` {#Operation.get_attr}

Returns the value of the attr of this op with the given `name`.

##### Args:


*  <b>`name`</b>: The name of the attr to fetch.

##### Returns:

  The value of the attr, as a Python object.

##### Raises:


*  <b>`ValueError`</b>: If this op does not have an attr with the given `name`.


- - -

#### `tf.Operation.traceback` {#Operation.traceback}

Returns the call stack from when this operation was constructed.



#### Other Methods
- - -

#### `tf.Operation.__init__(node_def, g, inputs=None, output_types=None, control_inputs=None, input_types=None, original_op=None, op_def=None)` {#Operation.__init__}

Creates an `Operation`.

NOTE: This constructor validates the name of the `Operation` (passed
as `node_def.name`). Valid `Operation` names match the following
regular expression:

    [A-Za-z0-9.][A-Za-z0-9_.\\-/]*

##### Args:


*  <b>`node_def`</b>: `node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`.
    Used for attributes of `node_def_pb2.NodeDef`, typically `name`,
    `op`, and `device`.  The `input` attribute is irrelevant here
    as it will be computed when generating the model.
*  <b>`g`</b>: `Graph`. The parent graph.
*  <b>`inputs`</b>: list of `Tensor` objects. The inputs to this `Operation`.
*  <b>`output_types`</b>: list of `DType` objects.  List of the types of the
    `Tensors` computed by this operation.  The length of this list indicates
    the number of output endpoints of the `Operation`.
*  <b>`control_inputs`</b>: list of operations or tensors from which to have a
    control dependency.
*  <b>`input_types`</b>: List of `DType` objects representing the
    types of the tensors accepted by the `Operation`.  By default
    uses `[x.dtype.base_dtype for x in inputs]`.  Operations that expect
    reference-typed inputs must specify these explicitly.
*  <b>`original_op`</b>: Optional. Used to associate the new `Operation` with an
    existing `Operation` (for example, a replica with the op that was
    replicated).
*  <b>`op_def`</b>: Optional. The `op_def_pb2.OpDef` proto that describes the
    op type that this `Operation` represents.

##### Raises:


*  <b>`TypeError`</b>: if control inputs are not Operations or Tensors,
    or if `node_def` is not a `NodeDef`,
    or if `g` is not a `Graph`,
    or if `inputs` are not tensors,
    or if `inputs` and `input_types` are incompatible.
*  <b>`ValueError`</b>: if the `node_def` name is not valid.


- - -

#### `tf.Operation.__str__()` {#Operation.__str__}




- - -

#### `tf.Operation.colocation_groups()` {#Operation.colocation_groups}

Returns the list of colocation groups of the op.


- - -

#### `tf.Operation.node_def` {#Operation.node_def}

Returns a serialized `NodeDef` representation of this operation.

##### Returns:

  A
  [`NodeDef`](https://www.tensorflow.org/code/tensorflow/core/framework/node_def.proto)
  protocol buffer.


- - -

#### `tf.Operation.op_def` {#Operation.op_def}

Returns the `OpDef` proto that represents the type of this op.

##### Returns:

  An
  [`OpDef`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto)
  protocol buffer.


- - -

#### `tf.Operation.values()` {#Operation.values}

DEPRECATED: Use outputs.


