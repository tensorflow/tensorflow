Transform a subgraph into another one.

By default, the constructor create a transform which copy a subgraph and
replaces inputs with placeholders. This behavior can be modified by changing
the handlers.
- - -

#### `tf.contrib.graph_editor.Transformer.__init__()` {#Transformer.__init__}

Transformer constructor.

The following members can be modified:
transform_op_handler: handle the transformation of a tf.Operation.
  This handler defaults to a simple copy.
assign_collections_handler: handle the assignment of collections.
  This handler defaults to assigning new collections created under the
  given name-scope.
transform_input_handler: handle the transform of the inputs to the given
  subgraph. This handler defaults to creating placeholders instead of the
  ops just before the input tensors of the subgraph.
transform_hidden_input_handler: handle the transform of the hidden inputs of
  the subgraph, that is, the inputs which are not listed in sgv.inputs.
  This handler defaults to a transform which keep the same input if the
  source and destination graphs are the same, otherwise use placeholders.
transform_original_op_hanlder: handle the transform of original_op. This
  handler defaults to transforming original_op only if they are in the
  subgraph, otherwise they are ignored.


- - -

#### `tf.contrib.graph_editor.Transformer.new_name(name)` {#Transformer.new_name}

Compute a destination name from a source name.

##### Args:


*  <b>`name`</b>: the name to be "transformed".

##### Returns:

  the transformed name.

##### Raises:


*  <b>`ValueError`</b>: if the source scope is used (that is, not an empty string)
    and the source name does not belong to the source scope.


