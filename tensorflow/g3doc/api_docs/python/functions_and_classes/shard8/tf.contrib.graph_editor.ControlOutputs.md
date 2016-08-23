The control outputs topology.
- - -

#### `tf.contrib.graph_editor.ControlOutputs.__init__(graph)` {#ControlOutputs.__init__}

Create a dictionary of control-output dependencies.

##### Args:


*  <b>`graph`</b>: a tf.Graph.

##### Returns:

  A dictionary where a key is a tf.Operation instance and the corresponding
    value is a list of all the ops which have the key as one of their
    control-input dependencies.

##### Raises:


*  <b>`TypeError`</b>: graph is not a tf.Graph.


- - -

#### `tf.contrib.graph_editor.ControlOutputs.get(op)` {#ControlOutputs.get}

return the control outputs of op.


- - -

#### `tf.contrib.graph_editor.ControlOutputs.get_all()` {#ControlOutputs.get_all}




- - -

#### `tf.contrib.graph_editor.ControlOutputs.graph` {#ControlOutputs.graph}




- - -

#### `tf.contrib.graph_editor.ControlOutputs.update()` {#ControlOutputs.update}

Update the control outputs if the graph has changed.


