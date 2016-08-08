### `tf.contrib.graph_editor.detach(sgv, control_inputs=False, control_outputs=None, control_ios=None)` {#detach}

Detach both the inputs and the outputs of a subgraph view.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of util.ControlOutputs or None. If not None,
    control outputs are enabled.
*  <b>`control_ios`</b>: An instance of util.ControlOutputs or None. If not None, both
    control inputs and control outputs are enabled. This is equivalent to set
    control_inputs to True and control_outputs to the util.ControlOutputs
    instance.

##### Returns:

  A new subgraph view of the detached subgraph.
    Note that sgv is also modified in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.

