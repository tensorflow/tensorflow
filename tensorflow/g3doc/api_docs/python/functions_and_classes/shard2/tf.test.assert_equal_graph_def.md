### `tf.test.assert_equal_graph_def(actual, expected)` {#assert_equal_graph_def}

Asserts that two `GraphDef`s are (mostly) the same.

Compares two `GraphDef` protos for equality, ignoring versions and ordering of
nodes, attrs, and control inputs.  Node names are used to match up nodes
between the graphs, so the naming of nodes must be consistent.

##### Args:


*  <b>`actual`</b>: The `GraphDef` we have.
*  <b>`expected`</b>: The `GraphDef` we expected.

##### Raises:


*  <b>`AssertionError`</b>: If the `GraphDef`s do not match.
*  <b>`TypeError`</b>: If either argument is not a `GraphDef`.

