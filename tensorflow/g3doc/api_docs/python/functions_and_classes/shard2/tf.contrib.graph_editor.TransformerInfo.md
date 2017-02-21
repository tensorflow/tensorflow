"Contains information about the result of a transform operation.
- - -

#### `tf.contrib.graph_editor.TransformerInfo.__init__(info)` {#TransformerInfo.__init__}

Constructor.

##### Args:


*  <b>`info`</b>: an instance of Transformer._TmpInfo containing various internal
    information about the transform operation.


- - -

#### `tf.contrib.graph_editor.TransformerInfo.__str__()` {#TransformerInfo.__str__}




- - -

#### `tf.contrib.graph_editor.TransformerInfo.original(transformed, missing_fn=None)` {#TransformerInfo.original}

Return the original op/tensor corresponding to the transformed one.

Note that the output of this function mimics the hierarchy
of its input argument `transformed`.
Given an iterable, it returns a list. Given an operation or a tensor,
it will return an operation or a tensor.

##### Args:


*  <b>`transformed`</b>: the transformed tensor/operation.
*  <b>`missing_fn`</b>: function handling the case where the counterpart
    cannot be found. By default, None is returned.

##### Returns:

  the original tensor/operation (or None if no match is found).


- - -

#### `tf.contrib.graph_editor.TransformerInfo.transformed(original, missing_fn=None)` {#TransformerInfo.transformed}

Return the transformed op/tensor corresponding to the original one.

Note that the output of this function mimics the hierarchy
of its input argument `original`.
Given an iterable, it returns a list. Given an operation or a tensor,
it will return an operation or a tensor.

##### Args:


*  <b>`original`</b>: the original tensor/operation.
*  <b>`missing_fn`</b>: function handling the case where the counterpart
    cannot be found. By default, None is returned.

##### Returns:

  the transformed tensor/operation (or None if no match is found).


