### `tf.contrib.layers.layer_norm(*args, **kwargs)` {#layer_norm}

Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.

  "Layer Normalization"

  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

Can be used as a normalizer function for conv2d and fully_connected.

##### Args:


*  <b>`inputs`</b>: a tensor with 2 or more dimensions. The normalization
          occurs over all but the first dimension.
*  <b>`center`</b>: If True, subtract `beta`. If False, `beta` is ignored.
*  <b>`scale`</b>: If True, multiply by `gamma`. If False, `gamma` is
    not used. When the next layer is linear (also e.g. `nn.relu`), this can be
    disabled since the scaling can be done by the next layer.
*  <b>`activation_fn`</b>: activation function, default set to None to skip it and
    maintain a linear activation.
*  <b>`reuse`</b>: whether or not the layer and its variables should be reused. To be
    able to reuse the layer scope must be given.
*  <b>`variables_collections`</b>: optional collections for the variables.
*  <b>`outputs_collections`</b>: collections to add the outputs.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for `variable_scope`.

##### Returns:

  A `Tensor` representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if rank or last dimension of `inputs` is undefined.

