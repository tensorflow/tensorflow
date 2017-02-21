### `tf.contrib.layers.one_hot_encoding(*args, **kwargs)` {#one_hot_encoding}

Transform numeric labels into onehot_labels using `tf.one_hot`.

##### Args:


*  <b>`labels`</b>: [batch_size] target labels.
*  <b>`num_classes`</b>: Total number of classes.
*  <b>`on_value`</b>: A scalar defining the on-value.
*  <b>`off_value`</b>: A scalar defining the off-value.
*  <b>`outputs_collections`</b>: Collection to add the outputs.
*  <b>`scope`</b>: Optional scope for name_scope.

##### Returns:

  One-hot encoding of the labels.

