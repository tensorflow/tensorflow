### `tf.contrib.layers.one_hot_encoding(*args, **kwargs)` {#one_hot_encoding}

Transform numeric labels into onehot_labels using tf.one_hot.

##### Args:


*  <b>`labels`</b>: [batch_size] target labels.
*  <b>`num_classes`</b>: total number of classes.
*  <b>`on_value`</b>: A scalar defining the on-value.
*  <b>`off_value`</b>: A scalar defining the off-value.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`scope`</b>: Optional scope for name_scope.

##### Returns:

  one hot encoding of the labels.

