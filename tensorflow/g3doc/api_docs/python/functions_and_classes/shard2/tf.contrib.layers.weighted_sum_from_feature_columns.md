### `tf.contrib.layers.weighted_sum_from_feature_columns(columns_to_tensors, feature_columns, num_outputs, weight_collections=None, trainable=True, scope=None)` {#weighted_sum_from_feature_columns}

A tf.contrib.layer style linear prediction builder based on FeatureColumns.

Generally a single example in training data is described with feature columns.
This function generates weighted sum for each num_outputs. Weighted sum refers
to logits in classification problems. It refers to prediction itself for
linear regression problems.

Example:

  ```
  # Building model for training
  feature_columns = (
      real_valued_column("my_feature1"),
      ...
  )
  columns_to_tensor = tf.parse_example(...)
  logits = weighted_sum_from_feature_columns(
      columns_to_tensors=columns_to_tensor,
      feature_columns=feature_columns,
      num_outputs=1)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
  ```

##### Args:


*  <b>`columns_to_tensors`</b>: A mapping from feature column to tensors. 'string' key
    means a base feature (not-transformed). It can have FeatureColumn as a
    key too. That means that FeatureColumn is already transformed by input
    pipeline. For example, `inflow` may have handled transformations.
*  <b>`feature_columns`</b>: A set containing all the feature columns. All items in the
    set should be instances of classes derived from FeatureColumn.
*  <b>`num_outputs`</b>: An integer specifying number of outputs. Default value is 1.
*  <b>`weight_collections`</b>: List of graph collections to which weights are added.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  A tuple containing:

    * A Tensor which represents predictions of a linear model.
    * A dictionary which maps feature_column to corresponding Variable.
    * A Variable which is used for bias.

##### Raises:


*  <b>`ValueError`</b>: if FeatureColumn cannot be used for linear predictions.

