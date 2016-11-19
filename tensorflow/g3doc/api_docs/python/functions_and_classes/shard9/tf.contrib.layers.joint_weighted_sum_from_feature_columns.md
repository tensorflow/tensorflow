### `tf.contrib.layers.joint_weighted_sum_from_feature_columns(columns_to_tensors, feature_columns, num_outputs, weight_collections=None, trainable=True, scope=None)` {#joint_weighted_sum_from_feature_columns}

A restricted linear prediction builder based on FeatureColumns.

As long as all feature columns are unweighted sparse columns this computes the
prediction of a linear model which stores all weights in a single variable.

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
    * A list of Variables storing the weights.
    * A Variable which is used for bias.

##### Raises:


*  <b>`ValueError`</b>: if FeatureColumn cannot be used for linear predictions.

