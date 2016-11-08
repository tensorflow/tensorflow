### `tf.contrib.layers.sequence_input_from_feature_columns(*args, **kwargs)` {#sequence_input_from_feature_columns}

Builds inputs for sequence models from `FeatureColumn`s. (experimental)

THIS FUNCTION IS EXPERIMENTAL. It may change or be removed at any time, and without warning.


  See documentation for `input_from_feature_columns`. The following types of
  `FeatureColumn` are permitted in `feature_columns`: `_OneHotColumn`,
  `_EmbeddingColumn`, `_HashedEmbeddingColumn`, `_RealValuedColumn`,
  `_DataFrameColumn`. In addition, columns in `feature_columns` may not be
  constructed using any of the following: `HashedEmbeddingColumn`,
  `BucketizedColumn`, `CrossedColumn`.

  Args:
    columns_to_tensors: A mapping from feature column to tensors. 'string' key
      means a base feature (not-transformed). It can have FeatureColumn as a
      key too. That means that FeatureColumn is already transformed by input
      pipeline. For example, `inflow` may have handled transformations.
    feature_columns: A set containing all the feature columns. All items in the
      set should be instances of classes derived by FeatureColumn.
    weight_collections: List of graph collections to which weights are added.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for variable_scope.

  Returns:
    A Tensor which can be consumed by hidden layers in the neural network.

  Raises:
    ValueError: if FeatureColumn cannot be consumed by a neural network.

