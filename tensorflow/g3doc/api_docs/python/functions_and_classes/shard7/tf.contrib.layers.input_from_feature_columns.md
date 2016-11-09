### `tf.contrib.layers.input_from_feature_columns(columns_to_tensors, feature_columns, weight_collections=None, trainable=True, scope=None)` {#input_from_feature_columns}

A tf.contrib.layer style input layer builder based on FeatureColumns.

Generally a single example in training data is described with feature columns.
At the first layer of the model, this column oriented data should be converted
to a single tensor. Each feature column needs a different kind of operation
during this conversion. For example sparse features need a totally different
handling than continuous features.

An example usage of input_from_feature_columns is as follows:

  # Building model for training
  columns_to_tensor = tf.parse_example(...)
  first_layer = input_from_feature_columns(
      columns_to_tensors=columns_to_tensor,
      feature_columns=feature_columns)
  second_layer = fully_connected(first_layer, ...)
  ...

  where feature_columns can be defined as follows:

  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")
  age = real_valued_column("age")
  age_buckets = bucketized_column(
      source_column=age,
      boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  feature_columns=[occupation_emb, age_buckets]

##### Args:


*  <b>`columns_to_tensors`</b>: A mapping from feature column to tensors. 'string' key
    means a base feature (not-transformed). It can have FeatureColumn as a
    key too. That means that FeatureColumn is already transformed by input
    pipeline. For example, `inflow` may have handled transformations.
*  <b>`feature_columns`</b>: A set containing all the feature columns. All items in the
    set should be instances of classes derived by FeatureColumn.
*  <b>`weight_collections`</b>: List of graph collections to which weights are added.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  A Tensor which can be consumed by hidden layers in the neural network.

##### Raises:


*  <b>`ValueError`</b>: if FeatureColumn cannot be consumed by a neural network.

