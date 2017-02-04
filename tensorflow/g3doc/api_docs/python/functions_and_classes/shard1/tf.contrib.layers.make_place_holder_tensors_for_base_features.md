### `tf.contrib.layers.make_place_holder_tensors_for_base_features(feature_columns)` {#make_place_holder_tensors_for_base_features}

Returns placeholder tensors for inference.

##### Args:


*  <b>`feature_columns`</b>: An iterable containing all the feature columns. All items
    should be instances of classes derived from _FeatureColumn.

##### Returns:

  A dict mapping feature keys to SparseTensors (sparse columns) or
  placeholder Tensors (dense columns).

