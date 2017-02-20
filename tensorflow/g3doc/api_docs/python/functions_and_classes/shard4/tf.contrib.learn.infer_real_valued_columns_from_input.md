### `tf.contrib.learn.infer_real_valued_columns_from_input(x)` {#infer_real_valued_columns_from_input}

Creates `FeatureColumn` objects for inputs defined by input `x`.

This interprets all inputs as dense, fixed-length float values.

##### Args:


*  <b>`x`</b>: Real-valued matrix of shape [n_samples, n_features...]. Can be
     iterator that returns arrays of features.

##### Returns:

  List of `FeatureColumn` objects.

