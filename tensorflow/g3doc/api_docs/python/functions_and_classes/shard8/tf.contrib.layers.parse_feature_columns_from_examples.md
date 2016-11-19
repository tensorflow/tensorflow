### `tf.contrib.layers.parse_feature_columns_from_examples(serialized, feature_columns, name=None, example_names=None)` {#parse_feature_columns_from_examples}

Parses tf.Examples to extract tensors for given feature_columns.

This is a wrapper of 'tf.parse_example'.

Example:

```python
columns_to_tensor = parse_feature_columns_from_examples(
    serialized=my_data,
    feature_columns=my_features)

# Where my_features are:
# Define features and transformations
sparse_feature_a = sparse_column_with_keys(
    column_name="sparse_feature_a", keys=["AB", "CD", ...])

embedding_feature_a = embedding_column(
    sparse_id_column=sparse_feature_a, dimension=3, combiner="sum")

sparse_feature_b = sparse_column_with_hash_bucket(
    column_name="sparse_feature_b", hash_bucket_size=1000)

embedding_feature_b = embedding_column(
    sparse_id_column=sparse_feature_b, dimension=16, combiner="sum")

crossed_feature_a_x_b = crossed_column(
    columns=[sparse_feature_a, sparse_feature_b], hash_bucket_size=10000)

real_feature = real_valued_column("real_feature")
real_feature_buckets = bucketized_column(
    source_column=real_feature, boundaries=[...])

my_features = [embedding_feature_b, real_feature_buckets, embedding_feature_a]
```

##### Args:


*  <b>`serialized`</b>: A vector (1-D Tensor) of strings, a batch of binary
    serialized `Example` protos.
*  <b>`feature_columns`</b>: An iterable containing all the feature columns. All items
    should be instances of classes derived from _FeatureColumn.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`example_names`</b>: A vector (1-D Tensor) of strings (optional), the names of
    the serialized protos in the batch.

##### Returns:

  A `dict` mapping FeatureColumn to `Tensor` and `SparseTensor` values.

