### `tf.contrib.layers.create_feature_spec_for_parsing(feature_columns)` {#create_feature_spec_for_parsing}

Helper that prepares features config from input feature_columns.

The returned feature config can be used as arg 'features' in tf.parse_example.

Typical usage example:

```python
# Define features and transformations
country = sparse_column_with_vocabulary_file("country", VOCAB_FILE)
age = real_valued_column("age")
click_bucket = bucketized_column(real_valued_column("historical_click_ratio"),
                                 boundaries=[i/10. for i in range(10)])
country_x_click = crossed_column([country, click_bucket], 10)

feature_columns = set([age, click_bucket, country_x_click])
batch_examples = tf.parse_example(
    serialized_examples,
    create_feature_spec_for_parsing(feature_columns))
```

For the above example, create_feature_spec_for_parsing would return the dict:
{"age": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
 "historical_click_ratio": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
 "country": parsing_ops.VarLenFeature(tf.string)}

##### Args:


*  <b>`feature_columns`</b>: An iterable containing all the feature columns. All items
    should be instances of classes derived from _FeatureColumn, unless
    feature_columns is a dict -- in which case, this should be true of all
    values in the dict.

##### Returns:

  A dict mapping feature keys to FixedLenFeature or VarLenFeature values.

