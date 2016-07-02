### `tf.contrib.learn.extract_pandas_labels(labels)` {#extract_pandas_labels}

Extract data from pandas.DataFrame for labels.

##### Args:


*  <b>`labels`</b>: `pandas.DataFrame` or `pandas.Series` containing one column of
    labels to be extracted.

##### Returns:

  A numpy `ndarray` of labels from the DataFrame.

##### Raises:


*  <b>`ValueError`</b>: if more than one column is found or type is not int, float or
    bool.

