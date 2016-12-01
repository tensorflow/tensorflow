### `tf.contrib.learn.extract_pandas_data(data)` {#extract_pandas_data}

Extract data from pandas.DataFrame for predictors.

Given a DataFrame, will extract the values and cast them to float. The
DataFrame is expected to contain values of type int, float or bool.

##### Args:


*  <b>`data`</b>: `pandas.DataFrame` containing the data to be extracted.

##### Returns:

  A numpy `ndarray` of the DataFrame's values as floats.

##### Raises:


*  <b>`ValueError`</b>: if data contains types other than int, float or bool.

