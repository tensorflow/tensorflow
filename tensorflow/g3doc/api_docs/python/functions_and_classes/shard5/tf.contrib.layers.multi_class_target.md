### `tf.contrib.layers.multi_class_target(*args, **kwargs)` {#multi_class_target}

Creates a _TargetColumn for multi class single label classification. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-11-12.
Instructions for updating:
This file will be removed after the deprecation date.Please switch to third_party/tensorflow/contrib/learn/python/learn/estimators/head.py

The target column uses softmax cross entropy loss.

##### Args:


*  <b>`n_classes`</b>: Integer, number of classes, must be >= 2
*  <b>`label_name`</b>: String, name of the key in label dict. Can be null if label
      is a tensor (single headed models).
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.

##### Returns:

  An instance of _MultiClassTargetColumn.

##### Raises:


*  <b>`ValueError`</b>: if n_classes is < 2

