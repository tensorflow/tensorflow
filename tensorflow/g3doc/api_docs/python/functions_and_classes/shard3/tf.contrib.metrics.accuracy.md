### `tf.contrib.metrics.accuracy(predictions, labels, weights=None)` {#accuracy}

Computes the percentage of times that predictions matches labels.

##### Args:


*  <b>`predictions`</b>: the predicted values, a `Tensor` whose dtype and shape
               matches 'labels'.
*  <b>`labels`</b>: the ground truth values, a `Tensor` of any shape and
          integer or string dtype.
*  <b>`weights`</b>: None or `Tensor` of float values to reweight the accuracy.

##### Returns:

  Accuracy `Tensor`.

##### Raises:


*  <b>`ValueError`</b>: if dtypes don't match or
              if dtype is not integer or string.

