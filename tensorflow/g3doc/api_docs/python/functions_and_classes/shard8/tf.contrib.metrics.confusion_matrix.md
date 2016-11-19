### `tf.contrib.metrics.confusion_matrix(predictions, labels, num_classes=None, dtype=tf.int32, name=None, weights=None)` {#confusion_matrix}

Computes the confusion matrix from predictions and labels.

Calculate the Confusion Matrix for a pair of prediction and
label 1-D int arrays.

The matrix rows represent the prediction labels and the columns
represents the real labels. The confusion matrix is always a 2-D array
of shape `[n, n]`, where `n` is the number of valid labels for a given
classification task. Both prediction and labels must be 1-D arrays of
the same shape in order for this function to work.

If `num_classes` is None, then `num_classes` will be set to the one plus
the maximum value in either predictions or labels.
Class labels are expected to start at 0. E.g., if `num_classes` was
three, then the possible labels would be `[0, 1, 2]`.

If `weights` is not `None`, then each prediction contributes its
corresponding weight to the total value of the confusion matrix cell.

For example:

```python
  tf.contrib.metrics.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
      [[0 0 0 0 0]
       [0 0 1 0 0]
       [0 0 1 0 0]
       [0 0 0 0 0]
       [0 0 0 0 1]]
```

Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`,
resulting in a 5x5 confusion matrix.

##### Args:


*  <b>`predictions`</b>: A 1-D array representing the predictions for a given
               classification.
*  <b>`labels`</b>: A 1-D representing the real labels for the classification task.
*  <b>`num_classes`</b>: The possible number of labels the classification task can
               have. If this value is not provided, it will be calculated
               using both predictions and labels array.
*  <b>`dtype`</b>: Data type of the confusion matrix.
*  <b>`name`</b>: Scope name.
*  <b>`weights`</b>: An optional `Tensor` whose shape matches `predictions`.

##### Returns:

  A k X k matrix representing the confusion matrix, where k is the number of
  possible labels in the classification task.

##### Raises:


*  <b>`ValueError`</b>: If both predictions and labels are not 1-D vectors and have
    mismatched shapes, or if `weights` is not `None` and its shape doesn't
    match `predictions`.

