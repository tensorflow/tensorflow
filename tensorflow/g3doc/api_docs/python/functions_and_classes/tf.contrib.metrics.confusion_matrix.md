### `tf.contrib.metrics.confusion_matrix(predictions, targets, num_classes=None, name=None)` {#confusion_matrix}

Computes the confusion matrix from predictions and targets

Calculate the Confusion Matrix for a pair of prediction and
target 1-D int arrays.

Considering a prediction array such as: `[1, 2, 3]`
And a target array such as: `[2, 2, 3]`

##### The confusion matrix returned would be the following one:

    [[0, 0, 0]
     [0, 1, 0]
     [0, 1, 0]
     [0, 0, 1]]

Where the matrix rows represent the prediction labels and the columns
represents the target labels. The confusion matrix is always a 2-D array
of shape [n, n], where n is the number of valid labels for a given
classification task. Both prediction and target must be 1-D arrays of
the same shape in order for this function to work.

##### Args:


*  <b>`predictions`</b>: A 1-D array represeting the predictions for a given
               classification.
*  <b>`targets`</b>: A 1-D represeting the real labels for the classification task.
*  <b>`num_classes`</b>: The possible number of labels the classification task can
               have. If this value is not provided, it will be calculated
               using both predictions and targets array.
*  <b>`name`</b>: Scope name.

##### Returns:

  A l X l matrix represeting the confusion matrix, where l in the number of
  possible labels in the classification task.

##### Raises:


*  <b>`ValueError`</b>: If both predictions and targets are not 1-D vectors and do not
              have the same size.

