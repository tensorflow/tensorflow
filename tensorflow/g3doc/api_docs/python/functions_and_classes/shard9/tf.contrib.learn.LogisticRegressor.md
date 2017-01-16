### `tf.contrib.learn.LogisticRegressor(model_fn, thresholds=None, model_dir=None, config=None, feature_engineering_fn=None)` {#LogisticRegressor}

Builds a logistic regression Estimator for binary classification.

This method provides a basic Estimator with some additional metrics for custom
binary classification models, including AUC, precision/recall and accuracy.

Example:

```python
  # See tf.contrib.learn.Estimator(...) for details on model_fn structure
  def my_model_fn(...):
    pass

  estimator = LogisticRegressor(model_fn=my_model_fn)

  # Input builders
  def input_fn_train:
    pass

  estimator.fit(input_fn=input_fn_train)
  estimator.predict(x=x)
```

##### Args:


*  <b>`model_fn`</b>: Model function with the signature:
    `(features, labels, mode) -> (predictions, loss, train_op)`.
    Expects the returned predictions to be probabilities in [0.0, 1.0].
*  <b>`thresholds`</b>: List of floating point thresholds to use for accuracy,
    precision, and recall metrics. If `None`, defaults to `[0.5]`.
*  <b>`model_dir`</b>: Directory to save model parameters, graphs, etc. This can also
    be used to load checkpoints from the directory into a estimator to
    continue training a previously saved model.
*  <b>`config`</b>: A RunConfig configuration object.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.

##### Returns:

  A `tf.contrib.learn.Estimator` instance.

