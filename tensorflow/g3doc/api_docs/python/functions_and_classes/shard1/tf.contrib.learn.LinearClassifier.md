Linear classifier model.

  Example:
  ```
  installed_app_id = sparse_column_with_hash_bucket("installed_id", 1e6)
  impression_app_id = sparse_column_with_hash_bucket("impression_id", 1e6)

  installed_x_impression = crossed_column(
      [installed_app_id, impression_app_id])

  estimator = LinearClassifier(
      feature_columns=[impression_app_id, installed_x_impression])

  # Input builders
  def input_fn_train: # returns X, Y
    ...
  def input_fn_eval: # returns X, Y
    ...
  estimator.train(input_fn_train)
  estimator.evaluate(input_fn_eval)
  estimator.predict(x)
  ```

  Input of `fit`, `train`, and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
      if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
      for each `column` in `feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
      - if `feauture_columns` is None, then `input` must contains only real
        valued `Tensor`.


Parameters:
  feature_columns: An iterable containing all the feature columns used by the
    model. All items in the set should be instances of classes derived from
    `FeatureColumn`.
  model_dir: Directory to save model parameters, graph and etc.
  n_classes: number of target classes. Default is binary classification.
  weight_column_name: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
  optimizer: An instance of `tf.Optimizer` used to train the model. If `None`,
    will use an Ftrl optimizer.
- - -

#### `tf.contrib.learn.LinearClassifier.__init__(feature_columns=None, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None)` {#LinearClassifier.__init__}




- - -

#### `tf.contrib.learn.LinearClassifier.bias_` {#LinearClassifier.bias_}




- - -

#### `tf.contrib.learn.LinearClassifier.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=32, steps=None, metrics=None, name=None)` {#LinearClassifier.evaluate}

Evaluates given model with provided evaluation data.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called. Called
    once per iteration.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32. Ignored
    if input_fn is set.
*  <b>`steps`</b>: Number of steps to evalute for.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metric functions
    are used; if {}, no metrics are used.
*  <b>`name`</b>: Name of the evaluation if user needs to run multiple evaluation on
    different data sets, such as evaluate on training data vs test data.

##### Returns:

  Returns `dict` with evaluation results.

##### Raises:


*  <b>`ValueError`</b>: If x or y are not None while input_fn or feed_fn is not None.


- - -

#### `tf.contrib.learn.LinearClassifier.fit(x, y, steps, batch_size=32, monitors=None)` {#LinearClassifier.fit}

Trains a model given training data X and y.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
     iterator that returns arrays of features. The training input
     samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
     iterator that returns array of targets. The training target values
     (class labels in classification, real numbers in regression).
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns final loss.


- - -

#### `tf.contrib.learn.LinearClassifier.get_params(deep=True)` {#LinearClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.LinearClassifier.get_variable_names()` {#LinearClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.LinearClassifier.get_variable_value(name)` {#LinearClassifier.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.LinearClassifier.linear_bias_` {#LinearClassifier.linear_bias_}

Returns bias of the linear part.


- - -

#### `tf.contrib.learn.LinearClassifier.linear_weights_` {#LinearClassifier.linear_weights_}

Returns weights per feature of the linear part.


- - -

#### `tf.contrib.learn.LinearClassifier.model_dir` {#LinearClassifier.model_dir}




- - -

#### `tf.contrib.learn.LinearClassifier.partial_fit(x, y, steps=1, batch_size=32, monitors=None)` {#LinearClassifier.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to 32.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns final loss.


- - -

#### `tf.contrib.learn.LinearClassifier.predict(x=None, input_fn=None, batch_size=None)` {#LinearClassifier.predict}

Returns predictions for given features.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted classes or regression values.


- - -

#### `tf.contrib.learn.LinearClassifier.predict_proba(x=None, input_fn=None, batch_size=None)` {#LinearClassifier.predict_proba}

Returns prediction probabilities for given features (classification).

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted probabilities.


- - -

#### `tf.contrib.learn.LinearClassifier.set_params(**params)` {#LinearClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Args:


*  <b>`**params`</b>: Parameters.

##### Returns:

  self

##### Raises:


*  <b>`ValueError`</b>: If params contain invalid names.


- - -

#### `tf.contrib.learn.LinearClassifier.train(input_fn, steps, monitors=None)` {#LinearClassifier.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.LinearClassifier.weights_` {#LinearClassifier.weights_}




