A regressor for TensorFlow DNN models.

  Example:
    ```
    installed_app_id = sparse_column_with_hash_bucket("installed_id", 1e6)
    impression_app_id = sparse_column_with_hash_bucket("impression_id", 1e6)

    installed_emb = embedding_column(installed_app_id, dimension=16,
                                     combiner="sum")
    impression_emb = embedding_column(impression_app_id, dimension=16,
                                      combiner="sum")

    estimator = DNNRegressor(
        feature_columns=[installed_emb, impression_emb],
        hidden_units=[1024, 512, 256])

    # Input builders
    def input_fn_train: # returns X, Y
      pass
    estimator.train(input_fn_train)

    def input_fn_eval: # returns X, Y
      pass
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
  hidden_units: List of hidden units per layer. All layers are fully
    connected. Ex. [64, 32] means first layer has 64 nodes and second one has
    32.
  feature_columns: An iterable containing all the feature columns used by the
    model. All items in the set should be instances of classes derived from
    `FeatureColumn`.
  model_dir: Directory to save model parameters, graph and etc.
  weight_column_name: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
  optimizer: An instance of `tf.Optimizer` used to train the model. If `None`,
    will use an Adagrad optimizer.
  activation_fn: Activation function applied to each layer. If `None`, will
    use `tf.nn.relu`.
- - -

#### `tf.contrib.learn.DNNRegressor.__init__(hidden_units, feature_columns=None, model_dir=None, weight_column_name=None, optimizer=None, activation_fn=relu)` {#DNNRegressor.__init__}




- - -

#### `tf.contrib.learn.DNNRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=32, steps=None, metrics=None, name=None)` {#DNNRegressor.evaluate}

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

#### `tf.contrib.learn.DNNRegressor.fit(x, y, steps, batch_size=32, monitors=None)` {#DNNRegressor.fit}

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

#### `tf.contrib.learn.DNNRegressor.get_params(deep=True)` {#DNNRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.DNNRegressor.get_variable_names()` {#DNNRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.DNNRegressor.get_variable_value(name)` {#DNNRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.DNNRegressor.linear_bias_` {#DNNRegressor.linear_bias_}

Returns bias of the linear part.


- - -

#### `tf.contrib.learn.DNNRegressor.linear_weights_` {#DNNRegressor.linear_weights_}

Returns weights per feature of the linear part.


- - -

#### `tf.contrib.learn.DNNRegressor.model_dir` {#DNNRegressor.model_dir}




- - -

#### `tf.contrib.learn.DNNRegressor.partial_fit(x, y, steps=1, batch_size=32, monitors=None)` {#DNNRegressor.partial_fit}

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

#### `tf.contrib.learn.DNNRegressor.predict(x=None, input_fn=None, batch_size=None)` {#DNNRegressor.predict}

Returns predictions for given features.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted classes or regression values.


- - -

#### `tf.contrib.learn.DNNRegressor.predict_proba(x=None, input_fn=None, batch_size=None)` {#DNNRegressor.predict_proba}

Returns prediction probabilities for given features (classification).

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted probabilities.


- - -

#### `tf.contrib.learn.DNNRegressor.set_params(**params)` {#DNNRegressor.set_params}

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

#### `tf.contrib.learn.DNNRegressor.train(input_fn, steps, monitors=None)` {#DNNRegressor.train}

Trains a model given input builder function.

##### Args:


*  <b>`input_fn`</b>: Input builder function, returns tuple of dicts or
            dict and Tensor.
*  <b>`steps`</b>: number of steps to train model for.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
            inside the training loop.

##### Returns:

  Returns self.


