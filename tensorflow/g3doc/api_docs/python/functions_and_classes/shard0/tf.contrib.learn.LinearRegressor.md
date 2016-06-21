Linear regressor model.

Train a linear regression model to predict target variable value given
observation of feature values.

Example:

```python
education = sparse_column_with_hash_bucket(column_name="education",
                                           hash_bucket_size=1000)
occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)

education_x_occupation = crossed_column(columns=[education, occupation],
                                        hash_bucket_size=10000)

estimator = LinearRegressor(
    feature_columns=[occupation, education_x_occupation])

# Input builders
def input_fn_train: # returns x, y, where y is a tensor of dimension 1
  ...
def input_fn_eval: # returns x, y, where y is a tensor of dimension 1
  ...
estimator.fit(input_fn=input_fn_train)
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x)
```

Input of `fit` and `evaluate` should have following features,
  otherwise there will be a KeyError:
    if `weight_column_name` is not `None`:
      key=weight_column_name, value=a `Tensor`
    for column in `feature_columns`:
    - if isinstance(column, `SparseColumn`):
        key=column.name, value=a `SparseTensor`
    - if isinstance(column, `RealValuedColumn`):
        key=column.name, value=a `Tensor`
    - if `feauture_columns` is `None`:
        input must contains only real valued `Tensor`.
- - -

#### `tf.contrib.learn.LinearRegressor.__init__(feature_columns=None, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None, config=None)` {#LinearRegressor.__init__}




- - -

#### `tf.contrib.learn.LinearRegressor.bias_` {#LinearRegressor.bias_}




- - -

#### `tf.contrib.learn.LinearRegressor.dnn_bias_` {#LinearRegressor.dnn_bias_}

Returns bias of deep neural network part.


- - -

#### `tf.contrib.learn.LinearRegressor.dnn_weights_` {#LinearRegressor.dnn_weights_}

Returns weights of deep neural network part.


- - -

#### `tf.contrib.learn.LinearRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#LinearRegressor.evaluate}

Evaluates given model with provided evaluation data.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function. If set, `x`, `y`, and `batch_size` must be
    `None`.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called. Called
    once per iteration.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`steps`</b>: Number of steps for which to evaluate model. If `None`, evaluate
    forever.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metric functions
    are used; if {}, no metrics are used.
*  <b>`name`</b>: Name of the evaluation if user needs to run multiple evaluation on
    different data sets, such as evaluate on training data vs test data.

##### Returns:

  Returns `dict` with evaluation results.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.


- - -

#### `tf.contrib.learn.LinearRegressor.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None)` {#LinearRegressor.fit}

Trains a model given training data `x` predictions and `y` targets.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
     iterator that returns arrays of features. The training input
     samples for fitting the model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
     iterator that returns array of targets. The training target values
     (class labels in classification, real numbers in regression). If set,
     `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x`, `y`, and `batch_size` must be
    `None`.
*  <b>`steps`</b>: Number of steps for which to train model. If `None`, train forever.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.

##### Returns:

  `self`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` and `y` is provided, and `input_fn` is
      provided.


- - -

#### `tf.contrib.learn.LinearRegressor.get_params(deep=True)` {#LinearRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.LinearRegressor.get_variable_names()` {#LinearRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.LinearRegressor.get_variable_value(name)` {#LinearRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.LinearRegressor.linear_bias_` {#LinearRegressor.linear_bias_}

Returns bias of the linear part.


- - -

#### `tf.contrib.learn.LinearRegressor.linear_weights_` {#LinearRegressor.linear_weights_}

Returns weights per feature of the linear part.


- - -

#### `tf.contrib.learn.LinearRegressor.model_dir` {#LinearRegressor.model_dir}




- - -

#### `tf.contrib.learn.LinearRegressor.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#LinearRegressor.partial_fit}

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
    samples for fitting the model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression). If set,
     `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x`, `y`, and `batch_size` must be
    `None`.
*  <b>`steps`</b>: Number of steps for which to train model. If `None`, train forever.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.

##### Returns:

  `self`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` and `y` is provided, and `input_fn` is
      provided.


- - -

#### `tf.contrib.learn.LinearRegressor.predict(x=None, input_fn=None, batch_size=None)` {#LinearRegressor.predict}

Returns predictions for given features.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.

##### Returns:

  Numpy array of predicted classes or regression values.


- - -

#### `tf.contrib.learn.LinearRegressor.set_params(**params)` {#LinearRegressor.set_params}

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

#### `tf.contrib.learn.LinearRegressor.weights_` {#LinearRegressor.weights_}




