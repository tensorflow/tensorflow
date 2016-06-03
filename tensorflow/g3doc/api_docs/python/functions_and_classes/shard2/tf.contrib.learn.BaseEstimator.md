Abstract BaseEstimator class to train and evaluate TensorFlow models.

Concrete implementation of this class should provide following functions:
  * _get_train_ops
  * _get_eval_ops
  * _get_predict_ops

`Estimator` implemented below is a good example of how to use this class.

Parameters:
  model_dir: Directory to save model parameters, graph and etc.
- - -

#### `tf.contrib.learn.BaseEstimator.__init__(model_dir=None, config=None)` {#BaseEstimator.__init__}




- - -

#### `tf.contrib.learn.BaseEstimator.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#BaseEstimator.evaluate}

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

#### `tf.contrib.learn.BaseEstimator.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None)` {#BaseEstimator.fit}

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

#### `tf.contrib.learn.BaseEstimator.get_params(deep=True)` {#BaseEstimator.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.BaseEstimator.get_variable_names()` {#BaseEstimator.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.BaseEstimator.get_variable_value(name)` {#BaseEstimator.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.BaseEstimator.model_dir` {#BaseEstimator.model_dir}




- - -

#### `tf.contrib.learn.BaseEstimator.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#BaseEstimator.partial_fit}

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

#### `tf.contrib.learn.BaseEstimator.predict(x=None, input_fn=None, batch_size=None, outputs=None)` {#BaseEstimator.predict}

Returns predictions for given features.

##### Args:


*  <b>`x`</b>: Features. If set, `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x` must be `None`.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
           If `None`, returns all.

##### Returns:

  Numpy array of predicted classes or regression values.

##### Raises:


*  <b>`ValueError`</b>: If x and input_fn are both provided or both `None`.


- - -

#### `tf.contrib.learn.BaseEstimator.set_params(**params)` {#BaseEstimator.set_params}

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


