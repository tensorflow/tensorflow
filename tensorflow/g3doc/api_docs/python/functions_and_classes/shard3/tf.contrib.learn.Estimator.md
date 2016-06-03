Estimator class is the basic TensorFlow model trainer/evaluator.

Parameters:
  model_fn: Model function, takes features and targets tensors or dicts of
            tensors and returns predictions and loss tensors.
            Supports next three signatures for the function:
              * `(features, targets) -> (predictions, loss, train_op)`
              * `(features, targets, mode) -> (predictions, loss, train_op)`
              * `(features, targets, mode, params) ->
                  (predictions, loss, train_op)`
            Where:
              * `features` are single `Tensor` or `dict` of `Tensor`s
                   (depending on data passed to `fit`),
              * `targets` are `Tensor` or
                  `dict` of `Tensor`s (for multi-head model).
              * `mode` represents if this training, evaluation or prediction.
                  See `ModeKeys` for example keys.
              * `params` is a `dict` of hyperparameters. Will receive what is
                  passed to Estimator in `params` parameter. This allows to
                  configure Estimators from hyper parameter tunning.
  model_dir: Directory to save model parameters, graph and etc.
  config: Configuration object.
  params: `dict` of hyper parameters that will be passed into `model_fn`.
          Keys are names of parameters, values are basic python types.
- - -

#### `tf.contrib.learn.Estimator.__init__(model_fn=None, model_dir=None, config=None, params=None)` {#Estimator.__init__}




- - -

#### `tf.contrib.learn.Estimator.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#Estimator.evaluate}

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

#### `tf.contrib.learn.Estimator.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None)` {#Estimator.fit}

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

#### `tf.contrib.learn.Estimator.get_params(deep=True)` {#Estimator.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.Estimator.get_variable_names()` {#Estimator.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.Estimator.get_variable_value(name)` {#Estimator.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.Estimator.model_dir` {#Estimator.model_dir}




- - -

#### `tf.contrib.learn.Estimator.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#Estimator.partial_fit}

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

#### `tf.contrib.learn.Estimator.predict(x=None, input_fn=None, batch_size=None, outputs=None)` {#Estimator.predict}

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

#### `tf.contrib.learn.Estimator.set_params(**params)` {#Estimator.set_params}

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


