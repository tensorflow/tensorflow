Abstract BaseEstimator class to train and evaluate TensorFlow models.

Concrete implementation of this class should provide the following functions:

  * _get_train_ops
  * _get_eval_ops
  * _get_predict_ops

`Estimator` implemented below is a good example of how to use this class.
- - -

#### `tf.contrib.learn.BaseEstimator.__init__(model_dir=None, config=None)` {#BaseEstimator.__init__}

Initializes a BaseEstimator instance.

##### Args:


*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator to
    continue training a previously saved model.
*  <b>`config`</b>: A RunConfig instance.


- - -

#### `tf.contrib.learn.BaseEstimator.__repr__()` {#BaseEstimator.__repr__}




- - -

#### `tf.contrib.learn.BaseEstimator.config` {#BaseEstimator.config}




- - -

#### `tf.contrib.learn.BaseEstimator.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#BaseEstimator.evaluate}

See `Evaluable`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.BaseEstimator.export(*args, **kwargs)` {#BaseEstimator.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn (and in most cases, input_feature_key) will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

    Args:
      export_dir: A string containing a directory to write the exported graph
        and checkpoints.
      input_fn: If `use_deprecated_input_fn` is true, then a function that given
        `Tensor` of `Example` strings, parses it into features that are then
        passed to the model. Otherwise, a function that takes no argument and
        returns a tuple of (features, targets), where features is a dict of
        string key to `Tensor` and targets is a `Tensor` that's currently not
        used (and so can be `None`).
      input_feature_key: Only used if `use_deprecated_input_fn` is false. String
        key into the features dict returned by `input_fn` that corresponds to
        the raw `Example` strings `Tensor` that the exported model will take as
        input. Can only be `None` if you're using a custom `signature_fn` that
        does not use the first arg (examples).
      use_deprecated_input_fn: Determines the signature format of `input_fn`.
      signature_fn: Function that returns a default signature and a named
        signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
        for features and `Tensor` or `dict` of `Tensor`s for predictions.
      prediction_key: The key for a tensor in the `predictions` dict (output
        from the `model_fn`) to use as the `predictions` input to the
        `signature_fn`. Optional. If `None`, predictions will pass to
        `signature_fn` without filtering.
      default_batch_size: Default batch size of the `Example` placeholder.
      exports_to_keep: Number of exports to keep.

    Returns:
      The string path to the exported directory. NB: this functionality was
      added ca. 2016/09/25; clients that depend on the return value may need
      to handle the case where this function returns None because subclasses
      are not returning a value.


- - -

#### `tf.contrib.learn.BaseEstimator.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#BaseEstimator.fit}

See `Trainable`.

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.BaseEstimator.get_params(deep=True)` {#BaseEstimator.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
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


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...]. Can be iterator that
     returns arrays of features. The training input samples for fitting the
     model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: Vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
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


*  <b>`ValueError`</b>: If at least one of `x` and `y` is provided, and `input_fn` is
      provided.


- - -

#### `tf.contrib.learn.BaseEstimator.predict(*args, **kwargs)` {#BaseEstimator.predict}

Returns predictions for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

    Args:
      x: Matrix of shape [n_samples, n_features...]. Can be iterator that
         returns arrays of features. The training input samples for fitting the
         model. If set, `input_fn` must be `None`.
      input_fn: Input function. If set, `x` and 'batch_size' must be `None`.
      batch_size: Override default batch size. If set, 'input_fn' must be
        'None'.
      outputs: list of `str`, name of the output to predict.
        If `None`, returns all.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      A numpy array of predicted classes or regression values if the
      constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict`
      of numpy arrays if `model_fn` returns a `dict`. Returns an iterable of
      predictions if as_iterable is True.

    Raises:
      ValueError: If x and input_fn are both provided or both `None`.


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


