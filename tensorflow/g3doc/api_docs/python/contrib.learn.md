<!-- This file is machine generated: DO NOT EDIT! -->

# Learn (contrib)
[TOC]

High level API for learning. See the @{$python/contrib.learn} guide.

- - -

### `class tf.contrib.learn.BaseEstimator` {#BaseEstimator}

Abstract BaseEstimator class to train and evaluate TensorFlow models.

Users should not instantiate or subclass this class. Instead, use `Estimator`.
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

#### `tf.contrib.learn.BaseEstimator.evaluate(*args, **kwargs)` {#BaseEstimator.evaluate}

See `Evaluable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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

##### Args:


*  <b>`export_dir`</b>: A string containing a directory to write the exported graph
    and checkpoints.
*  <b>`input_fn`</b>: If `use_deprecated_input_fn` is true, then a function that given
    `Tensor` of `Example` strings, parses it into features that are then
    passed to the model. Otherwise, a function that takes no argument and
    returns a tuple of (features, labels), where features is a dict of
    string key to `Tensor` and labels is a `Tensor` that's currently not
    used (and so can be `None`).
*  <b>`input_feature_key`</b>: Only used if `use_deprecated_input_fn` is false. String
    key into the features dict returned by `input_fn` that corresponds to a
    the raw `Example` strings `Tensor` that the exported model will take as
    input. Can only be `None` if you're using a custom `signature_fn` that
    does not use the first arg (examples).
*  <b>`use_deprecated_input_fn`</b>: Determines the signature format of `input_fn`.
*  <b>`signature_fn`</b>: Function that returns a default signature and a named
    signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
    for features and `Tensor` or `dict` of `Tensor`s for predictions.
*  <b>`prediction_key`</b>: The key for a tensor in the `predictions` dict (output
    from the `model_fn`) to use as the `predictions` input to the
    `signature_fn`. Optional. If `None`, predictions will pass to
    `signature_fn` without filtering.
*  <b>`default_batch_size`</b>: Default batch size of the `Example` placeholder.
*  <b>`exports_to_keep`</b>: Number of exports to keep.
*  <b>`checkpoint_path`</b>: the checkpoint path of the model to be exported. If it is
      `None` (which is default), will use the latest checkpoint in
      export_dir.

##### Returns:

  The string path to the exported directory. NB: this functionality was
  added ca. 2016/09/25; clients that depend on the return value may need
  to handle the case where this function returns None because subclasses
  are not returning a value.


- - -

#### `tf.contrib.learn.BaseEstimator.fit(*args, **kwargs)` {#BaseEstimator.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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

#### `tf.contrib.learn.BaseEstimator.partial_fit(*args, **kwargs)` {#BaseEstimator.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...]. Can be iterator that
     returns arrays of features. The training input samples for fitting the
     model. If set, `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x` and 'batch_size' must be `None`.
*  <b>`batch_size`</b>: Override default batch size. If set, 'input_fn' must be
    'None'.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns all.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  A numpy array of predicted classes or regression values if the
  constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict`
  of numpy arrays if `model_fn` returns a `dict`. Returns an iterable of
  predictions if as_iterable is True.

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



- - -

### `class tf.contrib.learn.Estimator` {#Estimator}

Estimator class is the basic TensorFlow model trainer/evaluator.
- - -

#### `tf.contrib.learn.Estimator.__init__(model_fn=None, model_dir=None, config=None, params=None, feature_engineering_fn=None)` {#Estimator.__init__}

Constructs an `Estimator` instance.

##### Args:


*  <b>`model_fn`</b>: Model function. Follows the signature:
    * Args:
      * `features`: single `Tensor` or `dict` of `Tensor`s
             (depending on data passed to `fit`),
      * `labels`: `Tensor` or `dict` of `Tensor`s (for multi-head
             models). If mode is `ModeKeys.INFER`, `labels=None` will be
             passed. If the `model_fn`'s signature does not accept
             `mode`, the `model_fn` must still be able to handle
             `labels=None`.
      * `mode`: Optional. Specifies if this training, evaluation or
             prediction. See `ModeKeys`.
      * `params`: Optional `dict` of hyperparameters.  Will receive what
             is passed to Estimator in `params` parameter. This allows
             to configure Estimators from hyper parameter tuning.
      * `config`: Optional configuration object. Will receive what is passed
             to Estimator in `config` parameter, or the default `config`.
             Allows updating things in your model_fn based on configuration
             such as `num_ps_replicas`.
      * `model_dir`: Optional directory where model parameters, graph etc
             are saved. Will receive what is passed to Estimator in
             `model_dir` parameter, or the default `model_dir`. Allows
             updating things in your model_fn that expect model_dir, such as
             training hooks.

    * Returns:
      `ModelFnOps`

    Also supports a legacy signature which returns tuple of:

      * predictions: `Tensor`, `SparseTensor` or dictionary of same.
          Can also be any type that is convertible to a `Tensor` or
          `SparseTensor`, or dictionary of same.
      * loss: Scalar loss `Tensor`.
      * train_op: Training update `Tensor` or `Operation`.

    Supports next three signatures for the function:

      * `(features, labels) -> (predictions, loss, train_op)`
      * `(features, labels, mode) -> (predictions, loss, train_op)`
      * `(features, labels, mode, params) -> (predictions, loss, train_op)`
      * `(features, labels, mode, params, config) ->
         (predictions, loss, train_op)`
      * `(features, labels, mode, params, config, model_dir) ->
         (predictions, loss, train_op)`


*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator to
    continue training a previously saved model.
*  <b>`config`</b>: Configuration object.
*  <b>`params`</b>: `dict` of hyper parameters that will be passed into `model_fn`.
          Keys are names of parameters, values are basic python types.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                          labels which are the output of `input_fn` and
                          returns features and labels which will be fed
                          into `model_fn`. Please check `model_fn` for
                          a definition of features and labels.

##### Raises:


*  <b>`ValueError`</b>: parameters of `model_fn` don't match `params`.


- - -

#### `tf.contrib.learn.Estimator.__repr__()` {#Estimator.__repr__}




- - -

#### `tf.contrib.learn.Estimator.config` {#Estimator.config}




- - -

#### `tf.contrib.learn.Estimator.evaluate(*args, **kwargs)` {#Estimator.evaluate}

See `Evaluable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.Estimator.export(*args, **kwargs)` {#Estimator.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn (and in most cases, input_feature_key) will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

##### Args:


*  <b>`export_dir`</b>: A string containing a directory to write the exported graph
    and checkpoints.
*  <b>`input_fn`</b>: If `use_deprecated_input_fn` is true, then a function that given
    `Tensor` of `Example` strings, parses it into features that are then
    passed to the model. Otherwise, a function that takes no argument and
    returns a tuple of (features, labels), where features is a dict of
    string key to `Tensor` and labels is a `Tensor` that's currently not
    used (and so can be `None`).
*  <b>`input_feature_key`</b>: Only used if `use_deprecated_input_fn` is false. String
    key into the features dict returned by `input_fn` that corresponds to a
    the raw `Example` strings `Tensor` that the exported model will take as
    input. Can only be `None` if you're using a custom `signature_fn` that
    does not use the first arg (examples).
*  <b>`use_deprecated_input_fn`</b>: Determines the signature format of `input_fn`.
*  <b>`signature_fn`</b>: Function that returns a default signature and a named
    signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
    for features and `Tensor` or `dict` of `Tensor`s for predictions.
*  <b>`prediction_key`</b>: The key for a tensor in the `predictions` dict (output
    from the `model_fn`) to use as the `predictions` input to the
    `signature_fn`. Optional. If `None`, predictions will pass to
    `signature_fn` without filtering.
*  <b>`default_batch_size`</b>: Default batch size of the `Example` placeholder.
*  <b>`exports_to_keep`</b>: Number of exports to keep.
*  <b>`checkpoint_path`</b>: the checkpoint path of the model to be exported. If it is
      `None` (which is default), will use the latest checkpoint in
      export_dir.

##### Returns:

  The string path to the exported directory. NB: this functionality was
  added ca. 2016/09/25; clients that depend on the return value may need
  to handle the case where this function returns None because subclasses
  are not returning a value.


- - -

#### `tf.contrib.learn.Estimator.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, checkpoint_path=None)` {#Estimator.export_savedmodel}

Exports inference graph as a SavedModel into given dir.

##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`serving_input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.  Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the most recent checkpoint found within the model directory is chosen.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.Estimator.fit(*args, **kwargs)` {#Estimator.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.Estimator.get_params(deep=True)` {#Estimator.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
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

#### `tf.contrib.learn.Estimator.partial_fit(*args, **kwargs)` {#Estimator.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

#### `tf.contrib.learn.Estimator.predict(*args, **kwargs)` {#Estimator.predict}

Returns predictions for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...]. Can be iterator that
     returns arrays of features. The training input samples for fitting the
     model. If set, `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x` and 'batch_size' must be `None`.
*  <b>`batch_size`</b>: Override default batch size. If set, 'input_fn' must be
    'None'.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns all.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  A numpy array of predicted classes or regression values if the
  constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict`
  of numpy arrays if `model_fn` returns a `dict`. Returns an iterable of
  predictions if as_iterable is True.

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



- - -

### `class tf.contrib.learn.Trainable` {#Trainable}

Interface for objects that are trainable by, e.g., `Experiment`.
- - -

#### `tf.contrib.learn.Trainable.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#Trainable.fit}

Trains a model given training data `x` predictions and `y` labels.

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...] or the dictionary of Matrices.
     Can be iterator that returns arrays of features or dictionary of arrays of features.
     The training input samples for fitting the model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: Vector or matrix [n_samples] or [n_samples, n_outputs] or the dictionary of same.
     Can be iterator that returns array of labels or dictionary of array of labels.
     The training label values (class labels in classification, real numbers in regression).
     If set, `input_fn` must be `None`. Note: For classification, label values must
     be integers representing the class index (i.e. values from 0 to
     n_classes-1).
*  <b>`input_fn`</b>: Input function returning a tuple of:
      features - `Tensor` or dictionary of string feature name to `Tensor`.
      labels - `Tensor` or dictionary of `Tensor` with labels.
    If input_fn is set, `x`, `y`, and `batch_size` must be `None`.
*  <b>`steps`</b>: Number of steps for which to train model. If `None`, train forever.
    'steps' works incrementally. If you call two times fit(steps=10) then
    training occurs in total 20 steps. If you don't want to have incremental
    behaviour please set `max_steps` instead. If set, `max_steps` must be
    `None`.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.
*  <b>`max_steps`</b>: Number of total steps for which to train model. If `None`,
    train forever. If set, `steps` must be `None`.

    Two calls to `fit(steps=100)` means 200 training
    iterations. On the other hand, two calls to `fit(max_steps=100)` means
    that the second call will not do any iteration since first call did
    all 100 steps.

##### Returns:

  `self`, for chaining.



- - -

### `class tf.contrib.learn.Evaluable` {#Evaluable}

Interface for objects that are evaluatable by, e.g., `Experiment`.
- - -

#### `tf.contrib.learn.Evaluable.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None, checkpoint_path=None, hooks=None)` {#Evaluable.evaluate}

Evaluates given model with provided evaluation data.

Stop conditions - we evaluate on the given input data until one of the
following:
- If `steps` is provided, and `steps` batches of size `batch_size` are
processed.
- If `input_fn` is provided, and it raises an end-of-input
exception (`OutOfRangeError` or `StopIteration`).
- If `x` is provided, and all items in `x` have been processed.

The return value is a dict containing the metrics specified in `metrics`, as
well as an entry `global_step` which contains the value of the global step
for which this evaluation was performed.

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...] or dictionary of many matrices
     containing the input samples for fitting the model. Can be iterator that returns
     arrays of features or dictionary of array of features. If set, `input_fn` must
     be `None`.
*  <b>`y`</b>: Vector or matrix [n_samples] or [n_samples, n_outputs] containing the
     label values (class labels in classification, real numbers in
     regression) or dictionary of multiple vectors/matrices. Can be iterator
     that returns array of targets or dictionary of array of targets. If set,
     `input_fn` must be `None`. Note: For classification, label values must
     be integers representing the class index (i.e. values from 0 to
     n_classes-1).
*  <b>`input_fn`</b>: Input function returning a tuple of:
      features - Dictionary of string feature name to `Tensor` or `Tensor`.
      labels - `Tensor` or dictionary of `Tensor` with labels.
    If input_fn is set, `x`, `y`, and `batch_size` must be `None`. If
    `steps` is not provided, this should raise `OutOfRangeError` or
    `StopIteration` after the desired amount of data (e.g., one epoch) has
    been provided. See "Stop conditions" above for specifics.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called. Called
    once per iteration. Must be `None` if `input_fn` is provided.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`, if specified. Must be `None` if `input_fn` is
    provided.
*  <b>`steps`</b>: Number of steps for which to evaluate model. If `None`, evaluate
    until `x` is consumed or `input_fn` raises an end-of-input exception.
    See "Stop conditions" above for specifics.
*  <b>`metrics`</b>: Dict of metrics to run. If None, the default metric functions
    are used; if {}, no metrics are used. Otherwise, `metrics` should map
    friendly names for the metric to a `MetricSpec` object defining which
    model outputs to evaluate against which labels with which metric
    function.

    Metric ops should support streaming, e.g., returning `update_op` and
    `value` tensors. For example, see the options defined in
    `../../../metrics/python/ops/metrics_ops.py`.

*  <b>`name`</b>: Name of the evaluation if user needs to run multiple evaluations on
    different data sets, such as on training data vs test data.
*  <b>`checkpoint_path`</b>: Path of a specific checkpoint to evaluate. If `None`, the
    latest checkpoint in `model_dir` is used.
*  <b>`hooks`</b>: List of `SessionRunHook` subclass instances. Used for callbacks
    inside the evaluation call.

##### Returns:

  Returns `dict` with evaluation results.


- - -

#### `tf.contrib.learn.Evaluable.model_dir` {#Evaluable.model_dir}

Returns a path in which the eval process will look for checkpoints.



- - -

### `class tf.contrib.learn.KMeansClustering` {#KMeansClustering}

An Estimator for K-Means clustering.
- - -

#### `tf.contrib.learn.KMeansClustering.__init__(num_clusters, model_dir=None, initial_clusters='random', distance_metric='squared_euclidean', random_seed=0, use_mini_batch=True, mini_batch_steps_per_iteration=1, kmeans_plus_plus_num_retries=2, relative_tolerance=None, config=None)` {#KMeansClustering.__init__}

Creates a model for running KMeans training and inference.

##### Args:


*  <b>`num_clusters`</b>: number of clusters to train.
*  <b>`model_dir`</b>: the directory to save the model results and log files.
*  <b>`initial_clusters`</b>: specifies how to initialize the clusters for training.
    See clustering_ops.kmeans for the possible values.
*  <b>`distance_metric`</b>: the distance metric used for clustering.
    See clustering_ops.kmeans for the possible values.
*  <b>`random_seed`</b>: Python integer. Seed for PRNG used to initialize centers.
*  <b>`use_mini_batch`</b>: If true, use the mini-batch k-means algorithm. Else assume
    full batch.
*  <b>`mini_batch_steps_per_iteration`</b>: number of steps after which the updated
    cluster centers are synced back to a master copy. See clustering_ops.py
    for more details.
*  <b>`kmeans_plus_plus_num_retries`</b>: For each point that is sampled during
    kmeans++ initialization, this parameter specifies the number of
    additional points to draw from the current distribution before selecting
    the best. If a negative value is specified, a heuristic is used to
    sample O(log(num_to_sample)) additional points.
*  <b>`relative_tolerance`</b>: A relative tolerance of change in the loss between
    iterations.  Stops learning if the loss changes less than this amount.
    Note that this may not work correctly if use_mini_batch=True.
*  <b>`config`</b>: See Estimator


- - -

#### `tf.contrib.learn.KMeansClustering.__repr__()` {#KMeansClustering.__repr__}




- - -

#### `tf.contrib.learn.KMeansClustering.clusters()` {#KMeansClustering.clusters}

Returns cluster centers.


- - -

#### `tf.contrib.learn.KMeansClustering.config` {#KMeansClustering.config}




- - -

#### `tf.contrib.learn.KMeansClustering.evaluate(*args, **kwargs)` {#KMeansClustering.evaluate}

See `Evaluable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.KMeansClustering.export(*args, **kwargs)` {#KMeansClustering.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn (and in most cases, input_feature_key) will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

##### Args:


*  <b>`export_dir`</b>: A string containing a directory to write the exported graph
    and checkpoints.
*  <b>`input_fn`</b>: If `use_deprecated_input_fn` is true, then a function that given
    `Tensor` of `Example` strings, parses it into features that are then
    passed to the model. Otherwise, a function that takes no argument and
    returns a tuple of (features, labels), where features is a dict of
    string key to `Tensor` and labels is a `Tensor` that's currently not
    used (and so can be `None`).
*  <b>`input_feature_key`</b>: Only used if `use_deprecated_input_fn` is false. String
    key into the features dict returned by `input_fn` that corresponds to a
    the raw `Example` strings `Tensor` that the exported model will take as
    input. Can only be `None` if you're using a custom `signature_fn` that
    does not use the first arg (examples).
*  <b>`use_deprecated_input_fn`</b>: Determines the signature format of `input_fn`.
*  <b>`signature_fn`</b>: Function that returns a default signature and a named
    signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
    for features and `Tensor` or `dict` of `Tensor`s for predictions.
*  <b>`prediction_key`</b>: The key for a tensor in the `predictions` dict (output
    from the `model_fn`) to use as the `predictions` input to the
    `signature_fn`. Optional. If `None`, predictions will pass to
    `signature_fn` without filtering.
*  <b>`default_batch_size`</b>: Default batch size of the `Example` placeholder.
*  <b>`exports_to_keep`</b>: Number of exports to keep.
*  <b>`checkpoint_path`</b>: the checkpoint path of the model to be exported. If it is
      `None` (which is default), will use the latest checkpoint in
      export_dir.

##### Returns:

  The string path to the exported directory. NB: this functionality was
  added ca. 2016/09/25; clients that depend on the return value may need
  to handle the case where this function returns None because subclasses
  are not returning a value.


- - -

#### `tf.contrib.learn.KMeansClustering.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, checkpoint_path=None)` {#KMeansClustering.export_savedmodel}

Exports inference graph as a SavedModel into given dir.

##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`serving_input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.  Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the most recent checkpoint found within the model directory is chosen.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.KMeansClustering.fit(*args, **kwargs)` {#KMeansClustering.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.KMeansClustering.get_params(deep=True)` {#KMeansClustering.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.KMeansClustering.get_variable_names()` {#KMeansClustering.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.KMeansClustering.get_variable_value(name)` {#KMeansClustering.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.KMeansClustering.model_dir` {#KMeansClustering.model_dir}




- - -

#### `tf.contrib.learn.KMeansClustering.partial_fit(*args, **kwargs)` {#KMeansClustering.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

#### `tf.contrib.learn.KMeansClustering.predict(*args, **kwargs)` {#KMeansClustering.predict}

Returns predictions for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...]. Can be iterator that
     returns arrays of features. The training input samples for fitting the
     model. If set, `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x` and 'batch_size' must be `None`.
*  <b>`batch_size`</b>: Override default batch size. If set, 'input_fn' must be
    'None'.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns all.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  A numpy array of predicted classes or regression values if the
  constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict`
  of numpy arrays if `model_fn` returns a `dict`. Returns an iterable of
  predictions if as_iterable is True.

##### Raises:


*  <b>`ValueError`</b>: If x and input_fn are both provided or both `None`.


- - -

#### `tf.contrib.learn.KMeansClustering.predict_cluster_idx(input_fn=None)` {#KMeansClustering.predict_cluster_idx}

Yields predicted cluster indices.


- - -

#### `tf.contrib.learn.KMeansClustering.score(input_fn=None, steps=None)` {#KMeansClustering.score}

Predict total sum of distances to nearest clusters.

Note that this function is different from the corresponding one in sklearn
which returns the negative of the sum of distances.

##### Args:


*  <b>`input_fn`</b>: see predict.
*  <b>`steps`</b>: see predict.

##### Returns:

  Total sum of distances to nearest clusters.


- - -

#### `tf.contrib.learn.KMeansClustering.set_params(**params)` {#KMeansClustering.set_params}

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

#### `tf.contrib.learn.KMeansClustering.transform(input_fn=None, as_iterable=False)` {#KMeansClustering.transform}

Transforms each element to distances to cluster centers.

Note that this function is different from the corresponding one in sklearn.
For SQUARED_EUCLIDEAN distance metric, sklearn transform returns the
EUCLIDEAN distance, while this function returns the SQUARED_EUCLIDEAN
distance.

##### Args:


*  <b>`input_fn`</b>: see predict.
*  <b>`as_iterable`</b>: see predict

##### Returns:

  Array with same number of rows as x, and num_clusters columns, containing
  distances to the cluster centers.



- - -

### `class tf.contrib.learn.ModeKeys` {#ModeKeys}

Standard names for model modes.

The following standard keys are defined:

* `TRAIN`: training mode.
* `EVAL`: evaluation mode.
* `INFER`: inference mode.

- - -

### `class tf.contrib.learn.ModelFnOps` {#ModelFnOps}

Ops returned from a model_fn.
- - -

#### `tf.contrib.learn.ModelFnOps.__getnewargs__()` {#ModelFnOps.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.contrib.learn.ModelFnOps.__getstate__()` {#ModelFnOps.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.contrib.learn.ModelFnOps.__new__(cls, mode, predictions=None, loss=None, train_op=None, eval_metric_ops=None, output_alternatives=None, training_chief_hooks=None, training_hooks=None, scaffold=None)` {#ModelFnOps.__new__}

Creates a validated `ModelFnOps` instance.

For a multi-headed model, the predictions dict here will contain the outputs
of all of the heads.  However: at serving time, requests will be made
specifically for one or more heads, and the RPCs used for these requests may
differ by problem type (i.e., regression, classification, other).  The
purpose of the output_alternatives dict is to aid in exporting a SavedModel
from which such head-specific queries can be served.  These
output_alternatives will be combined with input_alternatives (see
`saved_model_export_utils`) to produce a set of `SignatureDef`s specifying
the valid requests that can be served from this model.

For a single-headed model, it is still adviseable to provide
output_alternatives with a single entry, because this is how the problem
type is communicated for export and serving.  If output_alternatives is not
given, the resulting SavedModel will support only one head of unspecified
type.

##### Args:


*  <b>`mode`</b>: One of `ModeKeys`. Specifies if this training, evaluation or
    prediction.
*  <b>`predictions`</b>: Predictions `Tensor` or dict of `Tensor`.
*  <b>`loss`</b>: Training loss `Tensor`.
*  <b>`train_op`</b>: Op for the training step.
*  <b>`eval_metric_ops`</b>: Dict of metric results keyed by name. The values of the
    dict are the results of calling a metric function, such as `Tensor`.
*  <b>`output_alternatives`</b>: a dict of
    `{submodel_name: (problem_type, {tensor_name: Tensor})}`, where
    `submodel_name` is a submodel identifier that should be consistent
    across the pipeline (here likely taken from the name of each `Head`,
    for models that use them), `problem_type` is a `ProblemType`,
    `tensor_name` is a symbolic name for an output Tensor possibly but not
    necessarily taken from `PredictionKey`, and `Tensor` is the
    corresponding output Tensor itself.
*  <b>`training_chief_hooks`</b>: A list of `SessionRunHook` objects that will be
    run on the chief worker during training.
*  <b>`training_hooks`</b>: A list of `SessionRunHook` objects that will be run on
    all workers during training.
*  <b>`scaffold`</b>: A `tf.train.Scaffold` object that can be used to set
    initialization, saver, and more to be used in training.

##### Returns:

  A validated `ModelFnOps` object.

##### Raises:


*  <b>`ValueError`</b>: If validation fails.


- - -

#### `tf.contrib.learn.ModelFnOps.__repr__()` {#ModelFnOps.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.contrib.learn.ModelFnOps.eval_metric_ops` {#ModelFnOps.eval_metric_ops}

Alias for field number 3


- - -

#### `tf.contrib.learn.ModelFnOps.loss` {#ModelFnOps.loss}

Alias for field number 1


- - -

#### `tf.contrib.learn.ModelFnOps.output_alternatives` {#ModelFnOps.output_alternatives}

Alias for field number 4


- - -

#### `tf.contrib.learn.ModelFnOps.predictions` {#ModelFnOps.predictions}

Alias for field number 0


- - -

#### `tf.contrib.learn.ModelFnOps.scaffold` {#ModelFnOps.scaffold}

Alias for field number 7


- - -

#### `tf.contrib.learn.ModelFnOps.train_op` {#ModelFnOps.train_op}

Alias for field number 2


- - -

#### `tf.contrib.learn.ModelFnOps.training_chief_hooks` {#ModelFnOps.training_chief_hooks}

Alias for field number 5


- - -

#### `tf.contrib.learn.ModelFnOps.training_hooks` {#ModelFnOps.training_hooks}

Alias for field number 6



- - -

### `class tf.contrib.learn.MetricSpec` {#MetricSpec}

MetricSpec connects a model to metric functions.

The MetricSpec class contains all information necessary to connect the
output of a `model_fn` to the metrics (usually, streaming metrics) that are
used in evaluation.

It is passed in the `metrics` argument of `Estimator.evaluate`. The
`Estimator` then knows which predictions, labels, and weight to use to call a
given metric function.

When building the ops to run in evaluation, `Estimator` will call
`create_metric_ops`, which will connect the given `metric_fn` to the model
as detailed in the docstring for `create_metric_ops`, and return the metric.

Example:

Assuming a model has an input function which returns inputs containing
(among other things) a tensor with key "input_key", and a labels dictionary
containing "label_key". Let's assume that the `model_fn` for this model
returns a prediction with key "prediction_key".

In order to compute the accuracy of the "prediction_key" prediction, we
would add

```
"prediction accuracy": MetricSpec(metric_fn=prediction_accuracy_fn,
                                  prediction_key="prediction_key",
                                  label_key="label_key")
```

to the metrics argument to `evaluate`. `prediction_accuracy_fn` can be either
a predefined function in metric_ops (e.g., `streaming_accuracy`) or a custom
function you define.

If we would like the accuracy to be weighted by "input_key", we can add that
as the `weight_key` argument.

```
"prediction accuracy": MetricSpec(metric_fn=prediction_accuracy_fn,
                                  prediction_key="prediction_key",
                                  label_key="label_key",
                                  weight_key="input_key")
```

An end-to-end example is as follows:

```
estimator = tf.contrib.learn.Estimator(...)
estimator.fit(...)
_ = estimator.evaluate(
    input_fn=input_fn,
    steps=1,
    metrics={
        'prediction accuracy':
            metric_spec.MetricSpec(
                metric_fn=prediction_accuracy_fn,
                prediction_key="prediction_key",
                label_key="label_key")
    })
```
- - -

#### `tf.contrib.learn.MetricSpec.__init__(metric_fn, prediction_key=None, label_key=None, weight_key=None)` {#MetricSpec.__init__}

Constructor.

Creates a MetricSpec.

##### Args:


*  <b>`metric_fn`</b>: A function to use as a metric. See `_adapt_metric_fn` for
    rules on how `predictions`, `labels`, and `weights` are passed to this
    function. This must return either a single `Tensor`, which is
    interpreted as a value of this metric, or a pair
    `(value_op, update_op)`, where `value_op` is the op to call to
    obtain the value of the metric, and `update_op` should be run for
    each batch to update internal state.
*  <b>`prediction_key`</b>: The key for a tensor in the `predictions` dict (output
    from the `model_fn`) to use as the `predictions` input to the
    `metric_fn`. Optional. If `None`, the `model_fn` must return a single
    tensor or a dict with only a single entry as `predictions`.
*  <b>`label_key`</b>: The key for a tensor in the `labels` dict (output from the
    `input_fn`) to use as the `labels` input to the `metric_fn`.
    Optional. If `None`, the `input_fn` must return a single tensor or a
    dict with only a single entry as `labels`.
*  <b>`weight_key`</b>: The key for a tensor in the `inputs` dict (output from the
    `input_fn`) to use as the `weights` input to the `metric_fn`.
    Optional. If `None`, no weights will be passed to the `metric_fn`.


- - -

#### `tf.contrib.learn.MetricSpec.__str__()` {#MetricSpec.__str__}




- - -

#### `tf.contrib.learn.MetricSpec.create_metric_ops(inputs, labels, predictions)` {#MetricSpec.create_metric_ops}

Connect our `metric_fn` to the specified members of the given dicts.

This function will call the `metric_fn` given in our constructor as follows:

```
  metric_fn(predictions[self.prediction_key],
            labels[self.label_key],
            weights=weights[self.weight_key])
```

And returns the result. The `weights` argument is only passed if
`self.weight_key` is not `None`.

`predictions` and `labels` may be single tensors as well as dicts. If
`predictions` is a single tensor, `self.prediction_key` must be `None`. If
`predictions` is a single element dict, `self.prediction_key` is allowed to
be `None`. Conversely, if `labels` is a single tensor, `self.label_key` must
be `None`. If `labels` is a single element dict, `self.label_key` is allowed
to be `None`.

##### Args:


*  <b>`inputs`</b>: A dict of inputs produced by the `input_fn`
*  <b>`labels`</b>: A dict of labels or a single label tensor produced by the
    `input_fn`.
*  <b>`predictions`</b>: A dict of predictions or a single tensor produced by the
    `model_fn`.

##### Returns:

  The result of calling `metric_fn`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` or `labels` is a single `Tensor` and
    `self.prediction_key` or `self.label_key` is not `None`; or if
    `self.label_key` is `None` but `labels` is a dict with more than one
    element, or if `self.prediction_key` is `None` but `predictions` is a
    dict with more than one element.


- - -

#### `tf.contrib.learn.MetricSpec.label_key` {#MetricSpec.label_key}




- - -

#### `tf.contrib.learn.MetricSpec.metric_fn` {#MetricSpec.metric_fn}

Metric function.

This function accepts named args: `predictions`, `labels`, `weights`. It
returns a single `Tensor` or `(value_op, update_op)` pair. See `metric_fn`
constructor argument for more details.

##### Returns:

  Function, see `metric_fn` constructor argument for more details.


- - -

#### `tf.contrib.learn.MetricSpec.prediction_key` {#MetricSpec.prediction_key}




- - -

#### `tf.contrib.learn.MetricSpec.weight_key` {#MetricSpec.weight_key}





- - -

### `class tf.contrib.learn.PredictionKey` {#PredictionKey}



- - -

### `class tf.contrib.learn.DNNClassifier` {#DNNClassifier}

A classifier for TensorFlow DNN models.

Example:

```python
sparse_feature_a = sparse_column_with_hash_bucket(...)
sparse_feature_b = sparse_column_with_hash_bucket(...)

sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                        ...)
sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                        ...)

estimator = DNNClassifier(
    feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
    hidden_units=[1024, 512, 256])

# Or estimator using the ProximalAdagradOptimizer optimizer with
# regularization.
estimator = DNNClassifier(
    feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Input builders
def input_fn_train: # returns x, y (where y represents label's class index).
  pass
estimator.fit(input_fn=input_fn_train)

def input_fn_eval: # returns x, y (where y represents label's class index).
  pass
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x) # returns predicted labels (i.e. label's class index).
```

Input of `fit` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

* if `weight_column_name` is not `None`, a feature with
   `key=weight_column_name` whose value is a `Tensor`.
* for each `column` in `feature_columns`:
  - if `column` is a `SparseColumn`, a feature with `key=column.name`
    whose `value` is a `SparseTensor`.
  - if `column` is a `WeightedSparseColumn`, two features: the first with
    `key` the id column name, the second with `key` the weight column name.
    Both features' `value` must be a `SparseTensor`.
  - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
    whose `value` is a `Tensor`.
- - -

#### `tf.contrib.learn.DNNClassifier.__init__(hidden_units, feature_columns, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None, activation_fn=relu, dropout=None, gradient_clip_norm=None, enable_centered_bias=False, config=None, feature_engineering_fn=None, embedding_lr_multipliers=None, input_layer_min_slice_size=None)` {#DNNClassifier.__init__}

Initializes a DNNClassifier instance.

##### Args:


*  <b>`hidden_units`</b>: List of hidden units per layer. All layers are fully
    connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
    has 32.
*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator to
    continue training a previously saved model.
*  <b>`n_classes`</b>: number of label classes. Default is binary classification.
    It must be greater than 1. Note: Class labels are integers representing
    the class index (i.e. values from 0 to n_classes-1). For arbitrary
    label values (e.g. string labels), convert to class indices first.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
*  <b>`optimizer`</b>: An instance of `tf.Optimizer` used to train the model. If
    `None`, will use an Adagrad optimizer.
*  <b>`activation_fn`</b>: Activation function applied to each layer. If `None`, will
    use `tf.nn.relu`.
*  <b>`dropout`</b>: When not `None`, the probability we will drop out a given
    coordinate.
*  <b>`gradient_clip_norm`</b>: A float > 0. If provided, gradients are
    clipped to their global norm with this clipping ratio. See
    `tf.clip_by_global_norm` for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.
*  <b>`embedding_lr_multipliers`</b>: Optional. A dictionary from `EmbeddingColumn` to
      a `float` multiplier. Multiplier will be used to multiply with
      learning rate for the embedding variables.
*  <b>`input_layer_min_slice_size`</b>: Optional. The min slice size of input layer
      partitions. If not provided, will use the default of 64M.

##### Returns:

  A `DNNClassifier` estimator.

##### Raises:


*  <b>`ValueError`</b>: If `n_classes` < 2.


- - -

#### `tf.contrib.learn.DNNClassifier.__repr__()` {#DNNClassifier.__repr__}




- - -

#### `tf.contrib.learn.DNNClassifier.bias_` {#DNNClassifier.bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNClassifier.config` {#DNNClassifier.config}




- - -

#### `tf.contrib.learn.DNNClassifier.evaluate(*args, **kwargs)` {#DNNClassifier.evaluate}

See `Evaluable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.DNNClassifier.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#DNNClassifier.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.DNNClassifier.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, checkpoint_path=None)` {#DNNClassifier.export_savedmodel}

Exports inference graph as a SavedModel into given dir.

##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`serving_input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.  Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the most recent checkpoint found within the model directory is chosen.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.DNNClassifier.fit(*args, **kwargs)` {#DNNClassifier.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.DNNClassifier.get_params(deep=True)` {#DNNClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.DNNClassifier.get_variable_names()` {#DNNClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.DNNClassifier.get_variable_value(name)` {#DNNClassifier.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.DNNClassifier.model_dir` {#DNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.DNNClassifier.partial_fit(*args, **kwargs)` {#DNNClassifier.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

#### `tf.contrib.learn.DNNClassifier.predict(*args, **kwargs)` {#DNNClassifier.predict}

Returns predictions for given features. (deprecated arguments) (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2017-03-01.
Instructions for updating:
Please switch to predict_classes, or set `outputs` argument.

By default, returns predicted classes. But this default will be dropped
soon. Users should either pass `outputs`, or call `predict_classes` method.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns classes.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted classes with shape [batch_size] (or an iterable
  of predicted classes if as_iterable is True). Each predicted class is
  represented by its class index (i.e. integer from 0 to n_classes-1).
  If `outputs` is set, returns a dict of predictions.


- - -

#### `tf.contrib.learn.DNNClassifier.predict_classes(*args, **kwargs)` {#DNNClassifier.predict_classes}

Returns predicted classes for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted classes with shape [batch_size] (or an iterable
  of predicted classes if as_iterable is True). Each predicted class is
  represented by its class index (i.e. integer from 0 to n_classes-1).


- - -

#### `tf.contrib.learn.DNNClassifier.predict_proba(*args, **kwargs)` {#DNNClassifier.predict_proba}

Returns predicted probabilities for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted probabilities with shape [batch_size, n_classes]
  (or an iterable of predicted probabilities if as_iterable is True).


- - -

#### `tf.contrib.learn.DNNClassifier.set_params(**params)` {#DNNClassifier.set_params}

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

#### `tf.contrib.learn.DNNClassifier.weights_` {#DNNClassifier.weights_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().



- - -

### `class tf.contrib.learn.DNNRegressor` {#DNNRegressor}

A regressor for TensorFlow DNN models.

Example:

```python
sparse_feature_a = sparse_column_with_hash_bucket(...)
sparse_feature_b = sparse_column_with_hash_bucket(...)

sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                        ...)
sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                        ...)

estimator = DNNRegressor(
    feature_columns=[sparse_feature_a, sparse_feature_b],
    hidden_units=[1024, 512, 256])

# Or estimator using the ProximalAdagradOptimizer optimizer with
# regularization.
estimator = DNNRegressor(
    feature_columns=[sparse_feature_a, sparse_feature_b],
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Input builders
def input_fn_train: # returns x, y
  pass
estimator.fit(input_fn=input_fn_train)

def input_fn_eval: # returns x, y
  pass
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x)
```

Input of `fit` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

* if `weight_column_name` is not `None`, a feature with
  `key=weight_column_name` whose value is a `Tensor`.
* for each `column` in `feature_columns`:
  - if `column` is a `SparseColumn`, a feature with `key=column.name`
    whose `value` is a `SparseTensor`.
  - if `column` is a `WeightedSparseColumn`, two features: the first with
    `key` the id column name, the second with `key` the weight column name.
    Both features' `value` must be a `SparseTensor`.
  - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
    whose `value` is a `Tensor`.
- - -

#### `tf.contrib.learn.DNNRegressor.__init__(hidden_units, feature_columns, model_dir=None, weight_column_name=None, optimizer=None, activation_fn=relu, dropout=None, gradient_clip_norm=None, enable_centered_bias=False, config=None, feature_engineering_fn=None, label_dimension=1, embedding_lr_multipliers=None, input_layer_min_slice_size=None)` {#DNNRegressor.__init__}

Initializes a `DNNRegressor` instance.

##### Args:


*  <b>`hidden_units`</b>: List of hidden units per layer. All layers are fully
    connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
    has 32.
*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator to
    continue training a previously saved model.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
*  <b>`optimizer`</b>: An instance of `tf.Optimizer` used to train the model. If
    `None`, will use an Adagrad optimizer.
*  <b>`activation_fn`</b>: Activation function applied to each layer. If `None`, will
    use `tf.nn.relu`.
*  <b>`dropout`</b>: When not `None`, the probability we will drop out a given
    coordinate.
*  <b>`gradient_clip_norm`</b>: A `float` > 0. If provided, gradients are clipped
    to their global norm with this clipping ratio. See
    `tf.clip_by_global_norm` for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.
*  <b>`label_dimension`</b>: Number of regression targets per example. This is the
    size of the last dimension of the labels and logits `Tensor` objects
    (typically, these have shape `[batch_size, label_dimension]`).
*  <b>`embedding_lr_multipliers`</b>: Optional. A dictionary from `EbeddingColumn` to
      a `float` multiplier. Multiplier will be used to multiply with
      learning rate for the embedding variables.
*  <b>`input_layer_min_slice_size`</b>: Optional. The min slice size of input layer
      partitions. If not provided, will use the default of 64M.

##### Returns:

  A `DNNRegressor` estimator.


- - -

#### `tf.contrib.learn.DNNRegressor.__repr__()` {#DNNRegressor.__repr__}




- - -

#### `tf.contrib.learn.DNNRegressor.config` {#DNNRegressor.config}




- - -

#### `tf.contrib.learn.DNNRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None, checkpoint_path=None, hooks=None)` {#DNNRegressor.evaluate}

See evaluable.Evaluable.


- - -

#### `tf.contrib.learn.DNNRegressor.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#DNNRegressor.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.DNNRegressor.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, checkpoint_path=None)` {#DNNRegressor.export_savedmodel}

Exports inference graph as a SavedModel into given dir.

##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`serving_input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.  Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the most recent checkpoint found within the model directory is chosen.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.DNNRegressor.fit(*args, **kwargs)` {#DNNRegressor.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.DNNRegressor.get_params(deep=True)` {#DNNRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
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

#### `tf.contrib.learn.DNNRegressor.model_dir` {#DNNRegressor.model_dir}




- - -

#### `tf.contrib.learn.DNNRegressor.partial_fit(*args, **kwargs)` {#DNNRegressor.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

#### `tf.contrib.learn.DNNRegressor.predict(*args, **kwargs)` {#DNNRegressor.predict}

Returns predictions for given features. (deprecated arguments) (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2017-03-01.
Instructions for updating:
Please switch to predict_scores, or set `outputs` argument.

By default, returns predicted scores. But this default will be dropped
soon. Users should either pass `outputs`, or call `predict_scores` method.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns scores.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted scores (or an iterable of predicted scores if
  as_iterable is True). If `label_dimension == 1`, the shape of the output
  is `[batch_size]`, otherwise the shape is `[batch_size, label_dimension]`.
  If `outputs` is set, returns a dict of predictions.


- - -

#### `tf.contrib.learn.DNNRegressor.predict_scores(*args, **kwargs)` {#DNNRegressor.predict_scores}

Returns predicted scores for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted scores (or an iterable of predicted scores if
  as_iterable is True). If `label_dimension == 1`, the shape of the output
  is `[batch_size]`, otherwise the shape is `[batch_size, label_dimension]`.


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

### `class tf.contrib.learn.DNNLinearCombinedRegressor` {#DNNLinearCombinedRegressor}

A regressor for TensorFlow Linear and DNN joined training models.

Example:

```python
sparse_feature_a = sparse_column_with_hash_bucket(...)
sparse_feature_b = sparse_column_with_hash_bucket(...)

sparse_feature_a_x_sparse_feature_b = crossed_column(...)

sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                        ...)
sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                        ...)

estimator = DNNLinearCombinedRegressor(
    # common settings
    weight_column_name=weight_column_name,
    # wide settings
    linear_feature_columns=[sparse_feature_a_x_sparse_feature_b],
    linear_optimizer=tf.train.FtrlOptimizer(...),
    # deep settings
    dnn_feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
    dnn_hidden_units=[1000, 500, 100],
    dnn_optimizer=tf.train.ProximalAdagradOptimizer(...))

# To apply L1 and L2 regularization, you can set optimizers as follows:
tf.train.ProximalAdagradOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.001,
    l2_regularization_strength=0.001)
# It is same for FtrlOptimizer.

# Input builders
def input_fn_train: # returns x, y
  ...
def input_fn_eval: # returns x, y
  ...
estimator.train(input_fn_train)
estimator.evaluate(input_fn_eval)
estimator.predict(x)
```

Input of `fit`, `train`, and `evaluate` should have following features,
  otherwise there will be a `KeyError`:
    if `weight_column_name` is not `None`, a feature with
      `key=weight_column_name` whose value is a `Tensor`.
    for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn, a feature with `key=column.name`
      whose `value` is a `Tensor`.
- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.__init__(model_dir=None, weight_column_name=None, linear_feature_columns=None, linear_optimizer=None, _joint_linear_weights=False, dnn_feature_columns=None, dnn_optimizer=None, dnn_hidden_units=None, dnn_activation_fn=relu, dnn_dropout=None, gradient_clip_norm=None, enable_centered_bias=False, label_dimension=1, config=None, feature_engineering_fn=None, embedding_lr_multipliers=None, input_layer_min_slice_size=None)` {#DNNLinearCombinedRegressor.__init__}

Initializes a DNNLinearCombinedRegressor instance.

##### Args:


*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator
    to continue training a previously saved model.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
*  <b>`linear_feature_columns`</b>: An iterable containing all the feature columns
    used by linear part of the model. All items in the set must be
    instances of classes derived from `FeatureColumn`.
*  <b>`linear_optimizer`</b>: An instance of `tf.Optimizer` used to apply gradients to
    the linear part of the model. If `None`, will use a FTRL optimizer.
  _joint_linear_weights: If True a single (possibly partitioned) variable
    will be used to store the linear model weights. It's faster, but
    requires that all columns are sparse and have the 'sum' combiner.

*  <b>`dnn_feature_columns`</b>: An iterable containing all the feature columns used
    by deep part of the model. All items in the set must be instances of
    classes derived from `FeatureColumn`.
*  <b>`dnn_optimizer`</b>: An instance of `tf.Optimizer` used to apply gradients to
    the deep part of the model. If `None`, will use an Adagrad optimizer.
*  <b>`dnn_hidden_units`</b>: List of hidden units per layer. All layers are fully
    connected.
*  <b>`dnn_activation_fn`</b>: Activation function applied to each layer. If None,
    will use `tf.nn.relu`.
*  <b>`dnn_dropout`</b>: When not None, the probability we will drop out
    a given coordinate.
*  <b>`gradient_clip_norm`</b>: A float > 0. If provided, gradients are clipped
    to their global norm with this clipping ratio. See
    tf.clip_by_global_norm for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
*  <b>`label_dimension`</b>: Number of regression targets per example. This is the
    size of the last dimension of the labels and logits `Tensor` objects
    (typically, these have shape `[batch_size, label_dimension]`).
*  <b>`config`</b>: RunConfig object to configure the runtime settings.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.
*  <b>`embedding_lr_multipliers`</b>: Optional. A dictionary from `EmbeddingColumn` to
      a `float` multiplier. Multiplier will be used to multiply with
      learning rate for the embedding variables.
*  <b>`input_layer_min_slice_size`</b>: Optional. The min slice size of input layer
      partitions. If not provided, will use the default of 64M.


##### Raises:


*  <b>`ValueError`</b>: If both linear_feature_columns and dnn_features_columns are
    empty at the same time.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.__repr__()` {#DNNLinearCombinedRegressor.__repr__}




- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.config` {#DNNLinearCombinedRegressor.config}




- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None, checkpoint_path=None, hooks=None)` {#DNNLinearCombinedRegressor.evaluate}

See evaluable.Evaluable.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#DNNLinearCombinedRegressor.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, checkpoint_path=None)` {#DNNLinearCombinedRegressor.export_savedmodel}

Exports inference graph as a SavedModel into given dir.

##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`serving_input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.  Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the most recent checkpoint found within the model directory is chosen.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.fit(*args, **kwargs)` {#DNNLinearCombinedRegressor.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.get_params(deep=True)` {#DNNLinearCombinedRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.get_variable_names()` {#DNNLinearCombinedRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.get_variable_value(name)` {#DNNLinearCombinedRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.model_dir` {#DNNLinearCombinedRegressor.model_dir}




- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.partial_fit(*args, **kwargs)` {#DNNLinearCombinedRegressor.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

#### `tf.contrib.learn.DNNLinearCombinedRegressor.predict(*args, **kwargs)` {#DNNLinearCombinedRegressor.predict}

Returns predictions for given features. (deprecated arguments) (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2017-03-01.
Instructions for updating:
Please switch to predict_scores, or set `outputs` argument.

By default, returns predicted scores. But this default will be dropped
soon. Users should either pass `outputs`, or call `predict_scores` method.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns scores.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted scores (or an iterable of predicted scores if
  as_iterable is True). If `label_dimension == 1`, the shape of the output
  is `[batch_size]`, otherwise the shape is `[batch_size, label_dimension]`.
  If `outputs` is set, returns a dict of predictions.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.predict_scores(*args, **kwargs)` {#DNNLinearCombinedRegressor.predict_scores}

Returns predicted scores for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted scores (or an iterable of predicted scores if
  as_iterable is True). If `label_dimension == 1`, the shape of the output
  is `[batch_size]`, otherwise the shape is `[batch_size, label_dimension]`.


- - -

#### `tf.contrib.learn.DNNLinearCombinedRegressor.set_params(**params)` {#DNNLinearCombinedRegressor.set_params}

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

### `class tf.contrib.learn.DNNLinearCombinedClassifier` {#DNNLinearCombinedClassifier}

A classifier for TensorFlow Linear and DNN joined training models.

Example:

```python
sparse_feature_a = sparse_column_with_hash_bucket(...)
sparse_feature_b = sparse_column_with_hash_bucket(...)

sparse_feature_a_x_sparse_feature_b = crossed_column(...)

sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                        ...)
sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                        ...)

estimator = DNNLinearCombinedClassifier(
    # common settings
    n_classes=n_classes,
    weight_column_name=weight_column_name,
    # wide settings
    linear_feature_columns=[sparse_feature_a_x_sparse_feature_b],
    linear_optimizer=tf.train.FtrlOptimizer(...),
    # deep settings
    dnn_feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
    dnn_hidden_units=[1000, 500, 100],
    dnn_optimizer=tf.train.AdagradOptimizer(...))

# Input builders
def input_fn_train: # returns x, y (where y represents label's class index).
  ...
def input_fn_eval: # returns x, y (where y represents label's class index).
  ...
estimator.fit(input_fn=input_fn_train)
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x) # returns predicted labels (i.e. label's class index).
```

Input of `fit` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:
    if `weight_column_name` is not `None`, a feature with
      `key=weight_column_name` whose value is a `Tensor`.
    for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn, a feature with `key=column.name`
      whose `value` is a `Tensor`.
- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.__init__(model_dir=None, n_classes=2, weight_column_name=None, linear_feature_columns=None, linear_optimizer=None, _joint_linear_weights=False, dnn_feature_columns=None, dnn_optimizer=None, dnn_hidden_units=None, dnn_activation_fn=relu, dnn_dropout=None, gradient_clip_norm=None, enable_centered_bias=False, config=None, feature_engineering_fn=None, embedding_lr_multipliers=None, input_layer_min_slice_size=None)` {#DNNLinearCombinedClassifier.__init__}

Constructs a DNNLinearCombinedClassifier instance.

##### Args:


*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator
    to continue training a previously saved model.
*  <b>`n_classes`</b>: number of label classes. Default is binary classification.
    Note that class labels are integers representing the class index (i.e.
    values from 0 to n_classes-1). For arbitrary label values (e.g. string
    labels), convert to class indices first.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training.
    It will be multiplied by the loss of the example.
*  <b>`linear_feature_columns`</b>: An iterable containing all the feature columns
    used by linear part of the model. All items in the set must be
    instances of classes derived from `FeatureColumn`.
*  <b>`linear_optimizer`</b>: An instance of `tf.Optimizer` used to apply gradients to
    the linear part of the model. If `None`, will use a FTRL optimizer.
  _joint_linear_weights: If True a single (possibly partitioned) variable
    will be used to store the linear model weights. It's faster, but
    requires all columns are sparse and have the 'sum' combiner.

*  <b>`dnn_feature_columns`</b>: An iterable containing all the feature columns used
    by deep part of the model. All items in the set must be instances of
    classes derived from `FeatureColumn`.
*  <b>`dnn_optimizer`</b>: An instance of `tf.Optimizer` used to apply gradients to
    the deep part of the model. If `None`, will use an Adagrad optimizer.
*  <b>`dnn_hidden_units`</b>: List of hidden units per layer. All layers are fully
    connected.
*  <b>`dnn_activation_fn`</b>: Activation function applied to each layer. If `None`,
    will use `tf.nn.relu`.
*  <b>`dnn_dropout`</b>: When not None, the probability we will drop out
    a given coordinate.
*  <b>`gradient_clip_norm`</b>: A float > 0. If provided, gradients are clipped
    to their global norm with this clipping ratio. See
    tf.clip_by_global_norm for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
*  <b>`config`</b>: RunConfig object to configure the runtime settings.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.
*  <b>`embedding_lr_multipliers`</b>: Optional. A dictionary from `EmbeddingColumn` to
      a `float` multiplier. Multiplier will be used to multiply with
      learning rate for the embedding variables.
*  <b>`input_layer_min_slice_size`</b>: Optional. The min slice size of input layer
      partitions. If not provided, will use the default of 64M.

##### Raises:


*  <b>`ValueError`</b>: If `n_classes` < 2.
*  <b>`ValueError`</b>: If both `linear_feature_columns` and `dnn_features_columns`
    are empty at the same time.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.__repr__()` {#DNNLinearCombinedClassifier.__repr__}




- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.config` {#DNNLinearCombinedClassifier.config}




- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.dnn_bias_` {#DNNLinearCombinedClassifier.dnn_bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.dnn_weights_` {#DNNLinearCombinedClassifier.dnn_weights_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.evaluate(*args, **kwargs)` {#DNNLinearCombinedClassifier.evaluate}

See `Evaluable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#DNNLinearCombinedClassifier.export}

See BasEstimator.export.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, checkpoint_path=None)` {#DNNLinearCombinedClassifier.export_savedmodel}

Exports inference graph as a SavedModel into given dir.

##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`serving_input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.  Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the most recent checkpoint found within the model directory is chosen.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.fit(*args, **kwargs)` {#DNNLinearCombinedClassifier.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.get_params(deep=True)` {#DNNLinearCombinedClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.get_variable_names()` {#DNNLinearCombinedClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.get_variable_value(name)` {#DNNLinearCombinedClassifier.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.linear_bias_` {#DNNLinearCombinedClassifier.linear_bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.linear_weights_` {#DNNLinearCombinedClassifier.linear_weights_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.model_dir` {#DNNLinearCombinedClassifier.model_dir}




- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.partial_fit(*args, **kwargs)` {#DNNLinearCombinedClassifier.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

#### `tf.contrib.learn.DNNLinearCombinedClassifier.predict(*args, **kwargs)` {#DNNLinearCombinedClassifier.predict}

Returns predictions for given features. (deprecated arguments) (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2017-03-01.
Instructions for updating:
Please switch to predict_classes, or set `outputs` argument.

By default, returns predicted classes. But this default will be dropped
soon. Users should either pass `outputs`, or call `predict_classes` method.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns classes.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted classes with shape [batch_size] (or an iterable
  of predicted classes if as_iterable is True). Each predicted class is
  represented by its class index (i.e. integer from 0 to n_classes-1).
  If `outputs` is set, returns a dict of predictions.


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.predict_classes(*args, **kwargs)` {#DNNLinearCombinedClassifier.predict_classes}

Returns predicted classes for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted classes with shape [batch_size] (or an iterable
  of predicted classes if as_iterable is True). Each predicted class is
  represented by its class index (i.e. integer from 0 to n_classes-1).


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.predict_proba(*args, **kwargs)` {#DNNLinearCombinedClassifier.predict_proba}

Returns prediction probabilities for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted probabilities with shape [batch_size, n_classes]
  (or an iterable of predicted probabilities if as_iterable is True).


- - -

#### `tf.contrib.learn.DNNLinearCombinedClassifier.set_params(**params)` {#DNNLinearCombinedClassifier.set_params}

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

### `class tf.contrib.learn.LinearClassifier` {#LinearClassifier}

Linear classifier model.

Train a linear model to classify instances into one of multiple possible
classes. When number of possible classes is 2, this is binary classification.

Example:

```python
sparse_column_a = sparse_column_with_hash_bucket(...)
sparse_column_b = sparse_column_with_hash_bucket(...)

sparse_feature_a_x_sparse_feature_b = crossed_column(...)

# Estimator using the default optimizer.
estimator = LinearClassifier(
    feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

# Or estimator using the FTRL optimizer with regularization.
estimator = LinearClassifier(
    feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
    optimizer=tf.train.FtrlOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Or estimator using the SDCAOptimizer.
estimator = LinearClassifier(
   feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
   optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
     example_id_column='example_id',
     num_loss_partitions=...,
     symmetric_l2_regularization=2.0
   ))

# Input builders
def input_fn_train: # returns x, y (where y represents label's class index).
  ...
def input_fn_eval: # returns x, y (where y represents label's class index).
  ...
estimator.fit(input_fn=input_fn_train)
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x) # returns predicted labels (i.e. label's class index).
```

Input of `fit` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

* if `weight_column_name` is not `None`, a feature with
  `key=weight_column_name` whose value is a `Tensor`.
* for each `column` in `feature_columns`:
  - if `column` is a `SparseColumn`, a feature with `key=column.name`
    whose `value` is a `SparseTensor`.
  - if `column` is a `WeightedSparseColumn`, two features: the first with
    `key` the id column name, the second with `key` the weight column name.
    Both features' `value` must be a `SparseTensor`.
  - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
    whose `value` is a `Tensor`.
- - -

#### `tf.contrib.learn.LinearClassifier.__init__(feature_columns, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None, gradient_clip_norm=None, enable_centered_bias=False, _joint_weight=False, config=None, feature_engineering_fn=None)` {#LinearClassifier.__init__}

Construct a `LinearClassifier` estimator object.

##### Args:


*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator
    to continue training a previously saved model.
*  <b>`n_classes`</b>: number of label classes. Default is binary classification.
    Note that class labels are integers representing the class index (i.e.
    values from 0 to n_classes-1). For arbitrary label values (e.g. string
    labels), convert to class indices first.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
*  <b>`optimizer`</b>: The optimizer used to train the model. If specified, it should
    be either an instance of `tf.Optimizer` or the SDCAOptimizer. If `None`,
    the Ftrl optimizer will be used.
*  <b>`gradient_clip_norm`</b>: A `float` > 0. If provided, gradients are clipped
    to their global norm with this clipping ratio. See
    `tf.clip_by_global_norm` for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
  _joint_weight: If True, the weights for all columns will be stored in a
    single (possibly partitioned) variable. It's more efficient, but it's
    incompatible with SDCAOptimizer, and requires all feature columns are
    sparse and use the 'sum' combiner.

*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.

##### Returns:

  A `LinearClassifier` estimator.

##### Raises:


*  <b>`ValueError`</b>: if n_classes < 2.


- - -

#### `tf.contrib.learn.LinearClassifier.__repr__()` {#LinearClassifier.__repr__}




- - -

#### `tf.contrib.learn.LinearClassifier.bias_` {#LinearClassifier.bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.LinearClassifier.config` {#LinearClassifier.config}




- - -

#### `tf.contrib.learn.LinearClassifier.evaluate(*args, **kwargs)` {#LinearClassifier.evaluate}

See `Evaluable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.LinearClassifier.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#LinearClassifier.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.LinearClassifier.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, checkpoint_path=None)` {#LinearClassifier.export_savedmodel}

Exports inference graph as a SavedModel into given dir.

##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`serving_input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.  Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the most recent checkpoint found within the model directory is chosen.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.LinearClassifier.fit(*args, **kwargs)` {#LinearClassifier.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.LinearClassifier.get_params(deep=True)` {#LinearClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
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

#### `tf.contrib.learn.LinearClassifier.model_dir` {#LinearClassifier.model_dir}




- - -

#### `tf.contrib.learn.LinearClassifier.partial_fit(*args, **kwargs)` {#LinearClassifier.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

#### `tf.contrib.learn.LinearClassifier.predict(*args, **kwargs)` {#LinearClassifier.predict}

Returns predictions for given features. (deprecated arguments) (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2017-03-01.
Instructions for updating:
Please switch to predict_classes, or set `outputs` argument.

By default, returns predicted classes. But this default will be dropped
soon. Users should either pass `outputs`, or call `predict_classes` method.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns classes.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted classes with shape [batch_size] (or an iterable
  of predicted classes if as_iterable is True). Each predicted class is
  represented by its class index (i.e. integer from 0 to n_classes-1).
  If `outputs` is set, returns a dict of predictions.


- - -

#### `tf.contrib.learn.LinearClassifier.predict_classes(*args, **kwargs)` {#LinearClassifier.predict_classes}

Returns predicted classes for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted classes with shape [batch_size] (or an iterable
  of predicted classes if as_iterable is True). Each predicted class is
  represented by its class index (i.e. integer from 0 to n_classes-1).


- - -

#### `tf.contrib.learn.LinearClassifier.predict_proba(*args, **kwargs)` {#LinearClassifier.predict_proba}

Returns predicted probabilities for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x and y must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted probabilities with shape [batch_size, n_classes]
  (or an iterable of predicted probabilities if as_iterable is True).


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

#### `tf.contrib.learn.LinearClassifier.weights_` {#LinearClassifier.weights_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().



- - -

### `class tf.contrib.learn.LinearRegressor` {#LinearRegressor}

Linear regressor model.

Train a linear regression model to predict label value given observation of
feature values.

Example:

```python
sparse_column_a = sparse_column_with_hash_bucket(...)
sparse_column_b = sparse_column_with_hash_bucket(...)

sparse_feature_a_x_sparse_feature_b = crossed_column(...)

estimator = LinearRegressor(
    feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

# Input builders
def input_fn_train: # returns x, y
  ...
def input_fn_eval: # returns x, y
  ...
estimator.fit(input_fn=input_fn_train)
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x)
```

Input of `fit` and `evaluate` should have following features,
  otherwise there will be a KeyError:

* if `weight_column_name` is not `None`:
  key=weight_column_name, value=a `Tensor`
* for column in `feature_columns`:
  - if isinstance(column, `SparseColumn`):
      key=column.name, value=a `SparseTensor`
  - if isinstance(column, `WeightedSparseColumn`):
      {key=id column name, value=a `SparseTensor`,
       key=weight column name, value=a `SparseTensor`}
  - if isinstance(column, `RealValuedColumn`):
      key=column.name, value=a `Tensor`
- - -

#### `tf.contrib.learn.LinearRegressor.__init__(feature_columns, model_dir=None, weight_column_name=None, optimizer=None, gradient_clip_norm=None, enable_centered_bias=False, label_dimension=1, _joint_weights=False, config=None, feature_engineering_fn=None)` {#LinearRegressor.__init__}

Construct a `LinearRegressor` estimator object.

##### Args:


*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph, etc. This can
    also be used to load checkpoints from the directory into a estimator
    to continue training a previously saved model.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
*  <b>`optimizer`</b>: An instance of `tf.Optimizer` used to train the model. If
    `None`, will use an Ftrl optimizer.
*  <b>`gradient_clip_norm`</b>: A `float` > 0. If provided, gradients are clipped
    to their global norm with this clipping ratio. See
    `tf.clip_by_global_norm` for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
*  <b>`label_dimension`</b>: Number of regression targets per example. This is the
    size of the last dimension of the labels and logits `Tensor` objects
    (typically, these have shape `[batch_size, label_dimension]`).
  _joint_weights: If True use a single (possibly partitioned) variable to
    store the weights. It's faster, but requires all feature columns are
    sparse and have the 'sum' combiner. Incompatible with SDCAOptimizer.

*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.

##### Returns:

  A `LinearRegressor` estimator.


- - -

#### `tf.contrib.learn.LinearRegressor.__repr__()` {#LinearRegressor.__repr__}




- - -

#### `tf.contrib.learn.LinearRegressor.bias_` {#LinearRegressor.bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.LinearRegressor.config` {#LinearRegressor.config}




- - -

#### `tf.contrib.learn.LinearRegressor.evaluate(*args, **kwargs)` {#LinearRegressor.evaluate}

See `Evaluable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.LinearRegressor.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#LinearRegressor.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.LinearRegressor.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, checkpoint_path=None)` {#LinearRegressor.export_savedmodel}

Exports inference graph as a SavedModel into given dir.

##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`serving_input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.  Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the most recent checkpoint found within the model directory is chosen.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.LinearRegressor.fit(*args, **kwargs)` {#LinearRegressor.fit}

See `Trainable`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.LinearRegressor.get_params(deep=True)` {#LinearRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
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

#### `tf.contrib.learn.LinearRegressor.model_dir` {#LinearRegressor.model_dir}




- - -

#### `tf.contrib.learn.LinearRegressor.partial_fit(*args, **kwargs)` {#LinearRegressor.partial_fit}

Incremental fit on a batch of samples. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.

##### Example conversion:

  est = Estimator(...) -> est = SKCompat(Estimator(...))

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
     iterator that returns array of labels. The training label values
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

#### `tf.contrib.learn.LinearRegressor.predict(*args, **kwargs)` {#LinearRegressor.predict}

Returns predictions for given features. (deprecated arguments) (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2017-03-01.
Instructions for updating:
Please switch to predict_scores, or set `outputs` argument.

By default, returns predicted scores. But this default will be dropped
soon. Users should either pass `outputs`, or call `predict_scores` method.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`outputs`</b>: list of `str`, name of the output to predict.
    If `None`, returns scores.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted scores (or an iterable of predicted scores if
  as_iterable is True). If `label_dimension == 1`, the shape of the output
  is `[batch_size]`, otherwise the shape is `[batch_size, label_dimension]`.
  If `outputs` is set, returns a dict of predictions.


- - -

#### `tf.contrib.learn.LinearRegressor.predict_scores(*args, **kwargs)` {#LinearRegressor.predict_scores}

Returns predicted scores for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

##### Args:


*  <b>`x`</b>: features.
*  <b>`input_fn`</b>: Input function. If set, x must be None.
*  <b>`batch_size`</b>: Override default batch size.
*  <b>`as_iterable`</b>: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

##### Returns:

  Numpy array of predicted scores (or an iterable of predicted scores if
  as_iterable is True). If `label_dimension == 1`, the shape of the output
  is `[batch_size]`, otherwise the shape is `[batch_size, label_dimension]`.


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

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().



- - -

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



- - -

### `class tf.contrib.learn.Experiment` {#Experiment}

Experiment is a class containing all information needed to train a model.

After an experiment is created (by passing an Estimator and inputs for
training and evaluation), an Experiment instance knows how to invoke training
and eval loops in a sensible fashion for distributed training.
- - -

#### `tf.contrib.learn.Experiment.__init__(*args, **kwargs)` {#Experiment.__init__}

Constructor for `Experiment`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-23.
Instructions for updating:
local_eval_frequency is deprecated as local_run will be renamed to train_and_evaluate. Use min_eval_frequency and call train_and_evaluate instead. Note, however, that the default for min_eval_frequency is 1, meaning models will be evaluated every time a new checkpoint is available. In contrast, the default for local_eval_frequency is None, resulting in evaluation occurring only after training has completed. min_eval_frequency is ignored when calling the deprecated local_run.

Creates an Experiment instance. None of the functions passed to this
constructor are executed at construction time. They are stored and used
when a method is executed which requires it.

##### Args:


*  <b>`estimator`</b>: Object implementing `Trainable` and `Evaluable`.
*  <b>`train_input_fn`</b>: function, returns features and labels for training.
*  <b>`eval_input_fn`</b>: function, returns features and labels for evaluation. If
    `eval_steps` is `None`, this should be configured only to produce for a
    finite number of batches (generally, 1 epoch over the evaluation data).
*  <b>`eval_metrics`</b>: `dict` of string, metric function. If `None`, default set
    is used.
*  <b>`train_steps`</b>: Perform this many steps of training. `None`, the default,
    means train forever.
*  <b>`eval_steps`</b>: `evaluate` runs until input is exhausted (or another exception
    is raised), or for `eval_steps` steps, if specified.
*  <b>`train_monitors`</b>: A list of monitors to pass to the `Estimator`'s `fit`
    function.
*  <b>`eval_hooks`</b>: A list of `SessionRunHook` hooks to pass to the
    `Estimator`'s `evaluate` function.
*  <b>`local_eval_frequency`</b>: Frequency of running eval in steps,
    when running locally. If `None`, runs evaluation only at the end of
    training.
*  <b>`eval_delay_secs`</b>: Start evaluating after waiting for this many seconds.
*  <b>`continuous_eval_throttle_secs`</b>: Do not re-evaluate unless the last
    evaluation was started at least this many seconds ago for
    continuous_eval().
*  <b>`min_eval_frequency`</b>: (applies only to train_and_evaluate). the minimum
    number of steps between evaluations. Of course, evaluation does not
    occur if no new snapshot is available, hence, this is the minimum.
*  <b>`delay_workers_by_global_step`</b>: if `True` delays training workers
    based on global step instead of time.
*  <b>`export_strategies`</b>: A list of `ExportStrategy`s, or a single one, or None.

##### Raises:


*  <b>`ValueError`</b>: if `estimator` does not implement `Evaluable` and `Trainable`,
    or if export_strategies has the wrong type.


- - -

#### `tf.contrib.learn.Experiment.continuous_eval(delay_secs=None, throttle_delay_secs=None, evaluate_checkpoint_only_once=True, continuous_eval_predicate_fn=None)` {#Experiment.continuous_eval}




- - -

#### `tf.contrib.learn.Experiment.continuous_eval_on_train_data(delay_secs=None, throttle_delay_secs=None, continuous_eval_predicate_fn=None)` {#Experiment.continuous_eval_on_train_data}




- - -

#### `tf.contrib.learn.Experiment.estimator` {#Experiment.estimator}




- - -

#### `tf.contrib.learn.Experiment.eval_metrics` {#Experiment.eval_metrics}




- - -

#### `tf.contrib.learn.Experiment.eval_steps` {#Experiment.eval_steps}




- - -

#### `tf.contrib.learn.Experiment.evaluate(delay_secs=None)` {#Experiment.evaluate}

Evaluate on the evaluation data.

Runs evaluation on the evaluation data and returns the result. Runs for
`self._eval_steps` steps, or if it's `None`, then run until input is
exhausted or another exception is raised. Start the evaluation after
`delay_secs` seconds, or if it's `None`, defaults to using
`self._eval_delay_secs` seconds.

##### Args:


*  <b>`delay_secs`</b>: Start evaluating after this many seconds. If `None`, defaults
    to using `self._eval_delays_secs`.

##### Returns:

  The result of the `evaluate` call to the `Estimator`.


- - -

#### `tf.contrib.learn.Experiment.extend_train_hooks(additional_hooks)` {#Experiment.extend_train_hooks}

Extends the hooks for training.


- - -

#### `tf.contrib.learn.Experiment.local_run(*args, **kwargs)` {#Experiment.local_run}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-23.
Instructions for updating:
local_run will be renamed to train_and_evaluate and the new default behavior will be to run evaluation every time there is a new checkpoint.


- - -

#### `tf.contrib.learn.Experiment.reset_export_strategies(new_export_strategies=None)` {#Experiment.reset_export_strategies}

Resets the export strategies with the `new_export_strategies`.

##### Args:


*  <b>`new_export_strategies`</b>: A new list of `ExportStrategy`s, or a single one,
    or None.

##### Returns:

  The old export strategies.


- - -

#### `tf.contrib.learn.Experiment.run_std_server()` {#Experiment.run_std_server}

Starts a TensorFlow server and joins the serving thread.

Typically used for parameter servers.

##### Raises:


*  <b>`ValueError`</b>: if not enough information is available in the estimator's
    config to create a server.


- - -

#### `tf.contrib.learn.Experiment.test()` {#Experiment.test}

Tests training and evaluating the estimator both for a single step.

##### Returns:

  The result of the `evaluate` call to the `Estimator`.


- - -

#### `tf.contrib.learn.Experiment.train(delay_secs=None)` {#Experiment.train}

Fit the estimator using the training data.

Train the estimator for `self._train_steps` steps, after waiting for
`delay_secs` seconds. If `self._train_steps` is `None`, train forever.

##### Args:


*  <b>`delay_secs`</b>: Start training after this many seconds.

##### Returns:

  The trained estimator.


- - -

#### `tf.contrib.learn.Experiment.train_and_evaluate()` {#Experiment.train_and_evaluate}

Interleaves training and evaluation.

The frequency of evaluation is controlled by the contructor arg
`min_eval_frequency`. When this parameter is None or 0, evaluation happens
only after training has completed. Note that evaluation cannot happen
more frequently than checkpoints are taken. If no new snapshots are
available when evaluation is supposed to occur, then evaluation doesn't
happen for another `min_eval_frequency` steps (assuming a checkpoint is
available at that point). Thus, settings `min_eval_frequency` to 1 means
that the model will be evaluated everytime there is a new checkpoint.

This is particular useful for a "Master" task in the cloud, whose
responsibility it is to take checkpoints, evaluate those checkpoints,
and write out summaries. Participating in training as the supervisor
allows such a task to accomplish the first and last items, while
performing evaluation allows for the second.

##### Returns:

  The result of the `evaluate` call to the `Estimator`.


- - -

#### `tf.contrib.learn.Experiment.train_steps` {#Experiment.train_steps}





- - -

### `class tf.contrib.learn.ExportStrategy` {#ExportStrategy}

A class representing a type of model export.

Typically constructed by a utility function specific to the exporter, such as
`saved_model_export_utils.make_export_strategy()`.

The fields are:
  name: The directory name under the export base directory where exports of
    this type will be written.
  export_fn: A function that writes an export, given an estimator, a
    destination path, and optionally a checkpoint path and an evaluation
    result for that checkpoint.  This export_fn() may be run repeatedly during
    continuous training, or just once at the end of fixed-length training.
    Note the export_fn() may choose whether or not to export based on the eval
    result or based on an internal timer or any other criterion, if exports
    are not desired for every checkpoint.

    The signature of this function must be one of:
      * (estimator, export_path) -> export_path`
      * (estimator, export_path, checkpoint_path) -> export_path`
      * (estimator, export_path, checkpoint_path, eval_result) -> export_path`
- - -

#### `tf.contrib.learn.ExportStrategy.__getnewargs__()` {#ExportStrategy.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.contrib.learn.ExportStrategy.__getstate__()` {#ExportStrategy.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.contrib.learn.ExportStrategy.__new__(_cls, name, export_fn)` {#ExportStrategy.__new__}

Create new instance of ExportStrategy(name, export_fn)


- - -

#### `tf.contrib.learn.ExportStrategy.__repr__()` {#ExportStrategy.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.contrib.learn.ExportStrategy.export(estimator, export_path, checkpoint_path=None, eval_result=None)` {#ExportStrategy.export}

Exports the given Estimator to a specific format.

##### Args:


*  <b>`estimator`</b>: the Estimator to export.
*  <b>`export_path`</b>: A string containing a directory where to write the export.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the strategy may locate a checkpoint (e.g. the most recent) by itself.
*  <b>`eval_result`</b>: The output of Estimator.evaluate on this checkpoint.  This
    should be set only if checkpoint_path is provided (otherwise it is
    unclear which checkpoint this eval refers to).

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if the export_fn does not have the required signature


- - -

#### `tf.contrib.learn.ExportStrategy.export_fn` {#ExportStrategy.export_fn}

Alias for field number 1


- - -

#### `tf.contrib.learn.ExportStrategy.name` {#ExportStrategy.name}

Alias for field number 0



- - -

### `class tf.contrib.learn.TaskType` {#TaskType}




- - -

### `class tf.train.NanLossDuringTrainingError` {#NanLossDuringTrainingError}


- - -

#### `tf.train.NanLossDuringTrainingError.__str__()` {#NanLossDuringTrainingError.__str__}





- - -

### `class tf.contrib.learn.RunConfig` {#RunConfig}

This class specifies the configurations for an `Estimator` run.

If you're a Google-internal user using command line flags with
`learn_runner.py` (for instance, to do distributed training or to use
parameter servers), you probably want to use `learn_runner.EstimatorConfig`
instead.
- - -

#### `tf.contrib.learn.RunConfig.__init__(master=None, num_cores=0, log_device_placement=False, gpu_memory_fraction=1, tf_random_seed=None, save_summary_steps=100, save_checkpoints_secs=600, save_checkpoints_steps=None, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000, evaluation_master='')` {#RunConfig.__init__}

Constructor.

Note that the superclass `ClusterConfig` may set properties like
`cluster_spec`, `is_chief`, `master` (if `None` in the args),
`num_ps_replicas`, `task_id`, and `task_type` based on the `TF_CONFIG`
environment variable. See `ClusterConfig` for more details.

##### Args:


*  <b>`master`</b>: TensorFlow master. Defaults to empty string for local.
*  <b>`num_cores`</b>: Number of cores to be used. If 0, the system picks an
    appropriate number (default: 0).
*  <b>`log_device_placement`</b>: Log the op placement to devices (default: False).
*  <b>`gpu_memory_fraction`</b>: Fraction of GPU memory used by the process on
    each GPU uniformly on the same machine.
*  <b>`tf_random_seed`</b>: Random seed for TensorFlow initializers.
    Setting this value allows consistency between reruns.
*  <b>`save_summary_steps`</b>: Save summaries every this many steps.
*  <b>`save_checkpoints_secs`</b>: Save checkpoints every this many seconds. Can not
      be specified with `save_checkpoints_steps`.
*  <b>`save_checkpoints_steps`</b>: Save checkpoints every this many steps. Can not be
      specified with `save_checkpoints_secs`.
*  <b>`keep_checkpoint_max`</b>: The maximum number of recent checkpoint files to
    keep. As new files are created, older files are deleted. If None or 0,
    all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent
    checkpoint files are kept.)
*  <b>`keep_checkpoint_every_n_hours`</b>: Number of hours between each checkpoint
    to be saved. The default value of 10,000 hours effectively disables
    the feature.
*  <b>`evaluation_master`</b>: the master on which to perform evaluation.


- - -

#### `tf.contrib.learn.RunConfig.cluster_spec` {#RunConfig.cluster_spec}




- - -

#### `tf.contrib.learn.RunConfig.environment` {#RunConfig.environment}




- - -

#### `tf.contrib.learn.RunConfig.evaluation_master` {#RunConfig.evaluation_master}




- - -

#### `tf.contrib.learn.RunConfig.get_task_id()` {#RunConfig.get_task_id}

Returns task index from `TF_CONFIG` environmental variable.

If you have a ClusterConfig instance, you can just access its task_id
property instead of calling this function and re-parsing the environmental
variable.

##### Returns:

  `TF_CONFIG['task']['index']`. Defaults to 0.


- - -

#### `tf.contrib.learn.RunConfig.is_chief` {#RunConfig.is_chief}




- - -

#### `tf.contrib.learn.RunConfig.keep_checkpoint_every_n_hours` {#RunConfig.keep_checkpoint_every_n_hours}




- - -

#### `tf.contrib.learn.RunConfig.keep_checkpoint_max` {#RunConfig.keep_checkpoint_max}




- - -

#### `tf.contrib.learn.RunConfig.master` {#RunConfig.master}




- - -

#### `tf.contrib.learn.RunConfig.num_ps_replicas` {#RunConfig.num_ps_replicas}




- - -

#### `tf.contrib.learn.RunConfig.save_checkpoints_secs` {#RunConfig.save_checkpoints_secs}




- - -

#### `tf.contrib.learn.RunConfig.save_checkpoints_steps` {#RunConfig.save_checkpoints_steps}




- - -

#### `tf.contrib.learn.RunConfig.save_summary_steps` {#RunConfig.save_summary_steps}




- - -

#### `tf.contrib.learn.RunConfig.task_id` {#RunConfig.task_id}




- - -

#### `tf.contrib.learn.RunConfig.task_type` {#RunConfig.task_type}




- - -

#### `tf.contrib.learn.RunConfig.tf_config` {#RunConfig.tf_config}




- - -

#### `tf.contrib.learn.RunConfig.tf_random_seed` {#RunConfig.tf_random_seed}





- - -

### `tf.contrib.learn.evaluate(*args, **kwargs)` {#evaluate}

Evaluate a model loaded from a checkpoint. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

Given `graph`, a directory to write summaries to (`output_dir`), a checkpoint
to restore variables from, and a `dict` of `Tensor`s to evaluate, run an eval
loop for `max_steps` steps, or until an exception (generally, an
end-of-input signal from a reader operation) is raised from running
`eval_dict`.

In each step of evaluation, all tensors in the `eval_dict` are evaluated, and
every `log_every_steps` steps, they are logged. At the very end of evaluation,
a summary is evaluated (finding the summary ops using `Supervisor`'s logic)
and written to `output_dir`.

##### Args:


*  <b>`graph`</b>: A `Graph` to train. It is expected that this graph is not in use
    elsewhere.
*  <b>`output_dir`</b>: A string containing the directory to write a summary to.
*  <b>`checkpoint_path`</b>: A string containing the path to a checkpoint to restore.
    Can be `None` if the graph doesn't require loading any variables.
*  <b>`eval_dict`</b>: A `dict` mapping string names to tensors to evaluate. It is
    evaluated in every logging step. The result of the final evaluation is
    returned. If `update_op` is None, then it's evaluated in every step. If
    `max_steps` is `None`, this should depend on a reader that will raise an
    end-of-input exception when the inputs are exhausted.
*  <b>`update_op`</b>: A `Tensor` which is run in every step.
*  <b>`global_step_tensor`</b>: A `Variable` containing the global step. If `None`,
    one is extracted from the graph using the same logic as in `Supervisor`.
    Used to place eval summaries on training curves.
*  <b>`supervisor_master`</b>: The master string to use when preparing the session.
*  <b>`log_every_steps`</b>: Integer. Output logs every `log_every_steps` evaluation
    steps. The logs contain the `eval_dict` and timing information.
*  <b>`feed_fn`</b>: A function that is called every iteration to produce a `feed_dict`
    passed to `session.run` calls. Optional.
*  <b>`max_steps`</b>: Integer. Evaluate `eval_dict` this many times.

##### Returns:

  A tuple `(eval_results, global_step)`:

*  <b>`eval_results`</b>: A `dict` mapping `string` to numeric values (`int`, `float`)
    that are the result of running eval_dict in the last step. `None` if no
    eval steps were run.
*  <b>`global_step`</b>: The global step this evaluation corresponds to.

##### Raises:


*  <b>`ValueError`</b>: if `output_dir` is empty.


- - -

### `tf.contrib.learn.infer(*args, **kwargs)` {#infer}

Restore graph from `restore_checkpoint_path` and run `output_dict` tensors. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

If `restore_checkpoint_path` is supplied, restore from checkpoint. Otherwise,
init all variables.

##### Args:


*  <b>`restore_checkpoint_path`</b>: A string containing the path to a checkpoint to
    restore.
*  <b>`output_dict`</b>: A `dict` mapping string names to `Tensor` objects to run.
    Tensors must all be from the same graph.
*  <b>`feed_dict`</b>: `dict` object mapping `Tensor` objects to input values to feed.

##### Returns:

  Dict of values read from `output_dict` tensors. Keys are the same as
  `output_dict`, values are the results read from the corresponding `Tensor`
  in `output_dict`.

##### Raises:


*  <b>`ValueError`</b>: if `output_dict` or `feed_dicts` is None or empty.


- - -

### `tf.contrib.learn.run_feeds(*args, **kwargs)` {#run_feeds}

See run_feeds_iter(). Returns a `list` instead of an iterator. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.


- - -

### `tf.contrib.learn.run_n(*args, **kwargs)` {#run_n}

Run `output_dict` tensors `n` times, with the same `feed_dict` each run. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

##### Args:


*  <b>`output_dict`</b>: A `dict` mapping string names to tensors to run. Must all be
    from the same graph.
*  <b>`feed_dict`</b>: `dict` of input values to feed each run.
*  <b>`restore_checkpoint_path`</b>: A string containing the path to a checkpoint to
    restore.
*  <b>`n`</b>: Number of times to repeat.

##### Returns:

  A list of `n` `dict` objects, each containing values read from `output_dict`
  tensors.


- - -

### `tf.contrib.learn.train(*args, **kwargs)` {#train}

Train a model. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

Given `graph`, a directory to write outputs to (`output_dir`), and some ops,
run a training loop. The given `train_op` performs one step of training on the
model. The `loss_op` represents the objective function of the training. It is
expected to increment the `global_step_tensor`, a scalar integer tensor
counting training steps. This function uses `Supervisor` to initialize the
graph (from a checkpoint if one is available in `output_dir`), write summaries
defined in the graph, and write regular checkpoints as defined by
`supervisor_save_model_secs`.

Training continues until `global_step_tensor` evaluates to `max_steps`, or, if
`fail_on_nan_loss`, until `loss_op` evaluates to `NaN`. In that case the
program is terminated with exit code 1.

##### Args:


*  <b>`graph`</b>: A graph to train. It is expected that this graph is not in use
    elsewhere.
*  <b>`output_dir`</b>: A directory to write outputs to.
*  <b>`train_op`</b>: An op that performs one training step when run.
*  <b>`loss_op`</b>: A scalar loss tensor.
*  <b>`global_step_tensor`</b>: A tensor representing the global step. If none is given,
    one is extracted from the graph using the same logic as in `Supervisor`.
*  <b>`init_op`</b>: An op that initializes the graph. If `None`, use `Supervisor`'s
    default.
*  <b>`init_feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    This feed dictionary will be used when `init_op` is evaluated.
*  <b>`init_fn`</b>: Optional callable passed to Supervisor to initialize the model.
*  <b>`log_every_steps`</b>: Output logs regularly. The logs contain timing data and the
    current loss.
*  <b>`supervisor_is_chief`</b>: Whether the current process is the chief supervisor in
    charge of restoring the model and running standard services.
*  <b>`supervisor_master`</b>: The master string to use when preparing the session.
*  <b>`supervisor_save_model_secs`</b>: Save a checkpoint every
    `supervisor_save_model_secs` seconds when training.
*  <b>`keep_checkpoint_max`</b>: The maximum number of recent checkpoint files to
    keep. As new files are created, older files are deleted. If None or 0,
    all checkpoint files are kept. This is simply passed as the max_to_keep
    arg to tf.Saver constructor.
*  <b>`supervisor_save_summaries_steps`</b>: Save summaries every
    `supervisor_save_summaries_steps` seconds when training.
*  <b>`feed_fn`</b>: A function that is called every iteration to produce a `feed_dict`
    passed to `session.run` calls. Optional.
*  <b>`steps`</b>: Trains for this many steps (e.g. current global step + `steps`).
*  <b>`fail_on_nan_loss`</b>: If true, raise `NanLossDuringTrainingError` if `loss_op`
    evaluates to `NaN`. If false, continue training as if nothing happened.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.
*  <b>`max_steps`</b>: Number of total steps for which to train model. If `None`,
    train forever. Two calls fit(steps=100) means 200 training iterations.
    On the other hand two calls of fit(max_steps=100) means, second call
    will not do any iteration since first call did all 100 steps.

##### Returns:

  The final loss value.

##### Raises:


*  <b>`ValueError`</b>: If `output_dir`, `train_op`, `loss_op`, or `global_step_tensor`
    is not provided. See `tf.contrib.framework.get_global_step` for how we
    look up the latter if not provided explicitly.
*  <b>`NanLossDuringTrainingError`</b>: If `fail_on_nan_loss` is `True`, and loss ever
    evaluates to `NaN`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.



- - -

### `tf.contrib.learn.extract_dask_data(data)` {#extract_dask_data}

Extract data from dask.Series or dask.DataFrame for predictors.

Given a distributed dask.DataFrame or dask.Series containing columns or names
for one or more predictors, this operation returns a single dask.DataFrame or
dask.Series that can be iterated over.

##### Args:


*  <b>`data`</b>: A distributed dask.DataFrame or dask.Series.

##### Returns:

  A dask.DataFrame or dask.Series that can be iterated over.
  If the supplied argument is neither a dask.DataFrame nor a dask.Series this
  operation returns it without modification.


- - -

### `tf.contrib.learn.extract_dask_labels(labels)` {#extract_dask_labels}

Extract data from dask.Series or dask.DataFrame for labels.

Given a distributed dask.DataFrame or dask.Series containing exactly one
column or name, this operation returns a single dask.DataFrame or dask.Series
that can be iterated over.

##### Args:


*  <b>`labels`</b>: A distributed dask.DataFrame or dask.Series with exactly one
          column or name.

##### Returns:

  A dask.DataFrame or dask.Series that can be iterated over.
  If the supplied argument is neither a dask.DataFrame nor a dask.Series this
  operation returns it without modification.

##### Raises:


*  <b>`ValueError`</b>: If the supplied dask.DataFrame contains more than one
              column or the supplied dask.Series contains more than
              one name.


- - -

### `tf.contrib.learn.extract_pandas_data(data)` {#extract_pandas_data}

Extract data from pandas.DataFrame for predictors.

Given a DataFrame, will extract the values and cast them to float. The
DataFrame is expected to contain values of type int, float or bool.

##### Args:


*  <b>`data`</b>: `pandas.DataFrame` containing the data to be extracted.

##### Returns:

  A numpy `ndarray` of the DataFrame's values as floats.

##### Raises:


*  <b>`ValueError`</b>: if data contains types other than int, float or bool.


- - -

### `tf.contrib.learn.extract_pandas_labels(labels)` {#extract_pandas_labels}

Extract data from pandas.DataFrame for labels.

##### Args:


*  <b>`labels`</b>: `pandas.DataFrame` or `pandas.Series` containing one column of
    labels to be extracted.

##### Returns:

  A numpy `ndarray` of labels from the DataFrame.

##### Raises:


*  <b>`ValueError`</b>: if more than one column is found or type is not int, float or
    bool.


- - -

### `tf.contrib.learn.extract_pandas_matrix(data)` {#extract_pandas_matrix}

Extracts numpy matrix from pandas DataFrame.

##### Args:


*  <b>`data`</b>: `pandas.DataFrame` containing the data to be extracted.

##### Returns:

  A numpy `ndarray` of the DataFrame's values.


- - -

### `tf.contrib.learn.infer_real_valued_columns_from_input(x)` {#infer_real_valued_columns_from_input}

Creates `FeatureColumn` objects for inputs defined by input `x`.

This interprets all inputs as dense, fixed-length float values.

##### Args:


*  <b>`x`</b>: Real-valued matrix of shape [n_samples, n_features...]. Can be
     iterator that returns arrays of features.

##### Returns:

  List of `FeatureColumn` objects.


- - -

### `tf.contrib.learn.infer_real_valued_columns_from_input_fn(input_fn)` {#infer_real_valued_columns_from_input_fn}

Creates `FeatureColumn` objects for inputs defined by `input_fn`.

This interprets all inputs as dense, fixed-length float values. This creates
a local graph in which it calls `input_fn` to build the tensors, then discards
it.

##### Args:


*  <b>`input_fn`</b>: Input function returning a tuple of:
      features - Dictionary of string feature name to `Tensor` or `Tensor`.
      labels - `Tensor` of label values.

##### Returns:

  List of `FeatureColumn` objects.


- - -

### `tf.contrib.learn.read_batch_examples(file_pattern, batch_size, reader, randomize_input=True, num_epochs=None, queue_capacity=10000, num_threads=1, read_batch_size=1, parse_fn=None, name=None, seed=None)` {#read_batch_examples}

Adds operations to read, queue, batch `Example` protos.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size`.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Use `parse_fn` if you need to do parsing / processing on single examples.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`reader`</b>: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`num_epochs`</b>: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.global_variables_initializer()` and run the op in a session.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`num_threads`</b>: The number of threads enqueuing examples.
*  <b>`read_batch_size`</b>: An int or scalar `Tensor` specifying the number of
    records to read at once
*  <b>`parse_fn`</b>: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
*  <b>`name`</b>: Name of resulting op.
*  <b>`seed`</b>: An integer (optional). Seed used if randomize_input == True.

##### Returns:

  String `Tensor` of batched `Example` proto.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.


- - -

### `tf.contrib.learn.read_batch_features(file_pattern, batch_size, features, reader, randomize_input=True, num_epochs=None, queue_capacity=10000, feature_queue_capacity=100, reader_num_threads=1, parse_fn=None, name=None)` {#read_batch_features}

Adds operations to read, queue, batch and parse `Example` protos.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
*  <b>`reader`</b>: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`num_epochs`</b>: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.local_variables_initializer() and run the op in a session.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`feature_queue_capacity`</b>: Capacity of the parsed features queue. Set this
    value to a small number, for example 5 if the parsed features are large.
*  <b>`reader_num_threads`</b>: The number of threads to read examples.
*  <b>`parse_fn`</b>: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.


- - -

### `tf.contrib.learn.read_batch_record_features(file_pattern, batch_size, features, randomize_input=True, num_epochs=None, queue_capacity=10000, reader_num_threads=1, name='dequeue_record_examples')` {#read_batch_record_features}

Reads TFRecord, queues, batches and parses `Example` proto.

See more detailed description in `read_examples`.

##### Args:


*  <b>`file_pattern`</b>: List of files or pattern of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
*  <b>`batch_size`</b>: An int or scalar `Tensor` specifying the batch size to use.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
*  <b>`randomize_input`</b>: Whether the input should be randomized.
*  <b>`num_epochs`</b>: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.local_variables_initializer() and run the op in a session.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`reader_num_threads`</b>: The number of threads to read examples.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.



- - -

### `class tf.contrib.learn.InputFnOps` {#InputFnOps}

A return type for an input_fn.

This return type is currently only supported for serving input_fn.
Training and eval input_fn should return a `(features, labels)` tuple.

The expected return values are:
  features: A dict of string to `Tensor` or `SparseTensor`, specifying the
    features to be passed to the model.
  labels: A `Tensor`, `SparseTensor`, or a dict of string to `Tensor` or
    `SparseTensor`, specifying labels for training or eval. For serving, set
    `labels` to `None`.
  default_inputs: a dict of string to `Tensor` or `SparseTensor`, specifying
    the input placeholders (if any) that this input_fn expects to be fed.
    Typically, this is used by a serving input_fn, which expects to be fed
    serialized `tf.Example` protos.
- - -

#### `tf.contrib.learn.InputFnOps.__getnewargs__()` {#InputFnOps.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.contrib.learn.InputFnOps.__getstate__()` {#InputFnOps.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.contrib.learn.InputFnOps.__new__(_cls, features, labels, default_inputs)` {#InputFnOps.__new__}

Create new instance of InputFnOps(features, labels, default_inputs)


- - -

#### `tf.contrib.learn.InputFnOps.__repr__()` {#InputFnOps.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.contrib.learn.InputFnOps.default_inputs` {#InputFnOps.default_inputs}

Alias for field number 2


- - -

#### `tf.contrib.learn.InputFnOps.features` {#InputFnOps.features}

Alias for field number 0


- - -

#### `tf.contrib.learn.InputFnOps.labels` {#InputFnOps.labels}

Alias for field number 1



- - -

### `class tf.contrib.learn.ProblemType` {#ProblemType}

Enum-like values for the type of problem that the model solves.

These values are used when exporting the model to produce the appropriate
signature function for serving.

The following values are supported:
  UNSPECIFIED: Produces a predict signature_fn.
  CLASSIFICATION: Produces a classify signature_fn.
  LINEAR_REGRESSION: Produces a regression signature_fn.
  LOGISTIC_REGRESSION: Produces a classify signature_fn.

- - -

### `tf.contrib.learn.build_parsing_serving_input_fn(feature_spec, default_batch_size=None)` {#build_parsing_serving_input_fn}

Build an input_fn appropriate for serving, expecting fed tf.Examples.

Creates an input_fn that expects a serialized tf.Example fed into a string
placeholder.  The function parses the tf.Example according to the provided
feature_spec, and returns all parsed Tensors as features.  This input_fn is
for use at serving time, so the labels return value is always None.

##### Args:


*  <b>`feature_spec`</b>: a dict of string to `VarLenFeature`/`FixedLenFeature`.
*  <b>`default_batch_size`</b>: the number of query examples expected per batch.
      Leave unset for variable batch size (recommended).

##### Returns:

  An input_fn suitable for use in serving.


- - -

### `tf.contrib.learn.make_export_strategy(serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, exports_to_keep=5)` {#make_export_strategy}

Create an ExportStrategy for use with Experiment.

##### Args:


*  <b>`serving_input_fn`</b>: A function that takes no arguments and returns an
    `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when an
    incoming serving request does not explicitly request a specific head.
    Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`exports_to_keep`</b>: Number of exports to keep.  Older exports will be
    garbage-collected.  Defaults to 5.  Set to None to disable garbage
    collection.

##### Returns:

  An ExportStrategy that can be passed to the Experiment constructor.



## Other Functions and Classes
- - -

### `class tf.contrib.learn.NotFittedError` {#NotFittedError}

Exception class to raise if estimator is used before fitting.

This class inherits from both ValueError and AttributeError to help with
exception handling and backward compatibility.

Examples:
>>> from sklearn.svm import LinearSVC
>>> from sklearn.exceptions import NotFittedError
>>> try:
...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
... except NotFittedError as e:
...     print(repr(e))
...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
NotFittedError('This LinearSVC instance is not fitted yet',)

Copied from
https://github.com/scikit-learn/scikit-learn/master/sklearn/exceptions.py

