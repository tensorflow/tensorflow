Logistic regression Estimator for binary classification.

This class provides a basic Estimator with some additional metrics for custom
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
- - -

#### `tf.contrib.learn.LogisticRegressor.__init__(model_fn, thresholds=None, model_dir=None, config=None, feature_engineering_fn=None)` {#LogisticRegressor.__init__}

Initializes a LogisticRegressor.

##### Args:


*  <b>`model_fn`</b>: Model function. See superclass Estimator for more details. This
    expects the returned predictions to be probabilities in [0.0, 1.0].
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


- - -

#### `tf.contrib.learn.LogisticRegressor.__repr__()` {#LogisticRegressor.__repr__}




- - -

#### `tf.contrib.learn.LogisticRegressor.config` {#LogisticRegressor.config}




- - -

#### `tf.contrib.learn.LogisticRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None, checkpoint_path=None)` {#LogisticRegressor.evaluate}

Evaluates given model with provided evaluation data.

See superclass Estimator for more details.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: labels (must be 0 or 1).
*  <b>`input_fn`</b>: Input function.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called.
*  <b>`batch_size`</b>: minibatch size to use on the input.
*  <b>`steps`</b>: Number of steps for which to evaluate model.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metrics are used.
*  <b>`name`</b>: Name of the evaluation.
*  <b>`checkpoint_path`</b>: A specific checkpoint to use. By default, use the latest
    checkpoint in the `model_dir`.

##### Returns:

  Returns `dict` with evaluation results.


- - -

#### `tf.contrib.learn.LogisticRegressor.export(*args, **kwargs)` {#LogisticRegressor.export}

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

##### Returns:

  The string path to the exported directory. NB: this functionality was
  added ca. 2016/09/25; clients that depend on the return value may need
  to handle the case where this function returns None because subclasses
  are not returning a value.


- - -

#### `tf.contrib.learn.LogisticRegressor.export_savedmodel(*args, **kwargs)` {#LogisticRegressor.export_savedmodel}

Exports inference graph as a SavedModel into given dir. (experimental)

THIS FUNCTION IS EXPERIMENTAL. It may change or be removed at any time, and without warning.


##### Args:


*  <b>`export_dir_base`</b>: A string containing a directory to write the exported
    graph and checkpoints.
*  <b>`input_fn`</b>: A function that takes no argument and
    returns an `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when none is
    specified.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`exports_to_keep`</b>: Number of exports to keep.

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if an unrecognized export_type is requested.


- - -

#### `tf.contrib.learn.LogisticRegressor.fit(*args, **kwargs)` {#LogisticRegressor.fit}

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

#### `tf.contrib.learn.LogisticRegressor.get_default_metrics(cls, thresholds=None)` {#LogisticRegressor.get_default_metrics}

Returns a dictionary of basic metrics for logistic regression.

##### Args:


*  <b>`thresholds`</b>: List of floating point thresholds to use for accuracy,
    precision, and recall metrics. If None, defaults to [0.5].

##### Returns:

  Dictionary mapping metrics string names to metrics functions.


- - -

#### `tf.contrib.learn.LogisticRegressor.get_params(deep=True)` {#LogisticRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.LogisticRegressor.get_variable_names()` {#LogisticRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.LogisticRegressor.get_variable_value(name)` {#LogisticRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.LogisticRegressor.model_dir` {#LogisticRegressor.model_dir}




- - -

#### `tf.contrib.learn.LogisticRegressor.partial_fit(*args, **kwargs)` {#LogisticRegressor.partial_fit}

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

#### `tf.contrib.learn.LogisticRegressor.predict(*args, **kwargs)` {#LogisticRegressor.predict}

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

#### `tf.contrib.learn.LogisticRegressor.set_params(**params)` {#LogisticRegressor.set_params}

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


