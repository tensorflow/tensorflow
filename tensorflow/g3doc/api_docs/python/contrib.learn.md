<!-- This file is machine generated: DO NOT EDIT! -->

# Learn (contrib)
[TOC]

High level API for learning with TensorFlow.

## Estimators

Train and evaluate TensorFlow models.

- - -

### `class tf.contrib.learn.BaseEstimator` {#BaseEstimator}

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
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn and input_feature_key will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

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
        key into the features dict returned by `input_fn` that corresponds toa
        the raw `Example` strings `Tensor` that the exported model will take as
        input.
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



- - -

### `class tf.contrib.learn.Estimator` {#Estimator}

Estimator class is the basic TensorFlow model trainer/evaluator.
- - -

#### `tf.contrib.learn.Estimator.__init__(model_fn=None, model_dir=None, config=None, params=None, feature_engineering_fn=None)` {#Estimator.__init__}

Constructs an Estimator instance.

##### Args:


*  <b>`model_fn`</b>: Model function, takes features and targets tensors or dicts of
            tensors and returns predictions and loss tensors.
            Supports next three signatures for the function:

      * `(features, targets) -> (predictions, loss, train_op)`
      * `(features, targets, mode) -> (predictions, loss, train_op)`
      * `(features, targets, mode, params) -> (predictions, loss, train_op)`

    Where

      * `features` are single `Tensor` or `dict` of `Tensor`s
             (depending on data passed to `fit`),
      * `targets` are `Tensor` or `dict` of `Tensor`s (for multi-head
             models). If mode is `ModeKeys.INFER`, `targets=None` will be
             passed. If the `model_fn`'s signature does not accept
             `mode`, the `model_fn` must still be able to handle
             `targets=None`.
      * `mode` represents if this training, evaluation or
             prediction. See `ModeKeys`.
      * `params` is a `dict` of hyperparameters. Will receive what
             is passed to Estimator in `params` parameter. This allows
             to configure Estimators from hyper parameter tunning.


*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator to
    continue training a previously saved model.
*  <b>`config`</b>: Configuration object.
*  <b>`params`</b>: `dict` of hyper parameters that will be passed into `model_fn`.
          Keys are names of parameters, values are basic python types.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                          targets which are the output of `input_fn` and
                          returns features and targets which will be fed
                          into `model_fn`. Please check `model_fn` for
                          a definition of features and targets.

##### Raises:


*  <b>`ValueError`</b>: parameters of `model_fn` don't match `params`.


- - -

#### `tf.contrib.learn.Estimator.__repr__()` {#Estimator.__repr__}




- - -

#### `tf.contrib.learn.Estimator.config` {#Estimator.config}




- - -

#### `tf.contrib.learn.Estimator.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#Estimator.evaluate}

See `Evaluable`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.Estimator.export(*args, **kwargs)` {#Estimator.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn and input_feature_key will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

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
        key into the features dict returned by `input_fn` that corresponds toa
        the raw `Example` strings `Tensor` that the exported model will take as
        input.
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

#### `tf.contrib.learn.Estimator.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#Estimator.fit}

See `Trainable`.

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

#### `tf.contrib.learn.Estimator.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#Estimator.partial_fit}

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

#### `tf.contrib.learn.Estimator.predict(*args, **kwargs)` {#Estimator.predict}

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

### `class tf.contrib.learn.ModeKeys` {#ModeKeys}

Standard names for model modes.

The following standard keys are defined:

* `TRAIN`: training mode.
* `EVAL`: evaluation mode.
* `INFER`: inference mode.

- - -

### `class tf.contrib.learn.DNNClassifier` {#DNNClassifier}

A classifier for TensorFlow DNN models.

Example:

```python
education = sparse_column_with_hash_bucket(column_name="education",
                                           hash_bucket_size=1000)
occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)

education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                 combiner="sum")
occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                 combiner="sum")

estimator = DNNClassifier(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256])

# Or estimator using the ProximalAdagradOptimizer optimizer with
# regularization.
estimator = DNNClassifier(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Input builders
def input_fn_train: # returns x, Y
  pass
estimator.fit(input_fn=input_fn_train)

def input_fn_eval: # returns x, Y
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

#### `tf.contrib.learn.DNNClassifier.__init__(hidden_units, feature_columns, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None, activation_fn=relu, dropout=None, gradient_clip_norm=None, enable_centered_bias=None, config=None)` {#DNNClassifier.__init__}

Initializes a DNNClassifier instance.

##### Args:


*  <b>`hidden_units`</b>: List of hidden units per layer. All layers are fully
    connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
    has 32.
*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can also
    be used to load checkpoints from the directory into a estimator to continue
    training a previously saved model.
*  <b>`n_classes`</b>: number of target classes. Default is binary classification.
    It must be greater than 1.
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
    tf.clip_by_global_norm for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.

##### Returns:

  A `DNNClassifier` estimator.

##### Raises:


*  <b>`ValueError`</b>: If `n_classes` < 2.


- - -

#### `tf.contrib.learn.DNNClassifier.bias_` {#DNNClassifier.bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-13.
Instructions for updating:
This method inspects the private state of the object, and should not be used


- - -

#### `tf.contrib.learn.DNNClassifier.config` {#DNNClassifier.config}




- - -

#### `tf.contrib.learn.DNNClassifier.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#DNNClassifier.evaluate}

See evaluable.Evaluable.


- - -

#### `tf.contrib.learn.DNNClassifier.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#DNNClassifier.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.DNNClassifier.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#DNNClassifier.fit}

See trainable.Trainable.


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

  `Tensor` object.


- - -

#### `tf.contrib.learn.DNNClassifier.model_dir` {#DNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.DNNClassifier.predict(*args, **kwargs)` {#DNNClassifier.predict}

Returns predicted classes for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted classes (or an iterable of predicted classes if
      as_iterable is True).


- - -

#### `tf.contrib.learn.DNNClassifier.predict_proba(*args, **kwargs)` {#DNNClassifier.predict_proba}

Returns prediction probabilities for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted probabilities (or an iterable of predicted
      probabilities if as_iterable is True).


- - -

#### `tf.contrib.learn.DNNClassifier.weights_` {#DNNClassifier.weights_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-13.
Instructions for updating:
This method inspects the private state of the object, and should not be used



- - -

### `class tf.contrib.learn.DNNRegressor` {#DNNRegressor}

A regressor for TensorFlow DNN models.

Example:

```python
education = sparse_column_with_hash_bucket(column_name="education",
                                           hash_bucket_size=1000)
occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)

education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                 combiner="sum")
occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                 combiner="sum")

estimator = DNNRegressor(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256])

# Or estimator using the ProximalAdagradOptimizer optimizer with
# regularization.
estimator = DNNRegressor(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Input builders
def input_fn_train: # returns x, Y
  pass
estimator.fit(input_fn=input_fn_train)

def input_fn_eval: # returns x, Y
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

#### `tf.contrib.learn.DNNRegressor.__init__(hidden_units, feature_columns, model_dir=None, weight_column_name=None, optimizer=None, activation_fn=relu, dropout=None, gradient_clip_norm=None, enable_centered_bias=None, config=None)` {#DNNRegressor.__init__}

Initializes a `DNNRegressor` instance.

##### Args:


*  <b>`hidden_units`</b>: List of hidden units per layer. All layers are fully
    connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
    has 32.
*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can also
    be used to load checkpoints from the directory into a estimator to continue
    training a previously saved model.
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

##### Returns:

  A `DNNRegressor` estimator.


- - -

#### `tf.contrib.learn.DNNRegressor.__repr__()` {#DNNRegressor.__repr__}




- - -

#### `tf.contrib.learn.DNNRegressor.bias_` {#DNNRegressor.bias_}




- - -

#### `tf.contrib.learn.DNNRegressor.config` {#DNNRegressor.config}




- - -

#### `tf.contrib.learn.DNNRegressor.dnn_bias_` {#DNNRegressor.dnn_bias_}

Returns bias of deep neural network part.


- - -

#### `tf.contrib.learn.DNNRegressor.dnn_weights_` {#DNNRegressor.dnn_weights_}

Returns weights of deep neural network part.


- - -

#### `tf.contrib.learn.DNNRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#DNNRegressor.evaluate}

See `Evaluable`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.DNNRegressor.export(*args, **kwargs)` {#DNNRegressor.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn and input_feature_key will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

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
        key into the features dict returned by `input_fn` that corresponds toa
        the raw `Example` strings `Tensor` that the exported model will take as
        input.
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

#### `tf.contrib.learn.DNNRegressor.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#DNNRegressor.fit}

See `Trainable`.

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

#### `tf.contrib.learn.DNNRegressor.linear_bias_` {#DNNRegressor.linear_bias_}

Returns bias of the linear part.


- - -

#### `tf.contrib.learn.DNNRegressor.linear_weights_` {#DNNRegressor.linear_weights_}

Returns weights per feature of the linear part.


- - -

#### `tf.contrib.learn.DNNRegressor.model_dir` {#DNNRegressor.model_dir}




- - -

#### `tf.contrib.learn.DNNRegressor.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#DNNRegressor.partial_fit}

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

#### `tf.contrib.learn.DNNRegressor.predict(*args, **kwargs)` {#DNNRegressor.predict}

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

#### `tf.contrib.learn.DNNRegressor.weights_` {#DNNRegressor.weights_}





- - -

### `class tf.contrib.learn.TensorFlowEstimator` {#TensorFlowEstimator}

Base class for all TensorFlow estimators.
- - -

#### `tf.contrib.learn.TensorFlowEstimator.__init__(model_fn, n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, class_weight=None, continue_training=False, config=None, verbose=1)` {#TensorFlowEstimator.__init__}

Initializes a TensorFlowEstimator instance.

##### Args:


*  <b>`model_fn`</b>: Model function, that takes input `x`, `y` tensors and outputs
    prediction and loss tensors.
*  <b>`n_classes`</b>: Number of classes in the target.
*  <b>`batch_size`</b>: Mini batch size.
*  <b>`steps`</b>: Number of steps to run over data.
*  <b>`optimizer`</b>: Optimizer name (or class), for example "SGD", "Adam",
    "Adagrad".
*  <b>`learning_rate`</b>: If this is constant float value, no decay function is used.
    Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:

    ````python
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
    ````


*  <b>`clip_gradients`</b>: Clip norm of the gradients to this value to stop
    gradient explosion.
*  <b>`class_weight`</b>: None or list of n_classes floats. Weight associated with
    classes for loss computation. If not given, all classes are supposed to
    have weight one.
*  <b>`continue_training`</b>: when continue_training is True, once initialized
    model will be continuely trained on every call of fit.
*  <b>`config`</b>: RunConfig object that controls the configurations of the
    session, e.g. num_cores, gpu_memory_fraction, etc.
*  <b>`verbose`</b>: Controls the verbosity, possible values:

    * 0: the algorithm and debug information is muted.
    * 1: trainer prints the progress.
    * 2: log device placement is printed.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.__repr__()` {#TensorFlowEstimator.__repr__}




- - -

#### `tf.contrib.learn.TensorFlowEstimator.config` {#TensorFlowEstimator.config}




- - -

#### `tf.contrib.learn.TensorFlowEstimator.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#TensorFlowEstimator.evaluate}

Evaluates given model with provided evaluation data.

See superclass Estimator for more details.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called.
*  <b>`batch_size`</b>: minibatch size to use on the input.
*  <b>`steps`</b>: Number of steps for which to evaluate model.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metrics are used.
*  <b>`name`</b>: Name of the evaluation.

##### Returns:

  Returns `dict` with evaluation results.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.export(*args, **kwargs)` {#TensorFlowEstimator.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn and input_feature_key will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

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
        key into the features dict returned by `input_fn` that corresponds toa
        the raw `Example` strings `Tensor` that the exported model will take as
        input.
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

#### `tf.contrib.learn.TensorFlowEstimator.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowEstimator.fit}

Neural network model from provided `model_fn` and training data.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.
To restart learning, create new estimator.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class labels in classification, real numbers in regression).

*  <b>`steps`</b>: int, number of steps to train.
         If None or 0, train for `self.steps`.
*  <b>`monitors`</b>: List of `BaseMonitor` objects to print training progress and
    invoke early stopping.
*  <b>`logdir`</b>: the directory to save the log file that can be used for
  optional visualization.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_params(deep=True)` {#TensorFlowEstimator.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_tensor(name)` {#TensorFlowEstimator.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_variable_names()` {#TensorFlowEstimator.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_variable_value(name)` {#TensorFlowEstimator.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.model_dir` {#TensorFlowEstimator.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowEstimator.partial_fit(x, y)` {#TensorFlowEstimator.partial_fit}

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

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.predict(x, axis=1, batch_size=None)` {#TensorFlowEstimator.predict}

Predict class or regression for `x`.

For a classification model, the predicted class for each sample in `x` is
returned. For a regression model, the predicted value based on `x` is
returned.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
    By default axis 1 (next after batch) is used.
    Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member
    variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
  value.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.predict_proba(x, batch_size=None)` {#TensorFlowEstimator.predict_proba}

Predict class probability of the input samples `x`.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.restore(cls, path, config=None)` {#TensorFlowEstimator.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.

##### Raises:


*  <b>`ValueError`</b>: if `path` does not contain a model definition.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.save(path)` {#TensorFlowEstimator.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.set_params(**params)` {#TensorFlowEstimator.set_params}

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
education = sparse_column_with_hash_bucket(column_name="education",
                                           hash_bucket_size=1000)
occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)

education_x_occupation = crossed_column(columns=[education, occupation],
                                        hash_bucket_size=10000)

# Estimator using the default optimizer.
estimator = LinearClassifier(
    feature_columns=[occupation, education_x_occupation])

# Or estimator using the FTRL optimizer with regularization.
estimator = LinearClassifier(
    feature_columns=[occupation, education_x_occupation],
    optimizer=tf.train.FtrlOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Or estimator using the SDCAOptimizer.
estimator = LinearClassifier(
   feature_columns=[occupation, education_x_occupation],
   optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
     example_id_column='example_id',
     num_loss_partitions=...,
     symmetric_l2_regularization=2.0
   ))

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

#### `tf.contrib.learn.LinearClassifier.__init__(feature_columns, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None, gradient_clip_norm=None, enable_centered_bias=None, _joint_weight=False, config=None)` {#LinearClassifier.__init__}

Construct a `LinearClassifier` estimator object.

##### Args:


*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator
    to continue training a previously saved model.
*  <b>`n_classes`</b>: number of target classes. Default is binary classification.
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

##### Returns:

  A `LinearClassifier` estimator.

##### Raises:


*  <b>`ValueError`</b>: if n_classes < 2.


- - -

#### `tf.contrib.learn.LinearClassifier.bias_` {#LinearClassifier.bias_}




- - -

#### `tf.contrib.learn.LinearClassifier.config` {#LinearClassifier.config}




- - -

#### `tf.contrib.learn.LinearClassifier.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#LinearClassifier.evaluate}

See evaluable.Evaluable.


- - -

#### `tf.contrib.learn.LinearClassifier.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#LinearClassifier.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.LinearClassifier.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#LinearClassifier.fit}

See trainable.Trainable.


- - -

#### `tf.contrib.learn.LinearClassifier.get_estimator()` {#LinearClassifier.get_estimator}




- - -

#### `tf.contrib.learn.LinearClassifier.get_variable_names()` {#LinearClassifier.get_variable_names}




- - -

#### `tf.contrib.learn.LinearClassifier.get_variable_value(name)` {#LinearClassifier.get_variable_value}




- - -

#### `tf.contrib.learn.LinearClassifier.model_dir` {#LinearClassifier.model_dir}




- - -

#### `tf.contrib.learn.LinearClassifier.predict(x=None, input_fn=None, batch_size=None, as_iterable=False)` {#LinearClassifier.predict}

Runs inference to determine the predicted class.


- - -

#### `tf.contrib.learn.LinearClassifier.predict_proba(x=None, input_fn=None, batch_size=None, outputs=None, as_iterable=False)` {#LinearClassifier.predict_proba}

Runs inference to determine the class probability predictions.


- - -

#### `tf.contrib.learn.LinearClassifier.weights_` {#LinearClassifier.weights_}





- - -

### `class tf.contrib.learn.LinearRegressor` {#LinearRegressor}

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

#### `tf.contrib.learn.LinearRegressor.__init__(feature_columns, model_dir=None, weight_column_name=None, optimizer=None, gradient_clip_norm=None, enable_centered_bias=None, target_dimension=1, _joint_weights=False, config=None)` {#LinearRegressor.__init__}

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
*  <b>`target_dimension`</b>: dimension of the target for multilabels.
  _joint_weights: If True use a single (possibly partitioned) variable to
    store the weights. It's faster, but requires all feature columns are
    sparse and have the 'sum' combiner. Incompatible with SDCAOptimizer.

*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.

##### Returns:

  A `LinearRegressor` estimator.


- - -

#### `tf.contrib.learn.LinearRegressor.__repr__()` {#LinearRegressor.__repr__}




- - -

#### `tf.contrib.learn.LinearRegressor.bias_` {#LinearRegressor.bias_}




- - -

#### `tf.contrib.learn.LinearRegressor.config` {#LinearRegressor.config}




- - -

#### `tf.contrib.learn.LinearRegressor.dnn_bias_` {#LinearRegressor.dnn_bias_}

Returns bias of deep neural network part.


- - -

#### `tf.contrib.learn.LinearRegressor.dnn_weights_` {#LinearRegressor.dnn_weights_}

Returns weights of deep neural network part.


- - -

#### `tf.contrib.learn.LinearRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#LinearRegressor.evaluate}

See `Evaluable`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.LinearRegressor.export(*args, **kwargs)` {#LinearRegressor.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn and input_feature_key will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

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
        key into the features dict returned by `input_fn` that corresponds toa
        the raw `Example` strings `Tensor` that the exported model will take as
        input.
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

#### `tf.contrib.learn.LinearRegressor.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#LinearRegressor.fit}

See `Trainable`.

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

#### `tf.contrib.learn.LinearRegressor.predict(*args, **kwargs)` {#LinearRegressor.predict}

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





- - -

### `class tf.contrib.learn.TensorFlowRNNClassifier` {#TensorFlowRNNClassifier}

TensorFlow RNN Classifier model.
- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.__init__(rnn_size, n_classes, cell_type='gru', num_layers=1, input_op_fn=null_input_op_fn, initial_state=None, bidirectional=False, sequence_length=None, attn_length=None, attn_size=None, attn_vec_size=None, batch_size=32, steps=50, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRNNClassifier.__init__}

Initializes a TensorFlowRNNClassifier instance.

##### Args:


*  <b>`rnn_size`</b>: The size for rnn cell, e.g. size of your word embeddings.
*  <b>`cell_type`</b>: The type of rnn cell, including rnn, gru, and lstm.
*  <b>`num_layers`</b>: The number of layers of the rnn model.
*  <b>`input_op_fn`</b>: Function that will transform the input tensor, such as
    creating word embeddings, byte list, etc. This takes
    an argument x for input and returns transformed x.
*  <b>`bidirectional`</b>: boolean, Whether this is a bidirectional rnn.
*  <b>`sequence_length`</b>: If sequence_length is provided, dynamic calculation
    is performed. This saves computational time when unrolling past max
    sequence length.
*  <b>`initial_state`</b>: An initial state for the RNN. This must be a tensor of
    appropriate type and shape [batch_size x cell.state_size].
*  <b>`attn_length`</b>: integer, the size of attention vector attached to rnn cells.
*  <b>`attn_size`</b>: integer, the size of an attention window attached to rnn cells.
*  <b>`attn_vec_size`</b>: integer, the number of convolutional features calculated on
    attention state and the size of the hidden layer built from base cell state.
*  <b>`n_classes`</b>: Number of classes in the target.
*  <b>`batch_size`</b>: Mini batch size.
*  <b>`steps`</b>: Number of steps to run over data.
*  <b>`optimizer`</b>: Optimizer name (or class), for example "SGD", "Adam",
    "Adagrad".
*  <b>`learning_rate`</b>: If this is constant float value, no decay function is
    used. Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:

    ````python
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
    ````


*  <b>`class_weight`</b>: None or list of n_classes floats. Weight associated with
    classes for loss computation. If not given, all classes are
    supposed to have weight one.
*  <b>`continue_training`</b>: when continue_training is True, once initialized
    model will be continuely trained on every call of fit.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.__repr__()` {#TensorFlowRNNClassifier.__repr__}




- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.bias_` {#TensorFlowRNNClassifier.bias_}

Returns bias of the rnn layer.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.config` {#TensorFlowRNNClassifier.config}




- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#TensorFlowRNNClassifier.evaluate}

Evaluates given model with provided evaluation data.

See superclass Estimator for more details.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called.
*  <b>`batch_size`</b>: minibatch size to use on the input.
*  <b>`steps`</b>: Number of steps for which to evaluate model.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metrics are used.
*  <b>`name`</b>: Name of the evaluation.

##### Returns:

  Returns `dict` with evaluation results.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.export(*args, **kwargs)` {#TensorFlowRNNClassifier.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn and input_feature_key will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

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
        key into the features dict returned by `input_fn` that corresponds toa
        the raw `Example` strings `Tensor` that the exported model will take as
        input.
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

#### `tf.contrib.learn.TensorFlowRNNClassifier.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowRNNClassifier.fit}

Neural network model from provided `model_fn` and training data.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.
To restart learning, create new estimator.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class labels in classification, real numbers in regression).

*  <b>`steps`</b>: int, number of steps to train.
         If None or 0, train for `self.steps`.
*  <b>`monitors`</b>: List of `BaseMonitor` objects to print training progress and
    invoke early stopping.
*  <b>`logdir`</b>: the directory to save the log file that can be used for
  optional visualization.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_params(deep=True)` {#TensorFlowRNNClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_tensor(name)` {#TensorFlowRNNClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_variable_names()` {#TensorFlowRNNClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_variable_value(name)` {#TensorFlowRNNClassifier.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.model_dir` {#TensorFlowRNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.partial_fit(x, y)` {#TensorFlowRNNClassifier.partial_fit}

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

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.predict(x, axis=1, batch_size=None)` {#TensorFlowRNNClassifier.predict}

Predict class or regression for `x`.

For a classification model, the predicted class for each sample in `x` is
returned. For a regression model, the predicted value based on `x` is
returned.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
    By default axis 1 (next after batch) is used.
    Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member
    variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
  value.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.predict_proba(x, batch_size=None)` {#TensorFlowRNNClassifier.predict_proba}

Predict class probability of the input samples `x`.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.restore(cls, path, config=None)` {#TensorFlowRNNClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.

##### Raises:


*  <b>`ValueError`</b>: if `path` does not contain a model definition.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.save(path)` {#TensorFlowRNNClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.set_params(**params)` {#TensorFlowRNNClassifier.set_params}

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

#### `tf.contrib.learn.TensorFlowRNNClassifier.weights_` {#TensorFlowRNNClassifier.weights_}

Returns weights of the rnn layer.



- - -

### `class tf.contrib.learn.TensorFlowRNNRegressor` {#TensorFlowRNNRegressor}

TensorFlow RNN Regressor model.
- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.__init__(rnn_size, cell_type='gru', num_layers=1, input_op_fn=null_input_op_fn, initial_state=None, bidirectional=False, sequence_length=None, attn_length=None, attn_size=None, attn_vec_size=None, n_classes=0, batch_size=32, steps=50, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRNNRegressor.__init__}

Initializes a TensorFlowRNNRegressor instance.

##### Args:


*  <b>`rnn_size`</b>: The size for rnn cell, e.g. size of your word embeddings.
*  <b>`cell_type`</b>: The type of rnn cell, including rnn, gru, and lstm.
*  <b>`num_layers`</b>: The number of layers of the rnn model.
*  <b>`input_op_fn`</b>: Function that will transform the input tensor, such as
    creating word embeddings, byte list, etc. This takes
    an argument x for input and returns transformed x.
*  <b>`bidirectional`</b>: boolean, Whether this is a bidirectional rnn.
*  <b>`sequence_length`</b>: If sequence_length is provided, dynamic calculation
    is performed. This saves computational time when unrolling past max
    sequence length.
*  <b>`attn_length`</b>: integer, the size of attention vector attached to rnn cells.
*  <b>`attn_size`</b>: integer, the size of an attention window attached to rnn cells.
*  <b>`attn_vec_size`</b>: integer, the number of convolutional features calculated on
    attention state and the size of the hidden layer built from base cell state.
*  <b>`initial_state`</b>: An initial state for the RNN. This must be a tensor of
    appropriate type and shape [batch_size x cell.state_size].
*  <b>`batch_size`</b>: Mini batch size.
*  <b>`steps`</b>: Number of steps to run over data.
*  <b>`optimizer`</b>: Optimizer name (or class), for example "SGD", "Adam",
    "Adagrad".
*  <b>`learning_rate`</b>: If this is constant float value, no decay function is
    used. Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:

    ````python
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
    ````


*  <b>`continue_training`</b>: when continue_training is True, once initialized
    model will be continuely trained on every call of fit.
*  <b>`config`</b>: RunConfig object that controls the configurations of the
    session, e.g. num_cores, gpu_memory_fraction, etc.
*  <b>`verbose`</b>: Controls the verbosity, possible values:

    * 0: the algorithm and debug information is muted.
    * 1: trainer prints the progress.
    * 2: log device placement is printed.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.__repr__()` {#TensorFlowRNNRegressor.__repr__}




- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.bias_` {#TensorFlowRNNRegressor.bias_}

Returns bias of the rnn layer.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.config` {#TensorFlowRNNRegressor.config}




- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#TensorFlowRNNRegressor.evaluate}

Evaluates given model with provided evaluation data.

See superclass Estimator for more details.

##### Args:


*  <b>`x`</b>: features.
*  <b>`y`</b>: targets.
*  <b>`input_fn`</b>: Input function.
*  <b>`feed_fn`</b>: Function creating a feed dict every time it is called.
*  <b>`batch_size`</b>: minibatch size to use on the input.
*  <b>`steps`</b>: Number of steps for which to evaluate model.
*  <b>`metrics`</b>: Dict of metric ops to run. If None, the default metrics are used.
*  <b>`name`</b>: Name of the evaluation.

##### Returns:

  Returns `dict` with evaluation results.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.export(*args, **kwargs)` {#TensorFlowRNNRegressor.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn and input_feature_key will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

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
        key into the features dict returned by `input_fn` that corresponds toa
        the raw `Example` strings `Tensor` that the exported model will take as
        input.
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

#### `tf.contrib.learn.TensorFlowRNNRegressor.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowRNNRegressor.fit}

Neural network model from provided `model_fn` and training data.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.
To restart learning, create new estimator.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class labels in classification, real numbers in regression).

*  <b>`steps`</b>: int, number of steps to train.
         If None or 0, train for `self.steps`.
*  <b>`monitors`</b>: List of `BaseMonitor` objects to print training progress and
    invoke early stopping.
*  <b>`logdir`</b>: the directory to save the log file that can be used for
  optional visualization.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_params(deep=True)` {#TensorFlowRNNRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_tensor(name)` {#TensorFlowRNNRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_variable_names()` {#TensorFlowRNNRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_variable_value(name)` {#TensorFlowRNNRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.model_dir` {#TensorFlowRNNRegressor.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.partial_fit(x, y)` {#TensorFlowRNNRegressor.partial_fit}

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

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.predict(x, axis=1, batch_size=None)` {#TensorFlowRNNRegressor.predict}

Predict class or regression for `x`.

For a classification model, the predicted class for each sample in `x` is
returned. For a regression model, the predicted value based on `x` is
returned.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
    By default axis 1 (next after batch) is used.
    Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member
    variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
  value.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.predict_proba(x, batch_size=None)` {#TensorFlowRNNRegressor.predict_proba}

Predict class probability of the input samples `x`.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.restore(cls, path, config=None)` {#TensorFlowRNNRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.

##### Raises:


*  <b>`ValueError`</b>: if `path` does not contain a model definition.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.save(path)` {#TensorFlowRNNRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.set_params(**params)` {#TensorFlowRNNRegressor.set_params}

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

#### `tf.contrib.learn.TensorFlowRNNRegressor.weights_` {#TensorFlowRNNRegressor.weights_}

Returns weights of the rnn layer.




## Graph actions

Perform various training, evaluation, and inference actions on a graph.

- - -

### `class tf.contrib.learn.NanLossDuringTrainingError` {#NanLossDuringTrainingError}


- - -

#### `tf.contrib.learn.NanLossDuringTrainingError.__str__()` {#NanLossDuringTrainingError.__str__}





- - -

### `class tf.contrib.learn.RunConfig` {#RunConfig}

This class specifies the specific configurations for the run.

If you're a Google-internal user using command line flags with learn_runner.py
(for instance, to do distributed training or to use parameter servers), you
probably want to use learn_runner.EstimatorConfig instead.
- - -

#### `tf.contrib.learn.RunConfig.__init__(master=None, task=None, num_ps_replicas=None, num_cores=0, log_device_placement=False, gpu_memory_fraction=1, cluster_spec=None, tf_random_seed=None, save_summary_steps=100, save_checkpoints_secs=600, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000, job_name=None, is_chief=None, evaluation_master='')` {#RunConfig.__init__}

Constructor.

If set to None, `master`, `task`, `num_ps_replicas`, `cluster_spec`,
`job_name`, and `is_chief` are set based on the TF_CONFIG environment
variable, if the pertinent information is present; otherwise, the defaults
listed in the Args section apply.

The TF_CONFIG environment variable is a JSON object with two relevant
attributes: `task` and `cluster_spec`. `cluster_spec` is a JSON serialized
version of the Python dict described in server_lib.py. `task` has two
attributes: `type` and `index`, where `type` can be any of the task types
in the cluster_spec. When TF_CONFIG contains said information, the
following properties are set on this class:

  * `job_name` is set to [`task`][`type`]
  * `task` is set to [`task`][`index`]
  * `cluster_spec` is parsed from [`cluster`]
  * 'master' is determined by looking up `job_name` and `task` in the
    cluster_spec.
  * `num_ps_replicas` is set by counting the number of nodes listed
    in the `ps` job of `cluster_spec`.
  * `is_chief`: true when `job_name` == "master" and `task` == 0.

Example:
```
  cluster = {'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
  os.environ['TF_CONFIG'] = json.dumps({
      {'cluster': cluster,
       'task': {'type': 'worker', 'index': 1}}})
  config = RunConfig()
  assert config.master == 'host4:2222'
  assert config.task == 1
  assert config.num_ps_replicas == 2
  assert config.cluster_spec == server_lib.ClusterSpec(cluster)
  assert config.job_name == 'worker'
  assert not config.is_chief
```

##### Args:


*  <b>`master`</b>: TensorFlow master. Defaults to empty string for local.
*  <b>`task`</b>: Task id of the replica running the training (default: 0).
*  <b>`num_ps_replicas`</b>: Number of parameter server tasks to use (default: 0).
*  <b>`num_cores`</b>: Number of cores to be used. If 0, the system picks an
    appropriate number (default: 0).
*  <b>`log_device_placement`</b>: Log the op placement to devices (default: False).
*  <b>`gpu_memory_fraction`</b>: Fraction of GPU memory used by the process on
    each GPU uniformly on the same machine.
*  <b>`cluster_spec`</b>: a `tf.train.ClusterSpec` object that describes the cluster
    in the case of distributed computation. If missing, reasonable
    assumptions are made for the addresses of jobs.
*  <b>`tf_random_seed`</b>: Random seed for TensorFlow initializers.
    Setting this value allows consistency between reruns.
*  <b>`save_summary_steps`</b>: Save summaries every this many steps.
*  <b>`save_checkpoints_secs`</b>: Save checkpoints every this many seconds.
*  <b>`keep_checkpoint_max`</b>: The maximum number of recent checkpoint files to
    keep. As new files are created, older files are deleted. If None or 0,
    all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent
    checkpoint files are kept.)
*  <b>`keep_checkpoint_every_n_hours`</b>: Number of hours between each checkpoint
    to be saved. The default value of 10,000 hours effectively disables
    the feature.
*  <b>`job_name`</b>: the type of task, e.g., 'ps', 'worker', etc. The `job_name`
    must exist in the `cluster_spec.jobs`.
*  <b>`is_chief`</b>: whether or not this task (as identified by the other parameters)
    should be the chief task.
*  <b>`evaluation_master`</b>: the master on which to perform evaluation.

##### Raises:


*  <b>`ValueError`</b>: if num_ps_replicas and cluster_spec are set (cluster_spec
    may fome from the TF_CONFIG environment variable).


- - -

#### `tf.contrib.learn.RunConfig.is_chief` {#RunConfig.is_chief}




- - -

#### `tf.contrib.learn.RunConfig.job_name` {#RunConfig.job_name}





- - -

### `tf.contrib.learn.evaluate(graph, output_dir, checkpoint_path, eval_dict, update_op=None, global_step_tensor=None, supervisor_master='', log_every_steps=10, feed_fn=None, max_steps=None)` {#evaluate}

Evaluate a model loaded from a checkpoint.

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
    end-of-inupt exception when the inputs are exhausted.
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

### `tf.contrib.learn.infer(restore_checkpoint_path, output_dict, feed_dict=None)` {#infer}

Restore graph from `restore_checkpoint_path` and run `output_dict` tensors.

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

See run_feeds_iter(). Returns a `list` instead of an iterator.


- - -

### `tf.contrib.learn.run_n(output_dict, feed_dict=None, restore_checkpoint_path=None, n=1)` {#run_n}

Run `output_dict` tensors `n` times, with the same `feed_dict` each run.

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

### `tf.contrib.learn.train(graph, output_dir, train_op, loss_op, global_step_tensor=None, init_op=None, init_feed_dict=None, init_fn=None, log_every_steps=10, supervisor_is_chief=True, supervisor_master='', supervisor_save_model_secs=600, keep_checkpoint_max=5, supervisor_save_summaries_steps=100, feed_fn=None, steps=None, fail_on_nan_loss=True, monitors=None, max_steps=None)` {#train}

Train a model.

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



## Input processing

Queue and read batched input data.

- - -

### `tf.contrib.learn.extract_dask_data(data)` {#extract_dask_data}

Extract data from dask.Series or dask.DataFrame for predictors.


- - -

### `tf.contrib.learn.extract_dask_labels(labels)` {#extract_dask_labels}

Extract data from dask.Series for labels.


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

### `tf.contrib.learn.read_batch_examples(file_pattern, batch_size, reader, randomize_input=True, num_epochs=None, queue_capacity=10000, num_threads=1, read_batch_size=1, parse_fn=None, name=None)` {#read_batch_examples}

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
    `tf.initialize_all_variables()` as shown in the tests.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`num_threads`</b>: The number of threads enqueuing examples.
*  <b>`read_batch_size`</b>: An int or scalar `Tensor` specifying the number of
    records to read at once
*  <b>`parse_fn`</b>: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  String `Tensor` of batched `Example` proto.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.


- - -

### `tf.contrib.learn.read_batch_features(file_pattern, batch_size, features, reader, randomize_input=True, num_epochs=None, queue_capacity=10000, feature_queue_capacity=100, reader_num_threads=1, parser_num_threads=1, parse_fn=None, name=None)` {#read_batch_features}

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
    tf.initialize_local_variables() as shown in the tests.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`feature_queue_capacity`</b>: Capacity of the parsed features queue. Set this
    value to a small number, for example 5 if the parsed features are large.
*  <b>`reader_num_threads`</b>: The number of threads to read examples.
*  <b>`parser_num_threads`</b>: The number of threads to parse examples.
    records to read at once
*  <b>`parse_fn`</b>: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.


- - -

### `tf.contrib.learn.read_batch_record_features(file_pattern, batch_size, features, randomize_input=True, num_epochs=None, queue_capacity=10000, reader_num_threads=1, parser_num_threads=1, name='dequeue_record_examples')` {#read_batch_record_features}

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
    tf.initialize_local_variables() as shown in the tests.
*  <b>`queue_capacity`</b>: Capacity for input queue.
*  <b>`reader_num_threads`</b>: The number of threads to read examples.
*  <b>`parser_num_threads`</b>: The number of threads to parse examples.
*  <b>`name`</b>: Name of resulting op.

##### Returns:

  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

##### Raises:


*  <b>`ValueError`</b>: for invalid inputs.


