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




