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
*  <b>`label_dimension`</b>: Dimension of the label for multilabels. Defaults to 1.
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

#### `tf.contrib.learn.DNNRegressor.export_savedmodel(export_dir_base, serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False)` {#DNNRegressor.export_savedmodel}

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


