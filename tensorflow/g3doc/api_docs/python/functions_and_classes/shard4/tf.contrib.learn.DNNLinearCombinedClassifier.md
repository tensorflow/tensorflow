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


