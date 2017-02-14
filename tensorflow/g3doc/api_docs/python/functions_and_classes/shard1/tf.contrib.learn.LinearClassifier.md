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


