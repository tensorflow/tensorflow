An Estimator fo rK-Means clustering.
- - -

#### `tf.contrib.learn.KMeansClustering.__init__(num_clusters, model_dir=None, initial_clusters='random', distance_metric='squared_euclidean', random_seed=0, use_mini_batch=True, kmeans_plus_plus_num_retries=2, relative_tolerance=None, config=None)` {#KMeansClustering.__init__}

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


